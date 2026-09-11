/* dit.c — timestep conditioner and latent input projection for Irodori RF-DiT. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dit.h"
#include "ops.h"

enum {
    TIMESTEP_DIM = 512,
    MODEL_DIM = 1280,
    COND_DIM = 3 * MODEL_DIM,
    LATENT_DIM = 32,
    ADALN_RANK = 192,
    TEXT_DIM = 512,
    SPEAKER_DIM = 768,
    MLP_DIM = 3680,
};

static const float *require_f32(const IroSafetensors *st, const char *name,
                                int ndims, uint64_t d0, uint64_t d1) {
    const IroTensor *t = iro_st_get(st, name);
    if (!t) {
        fprintf(stderr, "dit: tensor tidak ada: %s\n", name);
        return NULL;
    }
    if (t->dtype != IRO_ST_F32 || t->ndims != ndims || t->shape[0] != d0 ||
        (ndims == 2 && t->shape[1] != d1)) {
        fprintf(stderr, "dit: dtype/shape tidak cocok: %s\n", name);
        return NULL;
    }
    return (const float *)t->data;
}

int iro_dit_init(IroDiT *d, const IroSafetensors *st) {
    if (!d || !st) return -1;
    memset(d, 0, sizeof(*d));
    d->cond_w0 = require_f32(st, "cond_module.0.weight", 2,
                             MODEL_DIM, TIMESTEP_DIM);
    d->cond_w1 = require_f32(st, "cond_module.2.weight", 2,
                             MODEL_DIM, MODEL_DIM);
    d->cond_w2 = require_f32(st, "cond_module.4.weight", 2,
                             COND_DIM, MODEL_DIM);
    d->in_w = require_f32(st, "in_proj.weight", 2, MODEL_DIM, LATENT_DIM);
    d->in_b = require_f32(st, "in_proj.bias", 1, MODEL_DIM, 0);
    d->out_norm = require_f32(st, "out_norm.weight", 1, MODEL_DIM, 0);
    d->out_w = require_f32(st, "out_proj.weight", 2, LATENT_DIM, MODEL_DIM);
    d->out_b = require_f32(st, "out_proj.bias", 1, LATENT_DIM, 0);
    d->norm_eps = 1e-5f;

    for (int layer = 0; layer < IRO_DIT_LAYERS; layer++) {
        IroAdaLN *sets[2] = { &d->attention_adaln[layer], &d->mlp_adaln[layer] };
        const char *parts[2] = { "attention_adaln", "mlp_adaln" };
        for (int set = 0; set < 2; set++) {
            IroAdaLN *a = sets[set];
            char name[128];
#define LOAD_ADALN(field, suffix, d0, d1) do {                                \
                snprintf(name, sizeof(name), "blocks.%d.%s.%s",              \
                         layer, parts[set], suffix);                            \
                a->field = require_f32(st, name, (d1) ? 2 : 1, d0, d1);        \
            } while (0)
            LOAD_ADALN(shift_down, "shift_down.weight", ADALN_RANK, MODEL_DIM);
            LOAD_ADALN(shift_up, "shift_up.weight", MODEL_DIM, ADALN_RANK);
            LOAD_ADALN(shift_bias, "shift_up.bias", MODEL_DIM, 0);
            LOAD_ADALN(scale_down, "scale_down.weight", ADALN_RANK, MODEL_DIM);
            LOAD_ADALN(scale_up, "scale_up.weight", MODEL_DIM, ADALN_RANK);
            LOAD_ADALN(scale_bias, "scale_up.bias", MODEL_DIM, 0);
            LOAD_ADALN(gate_down, "gate_down.weight", ADALN_RANK, MODEL_DIM);
            LOAD_ADALN(gate_up, "gate_up.weight", MODEL_DIM, ADALN_RANK);
            LOAD_ADALN(gate_bias, "gate_up.bias", MODEL_DIM, 0);
#undef LOAD_ADALN
        }

        IroJointAttention *a = &d->attention[layer];
        char name[128];
#define LOAD_ATTN(field, suffix, d0, d1) do {                                 \
            snprintf(name, sizeof(name), "blocks.%d.attention.%s",           \
                     layer, suffix);                                           \
            a->field = require_f32(st, name, 2, d0, d1);                       \
        } while (0)
        LOAD_ATTN(wq, "wq.weight", MODEL_DIM, MODEL_DIM);
        LOAD_ATTN(wk, "wk.weight", MODEL_DIM, MODEL_DIM);
        LOAD_ATTN(wv, "wv.weight", MODEL_DIM, MODEL_DIM);
        LOAD_ATTN(wo, "wo.weight", MODEL_DIM, MODEL_DIM);
        LOAD_ATTN(gate, "gate.weight", MODEL_DIM, MODEL_DIM);
        LOAD_ATTN(q_norm, "q_norm.weight", IRO_DIT_HEADS, IRO_DIT_HEAD_DIM);
        LOAD_ATTN(k_norm, "k_norm.weight", IRO_DIT_HEADS, IRO_DIT_HEAD_DIM);
        LOAD_ATTN(wk_text, "wk_text.weight", MODEL_DIM, TEXT_DIM);
        LOAD_ATTN(wv_text, "wv_text.weight", MODEL_DIM, TEXT_DIM);
        LOAD_ATTN(wk_speaker, "wk_speaker.weight", MODEL_DIM, SPEAKER_DIM);
        LOAD_ATTN(wv_speaker, "wv_speaker.weight", MODEL_DIM, SPEAKER_DIM);
        LOAD_ATTN(wk_caption, "wk_caption.weight", MODEL_DIM, TEXT_DIM);
        LOAD_ATTN(wv_caption, "wv_caption.weight", MODEL_DIM, TEXT_DIM);
#undef LOAD_ATTN

        IroSwiGLU *mlp = &d->mlp[layer];
#define LOAD_MLP(field, suffix, d0, d1) do {                                  \
            snprintf(name, sizeof(name), "blocks.%d.mlp.%s", layer, suffix); \
            mlp->field = require_f32(st, name, 2, d0, d1);                    \
        } while (0)
        LOAD_MLP(w1, "w1.weight", MLP_DIM, MODEL_DIM);
        LOAD_MLP(w2, "w2.weight", MODEL_DIM, MLP_DIM);
        LOAD_MLP(w3, "w3.weight", MLP_DIM, MODEL_DIM);
#undef LOAD_MLP
    }

    if (!d->cond_w0 || !d->cond_w1 || !d->cond_w2 || !d->in_w || !d->in_b ||
        !d->out_norm || !d->out_w || !d->out_b)
        return -1;
    for (int layer = 0; layer < IRO_DIT_LAYERS; layer++) {
        const IroAdaLN *sets[2] = {
            &d->attention_adaln[layer], &d->mlp_adaln[layer]
        };
        for (int set = 0; set < 2; set++) {
            const IroAdaLN *a = sets[set];
            if (!a->shift_down || !a->shift_up || !a->shift_bias ||
                !a->scale_down || !a->scale_up || !a->scale_bias ||
                !a->gate_down || !a->gate_up || !a->gate_bias)
                return -1;
        }
        const IroJointAttention *a = &d->attention[layer];
        if (!a->wq || !a->wk || !a->wv || !a->wo || !a->gate ||
            !a->q_norm || !a->k_norm || !a->wk_text || !a->wv_text ||
            !a->wk_speaker || !a->wv_speaker ||
            !a->wk_caption || !a->wv_caption)
            return -1;
        const IroSwiGLU *mlp = &d->mlp[layer];
        if (!mlp->w1 || !mlp->w2 || !mlp->w3) return -1;
    }
    return 0;
}

static void timestep_embedding(const float *t, int B, float *embedding) {
    const int half = TIMESTEP_DIM / 2;
    const float log_theta = logf(10000.0f);
    for (int b = 0; b < B; b++) {
        float *row = embedding + (size_t)b * TIMESTEP_DIM;
        for (int i = 0; i < half; i++) {
            float freq = 1000.0f * expf(-log_theta * (float)i / (float)half);
            float arg = t[b] * freq;
            row[i] = cosf(arg);
            row[half + i] = sinf(arg);
        }
    }
}

int iro_dit_condition(const IroDiT *d, const float *t, int B, float *cond) {
    if (!d || !t || B <= 0 || !cond) return -1;
    float *embedding = malloc(sizeof(float) * (size_t)B * TIMESTEP_DIM);
    float *hidden0 = malloc(sizeof(float) * (size_t)B * MODEL_DIM);
    float *hidden1 = malloc(sizeof(float) * (size_t)B * MODEL_DIM);
    if (!embedding || !hidden0 || !hidden1) {
        free(embedding); free(hidden0); free(hidden1);
        return -1;
    }
    timestep_embedding(t, B, embedding);
    iro_linear(embedding, d->cond_w0, NULL, hidden0,
               B, TIMESTEP_DIM, MODEL_DIM);
    iro_silu_inplace(hidden0, (size_t)B * MODEL_DIM);
    iro_linear(hidden0, d->cond_w1, NULL, hidden1,
               B, MODEL_DIM, MODEL_DIM);
    iro_silu_inplace(hidden1, (size_t)B * MODEL_DIM);
    iro_linear(hidden1, d->cond_w2, NULL, cond, B, MODEL_DIM, COND_DIM);
    free(embedding); free(hidden0); free(hidden1);
    return 0;
}

int iro_dit_input(const IroDiT *d, const float *x, int B, int S, float *projected) {
    if (!d || !x || B <= 0 || S <= 0 || !projected) return -1;
    iro_linear(x, d->in_w, d->in_b, projected, B * S, LATENT_DIM, MODEL_DIM);
    return 0;
}

static void adaln_branch(const float *cond, int branch, int B,
                         const float *down, const float *up, const float *bias,
                         float *act, float *low, float *output) {
    for (int b = 0; b < B; b++) {
        const float *input = cond + (size_t)b * COND_DIM + branch * MODEL_DIM;
        float *act_row = act + (size_t)b * MODEL_DIM;
        memcpy(act_row, input, sizeof(float) * MODEL_DIM);
        iro_silu_inplace(act_row, MODEL_DIM);
    }
    iro_linear(act, down, NULL, low, B, MODEL_DIM, ADALN_RANK);
    iro_linear(low, up, bias, output, B, ADALN_RANK, MODEL_DIM);
    for (int b = 0; b < B; b++) {
        const float *input = cond + (size_t)b * COND_DIM + branch * MODEL_DIM;
        float *output_row = output + (size_t)b * MODEL_DIM;
        for (int d = 0; d < MODEL_DIM; d++) output_row[d] += input[d];
    }
}

int iro_dit_adaln(const IroAdaLN *a, float eps, const float *x,
                  const float *cond, int B, int S, float *h, float *gate) {
    if (!a || !x || !cond || B <= 0 || S <= 0 || !h || !gate) return -1;
    size_t branch_n = (size_t)B * MODEL_DIM;
    float *act = malloc(sizeof(float) * branch_n);
    float *low = malloc(sizeof(float) * (size_t)B * ADALN_RANK);
    float *shift = malloc(sizeof(float) * branch_n);
    float *scale = malloc(sizeof(float) * branch_n);
    if (!act || !low || !shift || !scale) {
        free(act); free(low); free(shift); free(scale);
        return -1;
    }
    adaln_branch(cond, 0, B, a->shift_down, a->shift_up, a->shift_bias,
                 act, low, shift);
    adaln_branch(cond, 1, B,
                 a->scale_down, a->scale_up, a->scale_bias,
                 act, low, scale);
    adaln_branch(cond, 2, B,
                 a->gate_down, a->gate_up, a->gate_bias,
                 act, low, gate);
    for (size_t i = 0; i < branch_n; i++) gate[i] = tanhf(gate[i]);

    for (int b = 0; b < B; b++) {
        for (int s = 0; s < S; s++) {
            const float *row = x + ((size_t)b * S + s) * MODEL_DIM;
            float *out = h + ((size_t)b * S + s) * MODEL_DIM;
            float ms = 0.0f;
            for (int d = 0; d < MODEL_DIM; d++) ms += row[d] * row[d];
            float inv = 1.0f / sqrtf(ms / MODEL_DIM + eps);
            const float *shift_row = shift + (size_t)b * MODEL_DIM;
            const float *scale_row = scale + (size_t)b * MODEL_DIM;
            for (int d = 0; d < MODEL_DIM; d++)
                out[d] = row[d] * inv * (1.0f + scale_row[d]) + shift_row[d];
        }
    }
    free(act); free(low); free(shift); free(scale);
    return 0;
}

static void head_rmsnorm(float *x, const float *weight, int B, int S,
                         float eps) {
    for (int b = 0; b < B; b++) {
        for (int s = 0; s < S; s++) {
            float *token = x + ((size_t)b * S + s) * MODEL_DIM;
            for (int h = 0; h < IRO_DIT_HEADS; h++) {
                float *row = token + h * IRO_DIT_HEAD_DIM;
                const float *w = weight + h * IRO_DIT_HEAD_DIM;
                float ms = 0.0f;
                for (int d = 0; d < IRO_DIT_HEAD_DIM; d++)
                    ms += row[d] * row[d];
                float inv = 1.0f / sqrtf(ms / IRO_DIT_HEAD_DIM + eps);
                for (int d = 0; d < IRO_DIT_HEAD_DIM; d++)
                    row[d] *= inv * w[d];
            }
        }
    }
}

static void head_rmsnorm_rope_pair(float *q, const float *q_weight,
                                   float *k, const float *k_weight,
                                   int B, int S, float eps) {
    float cos_table[IRO_DIT_HEAD_DIM / 2];
    float sin_table[IRO_DIT_HEAD_DIM / 2];
    double cos_step[IRO_DIT_HEAD_DIM / 2];
    double sin_step[IRO_DIT_HEAD_DIM / 2];
    double cos_pos[IRO_DIT_HEAD_DIM / 2];
    double sin_pos[IRO_DIT_HEAD_DIM / 2];
    for (int pair = 0; pair < IRO_DIT_HEAD_DIM / 2; pair++) {
        double inv = 1.0 / pow(10000.0,
                               (double)(2 * pair) / IRO_DIT_HEAD_DIM);
        cos_step[pair] = cos(inv);
        sin_step[pair] = sin(inv);
        cos_pos[pair] = 1.0;
        sin_pos[pair] = 0.0;
    }
    for (int s = 0; s < S; s++) {
        for (int pair = 0; pair < IRO_DIT_HEAD_DIM / 2; pair++) {
            cos_table[pair] = (float)cos_pos[pair];
            sin_table[pair] = (float)sin_pos[pair];
        }
        for (int b = 0; b < B; b++) {
            float *q_token = q + ((size_t)b * S + s) * MODEL_DIM;
            float *k_token = k + ((size_t)b * S + s) * MODEL_DIM;
            for (int h = 0; h < IRO_DIT_HEADS; h++) {
                float *q_row = q_token + h * IRO_DIT_HEAD_DIM;
                float *k_row = k_token + h * IRO_DIT_HEAD_DIM;
                const float *qw = q_weight + h * IRO_DIT_HEAD_DIM;
                const float *kw = k_weight + h * IRO_DIT_HEAD_DIM;
                float q_ms = 0.0f, k_ms = 0.0f;
                for (int d = 0; d < IRO_DIT_HEAD_DIM; d++) {
                    q_ms += q_row[d] * q_row[d];
                    k_ms += k_row[d] * k_row[d];
                }
                float q_inv = 1.0f / sqrtf(q_ms / IRO_DIT_HEAD_DIM + eps);
                float k_inv = 1.0f / sqrtf(k_ms / IRO_DIT_HEAD_DIM + eps);

                if (h >= IRO_DIT_HEADS / 2) {
                    for (int d = 0; d < IRO_DIT_HEAD_DIM; d++) {
                        q_row[d] *= q_inv * qw[d];
                        k_row[d] *= k_inv * kw[d];
                    }
                    continue;
                }

                /* Upstream chunks on dimension -2: only the first half of heads. */
                for (int pair = 0; pair < IRO_DIT_HEAD_DIM / 2; pair++) {
                    int d = 2 * pair;
                    float q0 = q_row[d] * (q_inv * qw[d]);
                    float q1 = q_row[d + 1] * (q_inv * qw[d + 1]);
                    float k0 = k_row[d] * (k_inv * kw[d]);
                    float k1 = k_row[d + 1] * (k_inv * kw[d + 1]);
                    float c = cos_table[pair], sn = sin_table[pair];
                    q_row[d] = q0 * c - q1 * sn;
                    q_row[d + 1] = q0 * sn + q1 * c;
                    k_row[d] = k0 * c - k1 * sn;
                    k_row[d + 1] = k0 * sn + k1 * c;
                }
            }
        }
        for (int pair = 0; pair < IRO_DIT_HEAD_DIM / 2; pair++) {
            double c = cos_pos[pair];
            double sn = sin_pos[pair];
            cos_pos[pair] = c * cos_step[pair] - sn * sin_step[pair];
            sin_pos[pair] = sn * cos_step[pair] + c * sin_step[pair];
        }
    }
}

int iro_dit_text_kv(const IroJointAttention *a, float eps,
                    const float *text, int B, int T, float *k, float *v) {
    if (!a || !text || !k || !v || B <= 0 || T <= 0) return -1;
    iro_linear(text, a->wk_text, NULL, k, B * T, TEXT_DIM, MODEL_DIM);
    iro_linear(text, a->wv_text, NULL, v, B * T, TEXT_DIM, MODEL_DIM);
    head_rmsnorm(k, a->k_norm, B, T, eps);
    return 0;
}

int iro_dit_speaker_kv(const IroJointAttention *a, float eps,
                       const float *speaker, int B, int T, float *k, float *v) {
    if (!a || !speaker || !k || !v || B <= 0 || T <= 0) return -1;
    iro_linear(speaker, a->wk_speaker, NULL, k,
               B * T, SPEAKER_DIM, MODEL_DIM);
    iro_linear(speaker, a->wv_speaker, NULL, v,
               B * T, SPEAKER_DIM, MODEL_DIM);
    head_rmsnorm(k, a->k_norm, B, T, eps);
    return 0;
}

int iro_dit_caption_kv(const IroJointAttention *a, float eps,
                       const float *caption, int B, int T, float *k, float *v) {
    if (!a || !caption || !k || !v || B <= 0 || T <= 0) return -1;
    iro_linear(caption, a->wk_caption, NULL, k,
               B * T, TEXT_DIM, MODEL_DIM);
    iro_linear(caption, a->wv_caption, NULL, v,
               B * T, TEXT_DIM, MODEL_DIM);
    head_rmsnorm(k, a->k_norm, B, T, eps);
    return 0;
}

static int dit_attention(const IroJointAttention *a, float eps, const float *x,
                         const float *context_k, const float *context_v,
                         const uint8_t *context_mask, int context_batch,
                         int B, int S, int C, float *out) {
    if (!a || !x || !context_k || !context_v || !context_mask || !out ||
        (context_batch != 1 && context_batch != B) ||
        B <= 0 || S <= 0 || C <= 0)
        return -1;

    size_t latent_n = (size_t)B * S * MODEL_DIM;
    float *q = malloc(sizeof(float) * latent_n);
    float *k = malloc(sizeof(float) * latent_n);
    float *v = malloc(sizeof(float) * latent_n);
    float *gate = malloc(sizeof(float) * latent_n);
    float *mix = malloc(sizeof(float) * latent_n);
    size_t max_joined = (size_t)S + C;
    float *kh = malloc(sizeof(float) * max_joined * IRO_DIT_HEAD_DIM);
    float *vh = malloc(sizeof(float) * max_joined * IRO_DIT_HEAD_DIM);
    float *scores = malloc(sizeof(float) * (size_t)S * max_joined);
    if (!q || !k || !v || !gate || !mix || !kh || !vh || !scores) {
        free(q); free(k); free(v); free(gate); free(mix);
        free(kh); free(vh); free(scores);
        return -1;
    }

    iro_linear(x, a->wq, NULL, q, B * S, MODEL_DIM, MODEL_DIM);
    iro_linear(x, a->wk, NULL, k, B * S, MODEL_DIM, MODEL_DIM);
    iro_linear(x, a->wv, NULL, v, B * S, MODEL_DIM, MODEL_DIM);
    iro_linear(x, a->gate, NULL, gate, B * S, MODEL_DIM, MODEL_DIM);
    head_rmsnorm_rope_pair(q, a->q_norm, k, a->k_norm, B, S, eps);

    const float scale = 1.0f / sqrtf((float)IRO_DIT_HEAD_DIM);
    for (int b = 0; b < B; b++) {
        int valid_context = 0;
        for (int ci = 0; ci < C; ci++)
            if (context_mask[(size_t)b * C + ci]) valid_context++;
        int joined = S + valid_context;
        for (int hidx = 0; hidx < IRO_DIT_HEADS; hidx++) {
            for (int token = 0; token < S; token++) {
                memcpy(kh + (size_t)token * IRO_DIT_HEAD_DIM,
                       k + ((size_t)b * S + token) * MODEL_DIM +
                           hidx * IRO_DIT_HEAD_DIM,
                       sizeof(float) * IRO_DIT_HEAD_DIM);
                memcpy(vh + (size_t)token * IRO_DIT_HEAD_DIM,
                       v + ((size_t)b * S + token) * MODEL_DIM +
                           hidx * IRO_DIT_HEAD_DIM,
                       sizeof(float) * IRO_DIT_HEAD_DIM);
            }
            int packed = S;
            size_t context_row = context_batch == 1 ? 0 : (size_t)b * C;
            for (int ci = 0; ci < C; ci++) {
                if (!context_mask[(size_t)b * C + ci]) continue;
                memcpy(kh + (size_t)packed * IRO_DIT_HEAD_DIM,
                       context_k + (context_row + ci) * MODEL_DIM +
                                   hidx * IRO_DIT_HEAD_DIM,
                       sizeof(float) * IRO_DIT_HEAD_DIM);
                memcpy(vh + (size_t)packed * IRO_DIT_HEAD_DIM,
                       context_v + (context_row + ci) * MODEL_DIM +
                                   hidx * IRO_DIT_HEAD_DIM,
                       sizeof(float) * IRO_DIT_HEAD_DIM);
                packed++;
            }

            /* QK^T and P@V both go through the selected GEMM backend. */
            const float *q_head =
                q + (size_t)b * S * MODEL_DIM + hidx * IRO_DIT_HEAD_DIM;
            float *mix_head =
                mix + (size_t)b * S * MODEL_DIM + hidx * IRO_DIT_HEAD_DIM;
            iro_linear_strided_input(q_head, MODEL_DIM, kh, scores,
                                     S, IRO_DIT_HEAD_DIM, joined);
            for (size_t i = 0; i < (size_t)S * joined; i++) scores[i] *= scale;
            iro_softmax_row(scores, S, joined);
            iro_matmul_strided_output(scores, vh, mix_head, MODEL_DIM,
                                      S, joined, IRO_DIT_HEAD_DIM);
        }
    }

    iro_sigmoid_mul_inplace(mix, gate, latent_n);
    iro_linear(mix, a->wo, NULL, out, B * S, MODEL_DIM, MODEL_DIM);

    free(q); free(k); free(v); free(gate); free(mix);
    free(kh); free(vh); free(scores);
    return 0;
}

int iro_dit_attention(const IroJointAttention *a, float eps, const float *x,
                      const float *context_k, const float *context_v,
                      const uint8_t *context_mask, int B, int S, int C,
                      float *out) {
    return dit_attention(a, eps, x, context_k, context_v, context_mask,
                         B, B, S, C, out);
}

int iro_dit_attention_shared_context(const IroJointAttention *a, float eps,
                                     const float *x,
                                     const float *context_k,
                                     const float *context_v,
                                     const uint8_t *context_mask,
                                     int B, int S, int C, float *out) {
    return dit_attention(a, eps, x, context_k, context_v, context_mask,
                         1, B, S, C, out);
}

int iro_dit_swiglu(const IroSwiGLU *mlp, const float *x,
                   int B, int S, float *out) {
    if (!mlp || !x || !out || B <= 0 || S <= 0) return -1;
    size_t hidden_n = (size_t)B * S * MLP_DIM;
    float *w1 = malloc(sizeof(float) * hidden_n);
    float *w3 = malloc(sizeof(float) * hidden_n);
    if (!w1 || !w3) {
        free(w1); free(w3);
        return -1;
    }
    iro_linear(x, mlp->w1, NULL, w1, B * S, MODEL_DIM, MLP_DIM);
    iro_linear(x, mlp->w3, NULL, w3, B * S, MODEL_DIM, MLP_DIM);
    iro_silu_mul_inplace(w1, w3, hidden_n);
    iro_linear(w1, mlp->w2, NULL, out, B * S, MLP_DIM, MODEL_DIM);
    free(w1); free(w3);
    return 0;
}

static int dit_block(const IroDiT *d, int layer, float *x, const float *cond,
                     const float *context_k, const float *context_v,
                     const uint8_t *context_mask, int context_batch,
                     int B, int S, int C) {
    if (!d || layer < 0 || layer >= IRO_DIT_LAYERS || !x || !cond ||
        !context_k || !context_v || !context_mask || B <= 0 || S <= 0 || C <= 0)
        return -1;
    size_t latent_n = (size_t)B * S * MODEL_DIM;
    float *h = malloc(sizeof(float) * latent_n);
    float *branch = malloc(sizeof(float) * latent_n);
    float *gate = malloc(sizeof(float) * (size_t)B * MODEL_DIM);
    if (!h || !branch || !gate) {
        free(h); free(branch); free(gate);
        return -1;
    }

    if (iro_dit_adaln(&d->attention_adaln[layer], d->norm_eps,
                      x, cond, B, S, h, gate) != 0)
        goto fail;
    if (dit_attention(&d->attention[layer], d->norm_eps, h,
                      context_k, context_v, context_mask, context_batch,
                      B, S, C, branch) != 0)
        goto fail;
    for (int b = 0; b < B; b++) {
        const float *gate_row = gate + (size_t)b * MODEL_DIM;
        for (int s = 0; s < S; s++) {
            size_t row = ((size_t)b * S + s) * MODEL_DIM;
            for (int i = 0; i < MODEL_DIM; i++)
                x[row + i] += gate_row[i] * branch[row + i];
        }
    }

    if (iro_dit_adaln(&d->mlp_adaln[layer], d->norm_eps,
                      x, cond, B, S, h, gate) != 0)
        goto fail;
    if (iro_dit_swiglu(&d->mlp[layer], h, B, S, branch) != 0)
        goto fail;
    for (int b = 0; b < B; b++) {
        const float *gate_row = gate + (size_t)b * MODEL_DIM;
        for (int s = 0; s < S; s++) {
            size_t row = ((size_t)b * S + s) * MODEL_DIM;
            for (int i = 0; i < MODEL_DIM; i++)
                x[row + i] += gate_row[i] * branch[row + i];
        }
    }

    free(h); free(branch); free(gate);
    return 0;

fail:
    free(h); free(branch); free(gate);
    return -1;
}


int iro_dit_block(const IroDiT *d, int layer, float *x, const float *cond,
                  const float *context_k, const float *context_v,
                  const uint8_t *context_mask, int B, int S, int C) {
    return dit_block(d, layer, x, cond, context_k, context_v, context_mask,
                     B, B, S, C);
}

int iro_dit_block_shared_context(const IroDiT *d, int layer,
                                 float *x, const float *cond,
                                 const float *context_k,
                                 const float *context_v,
                                 const uint8_t *context_mask,
                                 int B, int S, int C) {
    return dit_block(d, layer, x, cond, context_k, context_v, context_mask,
                     1, B, S, C);
}

int iro_dit_output(const IroDiT *d, const float *x,
                   int B, int S, float *velocity) {
    if (!d || !x || !velocity || B <= 0 || S <= 0) return -1;
    size_t latent_n = (size_t)B * S * MODEL_DIM;
    float *norm = malloc(sizeof(float) * latent_n);
    if (!norm) return -1;
    iro_rmsnorm(x, d->out_norm, norm, B * S, MODEL_DIM, d->norm_eps);
    iro_linear(norm, d->out_w, d->out_b, velocity,
               B * S, MODEL_DIM, LATENT_DIM);
    free(norm);
    return 0;
}
