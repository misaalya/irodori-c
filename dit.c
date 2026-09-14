/* dit.c — timestep conditioner and latent input projection for Irodori RF-DiT. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

typedef struct {
    double seconds[IRO_DIT_PROFILE_BUCKET_COUNT];
    uint64_t calls[IRO_DIT_PROFILE_BUCKET_COUNT];
} IroDiTProfileCell;

static int dit_profile_state = -1;
/* index 0 = global layer -1, 1..12 = layers 0..11; batch index 1..4. */
static IroDiTProfileCell dit_profile_cells[IRO_DIT_LAYERS + 1][5];

static const char *const dit_profile_bucket_names[IRO_DIT_PROFILE_BUCKET_COUNT] = {
    "allocation",
    "conditioner",
    "input_projection",
    "context_prep",
    "kv_cache_prepare",
    "adaln_params",
    "adaln_apply",
    "qkvg_projection",
    "rmsnorm_rope",
    "kv_pack",
    "qk_gemm",
    "softmax",
    "pv_gemm",
    "attention_gate",
    "attention_output_projection",
    "w1_w3",
    "silu_gate",
    "w2",
    "residual",
    "final_output",
    "sampler_update",
};

static double dit_profile_now(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int iro_dit_profile_enabled(void) {
    if (dit_profile_state < 0) {
        const char *env = getenv("IRO_DIT_PROFILE");
        dit_profile_state = env && strcmp(env, "0") != 0;
    }
    return dit_profile_state;
}

void iro_dit_profile_reset(void) {
    if (!iro_dit_profile_enabled()) return;
    memset(dit_profile_cells, 0, sizeof(dit_profile_cells));
}

void iro_dit_profile_add(int layer, int batch, IroDiTProfileBucket bucket,
                         double seconds) {
    if (!iro_dit_profile_enabled() || bucket < 0 ||
        bucket >= IRO_DIT_PROFILE_BUCKET_COUNT || batch < 1 || batch > 4 ||
        layer < -1 || layer >= IRO_DIT_LAYERS)
        return;
    IroDiTProfileCell *cell = &dit_profile_cells[layer + 1][batch];
    cell->seconds[bucket] += seconds;
    cell->calls[bucket]++;
}

void iro_dit_profile_emit_json(const char *status, int steps,
                               int sequence_length, int context_tokens) {
    if (!iro_dit_profile_enabled()) return;
    fprintf(stderr,
            "{\"type\":\"iro_dit_profile\",\"status\":\"%s\","
            "\"steps\":%d,\"sequence_length\":%d,\"context_tokens\":%d,"
            "\"cells\":[",
            status ? status : "unknown", steps, sequence_length, context_tokens);
    int first_cell = 1;
    for (int li = 0; li <= IRO_DIT_LAYERS; li++) {
        for (int batch = 1; batch <= 4; batch++) {
            const IroDiTProfileCell *cell = &dit_profile_cells[li][batch];
            int any = 0;
            for (int bucket = 0; bucket < IRO_DIT_PROFILE_BUCKET_COUNT; bucket++)
                if (cell->calls[bucket]) { any = 1; break; }
            if (!any) continue;
            if (!first_cell) fputc(',', stderr);
            first_cell = 0;
            fprintf(stderr, "{\"layer\":%d,\"batch\":%d,\"buckets\":{",
                    li - 1, batch);
            int first_bucket = 1;
            for (int bucket = 0; bucket < IRO_DIT_PROFILE_BUCKET_COUNT; bucket++) {
                if (!cell->calls[bucket]) continue;
                if (!first_bucket) fputc(',', stderr);
                first_bucket = 0;
                fprintf(stderr,
                        "\"%s\":{\"calls\":%llu,\"seconds\":%.9f}",
                        dit_profile_bucket_names[bucket],
                        (unsigned long long)cell->calls[bucket],
                        cell->seconds[bucket]);
            }
            fputs("}}", stderr);
        }
    }
    fputs("]}\n", stderr);
}

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

int iro_dit_int8_quantize(IroDiTInt8 *dst, const IroDiT *d) {
    if (!dst || !d) return -1;
    memset(dst, 0, sizeof(*dst));
    for (int layer = 0; layer < IRO_DIT_LAYERS; layer++) {
        const IroJointAttention *a = &d->attention[layer];
        const IroSwiGLU *mlp = &d->mlp[layer];
        IroDiTInt8Layer *q = &dst->layer[layer];
        if (iro_int8_weight_quantize(&q->wq, a->wq, MODEL_DIM, MODEL_DIM) != 0 ||
            iro_int8_weight_quantize(&q->wk, a->wk, MODEL_DIM, MODEL_DIM) != 0 ||
            iro_int8_weight_quantize(&q->wv, a->wv, MODEL_DIM, MODEL_DIM) != 0 ||
            iro_int8_weight_quantize(&q->gate, a->gate, MODEL_DIM, MODEL_DIM) != 0 ||
            iro_int8_weight_quantize(&q->wo, a->wo, MODEL_DIM, MODEL_DIM) != 0 ||
            iro_int8_weight_quantize(&q->w1, mlp->w1, MLP_DIM, MODEL_DIM) != 0 ||
            iro_int8_weight_quantize(&q->w3, mlp->w3, MLP_DIM, MODEL_DIM) != 0 ||
            iro_int8_weight_quantize(&q->w2, mlp->w2, MODEL_DIM, MLP_DIM) != 0) {
            iro_dit_int8_free(dst);
            return -1;
        }
        dst->bytes += iro_int8_weight_bytes(&q->wq) + iro_int8_weight_bytes(&q->wk) +
                      iro_int8_weight_bytes(&q->wv) + iro_int8_weight_bytes(&q->gate) +
                      iro_int8_weight_bytes(&q->wo) + iro_int8_weight_bytes(&q->w1) +
                      iro_int8_weight_bytes(&q->w2) + iro_int8_weight_bytes(&q->w3);
    }
    return 0;
}

void iro_dit_int8_free(IroDiTInt8 *q) {
    if (!q) return;
    for (int layer = 0; layer < IRO_DIT_LAYERS; layer++) {
        IroDiTInt8Layer *l = &q->layer[layer];
        iro_int8_weight_free(&l->wq); iro_int8_weight_free(&l->wk);
        iro_int8_weight_free(&l->wv); iro_int8_weight_free(&l->gate);
        iro_int8_weight_free(&l->wo); iro_int8_weight_free(&l->w1);
        iro_int8_weight_free(&l->w2); iro_int8_weight_free(&l->w3);
    }
    memset(q, 0, sizeof(*q));
}

/* Diagnostic only: IRO_INT8_MASK selects which projection groups actually run
   through the integer path when int8 is enabled (1=Q/K/V/gate, 2=Wo,
   4=W1/W3, 8=W2).  Default keeps every group quantized; the mask exists to
   attribute audio error to a group, not as a product knob. */
enum {
    IRO_INT8_GROUP_QKVG = 1,
    IRO_INT8_GROUP_WO = 2,
    IRO_INT8_GROUP_W13 = 4,
    IRO_INT8_GROUP_W2 = 8,
};

static int dit_int8_mask(void) {
    static int mask = -1;
    if (mask < 0) {
        const char *env = getenv("IRO_INT8_MASK");
        mask = env && *env ? (int)strtol(env, NULL, 0) & 15 : 15;
    }
    return mask;
}

/* Diagnostic only: IRO_INT8_EMULATE=1 dequantizes the int8 weights and runs
   FP32 activations through the FP32 GEMM (weight-only error); =2 fake-quantizes
   the activations and uses the FP32 weights (activation-only error).  Both are
   slow and exist to attribute audio error, never for product use. */
static int dit_int8_emulate(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *env = getenv("IRO_INT8_EMULATE");
        mode = env && *env ? (int)strtol(env, NULL, 0) : 0;
    }
    return mode;
}

static int int8_linear_emulated(const float *x, const float *w_fp32,
                                const IroInt8Weight *w, float *y, int M) {
    int mode = dit_int8_emulate();
    size_t K = (size_t)w->K, N = (size_t)w->N;
    if (mode == 1) {
        float *deq = malloc(sizeof(float) * N * K);
        if (!deq) return -1;
        for (size_t n = 0; n < N; n++)
            for (size_t k = 0; k < K; k++)
                deq[n * K + k] = w->scale[n] * (float)w->data[n * K + k];
        iro_linear(x, deq, NULL, y, M, w->K, w->N);
        free(deq);
        return 0;
    }
    float *fake = malloc(sizeof(float) * (size_t)M * K);
    uint8_t *q = malloc((size_t)M * K);
    float *scale = malloc(sizeof(float) * (size_t)M);
    int32_t *zero = malloc(sizeof(int32_t) * (size_t)M);
    if (!fake || !q || !scale || !zero) {
        free(fake); free(q); free(scale); free(zero);
        return -1;
    }
    iro_int8_quantize_rows(x, M, w->K, q, scale, zero);
    for (size_t m = 0; m < (size_t)M; m++)
        for (size_t k = 0; k < K; k++)
            fake[m * K + k] = scale[m] * ((float)q[m * K + k] - (float)zero[m]);
    iro_linear(fake, w_fp32, NULL, y, M, w->K, w->N);
    free(fake); free(q); free(scale); free(zero);
    return 0;
}

/* Diagnostic only: IRO_INT8_STATS=1 prints per (layer, group) activation
   statistics at exit: mean row outlier ratio max|x|/rms(x), the share of rows
   whose largest channel is the globally most frequent one, and the number of
   channels holding the row maximum at least 5% of the time. */
typedef struct {
    double rows;
    double ratio_sum;
    uint32_t argmax_hist[MLP_DIM];
} IroInt8StatCell;

static IroInt8StatCell int8_stats[IRO_DIT_LAYERS][4];
static int int8_stats_enabled = -1;

static void int8_stats_dump(void) {
    static const char *const names[4] = { "qkvg", "wo", "w13", "w2" };
    for (int layer = 0; layer < IRO_DIT_LAYERS; layer++) {
        for (int g = 0; g < 4; g++) {
            IroInt8StatCell *c = &int8_stats[layer][g];
            if (c->rows <= 0) continue;
            uint32_t best = 0; int best_ch = -1, frequent = 0;
            for (int k = 0; k < MLP_DIM; k++) {
                if (c->argmax_hist[k] > best) { best = c->argmax_hist[k]; best_ch = k; }
                if ((double)c->argmax_hist[k] >= 0.05 * c->rows) frequent++;
            }
            fprintf(stderr,
                    "int8-stats layer=%2d group=%-4s rows=%.0f mean_max/rms=%.2f "
                    "top_channel=%d share=%.2f channels>=5%%=%d\n",
                    layer, names[g], c->rows, c->ratio_sum / c->rows,
                    best_ch, (double)best / c->rows, frequent);
        }
    }
}

static void int8_stats_record(int layer, int group, const float *x,
                              int M, int K) {
    if (int8_stats_enabled < 0) {
        const char *env = getenv("IRO_INT8_STATS");
        int8_stats_enabled = env && strcmp(env, "0") != 0;
        if (int8_stats_enabled) atexit(int8_stats_dump);
    }
    /* IRO_INT8_DUMP=dir also writes the first few raw activations per
       (layer, group) so quantization schemes can be simulated offline. */
    static const char *dump_dir;
    static int dump_checked;
    static int dump_counts[IRO_DIT_LAYERS][4];
    if (!dump_checked) { dump_dir = getenv("IRO_INT8_DUMP"); dump_checked = 1; }
    if (dump_dir && layer >= 0 && dump_counts[layer][group] < 3) {
        char path[512];
        snprintf(path, sizeof(path), "%s/act_L%02d_G%d_%d_M%d_K%d.f32", dump_dir,
                 layer, group, dump_counts[layer][group], M, K);
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(x, sizeof(float), (size_t)M * K, f); fclose(f); }
        dump_counts[layer][group]++;
    }
    if (!int8_stats_enabled || layer < 0) return;
    IroInt8StatCell *c = &int8_stats[layer][group];
    for (int m = 0; m < M; m++) {
        const float *row = x + (size_t)m * K;
        double ss = 0.0; float amax = 0.0f; int arg = 0;
        for (int k = 0; k < K; k++) {
            float a = fabsf(row[k]);
            ss += (double)row[k] * row[k];
            if (a > amax) { amax = a; arg = k; }
        }
        double rms = sqrt(ss / K);
        c->ratio_sum += rms > 0 ? amax / rms : 0.0;
        c->argmax_hist[arg]++;
        c->rows += 1;
    }
}

/* Scratch for one W8A8 activation: quantized rows plus the int32 GEMM output.
   Sized for the widest projection the caller will run from this input. */
typedef struct {
    uint8_t *q;
    float *scale;
    int32_t *zero;
    int32_t *work;
    /* Second residual term; allocated only when some group uses it. */
    float *residual;
    uint8_t *q2;
    float *scale2;
    int32_t *zero2;
} IroInt8Scratch;

/* Which groups run the two-term residual activation path.  Default: W2,
   whose SwiGLU-product input is the heaviest-tailed activation on this model
   (max/rms 15-21 versus 5-8 elsewhere).  On a six-text 8-step corpus with an
   exact VNNI backend this lowered the log-mel distance to FP32 from 1.91 to
   1.45 dB and raised STOI 0.969 -> 0.983 for about +20% sampling time;
   extending it to W1/W3 gained little more for +50%.  IRO_INT8_RESIDUAL
   (bitmask 1=Q/K/V/gate, 2=Wo, 4=W1/W3, 8=W2) overrides the policy. */
static int dit_int8_residual_mask(void) {
    static int mask = -1;
    if (mask < 0) {
        const char *env = getenv("IRO_INT8_RESIDUAL");
        mask = env && *env ? (int)strtol(env, NULL, 0) & 15 : IRO_INT8_GROUP_W2;
    }
    return mask;
}

static int int8_scratch_alloc(IroInt8Scratch *s, size_t rows, size_t K,
                              size_t max_N, int residual) {
    memset(s, 0, sizeof(*s));
    s->q = malloc(rows * K);
    s->scale = malloc(sizeof(float) * rows);
    s->zero = malloc(sizeof(int32_t) * rows);
    s->work = malloc(sizeof(int32_t) * rows * max_N);
    int ok = s->q && s->scale && s->zero && s->work;
    if (ok && residual) {
        s->residual = malloc(sizeof(float) * rows * K);
        s->q2 = malloc(rows * K);
        s->scale2 = malloc(sizeof(float) * rows);
        s->zero2 = malloc(sizeof(int32_t) * rows);
        ok = s->residual && s->q2 && s->scale2 && s->zero2;
    }
    if (!ok) {
        free(s->q); free(s->scale); free(s->zero); free(s->work);
        free(s->residual); free(s->q2); free(s->scale2); free(s->zero2);
        memset(s, 0, sizeof(*s));
        return -1;
    }
    return 0;
}

static void int8_scratch_free(IroInt8Scratch *s) {
    free(s->q); free(s->scale); free(s->zero); free(s->work);
    free(s->residual); free(s->q2); free(s->scale2); free(s->zero2);
    memset(s, 0, sizeof(*s));
}

/* Quantize x once (one or two terms) so several projections can share it. */
static void int8_scratch_quantize(IroInt8Scratch *s, const float *x, int M,
                                  int K, int residual) {
    if (residual)
        iro_int8_quantize_rows_residual(x, M, K, s->q, s->scale, s->zero,
                                        s->residual, s->q2, s->scale2,
                                        s->zero2);
    else
        iro_int8_quantize_rows(x, M, K, s->q, s->scale, s->zero);
}

static void int8_scratch_linear(const IroInt8Scratch *s,
                                const IroInt8Weight *w, float *y, int M,
                                int residual) {
    iro_int8_linear_ex(s->q, s->scale, s->zero, w, NULL, y, M, s->work, 0);
    if (residual)
        iro_int8_linear_ex(s->q2, s->scale2, s->zero2, w, NULL, y, M,
                           s->work, 1);
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
    int profile = iro_dit_profile_enabled();
    double t_alloc = profile ? dit_profile_now() : 0.0;
    float *embedding = malloc(sizeof(float) * (size_t)B * TIMESTEP_DIM);
    float *hidden0 = malloc(sizeof(float) * (size_t)B * MODEL_DIM);
    float *hidden1 = malloc(sizeof(float) * (size_t)B * MODEL_DIM);
    if (profile)
        iro_dit_profile_add(-1, B, IRO_DIT_PROFILE_ALLOCATION,
                            dit_profile_now() - t_alloc);
    if (!embedding || !hidden0 || !hidden1) {
        free(embedding); free(hidden0); free(hidden1);
        return -1;
    }
    double t_compute = profile ? dit_profile_now() : 0.0;
    timestep_embedding(t, B, embedding);
    iro_linear(embedding, d->cond_w0, NULL, hidden0,
               B, TIMESTEP_DIM, MODEL_DIM);
    iro_silu_inplace(hidden0, (size_t)B * MODEL_DIM);
    iro_linear(hidden0, d->cond_w1, NULL, hidden1,
               B, MODEL_DIM, MODEL_DIM);
    iro_silu_inplace(hidden1, (size_t)B * MODEL_DIM);
    iro_linear(hidden1, d->cond_w2, NULL, cond, B, MODEL_DIM, COND_DIM);
    if (profile)
        iro_dit_profile_add(-1, B, IRO_DIT_PROFILE_CONDITIONER,
                            dit_profile_now() - t_compute);
    free(embedding); free(hidden0); free(hidden1);
    return 0;
}

int iro_dit_input(const IroDiT *d, const float *x, int B, int S, float *projected) {
    if (!d || !x || B <= 0 || S <= 0 || !projected) return -1;
    int profile = iro_dit_profile_enabled();
    double t0 = profile ? dit_profile_now() : 0.0;
    iro_linear(x, d->in_w, d->in_b, projected, B * S, LATENT_DIM, MODEL_DIM);
    if (profile)
        iro_dit_profile_add(-1, B, IRO_DIT_PROFILE_INPUT_PROJECTION,
                            dit_profile_now() - t0);
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

static int dit_adaln(const IroAdaLN *a, float eps, const float *x,
                     const float *cond, int B, int S, float *h, float *gate,
                     int profile_layer) {
    if (!a || !x || !cond || B <= 0 || S <= 0 || !h || !gate) return -1;
    int profile = iro_dit_profile_enabled();
    size_t branch_n = (size_t)B * MODEL_DIM;
    double t_alloc = profile ? dit_profile_now() : 0.0;
    float *act = malloc(sizeof(float) * branch_n);
    float *low = malloc(sizeof(float) * (size_t)B * ADALN_RANK);
    float *shift = malloc(sizeof(float) * branch_n);
    float *scale = malloc(sizeof(float) * branch_n);
    if (profile)
        iro_dit_profile_add(profile_layer, B, IRO_DIT_PROFILE_ALLOCATION,
                            dit_profile_now() - t_alloc);
    if (!act || !low || !shift || !scale) {
        free(act); free(low); free(shift); free(scale);
        return -1;
    }
    double t_params = profile ? dit_profile_now() : 0.0;
    adaln_branch(cond, 0, B, a->shift_down, a->shift_up, a->shift_bias,
                 act, low, shift);
    adaln_branch(cond, 1, B,
                 a->scale_down, a->scale_up, a->scale_bias,
                 act, low, scale);
    adaln_branch(cond, 2, B,
                 a->gate_down, a->gate_up, a->gate_bias,
                 act, low, gate);
    for (size_t i = 0; i < branch_n; i++) gate[i] = tanhf(gate[i]);
    if (profile)
        iro_dit_profile_add(profile_layer, B, IRO_DIT_PROFILE_ADALN_PARAMS,
                            dit_profile_now() - t_params);

    double t_apply = profile ? dit_profile_now() : 0.0;
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
    if (profile)
        iro_dit_profile_add(profile_layer, B, IRO_DIT_PROFILE_ADALN_APPLY,
                            dit_profile_now() - t_apply);
    free(act); free(low); free(shift); free(scale);
    return 0;
}

int iro_dit_adaln(const IroAdaLN *a, float eps, const float *x,
                  const float *cond, int B, int S, float *h, float *gate) {
    return dit_adaln(a, eps, x, cond, B, S, h, gate, -1);
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
                         int B, int S, int C, float *out,
                         int profile_layer, const IroDiTInt8Layer *int8) {
    if (!a || !x || !context_k || !context_v || !context_mask || !out ||
        (context_batch != 1 && context_batch != B) ||
        B <= 0 || S <= 0 || C <= 0)
        return -1;

    int profile = iro_dit_profile_enabled();
    size_t latent_n = (size_t)B * S * MODEL_DIM;
    double t_alloc = profile ? dit_profile_now() : 0.0;
    float *q = malloc(sizeof(float) * latent_n);
    float *k = malloc(sizeof(float) * latent_n);
    float *v = malloc(sizeof(float) * latent_n);
    float *gate = malloc(sizeof(float) * latent_n);
    float *mix = malloc(sizeof(float) * latent_n);
    size_t max_joined = (size_t)S + C;
    float *kh = malloc(sizeof(float) * max_joined * IRO_DIT_HEAD_DIM);
    float *vh = malloc(sizeof(float) * max_joined * IRO_DIT_HEAD_DIM);
    float *scores = malloc(sizeof(float) * (size_t)S * max_joined);
    if (profile)
        iro_dit_profile_add(profile_layer, B, IRO_DIT_PROFILE_ALLOCATION,
                            dit_profile_now() - t_alloc);
    if (!q || !k || !v || !gate || !mix || !kh || !vh || !scores) {
        free(q); free(k); free(v); free(gate); free(mix);
        free(kh); free(vh); free(scores);
        return -1;
    }

    IroInt8Scratch scratch = {0};
    int residual_qkvg = (dit_int8_residual_mask() & IRO_INT8_GROUP_QKVG) != 0;
    int residual_wo = (dit_int8_residual_mask() & IRO_INT8_GROUP_WO) != 0;
    if (int8 && int8_scratch_alloc(&scratch, (size_t)B * S, MODEL_DIM,
                                   MODEL_DIM, residual_qkvg || residual_wo) != 0) {
        free(q); free(k); free(v); free(gate); free(mix);
        free(kh); free(vh); free(scores);
        return -1;
    }
    double t_proj = profile ? dit_profile_now() : 0.0;
    if (int8 && (dit_int8_mask() & IRO_INT8_GROUP_QKVG) && dit_int8_emulate()) {
        int8_linear_emulated(x, a->wq, &int8->wq, q, B * S);
        int8_linear_emulated(x, a->wk, &int8->wk, k, B * S);
        int8_linear_emulated(x, a->wv, &int8->wv, v, B * S);
        int8_linear_emulated(x, a->gate, &int8->gate, gate, B * S);
    } else if (int8 && (dit_int8_mask() & IRO_INT8_GROUP_QKVG)) {
        /* Q/K/V/gate share one quantized copy of the normalized input. */
        int8_stats_record(profile_layer, 0, x, B * S, MODEL_DIM);
        int8_scratch_quantize(&scratch, x, B * S, MODEL_DIM, residual_qkvg);
        int8_scratch_linear(&scratch, &int8->wq, q, B * S, residual_qkvg);
        int8_scratch_linear(&scratch, &int8->wk, k, B * S, residual_qkvg);
        int8_scratch_linear(&scratch, &int8->wv, v, B * S, residual_qkvg);
        int8_scratch_linear(&scratch, &int8->gate, gate, B * S, residual_qkvg);
    } else {
        iro_linear(x, a->wq, NULL, q, B * S, MODEL_DIM, MODEL_DIM);
        iro_linear(x, a->wk, NULL, k, B * S, MODEL_DIM, MODEL_DIM);
        iro_linear(x, a->wv, NULL, v, B * S, MODEL_DIM, MODEL_DIM);
        iro_linear(x, a->gate, NULL, gate, B * S, MODEL_DIM, MODEL_DIM);
    }
    if (profile)
        iro_dit_profile_add(profile_layer, B, IRO_DIT_PROFILE_QKVG_PROJECTION,
                            dit_profile_now() - t_proj);
    double t_norm_rope = profile ? dit_profile_now() : 0.0;
    head_rmsnorm_rope_pair(q, a->q_norm, k, a->k_norm, B, S, eps);
    if (profile)
        iro_dit_profile_add(profile_layer, B, IRO_DIT_PROFILE_RMSNORM_ROPE,
                            dit_profile_now() - t_norm_rope);

    const float scale = 1.0f / sqrtf((float)IRO_DIT_HEAD_DIM);
    for (int b = 0; b < B; b++) {
        int valid_context = 0;
        for (int ci = 0; ci < C; ci++)
            if (context_mask[(size_t)b * C + ci]) valid_context++;
        int joined = S + valid_context;
        for (int hidx = 0; hidx < IRO_DIT_HEADS; hidx++) {
            double t_pack = profile ? dit_profile_now() : 0.0;
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
            if (profile)
                iro_dit_profile_add(profile_layer, B, IRO_DIT_PROFILE_KV_PACK,
                                    dit_profile_now() - t_pack);

            /* QK^T and P@V both go through the selected GEMM backend. */
            const float *q_head =
                q + (size_t)b * S * MODEL_DIM + hidx * IRO_DIT_HEAD_DIM;
            float *mix_head =
                mix + (size_t)b * S * MODEL_DIM + hidx * IRO_DIT_HEAD_DIM;
            double t_qk = profile ? dit_profile_now() : 0.0;
            iro_linear_strided_input(q_head, MODEL_DIM, kh, scores,
                                     S, IRO_DIT_HEAD_DIM, joined);
            if (profile)
                iro_dit_profile_add(profile_layer, B, IRO_DIT_PROFILE_QK_GEMM,
                                    dit_profile_now() - t_qk);
            double t_softmax = profile ? dit_profile_now() : 0.0;
            for (size_t i = 0; i < (size_t)S * joined; i++) scores[i] *= scale;
            iro_softmax_row(scores, S, joined);
            if (profile)
                iro_dit_profile_add(profile_layer, B, IRO_DIT_PROFILE_SOFTMAX,
                                    dit_profile_now() - t_softmax);
            double t_pv = profile ? dit_profile_now() : 0.0;
            iro_matmul_strided_output(scores, vh, mix_head, MODEL_DIM,
                                      S, joined, IRO_DIT_HEAD_DIM);
            if (profile)
                iro_dit_profile_add(profile_layer, B, IRO_DIT_PROFILE_PV_GEMM,
                                    dit_profile_now() - t_pv);
        }
    }

    double t_gate = profile ? dit_profile_now() : 0.0;
    iro_sigmoid_mul_inplace(mix, gate, latent_n);
    if (profile)
        iro_dit_profile_add(profile_layer, B, IRO_DIT_PROFILE_ATTENTION_GATE,
                            dit_profile_now() - t_gate);
    double t_out = profile ? dit_profile_now() : 0.0;
    if (int8 && (dit_int8_mask() & IRO_INT8_GROUP_WO) && dit_int8_emulate()) {
        int8_linear_emulated(mix, a->wo, &int8->wo, out, B * S);
    } else if (int8 && (dit_int8_mask() & IRO_INT8_GROUP_WO)) {
        int8_stats_record(profile_layer, 1, mix, B * S, MODEL_DIM);
        int8_scratch_quantize(&scratch, mix, B * S, MODEL_DIM, residual_wo);
        int8_scratch_linear(&scratch, &int8->wo, out, B * S, residual_wo);
    } else {
        iro_linear(mix, a->wo, NULL, out, B * S, MODEL_DIM, MODEL_DIM);
    }
    if (profile)
        iro_dit_profile_add(profile_layer, B, IRO_DIT_PROFILE_ATTENTION_OUTPUT,
                            dit_profile_now() - t_out);

    int8_scratch_free(&scratch);
    free(q); free(k); free(v); free(gate); free(mix);
    free(kh); free(vh); free(scores);
    return 0;
}

int iro_dit_attention(const IroJointAttention *a, float eps, const float *x,
                      const float *context_k, const float *context_v,
                      const uint8_t *context_mask, int B, int S, int C,
                      float *out) {
    return dit_attention(a, eps, x, context_k, context_v, context_mask,
                         B, B, S, C, out, -1, NULL);
}

int iro_dit_attention_shared_context(const IroJointAttention *a, float eps,
                                     const float *x,
                                     const float *context_k,
                                     const float *context_v,
                                     const uint8_t *context_mask,
                                     int B, int S, int C, float *out) {
    return dit_attention(a, eps, x, context_k, context_v, context_mask,
                         1, B, S, C, out, -1, NULL);
}

static int dit_swiglu(const IroSwiGLU *mlp, const float *x,
                      int B, int S, float *out, int profile_layer,
                      IroPackedCache *cache, const IroDiTInt8Layer *int8) {
    if (!mlp || !x || !out || B <= 0 || S <= 0) return -1;
    int profile = iro_dit_profile_enabled();
    size_t hidden_n = (size_t)B * S * MLP_DIM;
    double t_alloc = profile ? dit_profile_now() : 0.0;
    float *w1 = malloc(sizeof(float) * hidden_n);
    float *w3 = malloc(sizeof(float) * hidden_n);
    IroInt8Scratch scratch = {0};
    int residual_w13 = (dit_int8_residual_mask() & IRO_INT8_GROUP_W13) != 0;
    int residual_w2 = (dit_int8_residual_mask() & IRO_INT8_GROUP_W2) != 0;
    /* The gated hidden state is the widest activation quantized here. */
    if (int8 && int8_scratch_alloc(&scratch, (size_t)B * S, MLP_DIM,
                                   MLP_DIM, residual_w13 || residual_w2) != 0) {
        free(w1); free(w3);
        return -1;
    }
    if (profile)
        iro_dit_profile_add(profile_layer, B, IRO_DIT_PROFILE_ALLOCATION,
                            dit_profile_now() - t_alloc);
    if (!w1 || !w3) {
        int8_scratch_free(&scratch);
        free(w1); free(w3);
        return -1;
    }
    double t_w13 = profile ? dit_profile_now() : 0.0;
    if (int8 && (dit_int8_mask() & IRO_INT8_GROUP_W13) && dit_int8_emulate()) {
        int8_linear_emulated(x, mlp->w1, &int8->w1, w1, B * S);
        int8_linear_emulated(x, mlp->w3, &int8->w3, w3, B * S);
    } else if (int8 && (dit_int8_mask() & IRO_INT8_GROUP_W13)) {
        int8_stats_record(profile_layer, 2, x, B * S, MODEL_DIM);
        int8_scratch_quantize(&scratch, x, B * S, MODEL_DIM, residual_w13);
        int8_scratch_linear(&scratch, &int8->w1, w1, B * S, residual_w13);
        int8_scratch_linear(&scratch, &int8->w3, w3, B * S, residual_w13);
    } else {
        iro_linear_cached(cache, x, mlp->w1, w1, B * S, MODEL_DIM, MLP_DIM);
        iro_linear_cached(cache, x, mlp->w3, w3, B * S, MODEL_DIM, MLP_DIM);
    }
    if (profile)
        iro_dit_profile_add(profile_layer, B, IRO_DIT_PROFILE_W1_W3,
                            dit_profile_now() - t_w13);
    double t_gate = profile ? dit_profile_now() : 0.0;
    iro_silu_mul_inplace(w1, w3, hidden_n);
    if (profile)
        iro_dit_profile_add(profile_layer, B, IRO_DIT_PROFILE_SILU_GATE,
                            dit_profile_now() - t_gate);
    double t_w2 = profile ? dit_profile_now() : 0.0;
    if (int8 && (dit_int8_mask() & IRO_INT8_GROUP_W2) && dit_int8_emulate()) {
        int8_linear_emulated(w1, mlp->w2, &int8->w2, out, B * S);
    } else if (int8 && (dit_int8_mask() & IRO_INT8_GROUP_W2)) {
        int8_stats_record(profile_layer, 3, w1, B * S, MLP_DIM);
        int8_scratch_quantize(&scratch, w1, B * S, MLP_DIM, residual_w2);
        int8_scratch_linear(&scratch, &int8->w2, out, B * S, residual_w2);
    } else {
        iro_linear(w1, mlp->w2, NULL, out, B * S, MLP_DIM, MODEL_DIM);
    }
    if (profile)
        iro_dit_profile_add(profile_layer, B, IRO_DIT_PROFILE_W2,
                            dit_profile_now() - t_w2);
    int8_scratch_free(&scratch);
    free(w1); free(w3);
    return 0;
}

int iro_dit_swiglu(const IroSwiGLU *mlp, const float *x,
                   int B, int S, float *out) {
    return dit_swiglu(mlp, x, B, S, out, -1, NULL, NULL);
}

static int dit_block(const IroDiT *d, int layer, float *x, const float *cond,
                     const float *context_k, const float *context_v,
                     const uint8_t *context_mask, int context_batch,
                     int B, int S, int C) {
    if (!d || layer < 0 || layer >= IRO_DIT_LAYERS || !x || !cond ||
        !context_k || !context_v || !context_mask || B <= 0 || S <= 0 || C <= 0)
        return -1;
    int profile = iro_dit_profile_enabled();
    size_t latent_n = (size_t)B * S * MODEL_DIM;
    double t_alloc = profile ? dit_profile_now() : 0.0;
    float *h = malloc(sizeof(float) * latent_n);
    float *branch = malloc(sizeof(float) * latent_n);
    float *gate = malloc(sizeof(float) * (size_t)B * MODEL_DIM);
    if (profile)
        iro_dit_profile_add(layer, B, IRO_DIT_PROFILE_ALLOCATION,
                            dit_profile_now() - t_alloc);
    if (!h || !branch || !gate) {
        free(h); free(branch); free(gate);
        return -1;
    }

    if (dit_adaln(&d->attention_adaln[layer], d->norm_eps,
                  x, cond, B, S, h, gate, layer) != 0)
        goto fail;
    const IroDiTInt8Layer *int8 = d->int8 ? &d->int8->layer[layer] : NULL;
    if (dit_attention(&d->attention[layer], d->norm_eps, h,
                      context_k, context_v, context_mask, context_batch,
                      B, S, C, branch, layer, int8) != 0)
        goto fail;
    double t_residual = profile ? dit_profile_now() : 0.0;
    for (int b = 0; b < B; b++) {
        const float *gate_row = gate + (size_t)b * MODEL_DIM;
        for (int s = 0; s < S; s++) {
            size_t row = ((size_t)b * S + s) * MODEL_DIM;
            for (int i = 0; i < MODEL_DIM; i++)
                x[row + i] += gate_row[i] * branch[row + i];
        }
    }
    if (profile)
        iro_dit_profile_add(layer, B, IRO_DIT_PROFILE_RESIDUAL,
                            dit_profile_now() - t_residual);

    if (dit_adaln(&d->mlp_adaln[layer], d->norm_eps,
                  x, cond, B, S, h, gate, layer) != 0)
        goto fail;
    if (dit_swiglu(&d->mlp[layer], h, B, S, branch, layer, d->packed_cache,
                   int8) != 0)
        goto fail;
    t_residual = profile ? dit_profile_now() : 0.0;
    for (int b = 0; b < B; b++) {
        const float *gate_row = gate + (size_t)b * MODEL_DIM;
        for (int s = 0; s < S; s++) {
            size_t row = ((size_t)b * S + s) * MODEL_DIM;
            for (int i = 0; i < MODEL_DIM; i++)
                x[row + i] += gate_row[i] * branch[row + i];
        }
    }
    if (profile)
        iro_dit_profile_add(layer, B, IRO_DIT_PROFILE_RESIDUAL,
                            dit_profile_now() - t_residual);

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
    int profile = iro_dit_profile_enabled();
    size_t latent_n = (size_t)B * S * MODEL_DIM;
    double t_alloc = profile ? dit_profile_now() : 0.0;
    float *norm = malloc(sizeof(float) * latent_n);
    if (profile)
        iro_dit_profile_add(-1, B, IRO_DIT_PROFILE_ALLOCATION,
                            dit_profile_now() - t_alloc);
    if (!norm) return -1;
    double t0 = profile ? dit_profile_now() : 0.0;
    iro_rmsnorm(x, d->out_norm, norm, B * S, MODEL_DIM, d->norm_eps);
    iro_linear(norm, d->out_w, d->out_b, velocity,
               B * S, MODEL_DIM, LATENT_DIM);
    if (profile)
        iro_dit_profile_add(-1, B, IRO_DIT_PROFILE_FINAL_OUTPUT,
                            dit_profile_now() - t0);
    free(norm);
    return 0;
}
