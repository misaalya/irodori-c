/* backbone.c — ModernBERT-ja forward (port transformers 5.16 modeling_modernbert).

Terverifikasi dari config + kode + tensor inventory:
- Embeddings: LayerNorm(tok_embeddings(ids)) tanpa bias
- Layer 0: attn_norm = Identity; layer lain: LayerNorm sebelum attn
- Wqkv [2304,768] → per posisi [3,12,64]: q,k,v; RoPE split-half (fp32),
  theta full=160000, sliding=10000
- Eager attention: score=qk*0.125; mask: key-only padding, sliding |q-k|<=64;
  softmax fp32; *v; @Wo
- MLP GLU: Wi [6144,768] chunk (in,gate) → gelu(in)*gate → Wo [768,3072]
- final LayerNorm; output * mask
*/
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "backbone.h"
#include "ops.h"

#define BB_MAX_POS 512

static void rope_build(float *cos_t, float *sin_t, int max_pos, int head_dim, double theta) {
    int half = head_dim / 2;
    for (int p = 0; p < max_pos; p++) {
        float *cos_p = cos_t + (size_t)p * head_dim;
        float *sin_p = sin_t + (size_t)p * head_dim;
        for (int d = 0; d < half; d++) {
            double inv_freq = 1.0 / pow(theta, (double)(2 * d) / head_dim);
            double arg = (double)p * inv_freq;
            float c = (float)cos(arg), s = (float)sin(arg);
            cos_p[d] = c;
            sin_p[d] = s;
            cos_p[half + d] = c; /* cat(freqs, freqs) */
            sin_p[half + d] = s;
        }
    }
}

static const IroTensor *bb_get(const IroBackbone *b, const char *fmt, ...) {
    char name[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    const IroTensor *t = iro_st_get(b->st, name);
    if (!t) fprintf(stderr, "backbone: tensor tidak ada: %s\n", name);
    else if (t->dtype != IRO_ST_F32) {
        fprintf(stderr, "backbone: %s harus F32, ditemukan %s\n",
                name, iro_dtype_name(t->dtype));
        return NULL;
    }
    return t;
}

int iro_backbone_init(IroBackbone *b, const IroSafetensors *st, int max_pos) {
    memset(b, 0, sizeof(*b));
    if (!st || max_pos <= 0) return -1;
    b->st = st;
    b->n_layers = 25;
    b->hidden = 768;
    b->heads = 12;
    b->head_dim = 64;
    b->inter = 3072;
    b->sliding = 64;
    b->norm_eps = 1e-5f;
    b->pad_id = 3;
    if (max_pos > BB_MAX_POS) max_pos = BB_MAX_POS;
    b->rope_len = max_pos;

    if (!bb_get(b, IRO_BB_PREFIX "embeddings.tok_embeddings.weight")) return -1;
    if (!bb_get(b, IRO_BB_PREFIX "embeddings.norm.weight")) return -1;
    if (!bb_get(b, IRO_BB_PREFIX "final_norm.weight")) return -1;
    for (int i = 0; i < b->n_layers; i++) {
        if (i > 0 && !bb_get(b, IRO_BB_PREFIX "layers.%d.attn_norm.weight", i)) return -1;
        if (!bb_get(b, IRO_BB_PREFIX "layers.%d.attn.Wqkv.weight", i)) return -1;
        if (!bb_get(b, IRO_BB_PREFIX "layers.%d.attn.Wo.weight", i)) return -1;
        if (!bb_get(b, IRO_BB_PREFIX "layers.%d.mlp.Wi.weight", i)) return -1;
        if (!bb_get(b, IRO_BB_PREFIX "layers.%d.mlp.Wo.weight", i)) return -1;
        if (!bb_get(b, IRO_BB_PREFIX "layers.%d.mlp_norm.weight", i)) return -1;
    }

    size_t n = (size_t)max_pos * b->head_dim;
    b->rope_cos_full = malloc(sizeof(float) * n);
    b->rope_sin_full = malloc(sizeof(float) * n);
    b->rope_cos_slide = malloc(sizeof(float) * n);
    b->rope_sin_slide = malloc(sizeof(float) * n);
    if (!b->rope_cos_full || !b->rope_sin_full || !b->rope_cos_slide || !b->rope_sin_slide) {
        iro_backbone_free(b);
        return -1;
    }
    rope_build(b->rope_cos_full, b->rope_sin_full, max_pos, b->head_dim, 160000.0);
    rope_build(b->rope_cos_slide, b->rope_sin_slide, max_pos, b->head_dim, 10000.0);
    return 0;
}

void iro_backbone_free(IroBackbone *b) {
    free(b->rope_cos_full);
    free(b->rope_sin_full);
    free(b->rope_cos_slide);
    free(b->rope_sin_slide);
    memset(b, 0, sizeof(*b));
}


/* layer_types modernbert-ja: full di index 0,3,6,... (setiap 3), sisanya sliding */
static int layer_is_full(int i) {
    return (i % 3) == 0;
}

/* satu head attention eager. hasil: out[qi*out_stride + d] */
static void attention_head(const float *q, const float *k, const float *v,
                           const uint8_t *mask, int S, int head_dim,
                           int sliding, int is_full_layer, float *scores,
                           float *out, int out_stride) {
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int qi = 0; qi < S; qi++) {
        float *row = scores + (size_t)qi * S;
        for (int ki = 0; ki < S; ki++) {
            float acc = 0.0f;
            for (int d = 0; d < head_dim; d++)
                acc += q[(size_t)qi * head_dim + d] * k[(size_t)ki * head_dim + d];
            acc *= scale;
            int allowed = mask[ki] && (is_full_layer || abs(qi - ki) <= sliding);
            row[ki] = allowed ? acc : -INFINITY;
        }
    }
    for (int qi = 0; qi < S; qi++) {
        float *row = scores + (size_t)qi * S;
        int any = 0;
        float mx = -INFINITY;
        for (int ki = 0; ki < S; ki++)
            if (row[ki] != -INFINITY) { any = 1; if (row[ki] > mx) mx = row[ki]; }
        float *outq = out + (size_t)qi * out_stride;
        if (!any) { memset(outq, 0, sizeof(float) * head_dim); continue; }
        float sum = 0.0f;
        for (int ki = 0; ki < S; ki++) {
            if (row[ki] == -INFINITY) { row[ki] = 0.0f; continue; }
            row[ki] = expf(row[ki] - mx);
            sum += row[ki];
        }
        memset(outq, 0, sizeof(float) * head_dim);
        for (int ki = 0; ki < S; ki++) {
            float w = row[ki] / sum;
            if (w == 0.0f) continue;
            const float *vk = v + (size_t)ki * head_dim;
            for (int d = 0; d < head_dim; d++) outq[d] += w * vk[d];
        }
    }
}


int iro_backbone_forward_trace(IroBackbone *b, const int32_t *ids,
                               const uint8_t *mask, int S, float *out,
                               IroBackboneTraceFn trace, void *trace_user) {
    if (!b || !ids || !mask || !out || S <= 0 || S > b->rope_len) return -1;
    int H = b->hidden, HD = b->head_dim, NH = b->heads, INTER = b->inter;
    int rc = -1;

    float *h      = malloc(sizeof(float) * (size_t)S * H);
    float *tmp    = malloc(sizeof(float) * (size_t)S * H);
    float *qkv    = malloc(sizeof(float) * (size_t)S * 3 * H);
    float *attnc  = malloc(sizeof(float) * (size_t)S * H);
    float *mlpw   = malloc(sizeof(float) * (size_t)S * 2 * INTER);
    float *mlpa   = malloc(sizeof(float) * (size_t)S * INTER);
    float *scores = malloc(sizeof(float) * (size_t)S * S);
    float *qh     = malloc(sizeof(float) * (size_t)S * HD);
    float *kh     = malloc(sizeof(float) * (size_t)S * HD);
    float *vh     = malloc(sizeof(float) * (size_t)S * HD);
    if (!h || !tmp || !qkv || !attnc || !mlpw || !mlpa ||
        !scores || !qh || !kh || !vh)
        goto cleanup;

    /* --- embeddings: LayerNorm(lookup) --- */
    const IroTensor *tok = bb_get(b, IRO_BB_PREFIX "embeddings.tok_embeddings.weight");
    const IroTensor *enorm = bb_get(b, IRO_BB_PREFIX "embeddings.norm.weight");
    for (int s = 0; s < S; s++) {
        int32_t id = ids[s];
        if (id < 0 || (uint64_t)id >= tok->shape[0]) goto cleanup;
        memcpy(tmp + (size_t)s * H,
               (const float *)tok->data + (size_t)id * H, sizeof(float) * H);
    }
    iro_layernorm(tmp, (const float *)enorm->data, NULL, h, S, H, b->norm_eps);
    if (trace && trace(trace_user, "bb_embed", h, S, H) != 0) goto cleanup;

    /* --- 25 layer --- */
    for (int i = 0; i < b->n_layers; i++) {
        int is_full = layer_is_full(i);

        if (i > 0) {
            const IroTensor *t = bb_get(b, IRO_BB_PREFIX "layers.%d.attn_norm.weight", i);
            iro_layernorm(h, (const float *)t->data, NULL, tmp, S, H, b->norm_eps);
        } else {
            memcpy(tmp, h, sizeof(float) * (size_t)S * H);
        }

        const IroTensor *wqkv = bb_get(b, IRO_BB_PREFIX "layers.%d.attn.Wqkv.weight", i);
        iro_linear(tmp, (const float *)wqkv->data, NULL, qkv, S, H, 3 * H);
        if (i == 0 && trace &&
            trace(trace_user, "bb0_qkv", qkv, S, 3 * H) != 0) goto cleanup;

        const float *cos_t = is_full ? b->rope_cos_full : b->rope_cos_slide;
        const float *sin_t = is_full ? b->rope_sin_full : b->rope_sin_slide;

        for (int hh = 0; hh < NH; hh++) {
            for (int s = 0; s < S; s++) {
                const float *row = qkv + (size_t)s * 3 * H + hh * HD;
                float *qr = qh + (size_t)s * HD;
                float *kr = kh + (size_t)s * HD;
                float *vr = vh + (size_t)s * HD;
                memcpy(qr, row, sizeof(float) * HD);
                memcpy(kr, row + H, sizeof(float) * HD);
                memcpy(vr, row + 2 * H, sizeof(float) * HD);
                const float *cs = cos_t + (size_t)s * HD;
                const float *sn = sin_t + (size_t)s * HD;
                for (int d = 0; d < HD / 2; d++) {
                    float q1 = qr[d], q2 = qr[d + HD / 2];
                    float k1 = kr[d], k2 = kr[d + HD / 2];
                    qr[d]          = q1 * cs[d] - q2 * sn[d];
                    qr[d + HD / 2] = q2 * cs[d + HD / 2] + q1 * sn[d + HD / 2];
                    kr[d]          = k1 * cs[d] - k2 * sn[d];
                    kr[d + HD / 2] = k2 * cs[d + HD / 2] + k1 * sn[d + HD / 2];
                }
            }
            attention_head(qh, kh, vh, mask, S, HD, b->sliding, is_full,
                           scores, attnc + hh * HD, H);
        }

        const IroTensor *wo = bb_get(b, IRO_BB_PREFIX "layers.%d.attn.Wo.weight", i);
        iro_linear(attnc, (const float *)wo->data, NULL, tmp, S, H, H);
        if (i == 0 && trace &&
            trace(trace_user, "bb0_attn_out", tmp, S, H) != 0) goto cleanup;
        for (size_t j = 0; j < (size_t)S * H; j++) h[j] += tmp[j];

        const IroTensor *mn = bb_get(b, IRO_BB_PREFIX "layers.%d.mlp_norm.weight", i);
        iro_layernorm(h, (const float *)mn->data, NULL, tmp, S, H, b->norm_eps);
        if (i == 0 && trace &&
            trace(trace_user, "bb0_mlp_norm", tmp, S, H) != 0) goto cleanup;

        const IroTensor *wi = bb_get(b, IRO_BB_PREFIX "layers.%d.mlp.Wi.weight", i);
        iro_linear(tmp, (const float *)wi->data, NULL, mlpw, S, H, 2 * INTER);
        if (i == 0 && trace &&
            trace(trace_user, "bb0_mlp_wi", mlpw, S, 2 * INTER) != 0) goto cleanup;
        for (int s = 0; s < S; s++) {
            const float *in = mlpw + (size_t)s * 2 * INTER;
            const float *gate = in + INTER;
            float *act = mlpa + (size_t)s * INTER;
            for (int d = 0; d < INTER; d++)
                act[d] = iro_gelu(in[d]) * gate[d];
        }
        if (i == 0 && trace &&
            trace(trace_user, "bb0_mlp_act", mlpa, S, INTER) != 0) goto cleanup;

        const IroTensor *mwo = bb_get(b, IRO_BB_PREFIX "layers.%d.mlp.Wo.weight", i);
        iro_linear(mlpa, (const float *)mwo->data, NULL, tmp, S, INTER, H);
        if (i == 0 && trace &&
            trace(trace_user, "bb0_mlp_out", tmp, S, H) != 0) goto cleanup;
        for (size_t j = 0; j < (size_t)S * H; j++) h[j] += tmp[j];
        if (trace) {
            char trace_name[32];
            snprintf(trace_name, sizeof(trace_name), "bb_layer%d", i);
            if (trace(trace_user, trace_name, h, S, H) != 0) goto cleanup;
        }
    }

    /* --- final norm + masking output --- */
    const IroTensor *fn = bb_get(b, IRO_BB_PREFIX "final_norm.weight");
    iro_layernorm(h, (const float *)fn->data, NULL, tmp, S, H, b->norm_eps);
    for (int s = 0; s < S; s++) {
        float m = mask[s] ? 1.0f : 0.0f;
        for (int d = 0; d < H; d++)
            out[(size_t)s * H + d] = tmp[(size_t)s * H + d] * m;
    }
    if (trace && trace(trace_user, "backbone_out", out, S, H) != 0) goto cleanup;
    rc = 0;

cleanup:
    free(h); free(tmp); free(qkv); free(attnc); free(mlpw); free(mlpa);
    free(scores); free(qh); free(kh); free(vh);
    return rc;
}

int iro_backbone_forward(IroBackbone *b, const int32_t *ids, const uint8_t *mask,
                         int S, float *out) {
    return iro_backbone_forward_trace(b, ids, mask, S, out, NULL, NULL);
}
