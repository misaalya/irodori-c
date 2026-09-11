/* speaker.c — pure-C 8-layer reference speaker Transformer. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "speaker.h"
#include "ops.h"

enum {
    LATENT_DIM = 32,
    PATCHED_DIM = LATENT_DIM * IRO_SPEAKER_PATCH,
    MLP_DIM = 1996,
};

static const float *require_f32(const IroSafetensors *st, const char *name,
                                int ndims, uint64_t d0, uint64_t d1) {
    const IroTensor *t = iro_st_get(st, name);
    if (!t) {
        fprintf(stderr, "speaker: tensor tidak ada: %s\n", name);
        return NULL;
    }
    if (t->dtype != IRO_ST_F32 || t->ndims != ndims || t->shape[0] != d0 ||
        (ndims == 2 && t->shape[1] != d1)) {
        fprintf(stderr, "speaker: dtype/shape tidak cocok: %s\n", name);
        return NULL;
    }
    return (const float *)t->data;
}

int iro_speaker_init(IroSpeakerEncoder *encoder, const IroSafetensors *st) {
    if (!encoder || !st) return -1;
    memset(encoder, 0, sizeof(*encoder));
    encoder->norm_eps = 1e-5f;
    encoder->in_w = require_f32(st, "speaker_encoder.in_proj.weight", 2,
                                IRO_SPEAKER_DIM, PATCHED_DIM);
    encoder->in_b = require_f32(st, "speaker_encoder.in_proj.bias", 1,
                                IRO_SPEAKER_DIM, 0);
    encoder->out_norm = require_f32(st, "speaker_norm.weight", 1,
                                    IRO_SPEAKER_DIM, 0);
    for (int layer = 0; layer < IRO_SPEAKER_LAYERS; layer++) {
        IroSpeakerBlock *b = &encoder->blocks[layer];
        char name[160];
#define LOAD1(field, suffix, d0) do {                                        \
            snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.%s",   \
                     layer, suffix);                                        \
            b->field = require_f32(st, name, 1, d0, 0);                     \
        } while (0)
#define LOAD2(field, suffix, d0, d1) do {                                   \
            snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.%s",   \
                     layer, suffix);                                        \
            b->field = require_f32(st, name, 2, d0, d1);                    \
        } while (0)
        LOAD1(attention_norm, "attention_norm.weight", IRO_SPEAKER_DIM);
        LOAD2(wq, "attention.wq.weight", IRO_SPEAKER_DIM, IRO_SPEAKER_DIM);
        LOAD2(wk, "attention.wk.weight", IRO_SPEAKER_DIM, IRO_SPEAKER_DIM);
        LOAD2(wv, "attention.wv.weight", IRO_SPEAKER_DIM, IRO_SPEAKER_DIM);
        LOAD2(wo, "attention.wo.weight", IRO_SPEAKER_DIM, IRO_SPEAKER_DIM);
        LOAD2(gate, "attention.gate.weight", IRO_SPEAKER_DIM, IRO_SPEAKER_DIM);
        LOAD2(q_norm, "attention.q_norm.weight", IRO_SPEAKER_HEADS,
              IRO_SPEAKER_HEAD_DIM);
        LOAD2(k_norm, "attention.k_norm.weight", IRO_SPEAKER_HEADS,
              IRO_SPEAKER_HEAD_DIM);
        LOAD1(mlp_norm, "mlp_norm.weight", IRO_SPEAKER_DIM);
        LOAD2(w1, "mlp.w1.weight", MLP_DIM, IRO_SPEAKER_DIM);
        LOAD2(w2, "mlp.w2.weight", IRO_SPEAKER_DIM, MLP_DIM);
        LOAD2(w3, "mlp.w3.weight", MLP_DIM, IRO_SPEAKER_DIM);
#undef LOAD1
#undef LOAD2
    }
    if (!encoder->in_w || !encoder->in_b || !encoder->out_norm) return -1;
    for (int i = 0; i < IRO_SPEAKER_LAYERS; i++) {
        const IroSpeakerBlock *b = &encoder->blocks[i];
        if (!b->attention_norm || !b->wq || !b->wk || !b->wv || !b->wo ||
            !b->gate || !b->q_norm || !b->k_norm || !b->mlp_norm ||
            !b->w1 || !b->w2 || !b->w3)
            return -1;
    }
    return 0;
}

static void head_rmsnorm(float *x, const float *weight, int frames, float eps) {
    for (int s = 0; s < frames; s++) {
        float *token = x + (size_t)s * IRO_SPEAKER_DIM;
        for (int h = 0; h < IRO_SPEAKER_HEADS; h++) {
            float *row = token + h * IRO_SPEAKER_HEAD_DIM;
            const float *w = weight + h * IRO_SPEAKER_HEAD_DIM;
            float ms = 0.0f;
            for (int d = 0; d < IRO_SPEAKER_HEAD_DIM; d++) ms += row[d] * row[d];
            float inv = 1.0f / sqrtf(ms / IRO_SPEAKER_HEAD_DIM + eps);
            for (int d = 0; d < IRO_SPEAKER_HEAD_DIM; d++) row[d] *= inv * w[d];
        }
    }
}

static void rope(float *x, int frames) {
    for (int s = 0; s < frames; s++) {
        float *token = x + (size_t)s * IRO_SPEAKER_DIM;
        for (int pair = 0; pair < IRO_SPEAKER_HEAD_DIM / 2; pair++) {
            double inv = 1.0 / pow(10000.0,
                                   (double)(2 * pair) / IRO_SPEAKER_HEAD_DIM);
            float c = (float)cos((double)s * inv);
            float sn = (float)sin((double)s * inv);
            for (int h = 0; h < IRO_SPEAKER_HEADS; h++) {
                float *row = token + h * IRO_SPEAKER_HEAD_DIM;
                int d = 2 * pair;
                float x0 = row[d], x1 = row[d + 1];
                row[d] = x0 * c - x1 * sn;
                row[d + 1] = x0 * sn + x1 * c;
            }
        }
    }
}

static int self_attention(const IroSpeakerBlock *b, float eps,
                          const float *x, int frames, float *out) {
    size_t state_n = (size_t)frames * IRO_SPEAKER_DIM;
    float *q = malloc(sizeof(float) * state_n);
    float *k = malloc(sizeof(float) * state_n);
    float *v = malloc(sizeof(float) * state_n);
    float *gate = malloc(sizeof(float) * state_n);
    float *mix = malloc(sizeof(float) * state_n);
    float *qh = malloc(sizeof(float) * (size_t)frames * IRO_SPEAKER_HEAD_DIM);
    float *kh = malloc(sizeof(float) * (size_t)frames * IRO_SPEAKER_HEAD_DIM);
    float *vh = malloc(sizeof(float) * (size_t)frames * IRO_SPEAKER_HEAD_DIM);
    float *scores = malloc(sizeof(float) * (size_t)frames * frames);
    float *mixh = malloc(sizeof(float) * (size_t)frames * IRO_SPEAKER_HEAD_DIM);
    if (!q || !k || !v || !gate || !mix || !qh || !kh || !vh || !scores || !mixh) {
        free(q); free(k); free(v); free(gate); free(mix);
        free(qh); free(kh); free(vh); free(scores); free(mixh);
        return -1;
    }
    iro_linear(x, b->wq, NULL, q, frames, IRO_SPEAKER_DIM, IRO_SPEAKER_DIM);
    iro_linear(x, b->wk, NULL, k, frames, IRO_SPEAKER_DIM, IRO_SPEAKER_DIM);
    iro_linear(x, b->wv, NULL, v, frames, IRO_SPEAKER_DIM, IRO_SPEAKER_DIM);
    iro_linear(x, b->gate, NULL, gate, frames, IRO_SPEAKER_DIM, IRO_SPEAKER_DIM);
    head_rmsnorm(q, b->q_norm, frames, eps);
    head_rmsnorm(k, b->k_norm, frames, eps);
    rope(q, frames);
    rope(k, frames);

    float scale = 1.0f / sqrtf((float)IRO_SPEAKER_HEAD_DIM);
    for (int h = 0; h < IRO_SPEAKER_HEADS; h++) {
        for (int s = 0; s < frames; s++) {
            memcpy(qh + (size_t)s * IRO_SPEAKER_HEAD_DIM,
                   q + (size_t)s * IRO_SPEAKER_DIM + h * IRO_SPEAKER_HEAD_DIM,
                   sizeof(float) * IRO_SPEAKER_HEAD_DIM);
            memcpy(kh + (size_t)s * IRO_SPEAKER_HEAD_DIM,
                   k + (size_t)s * IRO_SPEAKER_DIM + h * IRO_SPEAKER_HEAD_DIM,
                   sizeof(float) * IRO_SPEAKER_HEAD_DIM);
            memcpy(vh + (size_t)s * IRO_SPEAKER_HEAD_DIM,
                   v + (size_t)s * IRO_SPEAKER_DIM + h * IRO_SPEAKER_HEAD_DIM,
                   sizeof(float) * IRO_SPEAKER_HEAD_DIM);
        }
        iro_linear(qh, kh, NULL, scores, frames, IRO_SPEAKER_HEAD_DIM, frames);
        for (size_t i = 0; i < (size_t)frames * frames; i++) scores[i] *= scale;
        iro_softmax_row(scores, frames, frames);
        iro_matmul(scores, vh, mixh, frames, frames, IRO_SPEAKER_HEAD_DIM);
        for (int s = 0; s < frames; s++)
            memcpy(mix + (size_t)s * IRO_SPEAKER_DIM + h * IRO_SPEAKER_HEAD_DIM,
                   mixh + (size_t)s * IRO_SPEAKER_HEAD_DIM,
                   sizeof(float) * IRO_SPEAKER_HEAD_DIM);
    }
    iro_sigmoid_mul_inplace(mix, gate, state_n);
    iro_linear(mix, b->wo, NULL, out, frames,
               IRO_SPEAKER_DIM, IRO_SPEAKER_DIM);
    free(q); free(k); free(v); free(gate); free(mix);
    free(qh); free(kh); free(vh); free(scores); free(mixh);
    return 0;
}

static int run_block(const IroSpeakerBlock *b, float eps,
                     float *state, int frames, float *norm, float *branch,
                     float *w1, float *w3) {
    size_t state_n = (size_t)frames * IRO_SPEAKER_DIM;
    iro_rmsnorm(state, b->attention_norm, norm, frames, IRO_SPEAKER_DIM, eps);
    if (self_attention(b, eps, norm, frames, branch) != 0) return -1;
    for (size_t i = 0; i < state_n; i++) state[i] += branch[i];
    iro_rmsnorm(state, b->mlp_norm, norm, frames, IRO_SPEAKER_DIM, eps);
    iro_linear(norm, b->w1, NULL, w1, frames, IRO_SPEAKER_DIM, MLP_DIM);
    iro_linear(norm, b->w3, NULL, w3, frames, IRO_SPEAKER_DIM, MLP_DIM);
    iro_silu_mul_inplace(w1, w3, (size_t)frames * MLP_DIM);
    /* Accumulate the MLP projection directly into the residual state. */
    iro_linear_add(w1, b->w2, NULL, state, state,
                   frames, MLP_DIM, IRO_SPEAKER_DIM);
    return 0;
}

int iro_speaker_encode(const IroSpeakerEncoder *encoder,
                       const float *latent, int latent_frames,
                       float **output, int *output_frames,
                       IroSpeakerTraceFn trace, void *trace_user) {
    if (!encoder || !latent || latent_frames < IRO_SPEAKER_PATCH ||
        !output || !output_frames)
        return -1;
    *output = NULL;
    *output_frames = 0;
    int frames = latent_frames / IRO_SPEAKER_PATCH;
    size_t state_n = (size_t)frames * IRO_SPEAKER_DIM;
    float *patched = malloc(sizeof(float) * (size_t)frames * PATCHED_DIM);
    float *state = malloc(sizeof(float) * state_n);
    float *norm = malloc(sizeof(float) * state_n);
    float *branch = malloc(sizeof(float) * state_n);
    float *w1 = malloc(sizeof(float) * (size_t)frames * MLP_DIM);
    float *w3 = malloc(sizeof(float) * (size_t)frames * MLP_DIM);
    float *result = malloc(sizeof(float) * (size_t)(frames + 1) * IRO_SPEAKER_DIM);
    if (!patched || !state || !norm || !branch || !w1 || !w3 || !result) {
        free(patched); free(state); free(norm); free(branch); free(w1); free(w3); free(result);
        return -1;
    }
    /* Source [T,32] groups four consecutive frames into [S,128]. */
    memcpy(patched, latent, sizeof(float) * (size_t)frames * PATCHED_DIM);
    iro_linear(patched, encoder->in_w, encoder->in_b, state,
               frames, PATCHED_DIM, IRO_SPEAKER_DIM);
    for (size_t i = 0; i < state_n; i++) state[i] /= 6.0f;
    if (trace && trace(trace_user, "speaker_in", state, frames,
                       IRO_SPEAKER_DIM) != 0)
        goto fail;
    for (int layer = 0; layer < IRO_SPEAKER_LAYERS; layer++) {
        if (run_block(&encoder->blocks[layer], encoder->norm_eps,
                      state, frames, norm, branch, w1, w3) != 0)
            goto fail;
        if (trace) {
            char name[32];
            snprintf(name, sizeof(name), "speaker_b%d", layer);
            if (trace(trace_user, name, state, frames, IRO_SPEAKER_DIM) != 0)
                goto fail;
        }
    }
    iro_rmsnorm(state, encoder->out_norm, norm, frames,
                IRO_SPEAKER_DIM, encoder->norm_eps);
    if (trace && trace(trace_user, "speaker_norm", norm, frames,
                       IRO_SPEAKER_DIM) != 0)
        goto fail;
    memset(result, 0, sizeof(float) * IRO_SPEAKER_DIM);
    for (int s = 0; s < frames; s++)
        for (int d = 0; d < IRO_SPEAKER_DIM; d++)
            result[d] += norm[(size_t)s * IRO_SPEAKER_DIM + d];
    for (int d = 0; d < IRO_SPEAKER_DIM; d++) result[d] /= (float)frames;
    memcpy(result + IRO_SPEAKER_DIM, norm, sizeof(float) * state_n);
    if (trace && trace(trace_user, "speaker_state", result, frames + 1,
                       IRO_SPEAKER_DIM) != 0)
        goto fail;
    *output = result;
    *output_frames = frames + 1;
    free(patched); free(state); free(norm); free(branch); free(w1); free(w3);
    return 0;

fail:
    free(patched); free(state); free(norm); free(branch); free(w1); free(w3); free(result);
    return -1;
}
