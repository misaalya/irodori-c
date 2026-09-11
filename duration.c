/* duration.c — token_sum_dual_adarn_zero_no_aux inference. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "duration.h"
#include "ops.h"

enum {
    TEXT_DIM = 512,
    HIDDEN_DIM = 1024,
    SPEAKER_DIM = 768,
    MOD_DIM = 3 * HIDDEN_DIM,
};

static const float *require_f32(const IroSafetensors *st, const char *name,
                                int ndims, uint64_t d0, uint64_t d1) {
    const IroTensor *t = iro_st_get(st, name);
    if (!t) {
        fprintf(stderr, "duration: tensor tidak ada: %s\n", name);
        return NULL;
    }
    if (t->dtype != IRO_ST_F32 || t->ndims != ndims || t->shape[0] != d0 ||
        (ndims == 2 && t->shape[1] != d1)) {
        fprintf(stderr, "duration: dtype/shape tidak cocok: %s\n", name);
        return NULL;
    }
    return (const float *)t->data;
}

int iro_duration_init(IroDurationPredictor *p, const IroSafetensors *st) {
    if (!p || !st) return -1;
    memset(p, 0, sizeof(*p));
    p->null_speaker = require_f32(st, "duration_predictor.null_speaker", 1,
                                  SPEAKER_DIM, 0);
    p->null_caption = require_f32(st, "duration_predictor.null_caption", 1,
                                  TEXT_DIM, 0);
    p->input_w = require_f32(st, "duration_predictor.token_input_proj.weight", 2,
                             HIDDEN_DIM, TEXT_DIM);
    p->input_b = require_f32(st, "duration_predictor.token_input_proj.bias", 1,
                             HIDDEN_DIM, 0);

    for (int i = 0; i < IRO_DURATION_LAYERS; i++) {
        char name[128];
        IroDurationBlock *b = &p->blocks[i];
#define LOAD_BLOCK(field, suffix, nd, d0, d1) do {                         \
            snprintf(name, sizeof(name),                                  \
                     "duration_predictor.token_blocks.%d.%s", i, suffix); \
            b->field = require_f32(st, name, nd, d0, d1);                  \
        } while (0)
        LOAD_BLOCK(norm_w, "norm.weight", 1, HIDDEN_DIM, 0);
        LOAD_BLOCK(speaker_mod_w, "modulation.weight", 2, MOD_DIM, SPEAKER_DIM);
        LOAD_BLOCK(speaker_mod_b, "modulation.bias", 1, MOD_DIM, 0);
        LOAD_BLOCK(caption_mod_w, "caption_modulation.weight", 2, MOD_DIM, TEXT_DIM);
        LOAD_BLOCK(caption_mod_b, "caption_modulation.bias", 1, MOD_DIM, 0);
        LOAD_BLOCK(w1, "mlp.w1.weight", 2, HIDDEN_DIM, HIDDEN_DIM);
        LOAD_BLOCK(w2, "mlp.w2.weight", 2, HIDDEN_DIM, HIDDEN_DIM);
        LOAD_BLOCK(w3, "mlp.w3.weight", 2, HIDDEN_DIM, HIDDEN_DIM);
#undef LOAD_BLOCK
    }

    p->out_norm_w = require_f32(st, "duration_predictor.token_out_norm.weight", 1,
                                HIDDEN_DIM, 0);
    p->out_w = require_f32(st, "duration_predictor.token_out_proj.weight", 2,
                           1, HIDDEN_DIM);
    p->out_b = require_f32(st, "duration_predictor.token_out_proj.bias", 1, 1, 0);
    p->norm_eps = 1e-5f;

    if (!p->null_speaker || !p->null_caption || !p->input_w || !p->input_b ||
        !p->out_norm_w || !p->out_w || !p->out_b)
        return -1;
    for (int i = 0; i < IRO_DURATION_LAYERS; i++) {
        const IroDurationBlock *b = &p->blocks[i];
        if (!b->norm_w || !b->speaker_mod_w || !b->speaker_mod_b ||
            !b->caption_mod_w || !b->caption_mod_b ||
            !b->w1 || !b->w2 || !b->w3)
            return -1;
    }
    return 0;
}

static float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

int iro_duration_forward(const IroDurationPredictor *p,
                         const float *text_state,
                         const uint8_t *text_mask, int S,
                         const float *speaker_vec,
                         const float *caption_vec,
                         float *out_log_frames) {
    if (!p || !text_state || !text_mask || S <= 0 || !out_log_frames) return -1;

    size_t hidden_n = (size_t)S * HIDDEN_DIM;
    float *h = malloc(sizeof(float) * hidden_n);
    float *norm = malloc(sizeof(float) * hidden_n);
    float *mod_h = malloc(sizeof(float) * hidden_n);
    float *a = malloc(sizeof(float) * hidden_n);
    float *b = malloc(sizeof(float) * hidden_n);
    float *mlp = malloc(sizeof(float) * hidden_n);
    float *logits = malloc(sizeof(float) * (size_t)S);
    if (!h || !norm || !mod_h || !a || !b || !mlp || !logits) {
        free(h); free(norm); free(mod_h); free(a); free(b); free(mlp); free(logits);
        return -1;
    }

    float speaker[SPEAKER_DIM];
    float caption[TEXT_DIM];
    memcpy(speaker, speaker_vec ? speaker_vec : p->null_speaker, sizeof(speaker));
    memcpy(caption, caption_vec ? caption_vec : p->null_caption, sizeof(caption));
    iro_silu_inplace(speaker, SPEAKER_DIM);
    iro_silu_inplace(caption, TEXT_DIM);

    iro_linear(text_state, p->input_w, p->input_b,
               h, S, TEXT_DIM, HIDDEN_DIM);
    for (int layer = 0; layer < IRO_DURATION_LAYERS; layer++) {
        const IroDurationBlock *block = &p->blocks[layer];
        float speaker_mod[MOD_DIM], caption_mod[MOD_DIM];
        iro_linear(speaker, block->speaker_mod_w, block->speaker_mod_b,
                   speaker_mod, 1, SPEAKER_DIM, MOD_DIM);
        iro_linear(caption, block->caption_mod_w, block->caption_mod_b,
                   caption_mod, 1, TEXT_DIM, MOD_DIM);
        iro_rmsnorm(h, block->norm_w, norm, S, HIDDEN_DIM, p->norm_eps);

        const float *shift_s = speaker_mod;
        const float *scale_s = speaker_mod + HIDDEN_DIM;
        const float *gate_s = speaker_mod + 2 * HIDDEN_DIM;
        const float *shift_c = caption_mod;
        const float *scale_c = caption_mod + HIDDEN_DIM;
        const float *gate_c = caption_mod + 2 * HIDDEN_DIM;
        for (int s = 0; s < S; s++) {
            for (int d = 0; d < HIDDEN_DIM; d++) {
                size_t j = (size_t)s * HIDDEN_DIM + d;
                float shift = shift_s[d] + shift_c[d];
                float scale = scale_s[d] + scale_c[d];
                mod_h[j] = norm[j] * (1.0f + scale) + shift;
            }
        }

        iro_linear(mod_h, block->w1, NULL, a, S, HIDDEN_DIM, HIDDEN_DIM);
        iro_linear(mod_h, block->w3, NULL, b, S, HIDDEN_DIM, HIDDEN_DIM);
        iro_silu_mul_inplace(a, b, hidden_n);
        iro_linear(a, block->w2, NULL, mlp, S, HIDDEN_DIM, HIDDEN_DIM);
        for (int s = 0; s < S; s++) {
            for (int d = 0; d < HIDDEN_DIM; d++) {
                size_t j = (size_t)s * HIDDEN_DIM + d;
                float gate = tanhf(gate_s[d] + gate_c[d]);
                h[j] += gate * mlp[j];
            }
        }
    }

    iro_rmsnorm(h, p->out_norm_w, norm, S, HIDDEN_DIM, p->norm_eps);
    iro_linear(norm, p->out_w, p->out_b, logits, S, HIDDEN_DIM, 1);
    float total = 0.0f;
    for (int s = 0; s < S; s++)
        if (text_mask[s]) total += softplus(logits[s]);
    *out_log_frames = log1pf(fmaxf(total, 0.0f));

    free(h); free(norm); free(mod_h); free(a); free(b); free(mlp); free(logits);
    return 0;
}

int iro_duration_forward_text(const IroDurationPredictor *p,
                              const float *text_state,
                              const uint8_t *text_mask, int S,
                              float *out_log_frames) {
    return iro_duration_forward(p, text_state, text_mask, S,
                                NULL, NULL, out_log_frames);
}

int iro_duration_frame_count(float log_frames, float duration_scale,
                             float min_seconds, float max_seconds,
                             int sample_rate, int hop_length) {
    if (duration_scale <= 0.0f || min_seconds <= 0.0f ||
        max_seconds < min_seconds || sample_rate <= 0 || hop_length <= 0)
        return -1;
    double predicted = expm1((double)log_frames) * duration_scale;
    int frames = (int)nearbyint(predicted);
    int min_frames = (int)ceil((double)min_seconds * sample_rate / hop_length);
    int max_frames = (int)floor((double)max_seconds * sample_rate / hop_length);
    if (frames < min_frames) frames = min_frames;
    if (frames > max_frames) frames = max_frames;
    return frames;
}
