/* condition.c — Irodori residual_mlp projector and output RMSNorm. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "condition.h"
#include "ops.h"

enum {
    BACKBONE_DIM = 768,
    CONDITION_DIM = 512,
    RESIDUAL_DIM = 1024,
};

static const float *require_f32(const IroSafetensors *st, const char *name,
                                int ndims, uint64_t d0, uint64_t d1) {
    const IroTensor *t = iro_st_get(st, name);
    if (!t) {
        fprintf(stderr, "projector: tensor tidak ada: %s\n", name);
        return NULL;
    }
    if (t->dtype != IRO_ST_F32 || t->ndims != ndims || t->shape[0] != d0 ||
        (ndims == 2 && t->shape[1] != d1)) {
        fprintf(stderr, "projector: dtype/shape tidak cocok: %s\n", name);
        return NULL;
    }
    return (const float *)t->data;
}

static int projector_init(IroTextProjector *p, const IroSafetensors *st,
                          const char *prefix, const char *norm_name) {
    if (!p || !st) return -1;
    memset(p, 0, sizeof(*p));
    char name[128];
#define LOAD(field, suffix, nd, d0, d1) do {                         \
        snprintf(name, sizeof(name), "%s.%s", prefix, suffix);      \
        p->field = require_f32(st, name, nd, d0, d1);                \
    } while (0)
    LOAD(projector_w, "projector.weight", 2,
                                 CONDITION_DIM, BACKBONE_DIM);
    LOAD(projector_b, "projector.bias", 1,
                                 CONDITION_DIM, 0);
    LOAD(residual_norm_w, "residual_norm.weight", 1,
                                     BACKBONE_DIM, 0);
    LOAD(residual_up_w, "residual_up.weight", 2,
                                   RESIDUAL_DIM, BACKBONE_DIM);
    LOAD(residual_up_b, "residual_up.bias", 1,
                                   RESIDUAL_DIM, 0);
    LOAD(residual_down_w, "residual_down.weight", 2,
                                     CONDITION_DIM, RESIDUAL_DIM);
    LOAD(residual_down_b, "residual_down.bias", 1,
                                     CONDITION_DIM, 0);
#undef LOAD
    p->output_norm_w = require_f32(st, norm_name, 1, CONDITION_DIM, 0);
    p->norm_eps = 1e-5f;

    return p->projector_w && p->projector_b && p->residual_norm_w &&
           p->residual_up_w && p->residual_up_b && p->residual_down_w &&
           p->residual_down_b && p->output_norm_w ? 0 : -1;
}

int iro_text_projector_init(IroTextProjector *p, const IroSafetensors *st) {
    return projector_init(p, st, "text_encoder", "text_norm.weight");
}

int iro_caption_projector_init(IroTextProjector *p, const IroSafetensors *st) {
    return projector_init(p, st, "caption_encoder", "caption_norm.weight");
}

int iro_text_projector_forward(const IroTextProjector *p,
                               const float *backbone_state,
                               const uint8_t *mask, int S,
                               float *projected, float *text_state) {
    if (!p || !backbone_state || !mask || S <= 0 || !projected || !text_state)
        return -1;

    float *norm = malloc(sizeof(float) * (size_t)S * BACKBONE_DIM);
    float *up = malloc(sizeof(float) * (size_t)S * RESIDUAL_DIM);
    float *residual = malloc(sizeof(float) * (size_t)S * CONDITION_DIM);
    if (!norm || !up || !residual) {
        free(norm);
        free(up);
        free(residual);
        return -1;
    }

    iro_linear(backbone_state, p->projector_w, p->projector_b,
               projected, S, BACKBONE_DIM, CONDITION_DIM);
    iro_rmsnorm(backbone_state, p->residual_norm_w, norm,
                S, BACKBONE_DIM, p->norm_eps);
    iro_linear(norm, p->residual_up_w, p->residual_up_b,
               up, S, BACKBONE_DIM, RESIDUAL_DIM);
    iro_silu_inplace(up, (size_t)S * RESIDUAL_DIM);
    iro_linear(up, p->residual_down_w, p->residual_down_b,
               residual, S, RESIDUAL_DIM, CONDITION_DIM);

    for (int s = 0; s < S; s++) {
        float keep = mask[s] ? 1.0f : 0.0f;
        for (int d = 0; d < CONDITION_DIM; d++) {
            size_t i = (size_t)s * CONDITION_DIM + d;
            projected[i] = (projected[i] + residual[i]) * keep;
        }
    }
    iro_rmsnorm(projected, p->output_norm_w, text_state,
                S, CONDITION_DIM, p->norm_eps);

    free(norm);
    free(up);
    free(residual);
    return 0;
}
