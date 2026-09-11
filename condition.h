/* condition.h — text conditioning projection after ModernBERT. */
#ifndef IRO_CONDITION_H
#define IRO_CONDITION_H

#include <stdint.h>

#include "irodori.h"

typedef struct {
    const float *projector_w;
    const float *projector_b;
    const float *residual_norm_w;
    const float *residual_up_w;
    const float *residual_up_b;
    const float *residual_down_w;
    const float *residual_down_b;
    const float *output_norm_w;
    float norm_eps;
} IroTextProjector;

int iro_text_projector_init(IroTextProjector *p, const IroSafetensors *st);
int iro_caption_projector_init(IroTextProjector *p, const IroSafetensors *st);

/* backbone_state is [S,768]. projected and text_state are [S,512]. */
int iro_text_projector_forward(const IroTextProjector *p,
                               const float *backbone_state,
                               const uint8_t *mask, int S,
                               float *projected, float *text_state);

#endif
