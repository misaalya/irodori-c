/* duration.h — text-only token-sum duration predictor. */
#ifndef IRO_DURATION_H
#define IRO_DURATION_H

#include <stdint.h>

#include "irodori.h"

#define IRO_DURATION_LAYERS 3

typedef struct {
    const float *norm_w;
    const float *speaker_mod_w;
    const float *speaker_mod_b;
    const float *caption_mod_w;
    const float *caption_mod_b;
    const float *w1;
    const float *w2;
    const float *w3;
} IroDurationBlock;

typedef struct {
    const float *null_speaker;
    const float *null_caption;
    const float *input_w;
    const float *input_b;
    IroDurationBlock blocks[IRO_DURATION_LAYERS];
    const float *out_norm_w;
    const float *out_w;
    const float *out_b;
    float norm_eps;
} IroDurationPredictor;

int iro_duration_init(IroDurationPredictor *p, const IroSafetensors *st);

/* Conditioned duration path. speaker_vec is speaker_state[0] (the prepended
   global mean token); caption_vec is the masked mean caption token. A NULL
   pointer selects the checkpoint's learned null vector for that branch. */
int iro_duration_forward(const IroDurationPredictor *p,
                         const float *text_state,
                         const uint8_t *text_mask, int S,
                         const float *speaker_vec,
                         const float *caption_vec,
                         float *out_log_frames);

/* Text-only/no-caption compatibility wrapper. */
int iro_duration_forward_text(const IroDurationPredictor *p,
                              const float *text_state,
                              const uint8_t *text_mask, int S,
                              float *out_log_frames);

/* Runtime clamp/round policy for the 48 kHz, hop-1920 codec. */
int iro_duration_frame_count(float log_frames, float duration_scale,
                             float min_seconds, float max_seconds,
                             int sample_rate, int hop_length);

#endif
