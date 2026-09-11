/* sampler.h — cached rectified-flow Euler samplers. */
#ifndef IRO_SAMPLER_H
#define IRO_SAMPLER_H

#include "dit.h"

typedef struct {
    int steps;
    float init_scale;
    float cfg_scale_text;
    float cfg_scale_speaker;
    float cfg_scale_caption;
    float cfg_min_t;
    float cfg_max_t;
} IroEulerConfig;

typedef struct {
    int context_tokens;
    size_t context_cache_bytes;
} IroEulerStats;

/* Reusable scratch buffers for warm multi-request sampling. Initialize as {0}
   and release with iro_euler_workspace_free(). */
typedef struct {
    void *impl;
} IroEulerWorkspace;

void iro_euler_workspace_free(IroEulerWorkspace *workspace);

/* Called after each raw DiT velocity and before the Euler update.  x_t and
   velocity have shape [batch, sequence_length, 32] and are valid only during
   the callback.  Returning non-zero aborts sampling. */
typedef int (*IroEulerTraceFn)(void *user, int step, float timestep, int batch,
                              const float *x_t, const float *velocity);

/* Text-only independent CFG sampler.  The input text is [text_tokens,512],
   mask is [text_tokens], noise/output are [sequence_length,32]. */
int iro_sample_euler_text(const IroDiT *dit,
                          const float *text, const uint8_t *text_mask,
                          int text_tokens, const float *noise,
                          int sequence_length, const IroEulerConfig *config,
                          float *output, IroEulerTraceFn trace,
                          void *trace_user, IroEulerStats *stats);

/* Independent text+speaker CFG. speaker contains the already-normalized
   speaker context [speaker_tokens,768], including its prepended mean token. */
int iro_sample_euler_speaker(const IroDiT *dit,
                             const float *text, const uint8_t *text_mask,
                             int text_tokens,
                             const float *speaker, int speaker_tokens,
                             const float *noise, int sequence_length,
                             const IroEulerConfig *config, float *output,
                             IroEulerTraceFn trace, void *trace_user,
                             IroEulerStats *stats);

/* Full independent CFG over text + optional speaker + optional caption. */
int iro_sample_euler_conditions(const IroDiT *dit,
                                const float *text, const uint8_t *text_mask,
                                int text_tokens,
                                const float *speaker, int speaker_tokens,
                                const float *caption, int caption_tokens,
                                const float *noise, int sequence_length,
                                const IroEulerConfig *config, float *output,
                                IroEulerTraceFn trace, void *trace_user,
                                IroEulerStats *stats);

int iro_sample_euler_conditions_workspace(
    const IroDiT *dit,
    const float *text, const uint8_t *text_mask, int text_tokens,
    const float *speaker, int speaker_tokens,
    const float *caption, int caption_tokens,
    const float *noise, int sequence_length,
    const IroEulerConfig *config, float *output,
    IroEulerTraceFn trace, void *trace_user, IroEulerStats *stats,
    IroEulerWorkspace *workspace);

#endif
