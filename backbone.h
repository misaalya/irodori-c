/* backbone.h — ModernBERT-ja 25L (text/caption encoder) */
#ifndef IRO_BACKBONE_H
#define IRO_BACKBONE_H

#include "irodori.h"

#define IRO_BB_PREFIX "pretrained_text_backbone.backbone."

typedef struct {
    const IroSafetensors *st;
    int n_layers;   /* 25 */
    int hidden;     /* 768 */
    int heads;      /* 12 */
    int head_dim;   /* 64 */
    int inter;      /* 3072 */
    int sliding;    /* 64 (config.sliding_window = local_attention 128 / 2) */
    float norm_eps; /* 1e-5 */
    int pad_id;     /* 3 */
    /* rope tables per layer-type: cos/sin [max_pos][64] */
    float *rope_cos_full, *rope_sin_full;   /* theta 160000 */
    float *rope_cos_slide, *rope_sin_slide; /* theta 10000 */
    int rope_len;
} IroBackbone;

/* Optional zero-copy trace hook used by golden regression tests.  The data is
   valid only for the duration of the callback. */
typedef int (*IroBackboneTraceFn)(void *user, const char *name,
                                  const float *data, int rows, int cols);

int  iro_backbone_init(IroBackbone *b, const IroSafetensors *st, int max_pos);
/* forward: ids [S] + mask [S] (1=valid) → out [S,768] (sudah di-mask: out = h*mask) */
int  iro_backbone_forward(IroBackbone *b, const int32_t *ids, const uint8_t *mask,
                          int S, float *out);
int  iro_backbone_forward_trace(IroBackbone *b, const int32_t *ids,
                                const uint8_t *mask, int S, float *out,
                                IroBackboneTraceFn trace, void *trace_user);
void iro_backbone_free(IroBackbone *b);

#endif
