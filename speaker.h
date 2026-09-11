/* speaker.h — reference-latent speaker/style encoder. */
#ifndef IRO_SPEAKER_H
#define IRO_SPEAKER_H

#include "irodori.h"

#define IRO_SPEAKER_LAYERS 8
#define IRO_SPEAKER_DIM 768
#define IRO_SPEAKER_HEADS 12
#define IRO_SPEAKER_HEAD_DIM 64
#define IRO_SPEAKER_PATCH 4

typedef struct {
    const float *attention_norm;
    const float *wq;
    const float *wk;
    const float *wv;
    const float *wo;
    const float *gate;
    const float *q_norm;
    const float *k_norm;
    const float *mlp_norm;
    const float *w1;
    const float *w2;
    const float *w3;
} IroSpeakerBlock;

typedef struct {
    const float *in_w;
    const float *in_b;
    const float *out_norm;
    IroSpeakerBlock blocks[IRO_SPEAKER_LAYERS];
    float norm_eps;
} IroSpeakerEncoder;

typedef int (*IroSpeakerTraceFn)(void *user, const char *name,
                                const float *data, int frames, int channels);

int iro_speaker_init(IroSpeakerEncoder *encoder, const IroSafetensors *st);

/* Encode one unpadded DACVAE mean latent [latent_frames,32]. The final partial
   group is dropped like upstream. output is allocated as [patched_frames+1,768]
   and starts with the global mean token. */
int iro_speaker_encode(const IroSpeakerEncoder *encoder,
                       const float *latent, int latent_frames,
                       float **output, int *output_frames,
                       IroSpeakerTraceFn trace, void *trace_user);

#endif
