/* dacvae.h — deterministic text-path DACVAE decoder. */
#ifndef IRO_DACVAE_H
#define IRO_DACVAE_H

#include "irodori.h"

#define IRO_DACVAE_STAGES 4
#define IRO_DACVAE_RESIDUALS 3

typedef struct {
    const float *weight; /* Export-packed [out_channels,kernel,in_channels]. */
    const float *bias;
    int in_channels;
    int out_channels;
    int kernel;
    int stride;
    int dilation;
    int padding;
} IroConv1d;

typedef struct {
    /* Export-packed layout [in_channels,kernel,out_channels]. */
    const float *weight;
    const float *bias;
    int in_channels;
    int out_channels;
    int kernel;
    int stride;
    int padding;
    int output_padding;
} IroConvTranspose1d;

typedef struct {
    const float *alpha0;
    const float *alpha1;
    IroConv1d conv0;
    IroConv1d conv1;
} IroDACVAEResidual;

typedef struct {
    const float *alpha;
    IroConvTranspose1d upsample;
    IroDACVAEResidual residual[IRO_DACVAE_RESIDUALS];
} IroDACVAEStage;

typedef struct {
    IroConv1d quantizer_out;
    IroConv1d initial;
    IroDACVAEStage stage[IRO_DACVAE_STAGES];
    const float *final_alpha;
    IroConv1d final;
} IroDACVAEDecoder;

typedef struct {
    IroDACVAEResidual residual[IRO_DACVAE_RESIDUALS];
    const float *alpha;
    IroConv1d downsample;
} IroDACVAEEncoderStage;

typedef struct {
    IroConv1d initial;
    IroDACVAEEncoderStage stage[IRO_DACVAE_STAGES];
    const float *final_alpha;
    IroConv1d final;
    IroConv1d mean;
} IroDACVAEEncoder;

/* Optional golden/debug observer. data uses time-major [frames,channels]. */
typedef int (*IroDACVAETraceFn)(void *user, const char *name,
                               const float *data, int frames, int channels);

int iro_dacvae_init(IroDACVAEDecoder *decoder, const IroSafetensors *weights);
int iro_dacvae_encoder_init(IroDACVAEEncoder *encoder,
                            const IroSafetensors *weights);
int iro_conv1d_output_length(const IroConv1d *conv, int input_length);
int iro_convtranspose1d_output_length(const IroConvTranspose1d *conv,
                                      int input_length);
int iro_conv1d_forward(const IroConv1d *conv, const float *input,
                       int input_length, float *output);
int iro_convtranspose1d_forward(const IroConvTranspose1d *conv,
                                const float *input, int input_length,
                                float *output);
void iro_snake_forward(const float *input, float *output, int frames,
                       int channels, const float *alpha);
int iro_dacvae_output_length(int latent_frames);
int iro_dacvae_decode(const IroDACVAEDecoder *decoder, const float *latent,
                      int latent_frames, float *waveform,
                      IroDACVAETraceFn trace, void *trace_user);
/* Deterministic reference path: reflect-pad to hop 1920 and return VAE mean
   in time-major [latent_frames,32]. The caller owns *latent. */
int iro_dacvae_encode_mean(const IroDACVAEEncoder *encoder,
                           const float *waveform, size_t sample_count,
                           float **latent, int *latent_frames,
                           IroDACVAETraceFn trace, void *trace_user);

#endif
