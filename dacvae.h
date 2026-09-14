/* dacvae.h — deterministic text-path DACVAE decoder. */
#ifndef IRO_DACVAE_H
#define IRO_DACVAE_H

#include "irodori.h"
#include "ops.h"

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

/* Opt-in W8A8 copies of the decoder's dense convolutions, owned by the
   engine and borrowed by IroDACVAEDecoder.int8.  Conv weights keep the
   safetensors [out,kernel,in] layout flattened to [N, kernel*in]; the
   transposed convolution is re-laid out to [kernel*out, in].  Snake
   activations, biases, scatter-add, quantizer_out and the final 96->1 tap
   stay FP32. */
typedef struct {
    IroInt8Weight conv0; /* [C, 7*C] */
    IroInt8Weight conv1; /* [C, C] */
} IroDACVAEInt8Residual;

typedef struct {
    IroInt8Weight upsample; /* [kernel*out, in] */
    IroDACVAEInt8Residual residual[IRO_DACVAE_RESIDUALS];
} IroDACVAEInt8Stage;

typedef struct {
    IroInt8Weight initial; /* [1536, 7*1024] */
    IroDACVAEInt8Stage stage[IRO_DACVAE_STAGES];
    size_t bytes;
} IroDACVAEInt8;

typedef struct {
    IroConv1d quantizer_out;
    IroConv1d initial;
    IroDACVAEStage stage[IRO_DACVAE_STAGES];
    const float *final_alpha;
    IroConv1d final;
    const IroDACVAEInt8 *int8; /* Borrowed from owning engine; NULL = FP32. */
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
/* Quantize the decoder's dense convolutions into dst (zeroed by the call).
   decoder->int8 is left untouched; the caller assigns it. */
int iro_dacvae_int8_quantize(IroDACVAEInt8 *dst, const IroDACVAEDecoder *decoder);
void iro_dacvae_int8_free(IroDACVAEInt8 *q);
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
