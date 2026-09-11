#define _GNU_SOURCE
/* dacvae.c — BLAS-backed deterministic DACVAE decoder. */
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "dacvae.h"
#include "ops.h"

enum {
    CODEBOOK_DIM = 32,
    LATENT_DIM = 1024,
    DECODER_DIM = 1536,
    FINAL_CHANNELS = 96,
    CONV_WORK_BYTES = 8 * 1024 * 1024,
    CONV1D_MAX_ROWS = 512,
    CONV1D_DIRECT_MAX_ROWS = 8192,
    CONVTRANSPOSE_MAX_ROWS = 256,
};

typedef struct {
    double up_bias[IRO_DACVAE_STAGES];
    double up_snake[IRO_DACVAE_STAGES];
    double up_gemm[IRO_DACVAE_STAGES];
    double up_scatter[IRO_DACVAE_STAGES];
    double residual_snake0[IRO_DACVAE_STAGES][IRO_DACVAE_RESIDUALS];
    double residual_conv7_pack[IRO_DACVAE_STAGES][IRO_DACVAE_RESIDUALS];
    double residual_conv7_linear[IRO_DACVAE_STAGES][IRO_DACVAE_RESIDUALS];
    double residual_snake1[IRO_DACVAE_STAGES][IRO_DACVAE_RESIDUALS];
    double residual_conv1[IRO_DACVAE_STAGES][IRO_DACVAE_RESIDUALS];
} IroDACVAEOpProfile;

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void *dacvae_malloc(size_t bytes) {
    void *ptr = malloc(bytes);
#ifdef MADV_HUGEPAGE
    if (ptr && bytes >= 2u * 1024u * 1024u) {
        static long page_size;
        if (!page_size) page_size = sysconf(_SC_PAGESIZE);
        if (page_size > 0) {
            uintptr_t page = (uintptr_t)page_size;
            uintptr_t begin = ((uintptr_t)ptr + page - 1u) / page * page;
            uintptr_t end = ((uintptr_t)ptr + bytes) / page * page;
            if (end > begin)
                (void)madvise((void *)begin, end - begin, MADV_HUGEPAGE);
        }
    }
#endif
    return ptr;
}

static const float *require_f32(const IroSafetensors *st, const char *name,
                                int ndims, uint64_t d0, uint64_t d1,
                                uint64_t d2) {
    const IroTensor *tensor = iro_st_get(st, name);
    if (!tensor) {
        fprintf(stderr, "dacvae: tensor tidak ada: %s\n", name);
        return NULL;
    }
    if (tensor->dtype != IRO_ST_F32 || tensor->ndims != ndims ||
        tensor->shape[0] != d0 || (ndims >= 2 && tensor->shape[1] != d1) ||
        (ndims >= 3 && tensor->shape[2] != d2)) {
        fprintf(stderr, "dacvae: dtype/shape tidak cocok: %s\n", name);
        return NULL;
    }
    return tensor->data;
}

static int load_conv(IroConv1d *conv, const IroSafetensors *st,
                     const char *base, int in_channels, int out_channels,
                     int kernel, int stride, int dilation, int padding) {
    char name[256];
    snprintf(name, sizeof(name), "%s.weight", base);
    conv->weight = require_f32(st, name, 3, out_channels, kernel, in_channels);
    snprintf(name, sizeof(name), "%s.bias", base);
    conv->bias = require_f32(st, name, 1, out_channels, 0, 0);
    conv->in_channels = in_channels;
    conv->out_channels = out_channels;
    conv->kernel = kernel;
    conv->stride = stride;
    conv->dilation = dilation;
    conv->padding = padding;
    return conv->weight && conv->bias ? 0 : -1;
}

static int load_convtranspose(IroConvTranspose1d *conv,
                              const IroSafetensors *st, const char *base,
                              int in_channels, int out_channels, int kernel,
                              int stride) {
    char name[256];
    snprintf(name, sizeof(name), "%s.weight", base);
    conv->weight = require_f32(st, name, 3, in_channels, kernel, out_channels);
    snprintf(name, sizeof(name), "%s.bias", base);
    conv->bias = require_f32(st, name, 1, out_channels, 0, 0);
    conv->in_channels = in_channels;
    conv->out_channels = out_channels;
    conv->kernel = kernel;
    conv->stride = stride;
    conv->padding = (stride + 1) / 2;
    conv->output_padding = stride & 1;
    return conv->weight && conv->bias ? 0 : -1;
}

int iro_dacvae_init(IroDACVAEDecoder *decoder, const IroSafetensors *weights) {
    if (!decoder || !weights) return -1;
    memset(decoder, 0, sizeof(*decoder));
    const char *format = iro_st_meta(weights, "format");
    if (!format || strcmp(format, "irodori-dacvae-decoder-v2")) {
        fprintf(stderr, "dacvae: format decoder hasil export tidak cocok\n");
        return -1;
    }
    if (load_conv(&decoder->quantizer_out, weights, "quantizer.out_proj",
                  CODEBOOK_DIM, LATENT_DIM, 1, 1, 1, 0) != 0 ||
        load_conv(&decoder->initial, weights, "decoder.model.0",
                  LATENT_DIM, DECODER_DIM, 7, 1, 1, 3) != 0)
        return -1;

    const int input_channels[IRO_DACVAE_STAGES] = {1536, 768, 384, 192};
    const int output_channels[IRO_DACVAE_STAGES] = {768, 384, 192, 96};
    const int rates[IRO_DACVAE_STAGES] = {12, 10, 8, 2};
    const int residual_index[IRO_DACVAE_RESIDUALS] = {4, 5, 8};
    const int dilation[IRO_DACVAE_RESIDUALS] = {1, 3, 9};
    for (int stage = 0; stage < IRO_DACVAE_STAGES; stage++) {
        IroDACVAEStage *dst = &decoder->stage[stage];
        int module = stage + 1;
        int channels = output_channels[stage];
        char name[192];
        snprintf(name, sizeof(name), "decoder.model.%d.block.0.alpha", module);
        dst->alpha = require_f32(weights, name, 3, 1, input_channels[stage], 1);
        snprintf(name, sizeof(name), "decoder.model.%d.block.1", module);
        if (!dst->alpha ||
            load_convtranspose(&dst->upsample, weights, name,
                               input_channels[stage], channels,
                               2 * rates[stage], rates[stage]) != 0)
            return -1;
        for (int residual = 0; residual < IRO_DACVAE_RESIDUALS; residual++) {
            IroDACVAEResidual *unit = &dst->residual[residual];
            int block = residual_index[residual];
            snprintf(name, sizeof(name),
                     "decoder.model.%d.block.%d.block.0.alpha", module, block);
            unit->alpha0 = require_f32(weights, name, 3, 1, channels, 1);
            snprintf(name, sizeof(name),
                     "decoder.model.%d.block.%d.block.2.alpha", module, block);
            unit->alpha1 = require_f32(weights, name, 3, 1, channels, 1);
            snprintf(name, sizeof(name),
                     "decoder.model.%d.block.%d.block.1", module, block);
            if (!unit->alpha0 || !unit->alpha1 ||
                load_conv(&unit->conv0, weights, name, channels, channels,
                          7, 1, dilation[residual], 3 * dilation[residual]) != 0)
                return -1;
            snprintf(name, sizeof(name),
                     "decoder.model.%d.block.%d.block.3", module, block);
            if (load_conv(&unit->conv1, weights, name, channels, channels,
                          1, 1, 1, 0) != 0)
                return -1;
        }
    }
    decoder->final_alpha = require_f32(
        weights, "decoder.wm_model.encoder_block.pre.0.alpha",
        3, 1, FINAL_CHANNELS, 1);
    if (!decoder->final_alpha ||
        load_conv(&decoder->final, weights,
                  "decoder.wm_model.encoder_block.pre.1",
                  FINAL_CHANNELS, 1, 7, 1, 1, 3) != 0)
        return -1;
    return 0;
}

int iro_dacvae_encoder_init(IroDACVAEEncoder *encoder,
                            const IroSafetensors *weights) {
    if (!encoder || !weights) return -1;
    memset(encoder, 0, sizeof(*encoder));
    const char *format = iro_st_meta(weights, "format");
    if (!format || strcmp(format, "irodori-dacvae-encoder-v1")) {
        fprintf(stderr, "dacvae: format encoder hasil export tidak cocok\n");
        return -1;
    }
    if (load_conv(&encoder->initial, weights, "encoder.block.0",
                  1, 64, 7, 1, 1, 3) != 0)
        return -1;
    const int channels[IRO_DACVAE_STAGES] = {64, 128, 256, 512};
    const int rates[IRO_DACVAE_STAGES] = {2, 8, 10, 12};
    const int dilations[IRO_DACVAE_RESIDUALS] = {1, 3, 9};
    for (int stage = 0; stage < IRO_DACVAE_STAGES; stage++) {
        IroDACVAEEncoderStage *dst = &encoder->stage[stage];
        char name[192];
        for (int residual = 0; residual < IRO_DACVAE_RESIDUALS; residual++) {
            IroDACVAEResidual *unit = &dst->residual[residual];
            snprintf(name, sizeof(name),
                     "encoder.block.%d.block.%d.block.0.alpha",
                     stage + 1, residual);
            unit->alpha0 = require_f32(weights, name, 3, 1, channels[stage], 1);
            snprintf(name, sizeof(name),
                     "encoder.block.%d.block.%d.block.2.alpha",
                     stage + 1, residual);
            unit->alpha1 = require_f32(weights, name, 3, 1, channels[stage], 1);
            snprintf(name, sizeof(name),
                     "encoder.block.%d.block.%d.block.1",
                     stage + 1, residual);
            if (!unit->alpha0 || !unit->alpha1 ||
                load_conv(&unit->conv0, weights, name,
                          channels[stage], channels[stage], 7, 1,
                          dilations[residual], 3 * dilations[residual]) != 0)
                return -1;
            snprintf(name, sizeof(name),
                     "encoder.block.%d.block.%d.block.3",
                     stage + 1, residual);
            if (load_conv(&unit->conv1, weights, name,
                          channels[stage], channels[stage], 1, 1, 1, 0) != 0)
                return -1;
        }
        snprintf(name, sizeof(name), "encoder.block.%d.block.3.alpha", stage + 1);
        dst->alpha = require_f32(weights, name, 3, 1, channels[stage], 1);
        snprintf(name, sizeof(name), "encoder.block.%d.block.4", stage + 1);
        int rate = rates[stage];
        if (!dst->alpha ||
            load_conv(&dst->downsample, weights, name, channels[stage],
                      channels[stage] * 2, rate * 2, rate, 1, rate / 2) != 0)
            return -1;
    }
    encoder->final_alpha = require_f32(weights, "encoder.block.5.alpha",
                                       3, 1, 1024, 1);
    return encoder->final_alpha &&
           load_conv(&encoder->final, weights, "encoder.block.6",
                     1024, 1024, 3, 1, 1, 1) == 0 &&
           load_conv(&encoder->mean, weights, "quantizer.mean",
                     1024, CODEBOOK_DIM, 1, 1, 1, 0) == 0 ? 0 : -1;
}

int iro_conv1d_output_length(const IroConv1d *conv, int input_length) {
    if (!conv || input_length <= 0 || conv->stride <= 0 || conv->kernel <= 0 ||
        conv->dilation <= 0)
        return -1;
    long long effective = (long long)conv->dilation * (conv->kernel - 1) + 1;
    long long numerator = (long long)input_length + 2LL * conv->padding - effective;
    long long output = numerator < 0 ? -1 : numerator / conv->stride + 1;
    return output <= 0 || output > INT_MAX ? -1 : (int)output;
}

int iro_convtranspose1d_output_length(const IroConvTranspose1d *conv,
                                      int input_length) {
    if (!conv || input_length <= 0 || conv->stride <= 0 || conv->kernel <= 0)
        return -1;
    long long output = (long long)(input_length - 1) * conv->stride -
                       2LL * conv->padding + conv->kernel +
                       conv->output_padding;
    return output <= 0 || output > INT_MAX ? -1 : (int)output;
}

static int workspace_rows(size_t columns, int maximum) {
    if (!columns) return 0;
    size_t rows = CONV_WORK_BYTES / (sizeof(float) * columns);
    if (rows < 1) rows = 1;
    if (rows > (size_t)maximum) rows = (size_t)maximum;
    return (int)rows;
}

static int conv1d_forward_impl(const IroConv1d *conv, const float *input,
                               int input_length, float *output,
                               const float *bias,
                               const float *snake_alpha,
                               double *snake_seconds,
                               double *pack_seconds,
                               double *linear_seconds) {
    if (!conv || !input || !output) return -1;
    int output_length = iro_conv1d_output_length(conv, input_length);
    if (output_length <= 0) return -1;
    int inner = conv->in_channels * conv->kernel;
    if (conv->kernel == 1 && conv->stride == 1 && conv->dilation == 1 &&
        conv->padding == 0) {
        const float *projection_input = input;
        float *activated = NULL;
        if (snake_alpha) {
            activated = dacvae_malloc(sizeof(float) * (size_t)input_length *
                                      conv->in_channels);
            if (!activated) return -1;
            double t0 = snake_seconds ? monotonic_seconds() : 0.0;
            iro_snake_forward(input, activated, input_length,
                              conv->in_channels, snake_alpha);
            if (snake_seconds) *snake_seconds += monotonic_seconds() - t0;
            projection_input = activated;
        }
        iro_linear(projection_input, conv->weight, bias, output,
                   input_length, conv->in_channels, conv->out_channels);
        free(activated);
        return 0;
    }


    /* Large stride-1 residual Conv7 layers are faster as seven direct GEMMs
       over contiguous time slices than as hundreds of im2col GEMMs.  The
       safetensors layout is [out,kernel,in], so a fixed tap has a padded row
       stride of kernel*in and can be consumed without repacking weights. */
    if (iro_ops_has_cblas() && snake_alpha && conv->kernel == 7 &&
        conv->stride == 1 &&
        conv->in_channels == conv->out_channels &&
        output_length == input_length && input_length >= 100000) {
        int rows_cap = workspace_rows((size_t)conv->in_channels,
                                      CONV1D_DIRECT_MAX_ROWS);
        int halo = conv->dilation * (conv->kernel - 1);
        float *activated = dacvae_malloc(sizeof(float) *
                                         (size_t)(rows_cap + halo) *
                                         conv->in_channels);
        if (!activated) return -1;
        int weight_stride = conv->kernel * conv->in_channels;
        for (int start = 0; start < output_length; start += rows_cap) {
            int rows = output_length - start;
            if (rows > rows_cap) rows = rows_cap;
            int block_end = start + rows;
            int input_min = start - conv->padding;
            if (input_min < 0) input_min = 0;
            int input_max = block_end - 1 - conv->padding + halo;
            if (input_max >= input_length) input_max = input_length - 1;
            int activation_rows = input_max - input_min + 1;
            double t0 = snake_seconds ? monotonic_seconds() : 0.0;
            iro_snake_forward(input + (size_t)input_min * conv->in_channels,
                              activated, activation_rows,
                              conv->in_channels, snake_alpha);
            if (snake_seconds) *snake_seconds += monotonic_seconds() - t0;

            t0 = linear_seconds ? monotonic_seconds() : 0.0;
            float *block_output = output + (size_t)start * conv->out_channels;
            if (bias) {
                for (int row = 0; row < rows; row++)
                    memcpy(block_output + (size_t)row * conv->out_channels,
                           bias, sizeof(float) * (size_t)conv->out_channels);
            } else {
                memset(block_output, 0,
                       sizeof(float) * (size_t)rows * conv->out_channels);
            }
            for (int tap = 0; tap < conv->kernel; tap++) {
                int shift = tap * conv->dilation - conv->padding;
                int output_start = start;
                if (output_start < -shift) output_start = -shift;
                int output_end = input_length - shift;
                if (output_end > block_end) output_end = block_end;
                if (output_end <= output_start) continue;
                int input_start = output_start + shift;
                int tap_rows = output_end - output_start;
                iro_linear_strided_weight_add(
                    activated + (size_t)(input_start - input_min) *
                                    conv->in_channels,
                    conv->weight + (size_t)tap * conv->in_channels,
                    weight_stride,
                    output + (size_t)output_start * conv->out_channels,
                    tap_rows, conv->in_channels, conv->out_channels);
            }
            if (linear_seconds) *linear_seconds += monotonic_seconds() - t0;
        }
        free(activated);
        return 0;
    }

    int rows_cap = workspace_rows((size_t)inner, CONV1D_MAX_ROWS);
    float *columns = dacvae_malloc(sizeof(float) * (size_t)rows_cap * inner);
    size_t activation_rows_cap =
        (size_t)(rows_cap - 1) * conv->stride + 1u +
        (size_t)conv->dilation * (conv->kernel - 1);
    if (activation_rows_cap > (size_t)input_length)
        activation_rows_cap = (size_t)input_length;
    float *activated = snake_alpha
                           ? dacvae_malloc(sizeof(float) * activation_rows_cap *
                                           conv->in_channels)
                           : NULL;
    if (!columns || (snake_alpha && !activated)) {
        free(columns);
        free(activated);
        return -1;
    }
    for (int start = 0; start < output_length; start += rows_cap) {
        int rows = output_length - start;
        if (rows > rows_cap) rows = rows_cap;
        int valid_min = 0, valid_max = -1;
        if (snake_alpha) {
            long long raw_min = (long long)start * conv->stride - conv->padding;
            long long raw_max =
                (long long)(start + rows - 1) * conv->stride - conv->padding +
                (long long)(conv->kernel - 1) * conv->dilation;
            valid_min = raw_min > 0 ? (int)raw_min : 0;
            valid_max = raw_max < input_length ? (int)raw_max : input_length - 1;
            if (valid_max >= valid_min) {
                int activation_rows = valid_max - valid_min + 1;
                double t0 = snake_seconds ? monotonic_seconds() : 0.0;
                iro_snake_forward(input + (size_t)valid_min * conv->in_channels,
                                  activated, activation_rows,
                                  conv->in_channels, snake_alpha);
                if (snake_seconds)
                    *snake_seconds += monotonic_seconds() - t0;
            }
        }
        double t0 = pack_seconds ? monotonic_seconds() : 0.0;
        for (int row = 0; row < rows; row++) {
            int output_time = start + row;
            float *dst = columns + (size_t)row * inner;
            for (int tap = 0; tap < conv->kernel; tap++) {
                int input_time = output_time * conv->stride - conv->padding +
                                 tap * conv->dilation;
                float *tap_dst = dst + (size_t)tap * conv->in_channels;
                if (input_time >= 0 && input_time < input_length) {
                    const float *tap_src = snake_alpha
                                               ? activated +
                                                     (size_t)(input_time - valid_min) *
                                                         conv->in_channels
                                               : input +
                                                     (size_t)input_time *
                                                         conv->in_channels;
                    memcpy(tap_dst,
                           tap_src,
                           sizeof(float) * (size_t)conv->in_channels);
                } else {
                    memset(tap_dst, 0,
                           sizeof(float) * (size_t)conv->in_channels);
                }
            }
        }
        if (pack_seconds) *pack_seconds += monotonic_seconds() - t0;
        t0 = linear_seconds ? monotonic_seconds() : 0.0;
        iro_linear(columns, conv->weight, bias,
                   output + (size_t)start * conv->out_channels,
                   rows, inner, conv->out_channels);
        if (linear_seconds) *linear_seconds += monotonic_seconds() - t0;
    }
    free(columns);
    free(activated);
    return 0;
}

int iro_conv1d_forward(const IroConv1d *conv, const float *input,
                       int input_length, float *output) {
    return conv1d_forward_impl(conv, input, input_length, output, conv->bias,
                               NULL, NULL, NULL, NULL);
}

static int convtranspose1d_forward_impl(const IroConvTranspose1d *conv,
                                        const float *input, int input_length,
                                        float *output, const float *alpha,
                                        IroDACVAEOpProfile *profile,
                                        int profile_stage) {
    if (!conv || !input || !output) return -1;
    int output_length = iro_convtranspose1d_output_length(conv, input_length);
    if (output_length <= 0) return -1;
    double t0 = profile ? monotonic_seconds() : 0.0;
    for (int time = 0; time < output_length; time++)
        memcpy(output + (size_t)time * conv->out_channels, conv->bias,
               sizeof(float) * (size_t)conv->out_channels);
    if (profile) profile->up_bias[profile_stage] += monotonic_seconds() - t0;
    size_t expanded_columns = (size_t)conv->out_channels * conv->kernel;
    int rows_cap = workspace_rows(expanded_columns, CONVTRANSPOSE_MAX_ROWS);
    float *expanded = dacvae_malloc(sizeof(float) * (size_t)rows_cap *
                                    expanded_columns);
    float *activated = alpha
                           ? dacvae_malloc(sizeof(float) * (size_t)rows_cap *
                                           conv->in_channels)
                           : NULL;
    if (!expanded || (alpha && !activated)) {
        free(expanded); free(activated);
        return -1;
    }

    for (int start = 0; start < input_length; start += rows_cap) {
        int rows = input_length - start;
        if (rows > rows_cap) rows = rows_cap;
        const float *projection_input =
            input + (size_t)start * conv->in_channels;
        if (alpha) {
            t0 = profile ? monotonic_seconds() : 0.0;
            iro_snake_forward(projection_input, activated, rows,
                              conv->in_channels, alpha);
            if (profile)
                profile->up_snake[profile_stage] += monotonic_seconds() - t0;
            projection_input = activated;
        }
        t0 = profile ? monotonic_seconds() : 0.0;
        iro_matmul(projection_input, conv->weight,
                   expanded, rows, conv->in_channels, (int)expanded_columns);
        if (profile) profile->up_gemm[profile_stage] += monotonic_seconds() - t0;
        t0 = profile ? monotonic_seconds() : 0.0;
        for (int row = 0; row < rows; row++) {
            int output_base = (start + row) * conv->stride - conv->padding;
            const float *src = expanded + (size_t)row * expanded_columns;
            for (int tap = 0; tap < conv->kernel; tap++) {
                int output_time = output_base + tap;
                if (output_time < 0 || output_time >= output_length) continue;
                const float *kernel = src + (size_t)tap * conv->out_channels;
                float *dst = output + (size_t)output_time * conv->out_channels;
                for (int channel = 0; channel < conv->out_channels; channel++)
                    dst[channel] += kernel[channel];
            }
        }
        if (profile)
            profile->up_scatter[profile_stage] += monotonic_seconds() - t0;
    }
    free(expanded); free(activated);
    return 0;
}

int iro_convtranspose1d_forward(const IroConvTranspose1d *conv,
                                const float *input, int input_length,
                                float *output) {
    return convtranspose1d_forward_impl(conv, input, input_length,
                                        output, NULL, NULL, 0);
}

void iro_snake_forward(const float *input, float *output, int frames,
                       int channels, const float *alpha) {
    for (int time = 0; time < frames; time++) {
        for (int channel = 0; channel < channels; channel++) {
            size_t index = (size_t)time * channels + channel;
            float a = alpha[channel];
            float sine = sinf(a * input[index]);
            output[index] = input[index] + sine * sine / (a + 1e-9f);
        }
    }
}

static float *run_conv_profiled(const IroConv1d *conv, const float *input,
                                int input_length, int *output_length,
                                const float *bias,
                                double *pack_seconds,
                                double *linear_seconds) {
    *output_length = iro_conv1d_output_length(conv, input_length);
    if (*output_length <= 0) return NULL;
    float *output = dacvae_malloc(sizeof(float) * (size_t)*output_length *
                                  conv->out_channels);
    if (!output || conv1d_forward_impl(conv, input, input_length, output, bias,
                                       NULL, NULL, pack_seconds,
                                       linear_seconds) != 0) {
        free(output);
        return NULL;
    }
    return output;
}

static float *run_conv_snake_profiled(const IroConv1d *conv,
                                      const float *input, int input_length,
                                      int *output_length, const float *bias,
                                      const float *snake_alpha,
                                      double *snake_seconds,
                                      double *pack_seconds,
                                      double *linear_seconds) {
    *output_length = iro_conv1d_output_length(conv, input_length);
    if (*output_length <= 0) return NULL;
    float *output = dacvae_malloc(sizeof(float) * (size_t)*output_length *
                                  conv->out_channels);
    if (!output || conv1d_forward_impl(conv, input, input_length, output, bias,
                                       snake_alpha, snake_seconds,
                                       pack_seconds, linear_seconds) != 0) {
        free(output);
        return NULL;
    }
    return output;
}

static float *run_conv(const IroConv1d *conv, const float *input,
                       int input_length, int *output_length) {
    return run_conv_profiled(conv, input, input_length, output_length,
                             conv->bias, NULL, NULL);
}

static void snake_bias_forward(const float *input, float *output, int frames,
                               int channels, const float *bias,
                               const float *alpha) {
    for (int time = 0; time < frames; time++) {
        for (int channel = 0; channel < channels; channel++) {
            size_t index = (size_t)time * channels + channel;
            float x = input[index] + bias[channel];
            float a = alpha[channel];
            float sine = sinf(a * x);
            output[index] = x + sine * sine / (a + 1e-9f);
        }
    }
}

static float *run_residual(const IroDACVAEResidual *residual,
                           const float *input, int frames, int channels,
                           IroDACVAEOpProfile *profile,
                           int profile_stage, int profile_residual) {
    size_t n = (size_t)frames * channels;
    int hidden_frames = 0;
    double *snake0 = profile
                         ? &profile->residual_snake0[profile_stage][profile_residual]
                         : NULL;
    double *conv7_pack = profile
                             ? &profile->residual_conv7_pack[profile_stage]
                                                               [profile_residual]
                             : NULL;
    double *conv7_linear = profile
                               ? &profile->residual_conv7_linear[profile_stage]
                                                                   [profile_residual]
                               : NULL;
    float *hidden = run_conv_snake_profiled(
        &residual->conv0, input, frames, &hidden_frames, NULL,
        residual->alpha0, snake0, conv7_pack, conv7_linear);
    if (!hidden || hidden_frames != frames) {
        free(hidden);
        return NULL;
    }
    double t0 = profile ? monotonic_seconds() : 0.0;
    snake_bias_forward(hidden, hidden, frames, channels,
                       residual->conv0.bias, residual->alpha1);
    if (profile)
        profile->residual_snake1[profile_stage][profile_residual] +=
            monotonic_seconds() - t0;
    const IroConv1d *projection = &residual->conv1;
    int output_frames = iro_conv1d_output_length(projection, frames);
    float *output = dacvae_malloc(sizeof(float) * n);
    if (output && output_frames == frames && projection->kernel == 1 &&
        projection->stride == 1 && projection->dilation == 1 &&
        projection->padding == 0 && projection->in_channels == channels &&
        projection->out_channels == channels) {
        t0 = profile ? monotonic_seconds() : 0.0;
        iro_linear_add(hidden, projection->weight, projection->bias, input,
                       output, frames, channels, channels);
        if (profile)
            profile->residual_conv1[profile_stage][profile_residual] +=
                monotonic_seconds() - t0;
    } else {
        free(output);
        output = NULL;
    }
    free(hidden);
    if (!output || output_frames != frames) {
        free(output);
        return NULL;
    }
    return output;
}

static int emit_trace(IroDACVAETraceFn trace, void *user, const char *name,
                      const float *data, int frames, int channels) {
    return trace ? trace(user, name, data, frames, channels) : 0;
}

int iro_dacvae_encode_mean(const IroDACVAEEncoder *encoder,
                           const float *waveform, size_t sample_count,
                           float **latent, int *latent_frames,
                           IroDACVAETraceFn trace, void *trace_user) {
    enum { HOP_LENGTH = 1920 };
    if (!encoder || !waveform || !latent || !latent_frames ||
        sample_count == 0 || sample_count > INT_MAX)
        return -1;
    *latent = NULL;
    *latent_frames = 0;
    const char *profile_env = getenv("IRO_ENCODER_PROFILE");
    int profile = profile_env && strcmp(profile_env, "0") && !trace;
    IroDACVAEOpProfile encoder_op_profile = {0};
    double initial_seconds = 0.0;
    double residual_seconds[IRO_DACVAE_STAGES][IRO_DACVAE_RESIDUALS] = {{0}};
    double downsample_seconds[IRO_DACVAE_STAGES] = {0};
    double final_seconds = 0.0, mean_seconds = 0.0;
    size_t padding = (HOP_LENGTH - sample_count % HOP_LENGTH) % HOP_LENGTH;
    if (padding >= sample_count || sample_count > (size_t)INT_MAX - padding)
        return -1;
    size_t padded_count = sample_count + padding;
    float *padded = dacvae_malloc(sizeof(float) * padded_count);
    if (!padded) return -1;
    memcpy(padded, waveform, sizeof(float) * sample_count);
    for (size_t i = 0; i < padding; i++)
        padded[sample_count + i] = waveform[sample_count - 2u - i];

    int frames = 0;
    double t0 = profile ? monotonic_seconds() : 0.0;
    float *state = run_conv(&encoder->initial, padded, (int)padded_count, &frames);
    if (profile) initial_seconds += monotonic_seconds() - t0;
    free(padded);
    if (!state || emit_trace(trace, trace_user, "encoder_block0",
                             state, frames, 64) != 0)
        goto fail_encoder;
    int channels = 64;
    for (int stage = 0; stage < IRO_DACVAE_STAGES; stage++) {
        const IroDACVAEEncoderStage *block = &encoder->stage[stage];
        for (int residual = 0; residual < IRO_DACVAE_RESIDUALS; residual++) {
            t0 = profile ? monotonic_seconds() : 0.0;
            float *next = run_residual(&block->residual[residual], state,
                                       frames, channels,
                                       profile ? &encoder_op_profile : NULL,
                                       stage, residual);
            if (profile)
                residual_seconds[stage][residual] += monotonic_seconds() - t0;
            if (!next) goto fail_encoder;
            free(state);
            state = next;
        }
        iro_snake_forward(state, state, frames, channels, block->alpha);
        int next_frames = 0;
        t0 = profile ? monotonic_seconds() : 0.0;
        float *next = run_conv(&block->downsample, state, frames, &next_frames);
        if (profile) downsample_seconds[stage] += monotonic_seconds() - t0;
        free(state);
        state = next;
        channels *= 2;
        frames = next_frames;
        char trace_name[32];
        snprintf(trace_name, sizeof(trace_name), "encoder_block%d", stage + 1);
        if (!state || emit_trace(trace, trace_user, trace_name,
                                 state, frames, channels) != 0)
            goto fail_encoder;
    }
    iro_snake_forward(state, state, frames, 1024, encoder->final_alpha);
    if (emit_trace(trace, trace_user, "encoder_block5",
                   state, frames, 1024) != 0)
        goto fail_encoder;
    int final_frames = 0;
    t0 = profile ? monotonic_seconds() : 0.0;
    float *final_state = run_conv(&encoder->final, state, frames, &final_frames);
    if (profile) final_seconds += monotonic_seconds() - t0;
    free(state);
    state = final_state;
    if (!state || final_frames != frames ||
        emit_trace(trace, trace_user, "encoder_block6",
                   state, frames, 1024) != 0)
        goto fail_encoder;
    int mean_frames = 0;
    t0 = profile ? monotonic_seconds() : 0.0;
    float *mean = run_conv(&encoder->mean, state, frames, &mean_frames);
    if (profile) mean_seconds += monotonic_seconds() - t0;
    free(state);
    state = NULL;
    if (!mean || mean_frames != frames ||
        emit_trace(trace, trace_user, "encoder_mean",
                   mean, frames, CODEBOOK_DIM) != 0) {
        free(mean);
        return -1;
    }
    *latent = mean;
    *latent_frames = frames;
    if (profile) {
        fprintf(stderr, "DACVAE encoder: initial %.3f s\n", initial_seconds);
        for (int stage = 0; stage < IRO_DACVAE_STAGES; stage++) {
            double conv7_pack = 0.0, conv7_linear = 0.0, conv1 = 0.0;
            double snake0 = 0.0, snake1 = 0.0;
            for (int residual = 0; residual < IRO_DACVAE_RESIDUALS; residual++) {
                snake0 += encoder_op_profile.residual_snake0[stage][residual];
                conv7_pack += encoder_op_profile.residual_conv7_pack[stage][residual];
                conv7_linear += encoder_op_profile.residual_conv7_linear[stage][residual];
                snake1 += encoder_op_profile.residual_snake1[stage][residual];
                conv1 += encoder_op_profile.residual_conv1[stage][residual];
            }
            fprintf(stderr,
                    "DACVAE encoder stage%d: residual %.3f/%.3f/%.3f s "
                    "downsample %.3f s | ops snake0 %.3f pack %.3f "
                    "conv7 %.3f snake1 %.3f conv1 %.3f s\n",
                    stage, residual_seconds[stage][0], residual_seconds[stage][1],
                    residual_seconds[stage][2], downsample_seconds[stage],
                    snake0, conv7_pack, conv7_linear, snake1, conv1);
        }
        fprintf(stderr, "DACVAE encoder: final %.3f s mean %.3f s\n",
                final_seconds, mean_seconds);
    }
    return 0;

fail_encoder:
    free(state);
    return -1;
}

int iro_dacvae_output_length(int latent_frames) {
    return latent_frames > 0 && latent_frames <= INT_MAX / 1920
               ? latent_frames * 1920
               : -1;
}

int iro_dacvae_decode(const IroDACVAEDecoder *decoder, const float *latent,
                      int latent_frames, float *waveform,
                      IroDACVAETraceFn trace, void *trace_user) {
    if (!decoder || !latent || latent_frames <= 0 || !waveform) return -1;
    int frames = 0;
    float *state = run_conv(&decoder->quantizer_out, latent,
                            latent_frames, &frames);
    if (!state || emit_trace(trace, trace_user, "codec_quantizer_out",
                             state, frames, LATENT_DIM) != 0)
        goto fail;
    float *next = run_conv(&decoder->initial, state, frames, &frames);
    free(state);
    state = next;
    if (!state || emit_trace(trace, trace_user, "codec_decoder_initial",
                             state, frames, DECODER_DIM) != 0)
        goto fail;

    IroDACVAEOpProfile op_profile = {0};
    const char *op_profile_env = getenv("IRO_DACVAE_OP_PROFILE");
    IroDACVAEOpProfile *profile =
        op_profile_env && strcmp(op_profile_env, "0") && !trace
            ? &op_profile : NULL;
    for (int stage = 0; stage < IRO_DACVAE_STAGES; stage++) {
        const IroDACVAEStage *block = &decoder->stage[stage];
        int output_channels = block->upsample.out_channels;
        int output_frames = iro_convtranspose1d_output_length(&block->upsample,
                                                               frames);
        if (output_frames <= 0) {
            goto fail;
        }
        next = dacvae_malloc(sizeof(float) * (size_t)output_frames *
                             output_channels);
        if (!next || convtranspose1d_forward_impl(&block->upsample, state,
                                                  frames, next,
                                                  block->alpha, profile,
                                                  stage) != 0) {
            free(next);
            goto fail;
        }
        free(state);
        state = next;
        frames = output_frames;
        for (int residual = 0; residual < IRO_DACVAE_RESIDUALS; residual++) {
            next = run_residual(&block->residual[residual], state,
                                frames, output_channels, profile,
                                stage, residual);
            if (!next) goto fail;
            free(state);
            state = next;
        }
        char trace_name[32];
        snprintf(trace_name, sizeof(trace_name), "codec_decoder_stage%d", stage);
        if (emit_trace(trace, trace_user, trace_name, state,
                       frames, output_channels) != 0)
            goto fail;
    }

    size_t final_n = (size_t)frames * FINAL_CHANNELS;
    float *activated = dacvae_malloc(sizeof(float) * final_n);
    if (!activated) goto fail;
    iro_snake_forward(state, activated, frames, FINAL_CHANNELS,
                      decoder->final_alpha);
    int waveform_frames = 0;
    next = run_conv(&decoder->final, activated, frames, &waveform_frames);
    free(activated); free(state);
    state = next;
    if (!state || waveform_frames != iro_dacvae_output_length(latent_frames))
        goto fail;
    for (int i = 0; i < waveform_frames; i++) waveform[i] = tanhf(state[i]);
    if (emit_trace(trace, trace_user, "codec_decoder_out", waveform,
                   waveform_frames, 1) != 0)
        goto fail;
    if (profile) {
        for (int stage = 0; stage < IRO_DACVAE_STAGES; stage++) {
            double residual_snake0 = 0.0, residual_conv7_pack = 0.0;
            double residual_conv7_linear = 0.0;
            double residual_snake1 = 0.0, residual_conv1 = 0.0;
            for (int residual = 0; residual < IRO_DACVAE_RESIDUALS; residual++) {
                residual_snake0 += op_profile.residual_snake0[stage][residual];
                residual_conv7_pack +=
                    op_profile.residual_conv7_pack[stage][residual];
                residual_conv7_linear +=
                    op_profile.residual_conv7_linear[stage][residual];
                residual_snake1 += op_profile.residual_snake1[stage][residual];
                residual_conv1 += op_profile.residual_conv1[stage][residual];
            }
            fprintf(stderr,
                    "DACVAE op stage%d: up[bias %.3f snake %.3f gemm %.3f scatter %.3f] "
                    "res[snake0 %.3f conv7_pack %.3f conv7_linear %.3f "
                    "snake1 %.3f conv1 %.3f] s\n",
                    stage, op_profile.up_bias[stage], op_profile.up_snake[stage],
                    op_profile.up_gemm[stage], op_profile.up_scatter[stage],
                    residual_snake0, residual_conv7_pack,
                    residual_conv7_linear, residual_snake1, residual_conv1);
        }
    }
    free(state);
    return 0;

fail:
    free(state);
    return -1;
}
