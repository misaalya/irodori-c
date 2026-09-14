#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "dacvae.h"
#include "irodori.h"
#include "ops.h"

enum {
    CONV_WORK_BYTES = 8 * 1024 * 1024,
    IM2COL_MAX_ROWS = 512,
    DIRECT_MAX_ROWS = 8192,
};

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int parse_positive_int(const char *text, int *out) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || !end || *end != '\0' || value <= 0 || value > 1000000)
        return -1;
    *out = (int)value;
    return 0;
}

static int workspace_rows(size_t columns, int maximum) {
    size_t rows = CONV_WORK_BYTES / (sizeof(float) * columns);
    if (rows < 1) rows = 1;
    if (rows > (size_t)maximum) rows = (size_t)maximum;
    return (int)rows;
}

static float *load_rows(const char *path, int channels, int *frames_out) {
    struct stat info;
    if (stat(path, &info) != 0 || info.st_size <= 0 ||
        (uint64_t)info.st_size % (sizeof(float) * (uint64_t)channels) != 0) {
        fprintf(stderr, "bench_codec_conv7: ukuran tensor tidak valid: %s\n", path);
        return NULL;
    }
    uint64_t frames64 = (uint64_t)info.st_size /
                        (sizeof(float) * (uint64_t)channels);
    if (frames64 > 10000000) return NULL;
    size_t count = (size_t)frames64 * (size_t)channels;
    float *data = malloc(sizeof(float) * count);
    if (!data) return NULL;
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        free(data);
        return NULL;
    }
    size_t got = fread(data, sizeof(float), count, fp);
    int extra = fgetc(fp);
    int failed = got != count || extra != EOF || ferror(fp);
    fclose(fp);
    if (failed) {
        free(data);
        return NULL;
    }
    *frames_out = (int)frames64;
    return data;
}

static int run_direct(const IroConv1d *conv, const float *alpha,
                      const float *input, int frames, float *output,
                      int rows_cap) {
    int halo = conv->dilation * (conv->kernel - 1);
    float *activated = malloc(sizeof(float) *
                              (size_t)(rows_cap + halo) * conv->in_channels);
    if (!activated) return -1;
    int weight_stride = conv->kernel * conv->in_channels;
    for (int start = 0; start < frames; start += rows_cap) {
        int rows = frames - start;
        if (rows > rows_cap) rows = rows_cap;
        int block_end = start + rows;
        int input_min = start - conv->padding;
        if (input_min < 0) input_min = 0;
        int input_max = block_end - 1 - conv->padding + halo;
        if (input_max >= frames) input_max = frames - 1;
        int activation_rows = input_max - input_min + 1;
        iro_snake_forward(input + (size_t)input_min * conv->in_channels,
                          activated, activation_rows,
                          conv->in_channels, alpha);

        float *block_output = output + (size_t)start * conv->out_channels;
        memset(block_output, 0,
               sizeof(float) * (size_t)rows * conv->out_channels);
        for (int tap = 0; tap < conv->kernel; tap++) {
            int shift = tap * conv->dilation - conv->padding;
            int output_start = start;
            if (output_start < -shift) output_start = -shift;
            int output_end = frames - shift;
            if (output_end > block_end) output_end = block_end;
            if (output_end <= output_start) continue;
            int input_start = output_start + shift;
            int tap_rows = output_end - output_start;
            iro_linear_strided_weight_add(
                activated + (size_t)(input_start - input_min) * conv->in_channels,
                conv->weight + (size_t)tap * conv->in_channels,
                weight_stride,
                output + (size_t)output_start * conv->out_channels,
                tap_rows, conv->in_channels, conv->out_channels);
        }
    }
    free(activated);
    return 0;
}

static int run_im2col(const IroConv1d *conv, const float *alpha,
                      const float *input, int frames, float *output) {
    int inner = conv->in_channels * conv->kernel;
    int rows_cap = workspace_rows((size_t)inner, IM2COL_MAX_ROWS);
    float *columns = malloc(sizeof(float) * (size_t)rows_cap * inner);
    size_t activation_rows_cap = (size_t)(rows_cap - 1) + 1u +
                                 (size_t)conv->dilation * (conv->kernel - 1);
    if (activation_rows_cap > (size_t)frames) activation_rows_cap = (size_t)frames;
    float *activated = malloc(sizeof(float) * activation_rows_cap *
                              conv->in_channels);
    if (!columns || !activated) {
        free(columns);
        free(activated);
        return -1;
    }

    for (int start = 0; start < frames; start += rows_cap) {
        int rows = frames - start;
        if (rows > rows_cap) rows = rows_cap;
        long long raw_min = (long long)start - conv->padding;
        long long raw_max = (long long)(start + rows - 1) - conv->padding +
                            (long long)(conv->kernel - 1) * conv->dilation;
        int valid_min = raw_min > 0 ? (int)raw_min : 0;
        int valid_max = raw_max < frames ? (int)raw_max : frames - 1;
        if (valid_max >= valid_min) {
            int activation_rows = valid_max - valid_min + 1;
            iro_snake_forward(input + (size_t)valid_min * conv->in_channels,
                              activated, activation_rows,
                              conv->in_channels, alpha);
        }
        for (int row = 0; row < rows; row++) {
            int output_time = start + row;
            float *dst = columns + (size_t)row * inner;
            for (int tap = 0; tap < conv->kernel; tap++) {
                int input_time = output_time - conv->padding +
                                 tap * conv->dilation;
                float *tap_dst = dst + (size_t)tap * conv->in_channels;
                if (input_time >= 0 && input_time < frames) {
                    memcpy(tap_dst,
                           activated + (size_t)(input_time - valid_min) *
                                           conv->in_channels,
                           sizeof(float) * conv->in_channels);
                } else {
                    memset(tap_dst, 0, sizeof(float) * conv->in_channels);
                }
            }
        }
        iro_linear(columns, conv->weight, NULL,
                   output + (size_t)start * conv->out_channels,
                   rows, inner, conv->out_channels);
    }
    free(columns);
    free(activated);
    return 0;
}

static int cmp_double(const void *lhs, const void *rhs) {
    double a = *(const double *)lhs;
    double b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static double median_copy(const double *values, int count) {
    double *copy = malloc(sizeof(double) * (size_t)count);
    if (!copy) return NAN;
    memcpy(copy, values, sizeof(double) * (size_t)count);
    qsort(copy, (size_t)count, sizeof(double), cmp_double);
    double result = count & 1 ? copy[count / 2]
                              : 0.5 * (copy[count / 2 - 1] + copy[count / 2]);
    free(copy);
    return result;
}

static double checksum(const float *values, size_t count) {
    double sum = 0.0;
    size_t stride = count / 4096 + 1;
    for (size_t i = 0; i < count; i += stride)
        sum += (double)values[i] * (double)(1 + (i % 17));
    return sum;
}

static int bench_layer(const IroDACVAEDecoder *decoder, const char *golden_dir,
                       int stage, int residual, int pairs) {
    const IroDACVAEResidual *unit = &decoder->stage[stage].residual[residual];
    const IroConv1d *conv = &unit->conv0;
    char path[1024];
    snprintf(path, sizeof(path),
             "%s/codec_decoder_stage%d_residual%d_input.f32",
             golden_dir, stage, residual);
    int frames = 0;
    float *input = load_rows(path, conv->in_channels, &frames);
    size_t count = (size_t)frames * conv->out_channels;
    float *im2col = malloc(sizeof(float) * count);
    float *direct = malloc(sizeof(float) * count);
    double *im2col_times = malloc(sizeof(double) * (size_t)pairs);
    double *direct_times = malloc(sizeof(double) * (size_t)pairs);
    if (!input || !im2col || !direct || !im2col_times || !direct_times) {
        fprintf(stderr, "bench_codec_conv7: alokasi gagal stage=%d residual=%d\n",
                stage, residual);
        free(input); free(im2col); free(direct);
        free(im2col_times); free(direct_times);
        return -1;
    }
    int direct_rows = workspace_rows((size_t)conv->in_channels, DIRECT_MAX_ROWS);
    if (run_im2col(conv, unit->alpha0, input, frames, im2col) != 0 ||
        run_direct(conv, unit->alpha0, input, frames, direct, direct_rows) != 0) {
        free(input); free(im2col); free(direct);
        free(im2col_times); free(direct_times);
        return -1;
    }

    double max_abs = 0.0;
    double mae = 0.0;
    for (size_t i = 0; i < count; i++) {
        double err = fabs((double)im2col[i] - (double)direct[i]);
        if (err > max_abs) max_abs = err;
        mae += err;
    }
    mae /= (double)count;

    for (int pair = 0; pair < pairs; pair++) {
        double start;
        if ((pair & 1) == 0) {
            start = now_seconds();
            run_im2col(conv, unit->alpha0, input, frames, im2col);
            im2col_times[pair] = now_seconds() - start;
            start = now_seconds();
            run_direct(conv, unit->alpha0, input, frames, direct, direct_rows);
            direct_times[pair] = now_seconds() - start;
        } else {
            start = now_seconds();
            run_direct(conv, unit->alpha0, input, frames, direct, direct_rows);
            direct_times[pair] = now_seconds() - start;
            start = now_seconds();
            run_im2col(conv, unit->alpha0, input, frames, im2col);
            im2col_times[pair] = now_seconds() - start;
        }
        printf("{\"type\":\"codec_conv7_pair\",\"backend\":\"%s\","
               "\"stage\":%d,\"residual\":%d,\"frames\":%d,"
               "\"channels\":%d,\"dilation\":%d,\"direct_rows\":%d,"
               "\"pair\":%d,\"im2col_seconds\":%.9f,"
               "\"direct_seconds\":%.9f,\"speedup\":%.6f}\n",
               iro_ops_backend_name(), stage, residual, frames,
               conv->in_channels, conv->dilation, direct_rows, pair + 1,
               im2col_times[pair], direct_times[pair],
               im2col_times[pair] / direct_times[pair]);
    }
    double im2col_median = median_copy(im2col_times, pairs);
    double direct_median = median_copy(direct_times, pairs);
    printf("{\"type\":\"codec_conv7_summary\",\"backend\":\"%s\","
           "\"stage\":%d,\"residual\":%d,\"frames\":%d,"
           "\"channels\":%d,\"dilation\":%d,\"direct_rows\":%d,"
           "\"pairs\":%d,\"im2col_median_seconds\":%.9f,"
           "\"direct_median_seconds\":%.9f,\"speedup\":%.6f,"
           "\"max_abs\":%.9g,\"mae\":%.9g,\"checksum\":%.17g}\n",
           iro_ops_backend_name(), stage, residual, frames,
           conv->in_channels, conv->dilation, direct_rows, pairs,
           im2col_median, direct_median, im2col_median / direct_median,
           max_abs, mae, checksum(direct, count));

    free(input); free(im2col); free(direct);
    free(im2col_times); free(direct_times);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s DECODER.safetensors GOLDEN_DIR [pairs]\n", argv[0]);
        return 2;
    }
    int pairs = 3;
    if (argc == 4 && parse_positive_int(argv[3], &pairs) != 0) return 2;
    if (pairs < 3) return 2;
    int threads = 2;
    const char *threads_env = getenv("IRO_NUM_THREADS");
    if (threads_env && parse_positive_int(threads_env, &threads) != 0) return 2;
    iro_ops_set_threads(threads);
    fprintf(stderr, "bench_codec_conv7: backend=%s threads=%d active=%d\n",
            iro_ops_backend_name(), threads, iro_ops_get_threads());

    IroSafetensors st = {.fd = -1};
    if (iro_st_load(argv[1], &st) != 0) return 1;
    IroDACVAEDecoder decoder;
    if (iro_dacvae_init(&decoder, &st) != 0) {
        iro_st_free(&st);
        return 1;
    }
    int failed = 0;
    for (int stage = 2; stage <= 3; stage++) {
        for (int residual = 0; residual < IRO_DACVAE_RESIDUALS; residual++)
            failed |= bench_layer(&decoder, argv[2], stage, residual, pairs) != 0;
    }
    iro_st_free(&st);
    return failed ? 1 : 0;
}
