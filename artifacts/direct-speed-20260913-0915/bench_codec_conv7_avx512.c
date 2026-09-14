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
#include <immintrin.h>

enum { CONV_WORK_BYTES = 8 * 1024 * 1024, DIRECT_MAX_ROWS = 8192 };

typedef struct {
    float *data;
    size_t bytes;
    int channels;
} PackedWeights;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int workspace_rows(int channels) {
    size_t rows = CONV_WORK_BYTES / (sizeof(float) * (size_t)channels);
    if (rows < 1) rows = 1;
    if (rows > DIRECT_MAX_ROWS) rows = DIRECT_MAX_ROWS;
    return (int)rows;
}

static float *load_rows(const char *path, int channels, int *frames_out) {
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size <= 0 ||
        (uint64_t)st.st_size % (sizeof(float) * (uint64_t)channels) != 0)
        return NULL;
    uint64_t frames = (uint64_t)st.st_size / (sizeof(float) * (uint64_t)channels);
    if (frames > 10000000) return NULL;
    size_t count = (size_t)frames * (size_t)channels;
    float *data = malloc(sizeof(float) * count);
    if (!data) return NULL;
    FILE *fp = fopen(path, "rb");
    if (!fp) { free(data); return NULL; }
    size_t got = fread(data, sizeof(float), count, fp);
    int extra = fgetc(fp);
    int failed = got != count || extra != EOF || ferror(fp);
    fclose(fp);
    if (failed) { free(data); return NULL; }
    *frames_out = (int)frames;
    return data;
}

static int pack_init(PackedWeights *p, const IroConv1d *conv) {
    memset(p, 0, sizeof(*p));
    if (conv->kernel != 7 || conv->in_channels != conv->out_channels) return -1;
    p->channels = conv->in_channels;
    p->bytes = sizeof(float) * 7u * (size_t)conv->in_channels *
               (size_t)conv->out_channels;
    void *aligned = NULL;
    if (posix_memalign(&aligned, 64, p->bytes) != 0) return -1;
    p->data = aligned;
    int stride = conv->kernel * conv->in_channels;
    for (int tap = 0; tap < 7; tap++) {
        for (int k = 0; k < conv->in_channels; k++) {
            float *dst = p->data +
                         ((size_t)tap * conv->in_channels + k) *
                             conv->out_channels;
            for (int n = 0; n < conv->out_channels; n++)
                dst[n] = conv->weight[(size_t)n * stride +
                                      (size_t)tap * conv->in_channels + k];
        }
    }
    return 0;
}

static void pack_free(PackedWeights *p) {
    free(p->data);
    memset(p, 0, sizeof(*p));
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
                          activated, activation_rows, conv->in_channels, alpha);
        float *block_output = output + (size_t)start * conv->out_channels;
        memset(block_output, 0, sizeof(float) * (size_t)rows * conv->out_channels);
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

static void fused_conv7_avx512(const IroConv1d *conv,
                               const PackedWeights *packed,
                               const float *activated, int input_min,
                               int start, int rows, float *output) {
    const int channels = conv->in_channels;
    for (int row = 0; row < rows; row += 4) {
        for (int n0 = 0; n0 < channels; n0 += 48) {
            __m512 c00 = _mm512_setzero_ps();
            __m512 c01 = _mm512_setzero_ps();
            __m512 c02 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c11 = _mm512_setzero_ps();
            __m512 c12 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c21 = _mm512_setzero_ps();
            __m512 c22 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps();
            __m512 c31 = _mm512_setzero_ps();
            __m512 c32 = _mm512_setzero_ps();

            for (int tap = 0; tap < 7; tap++) {
                int shift = tap * conv->dilation - conv->padding;
                const float *x0 = activated +
                    (size_t)(start + row + shift - input_min) * channels;
                const float *x1 = x0 + channels;
                const float *x2 = x1 + channels;
                const float *x3 = x2 + channels;
                const float *w = packed->data +
                    ((size_t)tap * channels * channels) + n0;
                for (int k = 0; k < channels; k++, w += channels) {
                    __m512 w0 = _mm512_load_ps(w);
                    __m512 w1 = _mm512_load_ps(w + 16);
                    __m512 w2 = _mm512_load_ps(w + 32);
                    __m512 xv = _mm512_set1_ps(x0[k]);
                    c00 = _mm512_fmadd_ps(xv, w0, c00);
                    c01 = _mm512_fmadd_ps(xv, w1, c01);
                    c02 = _mm512_fmadd_ps(xv, w2, c02);
                    xv = _mm512_set1_ps(x1[k]);
                    c10 = _mm512_fmadd_ps(xv, w0, c10);
                    c11 = _mm512_fmadd_ps(xv, w1, c11);
                    c12 = _mm512_fmadd_ps(xv, w2, c12);
                    xv = _mm512_set1_ps(x2[k]);
                    c20 = _mm512_fmadd_ps(xv, w0, c20);
                    c21 = _mm512_fmadd_ps(xv, w1, c21);
                    c22 = _mm512_fmadd_ps(xv, w2, c22);
                    xv = _mm512_set1_ps(x3[k]);
                    c30 = _mm512_fmadd_ps(xv, w0, c30);
                    c31 = _mm512_fmadd_ps(xv, w1, c31);
                    c32 = _mm512_fmadd_ps(xv, w2, c32);
                }
            }

            float *y0 = output + (size_t)row * channels + n0;
            float *y1 = y0 + channels;
            float *y2 = y1 + channels;
            float *y3 = y2 + channels;
            _mm512_storeu_ps(y0, c00); _mm512_storeu_ps(y0 + 16, c01); _mm512_storeu_ps(y0 + 32, c02);
            _mm512_storeu_ps(y1, c10); _mm512_storeu_ps(y1 + 16, c11); _mm512_storeu_ps(y1 + 32, c12);
            _mm512_storeu_ps(y2, c20); _mm512_storeu_ps(y2 + 16, c21); _mm512_storeu_ps(y2 + 32, c22);
            _mm512_storeu_ps(y3, c30); _mm512_storeu_ps(y3 + 16, c31); _mm512_storeu_ps(y3 + 32, c32);
        }
    }
}

static int run_avx512(const IroConv1d *conv, const float *alpha,
                      const float *input, int frames, float *output,
                      int rows_cap, const PackedWeights *packed) {
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
                          activated, activation_rows, conv->in_channels, alpha);
        float *block_output = output + (size_t)start * conv->out_channels;
        memset(block_output, 0, sizeof(float) * (size_t)rows * conv->out_channels);

        int full_custom = rows == rows_cap && (rows % 4) == 0 &&
                          (conv->out_channels % 48) == 0;
        if (full_custom) {
            for (int tap = 0; tap < conv->kernel; tap++) {
                int shift = tap * conv->dilation - conv->padding;
                if (start + shift < 0 || block_end - 1 + shift >= frames) {
                    full_custom = 0;
                    break;
                }
            }
        }
        if (full_custom) {
            fused_conv7_avx512(conv, packed, activated, input_min,
                               start, rows, block_output);
            continue;
        }

        for (int tap = 0; tap < conv->kernel; tap++) {
            int shift = tap * conv->dilation - conv->padding;
            int output_start = start;
            if (output_start < -shift) output_start = -shift;
            int output_end = frames - shift;
            if (output_end > block_end) output_end = block_end;
            if (output_end <= output_start) continue;
            int input_start = output_start + shift;
            int tap_rows = output_end - output_start;
            const float *x = activated +
                             (size_t)(input_start - input_min) * conv->in_channels;
            float *y = output + (size_t)output_start * conv->out_channels;
            iro_linear_strided_weight_add(
                x, conv->weight + (size_t)tap * conv->in_channels,
                weight_stride, y, tap_rows,
                conv->in_channels, conv->out_channels);
        }
    }
    free(activated);
    return 0;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof(*v), cmp_double);
    return n & 1 ? v[n/2] : 0.5 * (v[n/2-1] + v[n/2]);
}

static int bench_layer(const IroDACVAEDecoder *decoder, const char *golden,
                       int stage, int residual, int pairs) {
    const IroDACVAEResidual *unit = &decoder->stage[stage].residual[residual];
    const IroConv1d *conv = &unit->conv0;
    char path[1024];
    snprintf(path, sizeof(path), "%s/codec_decoder_stage%d_residual%d_input.f32",
             golden, stage, residual);
    int frames = 0;
    float *input = load_rows(path, conv->in_channels, &frames);
    size_t count = (size_t)frames * conv->out_channels;
    float *base = malloc(sizeof(float) * count);
    float *cand = malloc(sizeof(float) * count);
    double *tb = malloc(sizeof(double) * (size_t)pairs);
    double *tc = malloc(sizeof(double) * (size_t)pairs);
    int rows = workspace_rows(conv->in_channels);
    PackedWeights packed;
    double pack_begin = now_seconds();
    if (!input || !base || !cand || !tb || !tc || pack_init(&packed, conv) != 0)
        return -1;
    double pack_seconds = now_seconds() - pack_begin;
    run_direct(conv, unit->alpha0, input, frames, base, rows);
    run_avx512(conv, unit->alpha0, input, frames, cand, rows, &packed);
    double max_abs = 0.0, mae = 0.0;
    for (size_t i = 0; i < count; i++) {
        double d = fabs((double)base[i] - cand[i]);
        if (d > max_abs) max_abs = d;
        mae += d;
    }
    mae /= (double)count;
    for (int p = 0; p < pairs; p++) {
        double t0;
        if (!(p & 1)) {
            t0 = now_seconds(); run_direct(conv, unit->alpha0, input, frames, base, rows); tb[p] = now_seconds()-t0;
            t0 = now_seconds(); run_avx512(conv, unit->alpha0, input, frames, cand, rows, &packed); tc[p] = now_seconds()-t0;
        } else {
            t0 = now_seconds(); run_avx512(conv, unit->alpha0, input, frames, cand, rows, &packed); tc[p] = now_seconds()-t0;
            t0 = now_seconds(); run_direct(conv, unit->alpha0, input, frames, base, rows); tb[p] = now_seconds()-t0;
        }
        printf("{\"type\":\"pair\",\"stage\":%d,\"residual\":%d,\"channels\":%d,\"dilation\":%d,\"pair\":%d,\"baseline\":%.9f,\"avx512\":%.9f,\"speedup\":%.6f}\n",
               stage, residual, conv->in_channels, conv->dilation, p+1,
               tb[p], tc[p], tb[p]/tc[p]);
        fflush(stdout);
    }
    double mb = median(tb, pairs), mc = median(tc, pairs);
    printf("{\"type\":\"summary\",\"stage\":%d,\"residual\":%d,\"frames\":%d,\"channels\":%d,\"dilation\":%d,\"rows\":%d,\"pairs\":%d,\"baseline_median\":%.9f,\"avx512_median\":%.9f,\"speedup\":%.6f,\"max_abs\":%.9g,\"mae\":%.9g,\"packed_bytes\":%zu,\"pack_seconds\":%.9f}\n",
           stage, residual, frames, conv->in_channels, conv->dilation, rows, pairs,
           mb, mc, mb/mc, max_abs, mae, packed.bytes, pack_seconds);
    fflush(stdout);
    pack_free(&packed);
    free(input); free(base); free(cand); free(tb); free(tc);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s DECODER GOLDEN_DIR PAIRS\n", argv[0]);
        return 2;
    }
    int pairs = atoi(argv[3]);
    if (pairs < 3) return 2;
    iro_ops_set_threads(2);
    fprintf(stderr, "backend=%s threads=%d\n", iro_ops_backend_name(), iro_ops_get_threads());
    IroSafetensors st = {.fd=-1};
    if (iro_st_load(argv[1], &st) != 0) return 1;
    IroDACVAEDecoder decoder;
    if (iro_dacvae_init(&decoder, &st) != 0) return 1;
    int failed = 0;
    for (int stage=2; stage<=3; stage++)
        for (int residual=0; residual<IRO_DACVAE_RESIDUALS; residual++)
            failed |= bench_layer(&decoder, argv[2], stage, residual, pairs) != 0;
    iro_st_free(&st);
    return failed ? 1 : 0;
}
