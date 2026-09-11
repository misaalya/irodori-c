#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { ROW_MAJOR = 101, NO_TRANS = 111, TRANS = 112 };

extern void scipy_cblas_sgemm(int layout, int transa, int transb,
                              int m, int n, int k, float alpha,
                              const float *a, int lda,
                              const float *b, int ldb,
                              float beta, float *c, int ldc);
extern void scipy_openblas_set_num_threads(int threads);

typedef struct {
    const int8_t *pw;
    const float *sw;
    const int32_t *wsum;
    const uint8_t *qx;
    const float *sx;
    float *y;
    int m, k, n, n0, n1;
} Job;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static uint64_t rng_state = 0x123456789abcdefULL;
static float rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    uint32_t u = (uint32_t)(rng_state >> 32);
    return ((float)(u & 0xffffu) / 32768.0f - 1.0f) * 0.35f;
}

static void quantize_weights(const float *w, int8_t *qw, float *scale,
                             int32_t *sum, int n, int k) {
    for (int r = 0; r < n; r++) {
        const float *wr = w + (size_t)r * k;
        int8_t *qr = qw + (size_t)r * k;
        float amax = 0.0f;
        for (int i = 0; i < k; i++) {
            float a = fabsf(wr[i]);
            if (a > amax) amax = a;
        }
        float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        scale[r] = s;
        int32_t total = 0;
        for (int i = 0; i < k; i++) {
            int q = (int)lrintf(wr[i] / s);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            qr[i] = (int8_t)q;
            total += q;
        }
        sum[r] = total;
    }
}

static void pack_weights_vnni16x4(const int8_t *qw, int8_t *pw,
                                  int n, int k) {
    for (int n0 = 0; n0 < n; n0 += 16) {
        for (int k0 = 0; k0 < k; k0 += 4) {
            int8_t *dst = pw +
                ((size_t)(n0 / 16) * (size_t)(k / 4) + (size_t)(k0 / 4)) * 64;
            for (int lane = 0; lane < 16; lane++) {
                const int8_t *src = qw + (size_t)(n0 + lane) * k + k0;
                for (int q = 0; q < 4; q++) dst[lane * 4 + q] = src[q];
            }
        }
    }
}

static void quantize_activation(const float *x, uint8_t *qx, float *scale,
                                int m, int k) {
    for (int r = 0; r < m; r++) {
        const float *xr = x + (size_t)r * k;
        uint8_t *qr = qx + (size_t)r * k;
        float amax = 0.0f;
        for (int i = 0; i < k; i++) {
            float a = fabsf(xr[i]);
            if (a > amax) amax = a;
        }
        float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        scale[r] = s;
        for (int i = 0; i < k; i++) {
            int q = (int)lrintf(xr[i] / s);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            qr[i] = (uint8_t)(q + 128);
        }
    }
}

static void *vnni_rows(void *arg) {
    Job *j = arg;
    for (int n0 = j->n0; n0 < j->n1; n0 += 16) {
        __m512i correction = _mm512_mullo_epi32(
            _mm512_loadu_si512((const void *)(j->wsum + n0)),
            _mm512_set1_epi32(128));
        __m512 weight_scale = _mm512_loadu_ps(j->sw + n0);
        for (int m0 = 0; m0 < j->m; m0 += 16) {
            int bm = j->m - m0;
            if (bm > 16) bm = 16;
            __m512i acc[16];
            for (int b = 0; b < bm; b++) acc[b] = _mm512_setzero_si512();
            for (int k0 = 0; k0 < j->k; k0 += 4) {
                const int8_t *packed = j->pw +
                    ((size_t)(n0 / 16) * (size_t)(j->k / 4) +
                     (size_t)(k0 / 4)) * 64;
                __m512i wv = _mm512_loadu_si512((const void *)packed);
                for (int b = 0; b < bm; b++) {
                    const uint8_t *x = j->qx + (size_t)(m0 + b) * j->k + k0;
                    uint32_t four;
                    __builtin_memcpy(&four, x, sizeof(four));
                    __m512i ux = _mm512_set1_epi32((int)four);
                    acc[b] = _mm512_dpbusd_epi32(acc[b], ux, wv);
                }
            }
            for (int b = 0; b < bm; b++) {
                __m512i corrected = _mm512_sub_epi32(acc[b], correction);
                __m512 yf = _mm512_cvtepi32_ps(corrected);
                yf = _mm512_mul_ps(yf, weight_scale);
                yf = _mm512_mul_ps(yf, _mm512_set1_ps(j->sx[m0 + b]));
                _mm512_storeu_ps(j->y + (size_t)(m0 + b) * j->n + n0, yf);
            }
        }
    }
    return NULL;
}

static void vnni_linear(const int8_t *pw, const float *sw, const int32_t *wsum,
                        const uint8_t *qx, const float *sx, float *y,
                        int m, int k, int n, int threads) {
    pthread_t tids[2];
    Job jobs[2];
    if (threads > 2) threads = 2;
    int nblocks = n / 16;
    for (int t = 0; t < threads; t++) {
        jobs[t] = (Job){pw, sw, wsum, qx, sx, y, m, k, n,
                        (t * nblocks / threads) * 16,
                        ((t + 1) * nblocks / threads) * 16};
        pthread_create(&tids[t], NULL, vnni_rows, &jobs[t]);
    }
    for (int t = 0; t < threads; t++) pthread_join(tids[t], NULL);
}

static int bench(int m, int k, int n) {
    size_t x_count = (size_t)m * k;
    size_t w_count = (size_t)n * k;
    size_t y_count = (size_t)m * n;
    float *x = malloc(sizeof(float) * x_count);
    float *w = malloc(sizeof(float) * w_count);
    float *ref = malloc(sizeof(float) * y_count);
    float *out = malloc(sizeof(float) * y_count);
    uint8_t *qx = malloc(x_count);
    int8_t *qw = malloc(w_count);
    int8_t *pw = malloc(w_count);
    float *sx = malloc(sizeof(float) * (size_t)m);
    float *sw = malloc(sizeof(float) * (size_t)n);
    int32_t *wsum = malloc(sizeof(int32_t) * (size_t)n);
    if (!x || !w || !ref || !out || !qx || !qw || !pw ||
        !sx || !sw || !wsum)
        return 1;
    for (size_t i = 0; i < x_count; i++) x[i] = rnd();
    for (size_t i = 0; i < w_count; i++) w[i] = rnd();
    quantize_weights(w, qw, sw, wsum, n, k);
    pack_weights_vnni16x4(qw, pw, n, k);

    scipy_cblas_sgemm(ROW_MAJOR, NO_TRANS, TRANS, m, n, k, 1.0f,
                      x, k, w, k, 0.0f, ref, n);
    quantize_activation(x, qx, sx, m, k);
    vnni_linear(pw, sw, wsum, qx, sx, out, m, k, n, 2);

    double max_abs = 0.0, mae = 0.0, ref_abs = 0.0, dot = 0.0, aa = 0.0, bb = 0.0;
    for (size_t i = 0; i < y_count; i++) {
        double d = fabs((double)out[i] - ref[i]);
        if (d > max_abs) max_abs = d;
        mae += d;
        ref_abs += fabs((double)ref[i]);
        dot += (double)out[i] * ref[i];
        aa += (double)out[i] * out[i];
        bb += (double)ref[i] * ref[i];
    }
    mae /= y_count;
    ref_abs /= y_count;

    const int repeats = 5;
    double fp32[repeats], int8[repeats], int8_kernel[repeats], quant[repeats];
    for (int r = 0; r < repeats; r++) {
        double t0;
        if ((r & 1) == 0) {
            t0 = now_seconds();
            scipy_cblas_sgemm(ROW_MAJOR, NO_TRANS, TRANS, m, n, k, 1.0f,
                              x, k, w, k, 0.0f, ref, n);
            fp32[r] = now_seconds() - t0;
            t0 = now_seconds();
            quantize_activation(x, qx, sx, m, k);
            quant[r] = now_seconds() - t0;
            t0 = now_seconds();
            vnni_linear(pw, sw, wsum, qx, sx, out, m, k, n, 2);
            int8_kernel[r] = now_seconds() - t0;
            int8[r] = quant[r] + int8_kernel[r];
        } else {
            t0 = now_seconds();
            quantize_activation(x, qx, sx, m, k);
            quant[r] = now_seconds() - t0;
            t0 = now_seconds();
            vnni_linear(pw, sw, wsum, qx, sx, out, m, k, n, 2);
            int8_kernel[r] = now_seconds() - t0;
            int8[r] = quant[r] + int8_kernel[r];
            t0 = now_seconds();
            scipy_cblas_sgemm(ROW_MAJOR, NO_TRANS, TRANS, m, n, k, 1.0f,
                              x, k, w, k, 0.0f, ref, n);
            fp32[r] = now_seconds() - t0;
        }
    }
    for (int i = 0; i < repeats; i++) {
        for (int j = i + 1; j < repeats; j++) {
            if (fp32[j] < fp32[i]) { double z = fp32[i]; fp32[i] = fp32[j]; fp32[j] = z; }
            if (int8[j] < int8[i]) { double z = int8[i]; int8[i] = int8[j]; int8[j] = z; }
            if (int8_kernel[j] < int8_kernel[i]) { double z = int8_kernel[i]; int8_kernel[i] = int8_kernel[j]; int8_kernel[j] = z; }
            if (quant[j] < quant[i]) { double z = quant[i]; quant[i] = quant[j]; quant[j] = z; }
        }
    }
    double fmed = fp32[repeats / 2], imed = int8[repeats / 2];
    printf("M=%d K=%d N=%d fp32=%.6f s vnni=%.6f s kernel=%.6f s quant=%.6f s speedup=%.3fx "
           "max=%.6g mae=%.6g rel_mae=%.4f%% cosine=%.9f\n",
           m, k, n, fmed, imed, int8_kernel[repeats / 2], quant[repeats / 2],
           fmed / imed, max_abs, mae,
           100.0 * mae / (ref_abs + 1e-30), dot / sqrt(aa * bb));
    free(x); free(w); free(ref); free(out); free(qx); free(qw); free(pw);
    free(sx); free(sw); free(wsum);
    return 0;
}

int main(void) {
    scipy_openblas_set_num_threads(2);
    int rc = 0;
    rc |= bench(357, 1280, 1280);
    rc |= bench(357, 1280, 3680);
    rc |= bench(357, 3680, 1280);
    rc |= bench(119, 1280, 1280);
    return rc;
}
