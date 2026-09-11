#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { ROW_MAJOR = 101, NO_TRANS = 111, TRANS = 112 };
extern void scipy_cblas_sgemm(int, int, int, int, int, int, float,
                              const float *, int, const float *, int,
                              float, float *, int);
extern void scipy_openblas_set_num_threads(int);

typedef struct {
    const float *x;
    const float *pw;
    float *y;
    int m, k, n, nb0, nb1;
} Job;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static uint64_t rs = 0x9e3779b97f4a7c15ULL;
static float rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return ((float)((uint32_t)(rs >> 32) & 65535) / 32768.0f - 1.0f) * 0.2f;
}

static void pack_w(const float *w, float *pw, int n, int k) {
    for (int n0 = 0; n0 < n; n0 += 16)
        for (int kk = 0; kk < k; kk++) {
            float *dst = pw + ((size_t)(n0 / 16) * k + kk) * 16;
            for (int lane = 0; lane < 16; lane++)
                dst[lane] = w[(size_t)(n0 + lane) * k + kk];
        }
}

static void *kernel(void *vp) {
    Job *j = vp;
    for (int nb = j->nb0; nb < j->nb1; nb++) {
        int n0 = nb * 16;
        const float *wp = j->pw + (size_t)nb * j->k * 16;
        for (int m0 = 0; m0 < j->m; m0 += 12) {
            int bm = j->m - m0;
            if (bm > 12) bm = 12;
            __m512 acc[12];
            for (int b = 0; b < bm; b++) acc[b] = _mm512_setzero_ps();
            for (int kk = 0; kk < j->k; kk++) {
                __m512 wv = _mm512_loadu_ps(wp + (size_t)kk * 16);
                for (int b = 0; b < bm; b++) {
                    __m512 xv = _mm512_set1_ps(j->x[(size_t)(m0 + b) * j->k + kk]);
                    acc[b] = _mm512_fmadd_ps(xv, wv, acc[b]);
                }
            }
            for (int b = 0; b < bm; b++)
                _mm512_storeu_ps(j->y + (size_t)(m0 + b) * j->n + n0, acc[b]);
        }
    }
    return NULL;
}

static void packed_linear(const float *x, const float *pw, float *y,
                          int m, int k, int n) {
    pthread_t th[2];
    Job jobs[2];
    int blocks = n / 16;
    for (int t = 0; t < 2; t++) {
        jobs[t] = (Job){x, pw, y, m, k, n,
                        t * blocks / 2, (t + 1) * blocks / 2};
        pthread_create(&th[t], NULL, kernel, &jobs[t]);
    }
    pthread_join(th[0], NULL);
    pthread_join(th[1], NULL);
}

static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void bench(int m, int k, int n) {
    size_t nx = (size_t)m * k, nw = (size_t)n * k, ny = (size_t)m * n;
    float *x = malloc(nx * 4), *w = malloc(nw * 4), *pw = malloc(nw * 4);
    float *ref = malloc(ny * 4), *out = malloc(ny * 4);
    for (size_t i = 0; i < nx; i++) x[i] = rnd();
    for (size_t i = 0; i < nw; i++) w[i] = rnd();
    pack_w(w, pw, n, k);
    scipy_cblas_sgemm(ROW_MAJOR, NO_TRANS, TRANS, m, n, k, 1, x, k, w, k, 0, ref, n);
    packed_linear(x, pw, out, m, k, n);
    double maxe = 0, mae = 0;
    for (size_t i = 0; i < ny; i++) {
        double e = fabs((double)ref[i] - out[i]);
        if (e > maxe) maxe = e;
        mae += e;
    }
    mae /= ny;
    double a[7], b[7];
    for (int r = 0; r < 7; r++) {
        double t = now_s();
        if (!(r & 1)) {
            scipy_cblas_sgemm(ROW_MAJOR, NO_TRANS, TRANS, m, n, k, 1, x, k, w, k, 0, ref, n);
            a[r] = now_s() - t; t = now_s(); packed_linear(x, pw, out, m, k, n); b[r] = now_s() - t;
        } else {
            packed_linear(x, pw, out, m, k, n); b[r] = now_s() - t; t = now_s();
            scipy_cblas_sgemm(ROW_MAJOR, NO_TRANS, TRANS, m, n, k, 1, x, k, w, k, 0, ref, n); a[r] = now_s() - t;
        }
    }
    qsort(a, 7, sizeof(double), cmpd); qsort(b, 7, sizeof(double), cmpd);
    printf("M=%d K=%d N=%d openblas=%.6f packed=%.6f speedup=%.3fx max=%.6g mae=%.6g\n",
           m, k, n, a[3], b[3], a[3] / b[3], maxe, mae);
    free(x); free(w); free(pw); free(ref); free(out);
}

int main(void) {
    scipy_openblas_set_num_threads(2);
    bench(357, 1280, 1280);
    bench(357, 1280, 3680);
    bench(357, 3680, 1280);
    bench(119, 1280, 1280);
    return 0;
}
