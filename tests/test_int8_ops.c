/* test_int8_ops.c — W8A8 primitive checks: reconstruction bounds, exact
   integer sums against a local reference, cross-M determinism, and float
   error against the FP32 oracle.  Runs on every backend. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ops.h"

static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static float rnd(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 40) & 0xFFFFFF) / 16777216.0f - 0.5f;
}

static void reference_i32(const uint8_t *a, const int8_t *b, int32_t *c,
                          int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)a[(size_t)m * K + k] * (int32_t)b[(size_t)n * K + k];
            c[(size_t)m * N + n] = acc;
        }
}

static int check(int ok, const char *what) {
    printf("  %s: %s\n", what, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int run_case(int M, int N, int K, int outlier) {
    int failures = 0;
    printf("case M=%d N=%d K=%d outlier=%d\n", M, N, K, outlier);
    float *w = malloc(sizeof(float) * (size_t)N * K);
    float *x = malloc(sizeof(float) * (size_t)M * K);
    float *y_ref = malloc(sizeof(float) * (size_t)M * N);
    float *y = malloc(sizeof(float) * (size_t)M * N);
    float *bias = malloc(sizeof(float) * (size_t)N);
    uint8_t *q = malloc((size_t)M * K);
    float *sx = malloc(sizeof(float) * (size_t)M);
    int32_t *zx = malloc(sizeof(int32_t) * (size_t)M);
    int32_t *work = malloc(sizeof(int32_t) * (size_t)M * N);
    int32_t *expect = malloc(sizeof(int32_t) * (size_t)M * N);
    if (!w || !x || !y_ref || !y || !bias || !q || !sx || !zx || !work || !expect)
        return 1;
    for (size_t i = 0; i < (size_t)N * K; i++) w[i] = rnd() * 0.08f;
    for (size_t i = 0; i < (size_t)M * K; i++) x[i] = rnd() * 3.0f;
    if (outlier)
        for (int m = 0; m < M; m++) x[(size_t)m * K + 17] = 20.0f + rnd();
    for (int n = 0; n < N; n++) bias[n] = rnd();

    IroInt8Weight qw;
    failures += check(iro_int8_weight_quantize(&qw, w, N, K) == 0, "weight quantize");
    /* Reconstruction bound: |w - s*q| <= s/2 and colsum matches. */
    int bound_ok = 1, colsum_ok = 1;
    for (int n = 0; n < N && bound_ok; n++) {
        int32_t sum = 0;
        for (int k = 0; k < K; k++) {
            float d = fabsf(w[(size_t)n * K + k] - qw.scale[n] * qw.data[(size_t)n * K + k]);
            if (d > qw.scale[n] * 0.5f + 1e-7f) bound_ok = 0;
            sum += qw.data[(size_t)n * K + k];
        }
        if (sum != qw.colsum[n]) colsum_ok = 0;
    }
    failures += check(bound_ok, "weight reconstruction bound");
    failures += check(colsum_ok, "weight colsum");

    iro_int8_quantize_rows(x, M, K, q, sx, zx);
    int act_ok = 1;
    for (int m = 0; m < M && act_ok; m++) {
        if (zx[m] < 0 || zx[m] > 255 || !(sx[m] > 0.0f)) act_ok = 0;
        for (int k = 0; k < K; k++) {
            float rec = sx[m] * ((float)q[(size_t)m * K + k] - (float)zx[m]);
            if (fabsf(rec - x[(size_t)m * K + k]) > sx[m] * 0.5f + 1e-5f) act_ok = 0;
        }
    }
    failures += check(act_ok, "activation reconstruction bound");

    iro_int8_linear(q, sx, zx, &qw, bias, y, M, work);
    reference_i32(q, qw.data, expect, M, N, K);
    failures += check(memcmp(work, expect, sizeof(int32_t) * (size_t)M * N) == 0,
                      "int32 sums bit-exact vs reference");

    /* Cross-M determinism: computing only the last 5 rows must reproduce the
       same integer sums as the full call. */
    int sub = M < 5 ? M : 5;
    int32_t *work_sub = malloc(sizeof(int32_t) * (size_t)sub * N);
    float *y_sub = malloc(sizeof(float) * (size_t)sub * N);
    iro_int8_linear(q + (size_t)(M - sub) * K, sx + (M - sub), zx + (M - sub),
                    &qw, bias, y_sub, sub, work_sub);
    failures += check(memcmp(work_sub, expect + (size_t)(M - sub) * N,
                             sizeof(int32_t) * (size_t)sub * N) == 0,
                      "cross-M integer sums identical");
    free(work_sub); free(y_sub);

    iro_linear_scalar(x, w, bias, y_ref, M, K, N);
    double num = 0.0, den = 0.0, worst = 0.0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        double d = (double)y[i] - (double)y_ref[i];
        num += d * d;
        den += (double)y_ref[i] * (double)y_ref[i];
        if (fabs(d) > worst) worst = fabs(d);
    }
    double rel = sqrt(num / (den > 0 ? den : 1.0));
    printf("  relative RMSE vs FP32 %.4f%% max abs %.5f\n", rel * 100.0, worst);
    failures += check(rel < (outlier ? 0.05 : 0.02), "float error within budget");

    iro_int8_weight_free(&qw);
    failures += check(qw.data == NULL && iro_int8_weight_bytes(&qw) == 0, "weight free resets");
    free(w); free(x); free(y_ref); free(y); free(bias); free(q); free(sx);
    free(zx); free(work); free(expect);
    return failures;
}

int main(void) {
    int failures = 0;
    printf("int8 fast backend: %s\n", iro_int8_fast_available() ? "yes" : "no (reference)");
    failures += check(iro_int8_selfcheck() == 0, "backend selfcheck (no int16 saturation)");
    failures += run_case(7, 32, 64, 0);
    failures += run_case(23, 96, 128, 1);
    failures += run_case(112, 160, 1280, 0);
    failures += run_case(37, 3680 / 20, 1280, 1);
    /* Degenerate rows: all zero and constant-positive. */
    {
        float x[2 * 16] = {0};
        for (int k = 0; k < 16; k++) x[16 + k] = 2.5f;
        uint8_t q[2 * 16]; float s[2]; int32_t z[2];
        iro_int8_quantize_rows(x, 2, 16, q, s, z);
        int ok = s[0] > 0.0f && z[0] >= 0 && z[0] <= 255 && q[16] == 255 && z[1] == 0;
        failures += check(ok, "degenerate rows");
    }
    printf("int8 ops: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
