/* ops.c — scalar oracle plus an optional CBLAS SGEMM backend. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ops.h"

#ifdef IRO_USE_CBLAS
/* Keep the runtime header-independent. These values and ABI are standardized
   by CBLAS and work with OpenBLAS, BLIS, Accelerate, and reference libblas. */
enum {
    IRO_CBLAS_ROW_MAJOR = 101,
    IRO_CBLAS_NO_TRANS = 111,
    IRO_CBLAS_TRANS = 112,
};
#ifdef IRO_USE_TORCH_MKL
/* PyTorch CPU wheels bundle oneMKL into libtorch_cpu.so but do not export the
   ordinary cblas_sgemm symbol.  The Fortran SGEMM ABI is exported, so map our
   row-major CBLAS calls to the equivalent column-major operation by swapping
   A/B and M/N.  This keeps every tensor and accumulation in FP32. */
extern void sgemm_(const char *transa, const char *transb,
                   const int *m, const int *n, const int *k,
                   const float *alpha, const float *a, const int *lda,
                   const float *b, const int *ldb, const float *beta,
                   float *c, const int *ldc);
extern int mkl_get_max_threads(void);

static void iro_torch_mkl_sgemm(int layout, int transa, int transb,
                                int m, int n, int k, float alpha,
                                const float *a, int lda,
                                const float *b, int ldb,
                                float beta, float *c, int ldc) {
    if (layout != IRO_CBLAS_ROW_MAJOR) return;
    const char ta = transb == IRO_CBLAS_TRANS ? 'T' : 'N';
    const char tb = transa == IRO_CBLAS_TRANS ? 'T' : 'N';
    sgemm_(&ta, &tb, &n, &m, &k,
           &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}
#define iro_cblas_sgemm iro_torch_mkl_sgemm
#elif defined(IRO_USE_SCIPY_OPENBLAS)
extern void scipy_cblas_sgemm(int layout, int transa, int transb,
                              int m, int n, int k, float alpha,
                              const float *a, int lda,
                              const float *b, int ldb,
                              float beta, float *c, int ldc);
#define iro_cblas_sgemm scipy_cblas_sgemm
#else
extern void cblas_sgemm(int layout, int transa, int transb,
                        int m, int n, int k, float alpha,
                        const float *a, int lda,
                        const float *b, int ldb,
                        float beta, float *c, int ldc);
#define iro_cblas_sgemm cblas_sgemm
#endif
#ifdef IRO_USE_OPENBLAS
extern void openblas_set_num_threads(int threads);
extern int openblas_get_num_threads(void);
#endif
#ifdef IRO_USE_SCIPY_OPENBLAS
extern void scipy_openblas_set_num_threads(int threads);
extern int scipy_openblas_get_num_threads(void);
#endif
#endif

int iro_ops_set_threads(int threads) {
#if defined(IRO_USE_CBLAS) && defined(IRO_USE_TORCH_MKL)
    /* The PyTorch wheel exposes no safe global MKL thread setter.  The
       irodori-mkl launcher maps IRO_NUM_THREADS to MKL_NUM_THREADS and
       OMP_NUM_THREADS before libtorch/MKL initialization. */
    (void)threads;
    return mkl_get_max_threads();
#elif defined(IRO_USE_CBLAS) && defined(IRO_USE_SCIPY_OPENBLAS)
    if (threads > 0) scipy_openblas_set_num_threads(threads);
    return scipy_openblas_get_num_threads();
#elif defined(IRO_USE_CBLAS) && defined(IRO_USE_OPENBLAS)
    if (threads > 0) openblas_set_num_threads(threads);
    return openblas_get_num_threads();
#else
    (void)threads;
    return 0;
#endif
}

int iro_ops_get_threads(void) {
#if defined(IRO_USE_CBLAS) && defined(IRO_USE_TORCH_MKL)
    return mkl_get_max_threads();
#elif defined(IRO_USE_CBLAS) && defined(IRO_USE_SCIPY_OPENBLAS)
    return scipy_openblas_get_num_threads();
#elif defined(IRO_USE_CBLAS) && defined(IRO_USE_OPENBLAS)
    return openblas_get_num_threads();
#else
    return 0;
#endif
}

int iro_ops_has_cblas(void) {
#ifdef IRO_USE_CBLAS
    return 1;
#else
    return 0;
#endif
}

void iro_linear_scalar(const float *x, const float *w, const float *bias,
                       float *y, int M, int K, int N) {
    /* y[M,N] = x[M,K] @ W[N,K]^T */
    for (int m = 0; m < M; m++) {
        const float *xm = x + (size_t)m * K;
        float *ym = y + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            const float *wn = w + (size_t)n * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += xm[k] * wn[k];
            ym[n] = acc + (bias ? bias[n] : 0.0f);
        }
    }
}

void iro_linear(const float *x, const float *w, const float *bias,
                float *y, int M, int K, int N) {
#ifdef IRO_USE_CBLAS
    iro_cblas_sgemm(IRO_CBLAS_ROW_MAJOR, IRO_CBLAS_NO_TRANS, IRO_CBLAS_TRANS,
                    M, N, K, 1.0f, x, K, w, K, 0.0f, y, N);
    if (bias) {
        for (int m = 0; m < M; m++) {
            float *row = y + (size_t)m * N;
            for (int n = 0; n < N; n++) row[n] += bias[n];
        }
    }
#else
    iro_linear_scalar(x, w, bias, y, M, K, N);
#endif
}

void iro_linear_strided_input(const float *x, int x_stride,
                              const float *w, float *y,
                              int M, int K, int N) {
#ifdef IRO_USE_CBLAS
    iro_cblas_sgemm(IRO_CBLAS_ROW_MAJOR, IRO_CBLAS_NO_TRANS, IRO_CBLAS_TRANS,
                    M, N, K, 1.0f, x, x_stride, w, K, 0.0f, y, N);
#else
    for (int m = 0; m < M; m++) {
        const float *xm = x + (size_t)m * x_stride;
        float *ym = y + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            const float *wn = w + (size_t)n * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += xm[k] * wn[k];
            ym[n] = acc;
        }
    }
#endif
}

void iro_linear_add(const float *x, const float *w, const float *bias,
                    const float *add, float *y, int M, int K, int N) {
#ifdef IRO_USE_CBLAS
    for (int m = 0; m < M; m++) {
        const float *add_row = add + (size_t)m * N;
        float *row = y + (size_t)m * N;
        for (int n = 0; n < N; n++)
            row[n] = add_row[n] + (bias ? bias[n] : 0.0f);
    }
    iro_cblas_sgemm(IRO_CBLAS_ROW_MAJOR, IRO_CBLAS_NO_TRANS, IRO_CBLAS_TRANS,
                    M, N, K, 1.0f, x, K, w, K, 1.0f, y, N);
#else
    for (int m = 0; m < M; m++) {
        const float *xm = x + (size_t)m * K;
        const float *add_row = add + (size_t)m * N;
        float *ym = y + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            const float *wn = w + (size_t)n * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += xm[k] * wn[k];
            ym[n] = acc + (bias ? bias[n] : 0.0f) + add_row[n];
        }
    }
#endif
}

void iro_linear_strided_weight_add(const float *x, const float *w,
                                   int w_stride, float *y,
                                   int M, int K, int N) {
#ifdef IRO_USE_CBLAS
    iro_cblas_sgemm(IRO_CBLAS_ROW_MAJOR, IRO_CBLAS_NO_TRANS, IRO_CBLAS_TRANS,
                    M, N, K, 1.0f, x, K, w, w_stride, 1.0f, y, N);
#else
    for (int m = 0; m < M; m++) {
        const float *xm = x + (size_t)m * K;
        float *ym = y + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            const float *wn = w + (size_t)n * w_stride;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += xm[k] * wn[k];
            ym[n] += acc;
        }
    }
#endif
}

void iro_matmul(const float *a, const float *b, float *c,
                int M, int K, int N) {
#ifdef IRO_USE_CBLAS
    iro_cblas_sgemm(IRO_CBLAS_ROW_MAJOR, IRO_CBLAS_NO_TRANS, IRO_CBLAS_NO_TRANS,
                    M, N, K, 1.0f, a, K, b, N, 0.0f, c, N);
#else
    for (int m = 0; m < M; m++) {
        const float *arow = a + (size_t)m * K;
        float *crow = c + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += arow[k] * b[(size_t)k * N + n];
            crow[n] = sum;
        }
    }
#endif
}

void iro_matmul_strided_output(const float *a, const float *b,
                               float *c, int c_stride,
                               int M, int K, int N) {
#ifdef IRO_USE_CBLAS
    iro_cblas_sgemm(IRO_CBLAS_ROW_MAJOR, IRO_CBLAS_NO_TRANS, IRO_CBLAS_NO_TRANS,
                    M, N, K, 1.0f, a, K, b, N, 0.0f, c, c_stride);
#else
    for (int m = 0; m < M; m++) {
        const float *arow = a + (size_t)m * K;
        float *crow = c + (size_t)m * c_stride;
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += arow[k] * b[(size_t)k * N + n];
            crow[n] = sum;
        }
    }
#endif
}

const char *iro_ops_backend_name(void) {
#ifdef IRO_USE_TORCH_MKL
    return "torch-mkl-sgemm";
#elif defined(IRO_USE_CBLAS)
    return "cblas-sgemm";
#else
    return "scalar";
#endif
}

void iro_layernorm(const float *x, const float *w, const float *b,
                   float *y, int M, int D, float eps) {
    for (int m = 0; m < M; m++) {
        const float *xm = x + (size_t)m * D;
        float *ym = y + (size_t)m * D;
        float mean = 0.0f;
        for (int d = 0; d < D; d++) mean += xm[d];
        mean /= (float)D;
        float var = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = xm[d] - mean;
            var += diff * diff;
        }
        var /= (float)D;
        float inv = 1.0f / sqrtf(var + eps);
        for (int d = 0; d < D; d++)
            ym[d] = (xm[d] - mean) * inv * w[d] + (b ? b[d] : 0.0f);
    }
}

void iro_rmsnorm(const float *x, const float *w, float *y, int M, int D, float eps) {
    for (int m = 0; m < M; m++) {
        const float *xm = x + (size_t)m * D;
        float *ym = y + (size_t)m * D;
        float ms = 0.0f;
        for (int d = 0; d < D; d++) ms += xm[d] * xm[d];
        float inv = 1.0f / sqrtf(ms / (float)D + eps);
        for (int d = 0; d < D; d++) ym[d] = xm[d] * inv * w[d];
    }
}

float iro_gelu(float x) {
    const float k = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + erff(x * k));
}

void iro_softmax_row(float *x, int R, int C) {
    for (int r = 0; r < R; r++) {
        float *xr = x + (size_t)r * C;
        float mx = xr[0];
        for (int c = 1; c < C; c++) if (xr[c] > mx) mx = xr[c];
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            xr[c] = expf(xr[c] - mx);
            sum += xr[c];
        }
        for (int c = 0; c < C; c++) xr[c] /= sum;
    }
}

float iro_silu(float x) {
    return x * iro_sigmoid(x);
}

float iro_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void iro_silu_inplace(float *x, size_t count) {
    for (size_t i = 0; i < count; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

void iro_silu_mul_inplace(float *x, const float *gate, size_t count) {
    for (size_t i = 0; i < count; i++)
        x[i] = (x[i] / (1.0f + expf(-x[i]))) * gate[i];
}

void iro_sigmoid_mul_inplace(float *x, const float *gate, size_t count) {
    for (size_t i = 0; i < count; i++)
        x[i] *= 1.0f / (1.0f + expf(-gate[i]));
}
