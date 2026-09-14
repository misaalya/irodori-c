/* ops.c — scalar oracle plus an optional CBLAS SGEMM backend. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ops.h"

#ifdef IRO_USE_ONEMKL
#include <mkl.h>
#endif

enum { IRO_GEMM_PROFILE_MAX_SHAPES = 256 };

typedef struct {
    const char *kind;
    int m, n, k;
    int lda, ldb, ldc;
    float beta;
    uint64_t calls;
    double seconds;
} IroGemmProfileRow;

static int gemm_profile_state = -1;
static IroGemmProfileRow gemm_profile_rows[IRO_GEMM_PROFILE_MAX_SHAPES];
static int gemm_profile_count;
static uint64_t gemm_profile_dropped;

#ifdef IRO_USE_CBLAS
static double gemm_profile_now(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

int iro_gemm_profile_enabled(void) {
    if (gemm_profile_state < 0) {
        const char *env = getenv("IRO_GEMM_PROFILE");
        gemm_profile_state = env && strcmp(env, "0") != 0;
    }
    return gemm_profile_state;
}

void iro_gemm_profile_reset(void) {
    if (!iro_gemm_profile_enabled()) return;
    memset(gemm_profile_rows, 0, sizeof(gemm_profile_rows));
    gemm_profile_count = 0;
    gemm_profile_dropped = 0;
}

#ifdef IRO_USE_CBLAS
static void gemm_profile_record(const char *kind,
                                int m, int n, int k,
                                int lda, int ldb, int ldc,
                                float beta, double seconds) {
    if (!iro_gemm_profile_enabled()) return;
    for (int i = 0; i < gemm_profile_count; i++) {
        IroGemmProfileRow *row = &gemm_profile_rows[i];
        if (row->kind == kind && row->m == m && row->n == n && row->k == k &&
            row->lda == lda && row->ldb == ldb && row->ldc == ldc &&
            row->beta == beta) {
            row->calls++;
            row->seconds += seconds;
            return;
        }
    }
    if (gemm_profile_count >= IRO_GEMM_PROFILE_MAX_SHAPES) {
        gemm_profile_dropped++;
        return;
    }
    IroGemmProfileRow *row = &gemm_profile_rows[gemm_profile_count++];
    row->kind = kind;
    row->m = m;
    row->n = n;
    row->k = k;
    row->lda = lda;
    row->ldb = ldb;
    row->ldc = ldc;
    row->beta = beta;
    row->calls = 1;
    row->seconds = seconds;
}
#endif

void iro_gemm_profile_emit_json(const char *status, int steps,
                                int sequence_length, int context_tokens) {
    if (!iro_gemm_profile_enabled()) return;
    fprintf(stderr,
            "{\"type\":\"iro_gemm_profile\",\"status\":\"%s\","
            "\"backend\":\"%s\",\"steps\":%d,\"sequence_length\":%d,"
            "\"context_tokens\":%d,\"dropped_shapes\":%llu,\"rows\":[",
            status ? status : "unknown", iro_ops_backend_name(), steps,
            sequence_length, context_tokens,
            (unsigned long long)gemm_profile_dropped);
    for (int i = 0; i < gemm_profile_count; i++) {
        const IroGemmProfileRow *row = &gemm_profile_rows[i];
        if (i) fputc(',', stderr);
        fprintf(stderr,
                "{\"kind\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,"
                "\"lda\":%d,\"ldb\":%d,\"ldc\":%d,\"beta\":%.1f,"
                "\"backend\":\"%s\",\"calls\":%llu,\"seconds\":%.9f}",
                row->kind, row->m, row->n, row->k,
                row->lda, row->ldb, row->ldc, row->beta,
                iro_ops_backend_name(), (unsigned long long)row->calls,
                row->seconds);
    }
    fputs("]}\n", stderr);
}

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
#elif defined(IRO_USE_ONEMKL)
static void iro_onemkl_sgemm(int layout, int transa, int transb,
                            int m, int n, int k, float alpha,
                            const float *a, int lda, const float *b, int ldb,
                            float beta, float *c, int ldc) {
    cblas_sgemm((CBLAS_LAYOUT)layout, (CBLAS_TRANSPOSE)transa,
                (CBLAS_TRANSPOSE)transb, m, n, k, alpha,
                a, lda, b, ldb, beta, c, ldc);
}
#define iro_cblas_sgemm iro_onemkl_sgemm
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
#if defined(IRO_USE_CBLAS) && defined(IRO_USE_ONEMKL)
    if (threads > 0) mkl_set_num_threads(threads);
    return mkl_get_max_threads();
#elif defined(IRO_USE_CBLAS) && defined(IRO_USE_TORCH_MKL)
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
#if defined(IRO_USE_CBLAS) && (defined(IRO_USE_TORCH_MKL) || defined(IRO_USE_ONEMKL))
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
    int profile = iro_gemm_profile_enabled();
    double t0 = profile ? gemm_profile_now() : 0.0;
    iro_cblas_sgemm(IRO_CBLAS_ROW_MAJOR, IRO_CBLAS_NO_TRANS, IRO_CBLAS_TRANS,
                    M, N, K, 1.0f, x, K, w, K, 0.0f, y, N);
    if (profile)
        gemm_profile_record("linear", M, N, K, K, K, N, 0.0f,
                            gemm_profile_now() - t0);
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

/* ---- W8A8 integer path ---------------------------------------------- */

int iro_int8_weight_quantize(IroInt8Weight *dst, const float *w, int N, int K) {
    if (!dst || !w || N <= 0 || K <= 0) return -1;
    memset(dst, 0, sizeof(*dst));
    dst->data = malloc((size_t)N * K);
    dst->scale = malloc(sizeof(float) * (size_t)N);
    dst->colsum = malloc(sizeof(int32_t) * (size_t)N);
    if (!dst->data || !dst->scale || !dst->colsum) {
        iro_int8_weight_free(dst);
        return -1;
    }
    for (int n = 0; n < N; n++) {
        const float *row = w + (size_t)n * K;
        int8_t *q = dst->data + (size_t)n * K;
        float amax = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = fabsf(row[k]);
            if (a > amax) amax = a;
        }
        float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        float inv = 1.0f / scale;
        int32_t sum = 0;
        for (int k = 0; k < K; k++) {
            int v = (int)nearbyintf(row[k] * inv);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            q[k] = (int8_t)v;
            sum += v;
        }
        dst->scale[n] = scale;
        dst->colsum[n] = sum;
    }
    dst->N = N;
    dst->K = K;
    return 0;
}

int iro_int8_weight_quantize_transposed(IroInt8Weight *dst, const float *w,
                                        int N, int K) {
    if (!dst || !w || N <= 0 || K <= 0) return -1;
    float *t = malloc(sizeof(float) * (size_t)N * K);
    if (!t) return -1;
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++)
            t[(size_t)n * K + k] = w[(size_t)k * N + n];
    int rc = iro_int8_weight_quantize(dst, t, N, K);
    free(t);
    return rc;
}

void iro_int8_weight_free(IroInt8Weight *w) {
    if (!w) return;
    free(w->data);
    free(w->scale);
    free(w->colsum);
    memset(w, 0, sizeof(*w));
}

size_t iro_int8_weight_bytes(const IroInt8Weight *w) {
    if (!w || !w->data) return 0;
    return (size_t)w->N * w->K + (size_t)w->N * (sizeof(float) + sizeof(int32_t));
}

void iro_int8_quantize_rows(const float *x, int M, int K,
                            uint8_t *q, float *scale, int32_t *zero) {
    for (int m = 0; m < M; m++) {
        const float *row = x + (size_t)m * K;
        uint8_t *qrow = q + (size_t)m * K;
        float lo = 0.0f, hi = 0.0f;
        for (int k = 0; k < K; k++) {
            lo = row[k] < lo ? row[k] : lo;
            hi = row[k] > hi ? row[k] : hi;
        }
        /* Range always includes zero so the zero point is representable. */
        float s = (hi - lo) / 255.0f;
        if (!(s > 0.0f)) s = 1.0f;
        float inv = 1.0f / s;
        int zp = (int)nearbyintf(-lo * inv);
        if (zp < 0) zp = 0;
        if (zp > 255) zp = 255;
        float zpf = (float)zp;
        for (int k = 0; k < K; k++) {
            float v = nearbyintf(row[k] * inv + zpf);
            v = v < 0.0f ? 0.0f : v;
            v = v > 255.0f ? 255.0f : v;
            qrow[k] = (uint8_t)v;
        }
        scale[m] = s;
        zero[m] = zp;
    }
}

#ifndef IRO_USE_ONEMKL
static void int8_gemm_reference(const uint8_t *a, const int8_t *b,
                                int32_t *c, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        const uint8_t *arow = a + (size_t)m * K;
        int32_t *crow = c + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            const int8_t *brow = b + (size_t)n * K;
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)arow[k] * (int32_t)brow[k];
            crow[n] = acc;
        }
    }
}
#endif

int iro_int8_fast_available(void) {
#ifdef IRO_USE_ONEMKL
    return 1;
#else
    return 0;
#endif
}

int iro_int8_selfcheck(void) {
    enum { M = 3, N = 5, K = 64 };
    uint8_t a[M * K];
    int8_t b[N * K];
    int32_t got[M * N], want[M * N];
    /* Extremes: 255*127 pairs sum to 64770 > INT16_MAX, so any int16
       intermediate saturates; mixed signs catch wrong offset handling. */
    for (int i = 0; i < M * K; i++) a[i] = (uint8_t)(i % 3 == 0 ? 255 : 200 + i % 50);
    for (int i = 0; i < N * K; i++) b[i] = (int8_t)(i % 4 == 0 ? 127 : (i % 4 == 1 ? -127 : 90 - i % 60));
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) acc += (int32_t)a[m * K + k] * (int32_t)b[n * K + k];
            want[m * N + n] = acc;
        }
    float wscale[N] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
    int32_t colsum[N] = { 0 };
    IroInt8Weight w = { .data = b, .scale = wscale, .colsum = colsum, .N = N, .K = K };
    float scale[M] = { 1.0f, 1.0f, 1.0f }, y[M * N];
    int32_t zero[M] = { 0, 0, 0 };
    /* Only the int32 work buffer is inspected; the float output is unused. */
    iro_int8_linear_ex(a, scale, zero, &w, NULL, y, M, got, 0);
    return memcmp(got, want, sizeof(got)) == 0 ? 0 : -1;
}

const char *iro_int8_prepare_backend(void) {
#ifdef IRO_USE_ONEMKL
    static const char *branch;
    if (branch) return branch;
    const char *env = getenv("MKL_CBWR");
    if (!env || !*env) {
        int rc = mkl_cbwr_set(MKL_CBWR_AVX512_E1);
        if (rc == MKL_CBWR_SUCCESS) {
            branch = "AVX512_E1";
            return branch;
        }
        fprintf(stderr, "int8: mkl_cbwr_set(AVX512_E1) gagal (%d); "
                        "GEMM int8 mungkin berjalan tanpa VNNI\n", rc);
        branch = "auto";
        return branch;
    }
    branch = env;
    if (!strcmp(env, "AVX2") || !strcmp(env, "AVX512") ||
        !strcmp(env, "COMPATIBLE") || !strncmp(env, "SSE", 3))
        fprintf(stderr, "int8: MKL_CBWR=%s tidak menyertakan VNNI; GEMM int8 "
                        "sekitar 2x lebih lambat dari AVX512_E1/AUTO\n", env);
    return branch;
#else
    return NULL;
#endif
}

void iro_int8_quantize_rows_residual(const float *x, int M, int K,
                                     uint8_t *q1, float *scale1, int32_t *zero1,
                                     float *residual,
                                     uint8_t *q2, float *scale2, int32_t *zero2) {
    iro_int8_quantize_rows(x, M, K, q1, scale1, zero1);
    for (int m = 0; m < M; m++) {
        const float *row = x + (size_t)m * K;
        const uint8_t *qrow = q1 + (size_t)m * K;
        float *rrow = residual + (size_t)m * K;
        const float s = scale1[m];
        const float zp = (float)zero1[m];
        for (int k = 0; k < K; k++)
            rrow[k] = row[k] - s * ((float)qrow[k] - zp);
    }
    iro_int8_quantize_rows(residual, M, K, q2, scale2, zero2);
}

void iro_int8_linear(const uint8_t *q, const float *scale,
                     const int32_t *zero, const IroInt8Weight *w,
                     const float *bias, float *y, int M, int32_t *work) {
    iro_int8_linear_ex(q, scale, zero, w, bias, y, M, work, 0);
}

void iro_int8_linear_ex(const uint8_t *q, const float *scale,
                        const int32_t *zero, const IroInt8Weight *w,
                        const float *bias, float *y, int M, int32_t *work,
                        int accumulate) {
    const int N = w->N, K = w->K;
#ifdef IRO_USE_ONEMKL
    int profile = iro_gemm_profile_enabled();
    double t0 = profile ? gemm_profile_now() : 0.0;
    const MKL_INT32 co = 0;
    cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset,
                       M, N, K, 1.0f, q, K, 0, w->data, K, 0, 0.0f,
                       work, N, &co);
    if (profile)
        gemm_profile_record("linear_int8", M, N, K, K, K, N, 0.0f,
                            gemm_profile_now() - t0);
#else
    int8_gemm_reference(q, w->data, work, M, N, K);
#endif
    for (int m = 0; m < M; m++) {
        const int32_t *crow = work + (size_t)m * N;
        float *yrow = y + (size_t)m * N;
        const float sx = scale[m];
        const int32_t zp = zero[m];
        if (accumulate && bias) {
            for (int n = 0; n < N; n++)
                yrow[n] += (float)(crow[n] - zp * w->colsum[n]) *
                               (sx * w->scale[n]) + bias[n];
        } else if (accumulate) {
            for (int n = 0; n < N; n++)
                yrow[n] += (float)(crow[n] - zp * w->colsum[n]) *
                           (sx * w->scale[n]);
        } else if (bias) {
            for (int n = 0; n < N; n++)
                yrow[n] = (float)(crow[n] - zp * w->colsum[n]) *
                              (sx * w->scale[n]) + bias[n];
        } else {
            for (int n = 0; n < N; n++)
                yrow[n] = (float)(crow[n] - zp * w->colsum[n]) *
                          (sx * w->scale[n]);
        }
    }
}

struct IroPackedCache {
    size_t budget, bytes;
    int count;
    struct {
        const float *weight;
        float *packed;
        int m, n, k, threads;
    } entries[64];
};

IroPackedCache *iro_packed_cache_create(size_t budget) {
#ifdef IRO_USE_ONEMKL
    if (!budget) return NULL;
    IroPackedCache *cache = calloc(1, sizeof(*cache));
    if (cache) cache->budget = budget;
    return cache;
#else
    (void)budget;
    return NULL;
#endif
}

void iro_packed_cache_free(IroPackedCache *cache) {
    if (!cache) return;
#ifdef IRO_USE_ONEMKL
    for (int i = 0; i < cache->count; i++) mkl_free(cache->entries[i].packed);
#endif
    free(cache);
}

size_t iro_packed_cache_bytes(const IroPackedCache *cache) {
    return cache ? cache->bytes : 0;
}

void iro_linear_cached(IroPackedCache *cache, const float *x, const float *w,
                        float *y, int M, int K, int N) {
#ifdef IRO_USE_ONEMKL
    /* Initial screen only supports the W1/W3 family and small/moderate M.
       Exact shape/thread keys avoid assuming packed-layout compatibility.
       Full cache or failed allocation falls back without eviction/repacking. */
    if (cache && M > 0 && M <= 345 && K == 1280 && N == 3680) {
        int threads = mkl_get_max_threads();
        int i;
        for (i = 0; i < cache->count; i++) {
            if (cache->entries[i].weight == w && cache->entries[i].m == M &&
                cache->entries[i].n == N && cache->entries[i].k == K &&
                cache->entries[i].threads == threads) break;
        }
        if (i == cache->count && cache->count < 64) {
            size_t bytes = cblas_sgemm_pack_get_size(CblasBMatrix, M, N, K);
            if (bytes && bytes <= cache->budget - cache->bytes) {
                float *packed = mkl_malloc(bytes, 64);
                if (packed) {
                    cblas_sgemm_pack(CblasRowMajor, CblasBMatrix, CblasTrans,
                                     M, N, K, 1.0f, w, K, packed);
                    cache->entries[i].weight = w;
                    cache->entries[i].packed = packed;
                    cache->entries[i].m = M;
                    cache->entries[i].n = N;
                    cache->entries[i].k = K;
                    cache->entries[i].threads = threads;
                    cache->bytes += bytes;
                    cache->count++;
                }
            }
        }
        if (i < cache->count) {
            int profile = iro_gemm_profile_enabled();
            double begin = profile ? gemm_profile_now() : 0.0;
            cblas_sgemm_compute(CblasRowMajor, CblasNoTrans, CblasPacked,
                                M, N, K, x, K, cache->entries[i].packed, K,
                                0.0f, y, N);
            if (profile)
                gemm_profile_record("linear_packed", M, N, K, K, K, N,
                                    0.0f, gemm_profile_now() - begin);
            return;
        }
    }
#else
    (void)cache;
#endif
    iro_linear(x, w, NULL, y, M, K, N);
}

void iro_linear_strided_input(const float *x, int x_stride,
                              const float *w, float *y,
                              int M, int K, int N) {
#ifdef IRO_USE_CBLAS
    int profile = iro_gemm_profile_enabled();
    double t0 = profile ? gemm_profile_now() : 0.0;
    iro_cblas_sgemm(IRO_CBLAS_ROW_MAJOR, IRO_CBLAS_NO_TRANS, IRO_CBLAS_TRANS,
                    M, N, K, 1.0f, x, x_stride, w, K, 0.0f, y, N);
    if (profile)
        gemm_profile_record("linear_strided_input", M, N, K,
                            x_stride, K, N, 0.0f, gemm_profile_now() - t0);
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
    int profile = iro_gemm_profile_enabled();
    double t0 = profile ? gemm_profile_now() : 0.0;
    iro_cblas_sgemm(IRO_CBLAS_ROW_MAJOR, IRO_CBLAS_NO_TRANS, IRO_CBLAS_TRANS,
                    M, N, K, 1.0f, x, K, w, K, 1.0f, y, N);
    if (profile)
        gemm_profile_record("linear_add", M, N, K, K, K, N, 1.0f,
                            gemm_profile_now() - t0);
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
    int profile = iro_gemm_profile_enabled();
    double t0 = profile ? gemm_profile_now() : 0.0;
    iro_cblas_sgemm(IRO_CBLAS_ROW_MAJOR, IRO_CBLAS_NO_TRANS, IRO_CBLAS_TRANS,
                    M, N, K, 1.0f, x, K, w, w_stride, 1.0f, y, N);
    if (profile)
        gemm_profile_record("linear_strided_weight_add", M, N, K,
                            K, w_stride, N, 1.0f, gemm_profile_now() - t0);
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
    int profile = iro_gemm_profile_enabled();
    double t0 = profile ? gemm_profile_now() : 0.0;
    iro_cblas_sgemm(IRO_CBLAS_ROW_MAJOR, IRO_CBLAS_NO_TRANS, IRO_CBLAS_NO_TRANS,
                    M, N, K, 1.0f, a, K, b, N, 0.0f, c, N);
    if (profile)
        gemm_profile_record("matmul", M, N, K, K, N, N, 0.0f,
                            gemm_profile_now() - t0);
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
    int profile = iro_gemm_profile_enabled();
    double t0 = profile ? gemm_profile_now() : 0.0;
    iro_cblas_sgemm(IRO_CBLAS_ROW_MAJOR, IRO_CBLAS_NO_TRANS, IRO_CBLAS_NO_TRANS,
                    M, N, K, 1.0f, a, K, b, N, 0.0f, c, c_stride);
    if (profile)
        gemm_profile_record("matmul_strided_output", M, N, K,
                            K, N, c_stride, 0.0f, gemm_profile_now() - t0);
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
#ifdef IRO_USE_ONEMKL
    return "onemkl-sgemm";
#elif defined(IRO_USE_TORCH_MKL)
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
