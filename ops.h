/* ops.h — operasi tensor f32 dengan scalar oracle dan backend CBLAS opsional. */
#ifndef IRO_OPS_H
#define IRO_OPS_H

#include <stddef.h>
#include <stdint.h>

/* Experimental engine-owned oneMKL packing cache. NULL means ordinary GEMM.
   Immutable weight addresses must remain valid until cache free. A cache is
   single-caller, like its engine; independent engines own independent caches. */
typedef struct IroPackedCache IroPackedCache;
IroPackedCache *iro_packed_cache_create(size_t budget);
void iro_packed_cache_free(IroPackedCache *cache);
size_t iro_packed_cache_bytes(const IroPackedCache *cache);
void iro_linear_cached(IroPackedCache *cache, const float *x, const float *w,
                        float *y, int M, int K, int N);

/* Opt-in W8A8 integer GEMM path.  Weights are quantized once per output
   channel (symmetric int8, scale[N]); activations are quantized per row on the
   fly (asymmetric uint8 with a per-row zero point).  Integer accumulation is
   exact, so every backend must return identical int32 sums for identical
   inputs; only the surrounding float scale math may differ by rounding. */
typedef struct {
    int8_t *data;    /* [N,K] row-major */
    float *scale;    /* [N] dequantization scale */
    int32_t *colsum; /* [N] sum over K of data[n][k], for zero-point removal */
    int N, K;
} IroInt8Weight;

int iro_int8_weight_quantize(IroInt8Weight *dst, const float *w, int N, int K);
/* Same, but the source is stored [K,N] row-major (e.g. ConvTranspose
   [in,kernel*out]); dst still holds the transposed [N,K] layout. */
int iro_int8_weight_quantize_transposed(IroInt8Weight *dst, const float *w,
                                        int N, int K);
void iro_int8_weight_free(IroInt8Weight *w);
size_t iro_int8_weight_bytes(const IroInt8Weight *w);

/* Per-row activation quantization: q[M,K] uint8, scale[M], zero[M]. */
void iro_int8_quantize_rows(const float *x, int M, int K,
                            uint8_t *q, float *scale, int32_t *zero);

/* y[M,N] = scale_x[m] * scale_w[n] * (q @ W^T - zero[m] * colsum[n]) + bias.
   work must hold M*N int32 values.  accumulate adds the result (and bias,
   when given) into y instead of overwriting it. */
void iro_int8_linear(const uint8_t *q, const float *scale,
                     const int32_t *zero, const IroInt8Weight *w,
                     const float *bias, float *y, int M, int32_t *work);
void iro_int8_linear_ex(const uint8_t *q, const float *scale,
                        const int32_t *zero, const IroInt8Weight *w,
                        const float *bias, float *y, int M, int32_t *work,
                        int accumulate);

/* Two-term residual quantization: x ~= deq(q1) + deq(q2) where q2 encodes the
   per-row remainder of the first pass with its own scale/zero point.  Running
   both terms through the integer GEMM gives ~16-bit activation precision for
   twice the GEMM cost.  residual is caller scratch of M*K floats. */
void iro_int8_quantize_rows_residual(const float *x, int M, int K,
                                     uint8_t *q1, float *scale1, int32_t *zero1,
                                     float *residual,
                                     uint8_t *q2, float *scale2, int32_t *zero2);

/* Non-zero when the build has a vectorized int8 GEMM (oneMKL VNNI). The
   scalar reference path still works elsewhere but is only meant for tests. */
int iro_int8_fast_available(void);

/* Call once before the first BLAS computation of a process that will run
   int8.  oneMKL's conditional-numerics branches AVX2/AVX512 exclude VNNI and
   halve int8 throughput, so when MKL_CBWR is not set this selects the
   AVX512_E1 branch (AVX-512 + VNNI, still a fixed reproducible branch).  An
   explicit MKL_CBWR is respected; a non-VNNI choice is reported on stderr.
   Returns the branch name in effect, or NULL when unknown/not applicable. */
const char *iro_int8_prepare_backend(void);

/* Run a small integer GEMM whose intermediate products overflow int16 and
   compare against the exact reference.  Returns 0 when the backend sums
   exactly; -1 when it saturates (oneMKL does on non-VNNI CBWR branches such
   as AVX2/AVX512) or fails.  Engines must refuse int8 on failure. */
int iro_int8_selfcheck(void);

/* Linear tanpa bias (PyTorch nn.Linear: y = x @ W^T, W [N,K] row-major) */
void iro_linear(const float *x, const float *w, const float *bias,
                float *y, int M, int K, int N);

/* Bias-free linear where logical input rows may have padding between them.
   Used by multi-head attention to read one head directly from [S, model_dim]
   without first packing it into a temporary contiguous [S, head_dim] buffer. */
void iro_linear_strided_input(const float *x, int x_stride,
                              const float *w, float *y,
                              int M, int K, int N);

/* y = x @ W^T + bias + add.  Useful for residual projections: the CBLAS
   backend seeds y once and accumulates GEMM with beta=1. */
void iro_linear_add(const float *x, const float *w, const float *bias,
                    const float *add, float *y, int M, int K, int N);

/* y += x @ W^T where logical W rows may have padding between them.  This is
   useful for direct convolution taps stored inside [N,kernel,K] weights. */
void iro_linear_strided_weight_add(const float *x, const float *w,
                                   int w_stride, float *y,
                                   int M, int K, int N);

/* Oracle portabel untuk golden test dan benchmark backend. */
void iro_linear_scalar(const float *x, const float *w, const float *bias,
                       float *y, int M, int K, int N);

/* C[M,N] = A[M,K] @ B[K,N], backend-dispatched seperti iro_linear. */
void iro_matmul(const float *a, const float *b, float *c, int M, int K, int N);

/* Matmul with a padded output row stride.  Attention can therefore write one
   head straight into [S, model_dim] instead of materializing/copying mixh. */
void iro_matmul_strided_output(const float *a, const float *b,
                               float *c, int c_stride,
                               int M, int K, int N);

const char *iro_ops_backend_name(void);

/* Explicit thread control avoids backend-specific environment parsing.
   Returns the active count when the backend exposes it, otherwise zero. */
int iro_ops_set_threads(int threads);
int iro_ops_get_threads(void);
int iro_ops_has_cblas(void);

/* Opt-in GEMM histogram for performance diagnosis.  IRO_GEMM_PROFILE=1
   enables timing; disabled builds keep only a cheap cached branch.  The
   sampler resets/emits this once per generation so hot GEMM loops never print. */
int iro_gemm_profile_enabled(void);
void iro_gemm_profile_reset(void);
void iro_gemm_profile_emit_json(const char *status, int steps,
                                int sequence_length, int context_tokens);

/* LayerNorm (PyTorch nn.LayerNorm, tanpa bias bila bias=NULL) */
void iro_layernorm(const float *x, const float *w, const float *b,
                   float *y, int M, int D, float eps);

/* RMSNorm (Irodori DiT) */
void iro_rmsnorm(const float *x, const float *w, float *y, int M, int D, float eps);

/* GELU exact (erf): 0.5x(1+erf(x/sqrt(2))) */
float iro_gelu(float x);

/* softmax in-place per baris [R, C] */
void iro_softmax_row(float *x, int R, int C);

/* swish/SiLU */
float iro_silu(float x);

/* Tensor kernels keep hot activation loops in one translation unit so native
   builds can emit vector exp instructions instead of millions of calls. */
void iro_silu_inplace(float *x, size_t count);
void iro_silu_mul_inplace(float *x, const float *gate, size_t count);
void iro_sigmoid_mul_inplace(float *x, const float *gate, size_t count);

/* sigmoid */
float iro_sigmoid(float x);

#endif
