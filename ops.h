/* ops.h — operasi tensor f32 dengan scalar oracle dan backend CBLAS opsional. */
#ifndef IRO_OPS_H
#define IRO_OPS_H

#include <stddef.h>
#include <stdint.h>

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
