/* dit.h — RF-DiT front-end primitives. */
#ifndef IRO_DIT_H
#define IRO_DIT_H

#include "irodori.h"
#include "ops.h"

#define IRO_DIT_LAYERS 12
#define IRO_DIT_MODEL_DIM 1280
#define IRO_DIT_HEADS 20
#define IRO_DIT_HEAD_DIM 64

typedef struct {
    const float *shift_down;
    const float *shift_up;
    const float *shift_bias;
    const float *scale_down;
    const float *scale_up;
    const float *scale_bias;
    const float *gate_down;
    const float *gate_up;
    const float *gate_bias;
} IroAdaLN;

typedef struct {
    const float *wq;
    const float *wk;
    const float *wv;
    const float *wo;
    const float *gate;
    const float *q_norm;
    const float *k_norm;
    const float *wk_text;
    const float *wv_text;
    const float *wk_speaker;
    const float *wv_speaker;
    const float *wk_caption;
    const float *wv_caption;
} IroJointAttention;

typedef struct {
    const float *w1;
    const float *w2;
    const float *w3;
} IroSwiGLU;

/* Opt-in W8A8 copy of the dense per-layer projections.  Owned by the engine,
   built once from the FP32 checkpoint, and borrowed by IroDiT.int8.  Small
   projections (AdaLN, conditioner, in/out, context K/V) and attention
   QK^T/PV stay FP32. */
typedef struct {
    IroInt8Weight wq, wk, wv, gate, wo, w1, w2, w3;
} IroDiTInt8Layer;

typedef struct {
    IroDiTInt8Layer layer[IRO_DIT_LAYERS];
    size_t bytes;
} IroDiTInt8;

typedef struct {
    const float *cond_w0;
    const float *cond_w1;
    const float *cond_w2;
    const float *in_w;
    const float *in_b;
    const float *out_norm;
    const float *out_w;
    const float *out_b;
    IroAdaLN attention_adaln[IRO_DIT_LAYERS];
    IroAdaLN mlp_adaln[IRO_DIT_LAYERS];
    IroJointAttention attention[IRO_DIT_LAYERS];
    IroSwiGLU mlp[IRO_DIT_LAYERS];
    float norm_eps;
    IroPackedCache *packed_cache; /* Borrowed from owning engine; optional. */
    const IroDiTInt8 *int8; /* Borrowed from owning engine; NULL keeps FP32. */
} IroDiT;

typedef enum {
    IRO_DIT_PROFILE_ALLOCATION = 0,
    IRO_DIT_PROFILE_CONDITIONER,
    IRO_DIT_PROFILE_INPUT_PROJECTION,
    IRO_DIT_PROFILE_CONTEXT_PREP,
    IRO_DIT_PROFILE_KV_CACHE_PREP,
    IRO_DIT_PROFILE_ADALN_PARAMS,
    IRO_DIT_PROFILE_ADALN_APPLY,
    IRO_DIT_PROFILE_QKVG_PROJECTION,
    IRO_DIT_PROFILE_RMSNORM_ROPE,
    IRO_DIT_PROFILE_KV_PACK,
    IRO_DIT_PROFILE_QK_GEMM,
    IRO_DIT_PROFILE_SOFTMAX,
    IRO_DIT_PROFILE_PV_GEMM,
    IRO_DIT_PROFILE_ATTENTION_GATE,
    IRO_DIT_PROFILE_ATTENTION_OUTPUT,
    IRO_DIT_PROFILE_W1_W3,
    IRO_DIT_PROFILE_SILU_GATE,
    IRO_DIT_PROFILE_W2,
    IRO_DIT_PROFILE_RESIDUAL,
    IRO_DIT_PROFILE_FINAL_OUTPUT,
    IRO_DIT_PROFILE_SAMPLER_UPDATE,
    IRO_DIT_PROFILE_BUCKET_COUNT
} IroDiTProfileBucket;

/* O1 profiler. IRO_DIT_PROFILE=1 enables monotonic timing and one JSON record
   per generation. layer=-1 denotes sampler/global work; batch is the active
   CFG batch (1..4). */
int iro_dit_profile_enabled(void);
void iro_dit_profile_reset(void);
void iro_dit_profile_add(int layer, int batch, IroDiTProfileBucket bucket,
                         double seconds);
void iro_dit_profile_emit_json(const char *status, int steps,
                               int sequence_length, int context_tokens);

int iro_dit_init(IroDiT *d, const IroSafetensors *st);

/* Quantize every dense projection of d into dst (dst must be zeroed). The
   FP32 tensors stay untouched; d->int8 is not modified by this call. */
int iro_dit_int8_quantize(IroDiTInt8 *dst, const IroDiT *d);
void iro_dit_int8_free(IroDiTInt8 *q);

/* t[B] -> timestep embedding [B,512] -> cond [B,3840]. */
int iro_dit_condition(const IroDiT *d, const float *t, int B, float *cond);

/* x[B,S,32] -> projected [B,S,1280]. */
int iro_dit_input(const IroDiT *d, const float *x, int B, int S, float *projected);

/* x[B,S,1280], cond[B,3840] -> h[B,S,1280], gate[B,1280]. */
int iro_dit_adaln(const IroAdaLN *a, float eps, const float *x,
                  const float *cond, int B, int S, float *h, float *gate);

/* Project static text context once for reuse by every Euler step. */
int iro_dit_text_kv(const IroJointAttention *a, float eps,
                    const float *text, int B, int T, float *k, float *v);

/* Project speaker context once for reuse by every Euler step. */
int iro_dit_speaker_kv(const IroJointAttention *a, float eps,
                       const float *speaker, int B, int T, float *k, float *v);
int iro_dit_caption_kv(const IroJointAttention *a, float eps,
                       const float *caption, int B, int T, float *k, float *v);

/* Joint attention over latent self tokens and a precomputed context KV cache. */
int iro_dit_attention(const IroJointAttention *a, float eps, const float *x,
                      const float *context_k, const float *context_v,
                      const uint8_t *context_mask, int B, int S, int C,
                      float *out);

/* Same attention, but context_k/context_v contain one shared [C,1280]
   context reused by every batch row.  Independent CFG rows differ only by
   context_mask, so this avoids replicating identical static K/V tensors. */
int iro_dit_attention_shared_context(const IroJointAttention *a, float eps,
                                     const float *x,
                                     const float *context_k,
                                     const float *context_v,
                                     const uint8_t *context_mask,
                                     int B, int S, int C, float *out);

/* SwiGLU: w2(silu(w1(x)) * w3(x)), hidden dimension 3680. */
int iro_dit_swiglu(const IroSwiGLU *mlp, const float *x,
                   int B, int S, float *out);

/* One complete residual DiffusionBlock, updating x in place. */
int iro_dit_block(const IroDiT *d, int layer, float *x, const float *cond,
                  const float *context_k, const float *context_v,
                  const uint8_t *context_mask, int B, int S, int C);

int iro_dit_block_shared_context(const IroDiT *d, int layer,
                                 float *x, const float *cond,
                                 const float *context_k,
                                 const float *context_v,
                                 const uint8_t *context_mask,
                                 int B, int S, int C);

/* Final RMSNorm + projection to patched latent velocity [B,S,32]. */
int iro_dit_output(const IroDiT *d, const float *x,
                   int B, int S, float *velocity);

#endif
