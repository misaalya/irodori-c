/* generate.h — text and optional reference-speaker inference pipeline. */
#ifndef IRO_GENERATE_H
#define IRO_GENERATE_H

#include <stddef.h>
#include <stdint.h>

typedef struct IroPreparedReference IroPreparedReference;

typedef struct {
    const char *model_path;
    const char *tokenizer_path;
    const char *decoder_path;
    const char *encoder_path;
    const char *reference_path; /* Optional WAV enabling speaker conditioning. */
    const IroPreparedReference *prepared_reference; /* Optional cached speaker. */
    const char *text;
    const char *caption; /* Optional style/emotion caption conditioning. */
    const char *output_path;
    const char *noise_path; /* Optional [frames,32] raw f32 regression input. */
    const char *dump_dir; /* Optional directory for end-to-end parity tensors. */
    uint64_t seed;
    int steps;
    /* Optional diagnostic hook, included in endpoint time. Synchronous;
       must not re-enter or free the engine/request. */
    void (*stage_observer)(void *user, const char *stage);
    void *stage_observer_user;
} IroGenerateConfig;

typedef struct {
    int tokens;
    int latent_frames;
    int output_samples;
    double encode_seconds;
    double sample_seconds;
    double decode_seconds;
    int speaker_tokens;
    int caption_tokens;
    int prepared_reference_used;
    size_t prepared_reference_bytes;
    uint64_t evicted_model_bytes;
    size_t packed_cache_bytes;
    int dit_precision;        /* Effective precision used for this request. */
    size_t dit_int8_bytes;    /* Resident quantized DiT payload, 0 for FP32. */
    int codec_precision;
    size_t codec_int8_bytes;
} IroGenerateStats;

/* Reusable engine for warm multi-request inference. The model, tokenizer,
   decoder, and optional reference encoder are loaded once and owned by the
   engine until iro_engine_free(). Initialize the public handle as {0}. */
typedef struct {
    const char *model_path;
    const char *tokenizer_path;
    const char *decoder_path;
    const char *encoder_path; /* Optional; required only for --ref requests. */
    /* Opt-in retention experiment: keep front-end pages eligible for reuse.
       Zero preserves existing eviction. The OS may still reclaim pages. */
    int retain_frontend_weights;
    size_t packed_cache_budget; /* Experimental oneMKL only; zero disables. */
    /* IRO_DIT_PRECISION_FP32 keeps the reference math.  IRO_DIT_PRECISION_INT8
       quantizes the dense DiT projections once at init (W8A8, dynamic per-row
       activations) and runs them through the integer GEMM backend.  Output is
       not bit-comparable with the FP32 golden fixtures; use the audio-level
       quality gate instead. */
    int dit_precision;
    /* Same choice for the DACVAE decoder's dense convolutions (Conv7/Conv1/
       ConvTranspose).  Unlike the DiT, the codec is deterministic given the
       latent, so int8-vs-FP32 error is directly measurable as waveform SNR. */
    int codec_precision;
} IroEngineConfig;

enum {
    IRO_DIT_PRECISION_FP32 = 0,
    IRO_DIT_PRECISION_INT8 = 1,
};

typedef struct {
    void *impl;
} IroEngine;

/* Immutable reference state prepared once for repeated requests on the same
   engine. It is mutually exclusive with IroGenerateConfig.reference_path. */
struct IroPreparedReference {
    void *impl;
};

typedef struct {
    int input_samples;
    int reference_frames;
    int speaker_tokens;
    size_t bytes;
    double prepare_seconds;
} IroPreparedReferenceStats;

int iro_engine_init(IroEngine *engine, const IroEngineConfig *config);
int iro_engine_prepare_reference(IroEngine *engine, const char *reference_path,
                                 IroPreparedReference *reference,
                                 IroPreparedReferenceStats *stats);
int iro_engine_generate(IroEngine *engine, const IroGenerateConfig *config,
                        IroGenerateStats *stats);
void iro_prepared_reference_free(IroPreparedReference *reference);
void iro_engine_free(IroEngine *engine);

/* Compatibility wrapper: create a one-shot engine, generate once, free it. */
int iro_generate_text(const IroGenerateConfig *config, IroGenerateStats *stats);

#endif
