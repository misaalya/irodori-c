/* generate.h — text and optional reference-speaker inference pipeline. */
#ifndef IRO_GENERATE_H
#define IRO_GENERATE_H

#include <stdint.h>

typedef struct {
    const char *model_path;
    const char *tokenizer_path;
    const char *decoder_path;
    const char *encoder_path;
    const char *reference_path; /* Optional WAV enabling speaker conditioning. */
    const char *text;
    const char *caption; /* Optional style/emotion caption conditioning. */
    const char *output_path;
    const char *noise_path; /* Optional [frames,32] raw f32 regression input. */
    const char *dump_dir; /* Optional directory for end-to-end parity tensors. */
    uint64_t seed;
    int steps;
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
    uint64_t evicted_model_bytes;
} IroGenerateStats;

/* Reusable engine for warm multi-request inference. The model, tokenizer,
   decoder, and optional reference encoder are loaded once and owned by the
   engine until iro_engine_free(). Initialize the public handle as {0}. */
typedef struct {
    const char *model_path;
    const char *tokenizer_path;
    const char *decoder_path;
    const char *encoder_path; /* Optional; required only for --ref requests. */
} IroEngineConfig;

typedef struct {
    void *impl;
} IroEngine;

int iro_engine_init(IroEngine *engine, const IroEngineConfig *config);
int iro_engine_generate(IroEngine *engine, const IroGenerateConfig *config,
                        IroGenerateStats *stats);
void iro_engine_free(IroEngine *engine);

/* Compatibility wrapper: create a one-shot engine, generate once, free it. */
int iro_generate_text(const IroGenerateConfig *config, IroGenerateStats *stats);

#endif
