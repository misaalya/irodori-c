#define _POSIX_C_SOURCE 200809L
/* generate.c — pure-C text/reference -> latent -> 48 kHz WAV orchestration. */
#include <limits.h>
#include <math.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "audio.h"
#include "backbone.h"
#include "condition.h"
#include "dacvae.h"
#include "duration.h"
#include "generate.h"
#include "normalize.h"
#include "sampler.h"
#include "speaker.h"
#include "tokenizer.h"

enum {
    TEXT_LENGTH = 256,
    CAPTION_LENGTH = 512,
    BACKBONE_DIM = 768,
    TEXT_DIM = 512,
    LATENT_DIM = 32,
    SAMPLE_RATE = 48000,
    HOP_LENGTH = 1920,
};

static double elapsed(struct timespec begin, struct timespec end) {
    return (double)(end.tv_sec - begin.tv_sec) +
           1e-9 * (double)(end.tv_nsec - begin.tv_nsec);
}

static uint32_t pcg32(uint64_t *state) {
    uint64_t old = *state;
    *state = old * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rotation = (uint32_t)(old >> 59u);
    return (xorshifted >> rotation) |
           (xorshifted << ((-(int)rotation) & 31));
}

static void gaussian_noise(float *output, size_t count, uint64_t seed) {
    uint64_t state = seed + 0x853c49e6748fea9bULL;
    (void)pcg32(&state);
    for (size_t i = 0; i < count; i += 2) {
        double u1 = ((double)pcg32(&state) + 1.0) / 4294967297.0;
        double u2 = ((double)pcg32(&state) + 0.5) / 4294967296.0;
        double radius = sqrt(-2.0 * log(u1));
        double angle = 6.2831853071795864769 * u2;
        output[i] = (float)(radius * cos(angle));
        if (i + 1 < count) output[i + 1] = (float)(radius * sin(angle));
    }
}

static int load_noise(const char *path, float *output, size_t count) {
    if (!path || !output || count > SIZE_MAX / (2u * sizeof(float))) return -1;
    FILE *file = fopen(path, "rb");
    if (!file) return -1;
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return -1;
    }
    long bytes = ftell(file);
    if (bytes < 0 ||
        ((size_t)bytes != count * sizeof(float) &&
         (size_t)bytes != 2u * count * sizeof(float)) ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return -1;
    }
    size_t read_count = fread(output, sizeof(float), count, file);
    int failed = read_count != count || ferror(file);
    fclose(file);
    return failed ? -1 : 0;
}

static int prepare_dump_dir(const char *directory) {
    if (!directory) return 0;
    if (mkdir(directory, 0777) == 0 || errno == EEXIST) {
        struct stat info;
        return stat(directory, &info) == 0 && S_ISDIR(info.st_mode) ? 0 : -1;
    }
    return -1;
}

static int dump_f32(const char *directory, const char *name,
                    const float *values, size_t count) {
    if (!directory) return 0;
    char path[PATH_MAX];
    int length = snprintf(path, sizeof(path), "%s/%s.f32", directory, name);
    if (length < 0 || (size_t)length >= sizeof(path)) return -1;
    FILE *file = fopen(path, "wb");
    if (!file) return -1;
    size_t written = fwrite(values, sizeof(*values), count, file);
    int failed = written != count || fflush(file) != 0 || ferror(file);
    if (fclose(file) != 0) failed = 1;
    return failed ? -1 : 0;
}

typedef struct {
    const char *directory;
    int sequence_length;
} IroGenerateTrace;

static int dump_euler_trace(void *user, int step, float timestep, int batch,
                            const float *x_t, const float *velocity) {
    (void)timestep;
    (void)x_t;
    IroGenerateTrace *trace = user;
    if (!trace || !trace->directory || batch <= 0 || trace->sequence_length <= 0)
        return -1;
    char name[32];
    snprintf(name, sizeof(name), "velocity_step%03d", step);
    return dump_f32(trace->directory, name, velocity,
                    (size_t)batch * trace->sequence_length * LATENT_DIM);
}

typedef struct {
    IroTokenizer tokenizer;
    IroSafetensors model_st;
    IroSafetensors decoder_st;
    IroSafetensors encoder_st;
    IroBackbone backbone;
    IroTextProjector projector;
    IroTextProjector caption_projector;
    IroDurationPredictor duration;
    IroDiT dit;
    IroSpeakerEncoder speaker_encoder;
    IroDACVAEDecoder decoder;
    IroDACVAEEncoder reference_encoder;
    IroEulerWorkspace sampler_workspace;
    int backbone_ready;
    int has_encoder;
    int retain_frontend_weights;
    int dit_precision;
    IroDiTInt8 *dit_int8;
    double dit_quantize_seconds;
    int codec_precision;
    IroDACVAEInt8 *codec_int8;
} IroEngineImpl;

typedef struct {
    IroEngineImpl *owner;
    float *reference_latent;
    float *speaker_state;
    int input_samples;
    int reference_frames;
    int speaker_tokens;
    size_t bytes;
} IroPreparedReferenceImpl;

typedef struct {
    char *normalized;
    int32_t *ids;
    uint8_t *mask;
    float *backbone_state, *projected, *text_state;
    int32_t *caption_ids;
    uint8_t *caption_mask;
    float *caption_backbone, *caption_projected;
    float *caption_state;
    float caption_vec[TEXT_DIM];
    float *noise, *latent, *waveform;
    float *reference_latent, *speaker_state;
    int token_count, caption_tokens, latent_frames, output_samples;
    int reference_frames, speaker_tokens;
    int reference_latent_borrowed, speaker_state_borrowed;
    size_t prepared_reference_bytes;
    uint64_t evicted_model_bytes;
    IroAudio reference_audio;
} IroGenerateRequest;

static void request_release_conditions(IroGenerateRequest *request) {
    if (!request) return;
    free(request->normalized); request->normalized = NULL;
    free(request->ids); request->ids = NULL;
    free(request->mask); request->mask = NULL;
    free(request->backbone_state); request->backbone_state = NULL;
    free(request->projected); request->projected = NULL;
    free(request->text_state); request->text_state = NULL;
    if (!request->reference_latent_borrowed) free(request->reference_latent);
    request->reference_latent = NULL;
    request->reference_latent_borrowed = 0;
    if (!request->speaker_state_borrowed) free(request->speaker_state);
    request->speaker_state = NULL;
    request->speaker_state_borrowed = 0;
    free(request->caption_ids); request->caption_ids = NULL;
    free(request->caption_mask); request->caption_mask = NULL;
    free(request->caption_backbone); request->caption_backbone = NULL;
    free(request->caption_projected); request->caption_projected = NULL;
    free(request->caption_state); request->caption_state = NULL;
    iro_audio_free(&request->reference_audio);
}

static void request_free(IroGenerateRequest *request) {
    if (!request) return;
    request_release_conditions(request);
    free(request->noise);
    free(request->latent);
    free(request->waveform);
    memset(request, 0, sizeof(*request));
}

static int prepare_conditions(IroEngineImpl *engine,
                              const IroGenerateConfig *config,
                              IroGenerateRequest *request,
                              IroGenerateStats *stats) {
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);

    request->normalized = iro_normalize(config->text);
    if (!request->normalized) return -1;
    int32_t token_buffer[IRO_TOK_MAX_IDS];
    request->token_count = iro_tok_encode(&engine->tokenizer, request->normalized,
                                          1, token_buffer, IRO_TOK_MAX_IDS);
    if (request->token_count <= 0 || request->token_count > TEXT_LENGTH) {
        fprintf(stderr, "generate: teks menghasilkan %d token (maksimum %d)\n",
                request->token_count, TEXT_LENGTH);
        return -1;
    }
    /* Padding queries never contribute to valid-token states: attention masks
       them as keys, while projector and duration operate independently per
       token.  Keep only the valid prefix instead of paying for 256 rows. */
    size_t tokens = (size_t)request->token_count;
    request->ids = malloc(sizeof(*request->ids) * tokens);
    request->mask = malloc(sizeof(*request->mask) * tokens);
    request->backbone_state = malloc(sizeof(float) * tokens * BACKBONE_DIM);
    request->projected = malloc(sizeof(float) * tokens * TEXT_DIM);
    request->text_state = malloc(sizeof(float) * tokens * TEXT_DIM);
    if (!request->ids || !request->mask || !request->backbone_state ||
        !request->projected || !request->text_state)
        return -1;
    memcpy(request->ids, token_buffer, sizeof(*request->ids) * tokens);
    memset(request->mask, 1, tokens);

    if (iro_backbone_forward(&engine->backbone, request->ids, request->mask,
                             request->token_count, request->backbone_state) != 0 ||
        iro_text_projector_forward(&engine->projector, request->backbone_state,
                                   request->mask, request->token_count,
                                   request->projected, request->text_state) != 0)
        return -1;
    memset(request->caption_vec, 0, sizeof(request->caption_vec));
    if (config->caption && config->caption[0]) {
        int32_t caption_buffer[IRO_TOK_MAX_IDS];
        request->caption_tokens = iro_tok_encode(&engine->tokenizer, config->caption,
                                                 1, caption_buffer, IRO_TOK_MAX_IDS);
        if (request->caption_tokens <= 0 || request->caption_tokens > CAPTION_LENGTH) {
            fprintf(stderr, "generate: caption menghasilkan %d token (maksimum %d)\n",
                    request->caption_tokens, CAPTION_LENGTH);
            return -1;
        }
        size_t caption_count = (size_t)request->caption_tokens;
        request->caption_ids = malloc(sizeof(*request->caption_ids) * caption_count);
        request->caption_mask = malloc(sizeof(*request->caption_mask) * caption_count);
        request->caption_backbone = malloc(sizeof(float) * caption_count * BACKBONE_DIM);
        request->caption_projected = malloc(sizeof(float) * caption_count * TEXT_DIM);
        request->caption_state = malloc(sizeof(float) * caption_count * TEXT_DIM);
        if (!request->caption_ids || !request->caption_mask ||
            !request->caption_backbone || !request->caption_projected ||
            !request->caption_state)
            return -1;
        memcpy(request->caption_ids, caption_buffer,
               sizeof(*request->caption_ids) * caption_count);
        memset(request->caption_mask, 1, caption_count);
        if (iro_backbone_forward(&engine->backbone, request->caption_ids,
                                 request->caption_mask, request->caption_tokens,
                                 request->caption_backbone) != 0 ||
            iro_text_projector_forward(&engine->caption_projector,
                                       request->caption_backbone,
                                       request->caption_mask,
                                       request->caption_tokens,
                                       request->caption_projected,
                                       request->caption_state) != 0)
            return -1;
        for (int s = 0; s < request->caption_tokens; s++)
            for (int d = 0; d < TEXT_DIM; d++)
                request->caption_vec[d] +=
                    request->caption_state[(size_t)s * TEXT_DIM + d];
        for (int d = 0; d < TEXT_DIM; d++)
            request->caption_vec[d] /= (float)request->caption_tokens;
        if (config->dump_dir) {
            float *padded_caption = calloc((size_t)CAPTION_LENGTH * TEXT_DIM,
                                           sizeof(float));
            if (!padded_caption) return -1;
            memcpy(padded_caption, request->caption_state,
                   sizeof(float) * caption_count * TEXT_DIM);
            int dump_failed = dump_f32(config->dump_dir, "caption_state",
                                       padded_caption,
                                       (size_t)CAPTION_LENGTH * TEXT_DIM);
            free(padded_caption);
            if (dump_failed != 0) return -1;
        }
    }
    if (config->prepared_reference) {
        const IroPreparedReferenceImpl *prepared =
            config->prepared_reference->impl;
        if (!prepared || prepared->owner != engine ||
            !prepared->reference_latent || !prepared->speaker_state ||
            prepared->reference_frames <= 0 || prepared->speaker_tokens <= 0) {
            fprintf(stderr,
                    "generate: prepared reference bukan milik engine ini\n");
            return -1;
        }
        request->reference_latent = prepared->reference_latent;
        request->speaker_state = prepared->speaker_state;
        request->reference_frames = prepared->reference_frames;
        request->speaker_tokens = prepared->speaker_tokens;
        request->reference_latent_borrowed = 1;
        request->speaker_state_borrowed = 1;
        request->prepared_reference_bytes = prepared->bytes;
        if (dump_f32(config->dump_dir, "reference_latent",
                     request->reference_latent,
                     (size_t)request->reference_frames * LATENT_DIM) != 0 ||
            dump_f32(config->dump_dir, "speaker_state", request->speaker_state,
                     (size_t)request->speaker_tokens * BACKBONE_DIM) != 0) {
            fprintf(stderr, "generate: gagal menulis tensor reference\n");
            return -1;
        }
    } else if (config->reference_path) {
        if (!engine->has_encoder) {
            fprintf(stderr, "generate: engine tidak memiliki reference encoder\n");
            return -1;
        }
        if (iro_prepare_reference_wav(config->reference_path, -16.0f,
                                      &request->reference_audio, NULL, NULL) != 0 ||
            iro_dacvae_encode_mean(&engine->reference_encoder,
                                   request->reference_audio.samples,
                                   request->reference_audio.sample_count,
                                   &request->reference_latent,
                                   &request->reference_frames,
                                   NULL, NULL) != 0 ||
            iro_speaker_encode(&engine->speaker_encoder,
                               request->reference_latent,
                               request->reference_frames,
                               &request->speaker_state,
                               &request->speaker_tokens, NULL, NULL) != 0)
            return -1;
        if (dump_f32(config->dump_dir, "reference_latent", request->reference_latent,
                     (size_t)request->reference_frames * LATENT_DIM) != 0 ||
            dump_f32(config->dump_dir, "speaker_state", request->speaker_state,
                     (size_t)request->speaker_tokens * BACKBONE_DIM) != 0) {
            fprintf(stderr, "generate: gagal menulis tensor reference\n");
            return -1;
        }
        free(request->reference_latent); request->reference_latent = NULL;
        iro_audio_free(&request->reference_audio);
    }
    float log_frames = 0.0f;
    if (iro_duration_forward(&engine->duration, request->text_state,
                             request->mask, request->token_count,
                             request->speaker_state,
                             request->caption_tokens > 0 ? request->caption_vec : NULL,
                             &log_frames) != 0)
        return -1;
    if (dump_f32(config->dump_dir, "duration_log_frames", &log_frames, 1) != 0) {
        fprintf(stderr, "generate: gagal menulis duration fixture\n");
        return -1;
    }
    request->latent_frames = iro_duration_frame_count(
        log_frames, 1.0f, 0.5f, 30.0f, SAMPLE_RATE, HOP_LENGTH);
    if (request->latent_frames <= 0) return -1;
    /* The DiT never reads encoder/duration tensors again. Evict their clean
       mmap pages before batch-3 CFG so resident weights do not accumulate. */
    if (!engine->retain_frontend_weights) {
        request->evicted_model_bytes += iro_st_drop_prefix(
            &engine->model_st, "pretrained_text_backbone.");
        request->evicted_model_bytes += iro_st_drop_prefix(&engine->model_st, "text_encoder.");
        request->evicted_model_bytes += iro_st_drop_prefix(&engine->model_st, "text_norm.");
        request->evicted_model_bytes += iro_st_drop_prefix(&engine->model_st, "caption_encoder.");
        request->evicted_model_bytes += iro_st_drop_prefix(&engine->model_st, "caption_norm.");
        request->evicted_model_bytes += iro_st_drop_prefix(&engine->model_st, "speaker_encoder.");
        request->evicted_model_bytes += iro_st_drop_prefix(&engine->model_st, "speaker_norm.");
        request->evicted_model_bytes += iro_st_drop_prefix(&engine->model_st, "duration_predictor.");
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    if (stats) {
        stats->tokens = request->token_count;
        stats->latent_frames = request->latent_frames;
        stats->speaker_tokens = request->speaker_tokens;
        stats->caption_tokens = request->caption_tokens;
        stats->prepared_reference_used = config->prepared_reference != NULL;
        stats->prepared_reference_bytes = request->prepared_reference_bytes;
        stats->evicted_model_bytes = request->evicted_model_bytes;
        stats->encode_seconds = elapsed(begin, end);
    }
    return 0;
}

static int sample_latent(IroEngineImpl *engine,
                         const IroGenerateConfig *config,
                         IroGenerateRequest *request,
                         IroGenerateStats *stats) {
    size_t latent_n = (size_t)request->latent_frames * LATENT_DIM;
    request->noise = malloc(sizeof(float) * latent_n);
    request->latent = malloc(sizeof(float) * latent_n);
    if (!request->noise || !request->latent) return -1;
    if (config->noise_path) {
        if (load_noise(config->noise_path, request->noise, latent_n) != 0) {
            fprintf(stderr, "generate: noise fixture tidak cocok: %s\n",
                    config->noise_path);
            return -1;
        }
    } else {
        gaussian_noise(request->noise, latent_n, config->seed);
    }
    if (dump_f32(config->dump_dir, "noise", request->noise, latent_n) != 0) {
        fprintf(stderr, "generate: gagal menulis noise fixture\n");
        return -1;
    }
    IroEulerConfig sampler_config = {
        .steps = config->steps,
        .init_scale = 0.999f,
        .cfg_scale_text = 3.0f,
        .cfg_scale_speaker = request->speaker_tokens > 0 ? 5.0f : 0.0f,
        .cfg_scale_caption = request->caption_tokens > 0 ? 3.0f : 0.0f,
        .cfg_min_t = 0.5f,
        .cfg_max_t = 1.0f,
    };
    IroGenerateTrace euler_trace = {
        .directory = config->dump_dir,
        .sequence_length = request->latent_frames,
    };
    IroEulerTraceFn trace_fn = config->dump_dir ? dump_euler_trace : NULL;
    void *trace_user = config->dump_dir ? &euler_trace : NULL;
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    if (iro_sample_euler_conditions_workspace(
            &engine->dit, request->text_state, request->mask,
            request->token_count, request->speaker_state,
            request->speaker_tokens, request->caption_state,
            request->caption_tokens, request->noise,
            request->latent_frames, &sampler_config, request->latent,
            trace_fn, trace_user, NULL, &engine->sampler_workspace) != 0) {
        return -1;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    if (dump_f32(config->dump_dir, "latent_final", request->latent, latent_n) != 0) {
        fprintf(stderr, "generate: gagal menulis latent fixture\n");
        return -1;
    }
    if (stats) stats->sample_seconds = elapsed(begin, end);
    return 0;
}

static int decode_output(IroEngineImpl *engine,
                         const IroGenerateConfig *config,
                         IroGenerateRequest *request,
                         IroGenerateStats *stats) {
    int decoded_samples = iro_dacvae_output_length(request->latent_frames);
    if (decoded_samples <= 0) return -1;
    request->waveform = malloc(sizeof(float) * (size_t)decoded_samples);
    if (!request->waveform) return -1;
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    if (iro_dacvae_decode(&engine->decoder, request->latent,
                          request->latent_frames, request->waveform,
                          NULL, NULL) != 0)
        return -1;
    clock_gettime(CLOCK_MONOTONIC, &end);
    size_t trimmed = iro_trimmed_sample_count(
        request->latent, request->latent_frames, LATENT_DIM,
        (size_t)decoded_samples,
        HOP_LENGTH, 20, 0.05f, 0.1f);
    if (trimmed > INT_MAX ||
        iro_write_wav_pcm16(config->output_path, request->waveform, trimmed,
                            SAMPLE_RATE) != 0)
        return -1;
    if (dump_f32(config->dump_dir, "waveform", request->waveform, trimmed) != 0) {
        fprintf(stderr, "generate: gagal menulis waveform fixture\n");
        return -1;
    }
    request->output_samples = (int)trimmed;
    if (stats) {
        stats->output_samples = request->output_samples;
        stats->decode_seconds = elapsed(begin, end);
    }
    return 0;
}

int iro_engine_init(IroEngine *engine, const IroEngineConfig *config) {
    if (!engine || !config || !config->model_path || !config->tokenizer_path ||
        !config->decoder_path || engine->impl ||
        (config->retain_frontend_weights != 0 && config->retain_frontend_weights != 1) ||
        (config->dit_precision != IRO_DIT_PRECISION_FP32 &&
         config->dit_precision != IRO_DIT_PRECISION_INT8) ||
        (config->codec_precision != IRO_DIT_PRECISION_FP32 &&
         config->codec_precision != IRO_DIT_PRECISION_INT8))
        return -1;
    IroEngineImpl *impl = calloc(1, sizeof(*impl));
    if (!impl) return -1;
    impl->tokenizer.fd = -1;
    impl->model_st.fd = -1;
    impl->decoder_st.fd = -1;
    impl->encoder_st.fd = -1;
    impl->retain_frontend_weights = config->retain_frontend_weights;

    if (iro_tok_load(config->tokenizer_path, &impl->tokenizer) != 0 ||
        iro_st_load(config->model_path, &impl->model_st) != 0 ||
        iro_st_load(config->decoder_path, &impl->decoder_st) != 0 ||
        iro_backbone_init(&impl->backbone, &impl->model_st, CAPTION_LENGTH) != 0)
        goto fail;
    impl->backbone_ready = 1;
    if (iro_text_projector_init(&impl->projector, &impl->model_st) != 0 ||
        iro_caption_projector_init(&impl->caption_projector, &impl->model_st) != 0 ||
        iro_duration_init(&impl->duration, &impl->model_st) != 0 ||
        iro_dit_init(&impl->dit, &impl->model_st) != 0 ||
        iro_speaker_init(&impl->speaker_encoder, &impl->model_st) != 0 ||
        iro_dacvae_init(&impl->decoder, &impl->decoder_st) != 0)
        goto fail;
    if (config->encoder_path && config->encoder_path[0]) {
        if (iro_st_load(config->encoder_path, &impl->encoder_st) != 0 ||
            iro_dacvae_encoder_init(&impl->reference_encoder,
                                    &impl->encoder_st) != 0)
            goto fail;
        impl->has_encoder = 1;
    }
    if (config->packed_cache_budget) {
        impl->dit.packed_cache = iro_packed_cache_create(config->packed_cache_budget);
        if (!impl->dit.packed_cache) goto fail;
    }
    impl->dit_precision = config->dit_precision;
    if (config->dit_precision == IRO_DIT_PRECISION_INT8) {
        struct timespec q0, q1;
        (void)iro_int8_prepare_backend();
        if (iro_int8_selfcheck() != 0) {
            fprintf(stderr,
                    "generate: GEMM int8 backend ini tidak eksak (saturasi); "
                    "pakai MKL_CBWR=AVX512_E1/AUTO pada CPU dengan VNNI, atau "
                    "dit_precision fp32\n");
            goto fail;
        }
        clock_gettime(CLOCK_MONOTONIC, &q0);
        impl->dit_int8 = calloc(1, sizeof(*impl->dit_int8));
        if (!impl->dit_int8 ||
            iro_dit_int8_quantize(impl->dit_int8, &impl->dit) != 0)
            goto fail;
        impl->dit.int8 = impl->dit_int8;
        /* The quantized copies replace these FP32 tensors for the engine's
           lifetime; release their freshly-touched pages so peak RSS does not
           carry both representations.  Pointers stay valid. */
        if (!impl->retain_frontend_weights) {
            static const char *const replaced[] = {
                "attention.wq.", "attention.wk.", "attention.wv.",
                "attention.gate.", "attention.wo.", "mlp.",
            };
            for (int layer = 0; layer < IRO_DIT_LAYERS; layer++) {
                for (size_t i = 0; i < sizeof(replaced) / sizeof(replaced[0]); i++) {
                    char prefix[64];
                    snprintf(prefix, sizeof(prefix), "blocks.%d.%s", layer,
                             replaced[i]);
                    (void)iro_st_drop_prefix(&impl->model_st, prefix);
                }
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &q1);
        impl->dit_quantize_seconds = elapsed(q0, q1);
    }
    impl->codec_precision = config->codec_precision;
    if (config->codec_precision == IRO_DIT_PRECISION_INT8) {
        (void)iro_int8_prepare_backend();
        if (iro_int8_selfcheck() != 0) {
            fprintf(stderr,
                    "generate: GEMM int8 backend ini tidak eksak (saturasi); "
                    "codec_precision int8 ditolak\n");
            goto fail;
        }
        impl->codec_int8 = calloc(1, sizeof(*impl->codec_int8));
        if (!impl->codec_int8 ||
            iro_dacvae_int8_quantize(impl->codec_int8, &impl->decoder) != 0)
            goto fail;
        impl->decoder.int8 = impl->codec_int8;
        if (!impl->retain_frontend_weights)
            (void)iro_st_drop_prefix(&impl->decoder_st, "decoder.model.");
    }
    engine->impl = impl;
    return 0;

fail:
    iro_dacvae_int8_free(impl->codec_int8);
    free(impl->codec_int8);
    iro_dit_int8_free(impl->dit_int8);
    free(impl->dit_int8);
    iro_packed_cache_free(impl->dit.packed_cache);
    if (impl->backbone_ready) iro_backbone_free(&impl->backbone);
    iro_tok_free(&impl->tokenizer);
    iro_st_free(&impl->model_st);
    iro_st_free(&impl->decoder_st);
    iro_st_free(&impl->encoder_st);
    free(impl);
    return -1;
}

int iro_engine_prepare_reference(IroEngine *engine, const char *reference_path,
                                 IroPreparedReference *reference,
                                 IroPreparedReferenceStats *stats) {
    if (!engine || !engine->impl || !reference_path || !reference_path[0] ||
        !reference || reference->impl)
        return -1;
    if (stats) memset(stats, 0, sizeof(*stats));
    IroEngineImpl *owner = engine->impl;
    if (!owner->has_encoder) {
        fprintf(stderr, "generate: engine tidak memiliki reference encoder\n");
        return -1;
    }

    IroPreparedReferenceImpl *prepared = calloc(1, sizeof(*prepared));
    if (!prepared) return -1;
    IroAudio audio = {0};
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    if (iro_prepare_reference_wav(reference_path, -16.0f, &audio, NULL, NULL) != 0 ||
        audio.sample_count > INT_MAX ||
        iro_dacvae_encode_mean(&owner->reference_encoder, audio.samples,
                               audio.sample_count, &prepared->reference_latent,
                               &prepared->reference_frames, NULL, NULL) != 0 ||
        iro_speaker_encode(&owner->speaker_encoder, prepared->reference_latent,
                           prepared->reference_frames, &prepared->speaker_state,
                           &prepared->speaker_tokens, NULL, NULL) != 0)
        goto fail;
    clock_gettime(CLOCK_MONOTONIC, &end);
    prepared->owner = owner;
    prepared->input_samples = (int)audio.sample_count;
    prepared->bytes = sizeof(float) *
        ((size_t)prepared->reference_frames * LATENT_DIM +
         (size_t)prepared->speaker_tokens * BACKBONE_DIM);
    iro_audio_free(&audio);
    reference->impl = prepared;
    if (stats) {
        stats->input_samples = prepared->input_samples;
        stats->reference_frames = prepared->reference_frames;
        stats->speaker_tokens = prepared->speaker_tokens;
        stats->bytes = prepared->bytes;
        stats->prepare_seconds = elapsed(begin, end);
    }
    return 0;

fail:
    iro_audio_free(&audio);
    free(prepared->reference_latent);
    free(prepared->speaker_state);
    free(prepared);
    return -1;
}

void iro_prepared_reference_free(IroPreparedReference *reference) {
    if (!reference || !reference->impl) return;
    IroPreparedReferenceImpl *prepared = reference->impl;
    free(prepared->reference_latent);
    free(prepared->speaker_state);
    free(prepared);
    reference->impl = NULL;
}

void iro_engine_free(IroEngine *engine) {
    if (!engine || !engine->impl) return;
    IroEngineImpl *impl = engine->impl;
    iro_dacvae_int8_free(impl->codec_int8);
    free(impl->codec_int8);
    iro_dit_int8_free(impl->dit_int8);
    free(impl->dit_int8);
    iro_packed_cache_free(impl->dit.packed_cache);
    iro_euler_workspace_free(&impl->sampler_workspace);
    if (impl->backbone_ready) iro_backbone_free(&impl->backbone);
    iro_tok_free(&impl->tokenizer);
    iro_st_free(&impl->model_st);
    iro_st_free(&impl->decoder_st);
    iro_st_free(&impl->encoder_st);
    free(impl);
    engine->impl = NULL;
}

int iro_engine_generate(IroEngine *engine, const IroGenerateConfig *config,
                        IroGenerateStats *stats) {
    if (!engine || !engine->impl || !config || !config->text ||
        !config->output_path || config->steps <= 0 ||
        (config->reference_path && config->prepared_reference))
        return -1;
    if (config->prepared_reference) {
        const IroPreparedReferenceImpl *prepared =
            config->prepared_reference->impl;
        if (!prepared || prepared->owner != (IroEngineImpl *)engine->impl) {
            fprintf(stderr,
                    "generate: prepared reference bukan milik engine ini\n");
            return -1;
        }
    }
    if (config->reference_path &&
        !((IroEngineImpl *)engine->impl)->has_encoder) {
        fprintf(stderr, "generate: reference meminta encoder yang tidak diload\n");
        return -1;
    }
    if (prepare_dump_dir(config->dump_dir) != 0) {
        fprintf(stderr, "generate: dump directory tidak valid: %s\n",
                config->dump_dir);
        return -1;
    }
    if (stats) memset(stats, 0, sizeof(*stats));

    IroEngineImpl *impl = engine->impl;
    IroGenerateRequest request = {0};
    int result = -1;
    if (prepare_conditions(impl, config, &request, stats) != 0) goto cleanup;
    if (config->stage_observer)
        config->stage_observer(config->stage_observer_user, "conditions");
    if (sample_latent(impl, config, &request, stats) != 0) goto cleanup;
    if (config->stage_observer)
        config->stage_observer(config->stage_observer_user, "sampling");
    request_release_conditions(&request);
    if (decode_output(impl, config, &request, stats) != 0) goto cleanup;
    if (config->stage_observer)
        config->stage_observer(config->stage_observer_user, "decode");
    result = 0;

cleanup:
    if (stats) {
        stats->packed_cache_bytes = iro_packed_cache_bytes(impl->dit.packed_cache);
        stats->dit_precision = impl->dit_precision;
        stats->dit_int8_bytes = impl->dit_int8 ? impl->dit_int8->bytes : 0;
        stats->codec_precision = impl->codec_precision;
        stats->codec_int8_bytes = impl->codec_int8 ? impl->codec_int8->bytes : 0;
    }
    request_free(&request);
    if (config->stage_observer)
        config->stage_observer(config->stage_observer_user, "cleanup");
    return result;
}

int iro_generate_text(const IroGenerateConfig *config, IroGenerateStats *stats) {
    if (!config || !config->model_path || !config->tokenizer_path ||
        !config->decoder_path || !config->text || !config->output_path ||
        config->steps <= 0 || config->prepared_reference ||
        (config->reference_path && !config->encoder_path))
        return -1;
    IroEngineConfig engine_config = {
        .model_path = config->model_path,
        .tokenizer_path = config->tokenizer_path,
        .decoder_path = config->decoder_path,
        .encoder_path = config->reference_path ? config->encoder_path : NULL,
    };
    IroEngine engine = {0};
    if (iro_engine_init(&engine, &engine_config) != 0) return -1;
    int result = iro_engine_generate(&engine, config, stats);
    iro_engine_free(&engine);
    return result;
}
