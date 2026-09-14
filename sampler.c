/* sampler.c — cached independent-CFG rectified-flow Euler loop. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sampler.h"
#include "ops.h"

enum {
    TEXT_DIM = 512,
    MODEL_DIM = IRO_DIT_MODEL_DIM,
    COND_DIM = 3 * IRO_DIT_MODEL_DIM,
    LATENT_DIM = 32,
    MAX_BATCH = 4,
};

typedef struct {
    float *compact_text;
    float *cache_k;
    float *cache_v;
    uint8_t *mask_cfg;
    float *state;
    float *x_batch;
    float *projected;
    float *cond;
    float *velocity;
    size_t compact_text_capacity;
    size_t cache_k_capacity;
    size_t cache_v_capacity;
    size_t mask_capacity;
    size_t state_capacity;
    size_t x_batch_capacity;
    size_t projected_capacity;
    size_t cond_capacity;
    size_t velocity_capacity;
} IroEulerWorkspaceImpl;

static int reserve_f32(float **buffer, size_t *capacity, size_t count) {
    if (!buffer || !capacity || count > SIZE_MAX / sizeof(float)) return -1;
    if (count <= *capacity) return 0;
    float *grown = realloc(*buffer, count * sizeof(float));
    if (!grown) return -1;
    *buffer = grown;
    *capacity = count;
    return 0;
}

static int reserve_u8(uint8_t **buffer, size_t *capacity, size_t count) {
    if (!buffer || !capacity) return -1;
    if (count <= *capacity) return 0;
    uint8_t *grown = realloc(*buffer, count);
    if (!grown) return -1;
    *buffer = grown;
    *capacity = count;
    return 0;
}

static int checked_mul(size_t left, size_t right, size_t *out) {
    if (!out || (right != 0 && left > SIZE_MAX / right)) return -1;
    *out = left * right;
    return 0;
}

static double sampler_profile_now(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void workspace_impl_free(IroEulerWorkspaceImpl *workspace) {
    if (!workspace) return;
    free(workspace->compact_text);
    free(workspace->cache_k);
    free(workspace->cache_v);
    free(workspace->mask_cfg);
    free(workspace->state);
    free(workspace->x_batch);
    free(workspace->projected);
    free(workspace->cond);
    free(workspace->velocity);
    free(workspace);
}

void iro_euler_workspace_free(IroEulerWorkspace *workspace) {
    if (!workspace) return;
    workspace_impl_free(workspace->impl);
    workspace->impl = NULL;
}

static IroEulerWorkspaceImpl *workspace_get(IroEulerWorkspace *workspace) {
    if (!workspace) return NULL;
    if (!workspace->impl) {
        workspace->impl = calloc(1, sizeof(IroEulerWorkspaceImpl));
        if (!workspace->impl) return NULL;
    }
    return workspace->impl;
}

static int sample_euler(const IroDiT *dit,
                        const float *text, const uint8_t *text_mask,
                        int text_tokens,
                        const float *speaker, int speaker_tokens,
                        const float *caption, int caption_tokens,
                        const float *noise, int sequence_length,
                        const IroEulerConfig *config, float *output,
                        IroEulerTraceFn trace, void *trace_user,
                        IroEulerStats *stats,
                        IroEulerWorkspace *workspace) {
    if (!dit || !text || !text_mask || text_tokens <= 0 || !noise ||
        sequence_length <= 0 || !config || config->steps <= 0 ||
        !isfinite(config->init_scale) || config->init_scale <= 0.0f ||
        !isfinite(config->cfg_scale_text) ||
        !isfinite(config->cfg_scale_speaker) ||
        !isfinite(config->cfg_scale_caption) ||
        !isfinite(config->cfg_min_t) || !isfinite(config->cfg_max_t) ||
        config->cfg_min_t > config->cfg_max_t || !output ||
        speaker_tokens < 0 || (speaker_tokens > 0 && !speaker) ||
        caption_tokens < 0 || (caption_tokens > 0 && !caption) ||
        (speaker_tokens == 0 && config->cfg_scale_speaker != 0.0f) ||
        (caption_tokens == 0 && config->cfg_scale_caption != 0.0f))
        return -1;

    int valid_text = 0;
    for (int i = 0; i < text_tokens; i++)
        if (text_mask[i]) valid_text++;
    if (valid_text <= 0) return -1;

    int text_cfg = config->cfg_scale_text > 0.0f;
    int speaker_cfg = speaker_tokens > 0 && config->cfg_scale_speaker > 0.0f;
    int caption_cfg = caption_tokens > 0 && config->cfg_scale_caption > 0.0f;
    int max_batch = 1 + text_cfg + speaker_cfg + caption_cfg;
    int text_uncond_row = text_cfg ? 1 : -1;
    int speaker_uncond_row = speaker_cfg ? 1 + text_cfg : -1;
    int caption_uncond_row = caption_cfg ? 1 + text_cfg + speaker_cfg : -1;
    if (valid_text > INT32_MAX - speaker_tokens ||
        valid_text + speaker_tokens > INT32_MAX - caption_tokens)
        return -1;
    int context_tokens = valid_text + speaker_tokens + caption_tokens;
    int dit_profile = iro_dit_profile_enabled();
    int gemm_profile = iro_gemm_profile_enabled();
    if (dit_profile) iro_dit_profile_reset();
    if (gemm_profile) iro_gemm_profile_reset();
    size_t compact_text_n, context_one_n, cache_n, latent_one_n;
    size_t latent_batch_n, projected_one_n, projected_n, cond_n, mask_n;
    if (checked_mul((size_t)valid_text, TEXT_DIM, &compact_text_n) != 0 ||
        checked_mul((size_t)context_tokens, MODEL_DIM, &context_one_n) != 0 ||
        checked_mul((size_t)IRO_DIT_LAYERS, context_one_n, &cache_n) != 0 ||
        checked_mul((size_t)sequence_length, LATENT_DIM, &latent_one_n) != 0 ||
        checked_mul((size_t)max_batch, latent_one_n, &latent_batch_n) != 0 ||
        checked_mul((size_t)sequence_length, MODEL_DIM, &projected_one_n) != 0 ||
        checked_mul((size_t)max_batch, projected_one_n, &projected_n) != 0 ||
        checked_mul((size_t)max_batch, COND_DIM, &cond_n) != 0 ||
        checked_mul((size_t)max_batch, (size_t)context_tokens, &mask_n) != 0)
        return -1;

    double t_alloc = dit_profile ? sampler_profile_now() : 0.0;
    IroEulerWorkspaceImpl *ws = workspace_get(workspace);
    if (!ws ||
        reserve_f32(&ws->compact_text, &ws->compact_text_capacity, compact_text_n) != 0 ||
        reserve_f32(&ws->cache_k, &ws->cache_k_capacity, cache_n) != 0 ||
        reserve_f32(&ws->cache_v, &ws->cache_v_capacity, cache_n) != 0 ||
        reserve_u8(&ws->mask_cfg, &ws->mask_capacity, mask_n) != 0 ||
        reserve_f32(&ws->state, &ws->state_capacity, latent_one_n) != 0 ||
        reserve_f32(&ws->x_batch, &ws->x_batch_capacity, latent_batch_n) != 0 ||
        reserve_f32(&ws->projected, &ws->projected_capacity, projected_n) != 0 ||
        reserve_f32(&ws->cond, &ws->cond_capacity, cond_n) != 0 ||
        reserve_f32(&ws->velocity, &ws->velocity_capacity, latent_batch_n) != 0) {
        if (dit_profile)
            iro_dit_profile_add(-1, max_batch, IRO_DIT_PROFILE_ALLOCATION,
                                sampler_profile_now() - t_alloc);
        goto fail;
    }
    if (dit_profile)
        iro_dit_profile_add(-1, max_batch, IRO_DIT_PROFILE_ALLOCATION,
                            sampler_profile_now() - t_alloc);

    float *compact_text = ws->compact_text;
    float *cache_k = ws->cache_k;
    float *cache_v = ws->cache_v;
    uint8_t *mask_cfg = ws->mask_cfg;
    float *state = ws->state;
    float *x_batch = ws->x_batch;
    float *projected = ws->projected;
    float *cond = ws->cond;
    float *velocity = ws->velocity;

    double t_context = dit_profile ? sampler_profile_now() : 0.0;
    int compact_index = 0;
    for (int i = 0; i < text_tokens; i++) {
        if (!text_mask[i]) continue;
        memcpy(compact_text + (size_t)compact_index * TEXT_DIM,
               text + (size_t)i * TEXT_DIM, sizeof(float) * TEXT_DIM);
        compact_index++;
    }
    /* Conditional row has every compact context token. Each independent CFG
       row masks exactly one branch, matching upstream bundle ordering. */
    memset(mask_cfg, 1, (size_t)max_batch * context_tokens);
    if (text_uncond_row >= 0)
        memset(mask_cfg + (size_t)text_uncond_row * context_tokens,
               0, (size_t)valid_text);
    if (speaker_uncond_row >= 0)
        memset(mask_cfg + (size_t)speaker_uncond_row * context_tokens + valid_text,
               0, (size_t)speaker_tokens);
    if (caption_uncond_row >= 0)
        memset(mask_cfg + (size_t)caption_uncond_row * context_tokens +
                   valid_text + speaker_tokens,
               0, (size_t)caption_tokens);
    if (dit_profile)
        iro_dit_profile_add(-1, max_batch, IRO_DIT_PROFILE_CONTEXT_PREP,
                            sampler_profile_now() - t_context);

    /* Project each static condition once per layer. Independent-CFG rows share
       the same K/V values and differ only by mask, so keeping one copy avoids
       replicating the full static cache three/four times. */
    double t_kv = dit_profile ? sampler_profile_now() : 0.0;
    for (int layer = 0; layer < IRO_DIT_LAYERS; layer++) {
        float *layer_k = cache_k + (size_t)layer * context_one_n;
        float *layer_v = cache_v + (size_t)layer * context_one_n;
        if (iro_dit_text_kv(&dit->attention[layer], dit->norm_eps,
                            compact_text, 1, valid_text,
                            layer_k, layer_v) != 0)
            goto fail;
        if (speaker_tokens > 0 &&
            iro_dit_speaker_kv(&dit->attention[layer], dit->norm_eps,
                               speaker, 1, speaker_tokens,
                               layer_k + (size_t)valid_text * MODEL_DIM,
                               layer_v + (size_t)valid_text * MODEL_DIM) != 0)
            goto fail;
        if (caption_tokens > 0 &&
            iro_dit_caption_kv(&dit->attention[layer], dit->norm_eps,
                               caption, 1, caption_tokens,
                               layer_k + (size_t)(valid_text + speaker_tokens) * MODEL_DIM,
                               layer_v + (size_t)(valid_text + speaker_tokens) * MODEL_DIM) != 0)
            goto fail;
    }
    if (dit_profile)
        iro_dit_profile_add(-1, max_batch, IRO_DIT_PROFILE_KV_CACHE_PREP,
                            sampler_profile_now() - t_kv);
    memcpy(state, noise, sizeof(float) * latent_one_n);

    for (int step = 0; step < config->steps; step++) {
        float u = (float)step / (float)config->steps;
        float u_next = (float)(step + 1) / (float)config->steps;
        float timestep = (1.0f - u) * config->init_scale;
        float timestep_next = (1.0f - u_next) * config->init_scale;
        int use_cfg = max_batch > 1 && timestep >= config->cfg_min_t &&
                      timestep <= config->cfg_max_t;
        int batch = use_cfg ? max_batch : 1;
        for (int b = 0; b < batch; b++)
            memcpy(x_batch + (size_t)b * latent_one_n, state,
                   sizeof(float) * latent_one_n);
        float timesteps[MAX_BATCH] = { timestep, timestep, timestep, timestep };

        /* Every independent-CFG row starts a step from the same latent and
           uses the same timestep. Project those shared inputs once, then
           broadcast them before context-specific blocks make rows diverge. */
        if (iro_dit_condition(dit, timesteps, 1, cond) != 0 ||
            iro_dit_input(dit, x_batch, 1, sequence_length, projected) != 0)
            goto fail;
        for (int b = 1; b < batch; b++) {
            memcpy(cond + (size_t)b * COND_DIM, cond,
                   sizeof(float) * COND_DIM);
            memcpy(projected + (size_t)b * sequence_length * MODEL_DIM,
                   projected,
                   sizeof(float) * (size_t)sequence_length * MODEL_DIM);
        }
        for (int layer = 0; layer < IRO_DIT_LAYERS; layer++) {
            size_t offset = (size_t)layer * context_one_n;
            if (iro_dit_block_shared_context(
                    dit, layer, projected, cond,
                    cache_k + offset, cache_v + offset, mask_cfg,
                    batch, sequence_length, context_tokens) != 0)
                goto fail;
        }
        if (iro_dit_output(dit, projected, batch, sequence_length, velocity) != 0)
            goto fail;
        if (trace && trace(trace_user, step, timestep, batch,
                           x_batch, velocity) != 0)
            goto fail;

        double t_update = dit_profile ? sampler_profile_now() : 0.0;
        float dt = timestep_next - timestep;
        for (size_t i = 0; i < latent_one_n; i++) {
            float guided = velocity[i];
            if (use_cfg && text_uncond_row >= 0)
                guided += config->cfg_scale_text *
                    (velocity[i] - velocity[(size_t)text_uncond_row * latent_one_n + i]);
            if (use_cfg && speaker_uncond_row >= 0)
                guided += config->cfg_scale_speaker *
                    (velocity[i] - velocity[(size_t)speaker_uncond_row * latent_one_n + i]);
            if (use_cfg && caption_uncond_row >= 0)
                guided += config->cfg_scale_caption *
                    (velocity[i] - velocity[(size_t)caption_uncond_row * latent_one_n + i]);
            state[i] += guided * dt;
        }
        if (dit_profile)
            iro_dit_profile_add(-1, batch, IRO_DIT_PROFILE_SAMPLER_UPDATE,
                                sampler_profile_now() - t_update);
    }

    memcpy(output, state, sizeof(float) * latent_one_n);
    if (stats) {
        stats->context_tokens = context_tokens;
        stats->context_cache_bytes = 2 * sizeof(float) * cache_n;
    }
    if (dit_profile)
        iro_dit_profile_emit_json("ok", config->steps,
                                  sequence_length, context_tokens);
    if (gemm_profile)
        iro_gemm_profile_emit_json("ok", config->steps,
                                   sequence_length, context_tokens);
    return 0;

fail:
    if (dit_profile)
        iro_dit_profile_emit_json("error", config->steps,
                                  sequence_length, context_tokens);
    if (gemm_profile)
        iro_gemm_profile_emit_json("error", config->steps,
                                   sequence_length, context_tokens);
    return -1;
}

int iro_sample_euler_text(const IroDiT *dit,
                          const float *text, const uint8_t *text_mask,
                          int text_tokens, const float *noise,
                          int sequence_length, const IroEulerConfig *config,
                          float *output, IroEulerTraceFn trace,
                          void *trace_user, IroEulerStats *stats) {
    if (!config) return -1;
    IroEulerWorkspace workspace = {0};
    IroEulerConfig text_config = *config;
    text_config.cfg_scale_speaker = 0.0f;
    text_config.cfg_scale_caption = 0.0f;
    int result = sample_euler(dit, text, text_mask, text_tokens,
                              NULL, 0, NULL, 0, noise, sequence_length,
                              &text_config, output, trace, trace_user, stats,
                              &workspace);
    iro_euler_workspace_free(&workspace);
    return result;
}

int iro_sample_euler_speaker(const IroDiT *dit,
                             const float *text, const uint8_t *text_mask,
                             int text_tokens,
                             const float *speaker, int speaker_tokens,
                             const float *noise, int sequence_length,
                             const IroEulerConfig *config, float *output,
                             IroEulerTraceFn trace, void *trace_user,
                             IroEulerStats *stats) {
    if (!config || !speaker || speaker_tokens <= 0) return -1;
    IroEulerWorkspace workspace = {0};
    IroEulerConfig speaker_config = *config;
    speaker_config.cfg_scale_caption = 0.0f;
    int result = sample_euler(dit, text, text_mask, text_tokens,
                              speaker, speaker_tokens, NULL, 0,
                              noise, sequence_length, &speaker_config,
                              output, trace, trace_user, stats, &workspace);
    iro_euler_workspace_free(&workspace);
    return result;
}

int iro_sample_euler_conditions(const IroDiT *dit,
                                const float *text, const uint8_t *text_mask,
                                int text_tokens,
                                const float *speaker, int speaker_tokens,
                                const float *caption, int caption_tokens,
                                const float *noise, int sequence_length,
                                const IroEulerConfig *config, float *output,
                                IroEulerTraceFn trace, void *trace_user,
                                IroEulerStats *stats) {
    if (!config) return -1;
    IroEulerWorkspace workspace = {0};
    int result = sample_euler(dit, text, text_mask, text_tokens,
                              speaker, speaker_tokens, caption, caption_tokens,
                              noise, sequence_length, config, output,
                              trace, trace_user, stats, &workspace);
    iro_euler_workspace_free(&workspace);
    return result;
}

int iro_sample_euler_conditions_workspace(
    const IroDiT *dit,
    const float *text, const uint8_t *text_mask, int text_tokens,
    const float *speaker, int speaker_tokens,
    const float *caption, int caption_tokens,
    const float *noise, int sequence_length,
    const IroEulerConfig *config, float *output,
    IroEulerTraceFn trace, void *trace_user, IroEulerStats *stats,
    IroEulerWorkspace *workspace) {
    if (!workspace) return -1;
    return sample_euler(dit, text, text_mask, text_tokens,
                        speaker, speaker_tokens, caption, caption_tokens,
                        noise, sequence_length, config, output,
                        trace, trace_user, stats, workspace);
}
