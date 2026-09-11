#define _POSIX_C_SOURCE 200809L
/* main.c — CLI Irodori C engine. Phase 0: inspeksi checkpoint. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "irodori.h"
#include "tokenizer.h"
#include "normalize.h"
#include "backbone.h"
#include "condition.h"
#include "duration.h"
#include "dit.h"
#include "sampler.h"
#include "dacvae.h"
#include "speaker.h"
#include "audio.h"
#include "generate.h"
#include "ops.h"

static void usage(void) {
    fprintf(stderr,
        "irodori — Irodori-TTS pure C engine\n\n"
        "Usage:\n"
        "  irodori --text <text> [--model PATH] [--tokenizer PATH]\n"
        "          [--decoder PATH] [--ref WAV --encoder PATH] [--caption TEXT]\n"
        "          [--seed N] [--steps N] [--out PATH] [--dump-dir PATH]\n"
        "                                                       text -> WAV\n"
        "  irodori --list-tensors <model.safetensors>         inventory tensor\n"
        "  irodori --info <model.safetensors>                  metadata checkpoint\n"
        "  irodori --check <model.safetensors>                 sanity load + shapes\n"
        "  irodori --tokenize <tokenizer.bin> <text>           encode teks -> ids\n"
        "  irodori --test-tokenizer <tokenizer.bin> <vectors.tsv>  gate: bandingkan\n"
        "                                                       dengan referensi HF\n"
        "  irodori --test-backbone <model.safetensors> <golden-dir>\n"
        "                                                       gate ModernBERT FP32\n"
        "  irodori --test-projector <model.safetensors> <golden-dir>\n"
        "                                                       gate text projector\n"
        "  irodori --test-duration <model.safetensors> <golden-dir>\n"
        "                                                       one duration probe\n"
        "  irodori --test-duration-vectors <model.safetensors> <vectors.bin>\n"
        "                                                       20-case duration gate\n"
        "  irodori --test-dit-prepare <model.safetensors> <golden-dir>\n"
        "                                                       timestep + input gate\n"
        "  irodori --test-dit-adaln <model.safetensors> <golden-dir>\n"
        "                                                       block-0 AdaLN gate\n"
        "  irodori --test-dit-attention <model.safetensors> <golden-dir>\n"
        "                                                       block-0 joint attention\n"
        "  irodori --test-dit-mlp <model.safetensors> <golden-dir>\n"
        "                                                       block-0 AdaLN + SwiGLU\n"
        "  irodori --test-dit-forward <model.safetensors> <golden-dir>\n"
        "                                                       12 blocks + velocity\n"
        "  irodori --test-euler <model.safetensors> <golden-dir>\n"
        "                                                       text CFG Euler loop\n"
        "  irodori --test-euler-speaker <model.safetensors> <golden-dir>\n"
        "                                                       text+speaker CFG loop\n"
        "  irodori --test-codec <decoder.safetensors> <golden-dir>\n"
        "                                                       DACVAE decoder gate\n"
        "  irodori --test-encoder <encoder.safetensors> <golden-dir>\n"
        "                                                       DACVAE encoder gate\n"
        "  irodori --test-speaker <model.safetensors> <golden-dir>\n"
        "                                                       speaker encoder gate\n"
        "  irodori --encode-reference <encoder.safetensors> <input.wav> <out.f32>\n"
        "                                                       WAV -> mean latent\n"
        "  irodori --test-wav <golden-dir> <output.wav>         trim + PCM16 WAV gate\n"
        "  irodori --test-pipeline <model> <decoder> <golden-dir> <output.wav>\n"
        "                                                       Euler -> WAV gate\n"
        "  irodori --bench-linear                               benchmark GEMM kernel\n");
}

static uint32_t bench_rng = 0x243f6a88u;

static float bench_random(void) {
    bench_rng ^= bench_rng << 13;
    bench_rng ^= bench_rng >> 17;
    bench_rng ^= bench_rng << 5;
    return ((float)(bench_rng & 0xffffu) / 32768.0f - 1.0f) * 0.02f;
}

static int cmd_bench_linear(void) {
    enum { M = 224, K = 1280, N = 3680, REPEATS = 3 };
    size_t x_n = (size_t)M * K;
    size_t w_n = (size_t)N * K;
    size_t y_n = (size_t)M * N;
    float *x = malloc(sizeof(float) * x_n);
    float *w = malloc(sizeof(float) * w_n);
    float *y = malloc(sizeof(float) * y_n);
    if (!x || !w || !y) {
        fprintf(stderr, "bench-linear: alokasi gagal\n");
        free(x); free(w); free(y);
        return 1;
    }
    for (size_t i = 0; i < x_n; i++) x[i] = bench_random();
    for (size_t i = 0; i < w_n; i++) w[i] = bench_random();

    /* One warm-up removes first-touch/page-fault noise from the timed pass. */
    iro_linear(x, w, NULL, y, M, K, N);
    double samples[REPEATS];
    for (int run = 0; run < REPEATS; run++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        iro_linear(x, w, NULL, y, M, K, N);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        samples[run] = (double)(t1.tv_sec - t0.tv_sec) +
                       1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    }
    for (int i = 1; i < REPEATS; i++) {
        double sample = samples[i];
        int j = i;
        while (j > 0 && samples[j - 1] > sample) {
            samples[j] = samples[j - 1];
            j--;
        }
        samples[j] = sample;
    }
    double secs = samples[REPEATS / 2];
    double gflops = 2.0 * M * K * N / (secs * 1e9);
    double checksum = 0.0;
    for (size_t i = 0; i < y_n; i += 97) checksum += y[i];
    int threads = iro_ops_get_threads();
    printf("linear backend=%s threads=%d shape=%dx%dx%d median=%.3f s best=%.3f s "
           "%.2f GFLOP/s checksum=%.9g\n",
           iro_ops_backend_name(), threads > 0 ? threads : 1,
           M, K, N, secs, samples[0], gflops,
           checksum);
    free(x); free(w); free(y);
    return 0;
}

static int cmd_list(const char *path) {
    IroSafetensors st;
    if (iro_st_load(path, &st) != 0) return 1;
    printf("== %s ==\n", path);
    printf("tensors: %d | metadata keys: %d | file: %.2f GB\n\n",
           st.n_tensors, st.n_meta, (double)st.map_size / (1024.0 * 1024.0 * 1024.0));
    uint64_t total = 0;
    for (int i = 0; i < st.n_tensors; i++) {
        const IroTensor *t = &st.tensors[i];
        printf("%-72s %-4s [", t->name, iro_dtype_name(t->dtype));
        for (int d = 0; d < t->ndims; d++)
            printf("%llu%s", (unsigned long long)t->shape[d], d + 1 < t->ndims ? "," : "");
        printf("] %llu B\n", (unsigned long long)t->nbytes);
        total += t->nbytes;
    }
    printf("\ntotal tensor bytes: %.2f GB\n", (double)total / (1024.0 * 1024.0 * 1024.0));
    iro_st_free(&st);
    return 0;
}

static int cmd_info(const char *path) {
    IroSafetensors st;
    if (iro_st_load(path, &st) != 0) return 1;
    printf("== metadata ==\n");
    for (int i = 0; i < st.n_meta; i++) {
        printf("--- %s ---\n%s\n", st.meta_keys[i], st.meta_values[i]);
    }
    iro_st_free(&st);
    return 0;
}

static int cmd_check(const char *path) {
    IroSafetensors st;
    if (iro_st_load(path, &st) != 0) return 1;
    int bad = 0;
    for (int i = 0; i < st.n_tensors; i++) {
        const IroTensor *t = &st.tensors[i];
        int64_t numel = iro_tensor_numel(t);
        if (t->dtype == IRO_ST_UNKNOWN) { printf("dtype UNKNOWN: %s\n", t->name); bad++; }
        if (t->nbytes != (uint64_t)(numel * 4)) {
            /* fp32 expected untuk checkpoint utama */
            if (t->dtype == IRO_ST_F32) {
                printf("size mismatch: %s numel=%lld bytes=%llu\n", t->name,
                       (long long)numel, (unsigned long long)t->nbytes);
                bad++;
            }
        }
        if (t->dtype == IRO_ST_F32 && iro_tensor_f32_all_finite(t) != 1) {
            printf("non-finite F32: %s\n", t->name);
            bad++;
        }
        if (t->data_begin + t->nbytes > st.map_size) { printf("out of range: %s\n", t->name); bad++; }
    }
    printf("check: %d tensor, %d masalah\n", st.n_tensors, bad);
    iro_st_free(&st);
    return bad ? 1 : 0;
}

/* hex string -> bytes */
static int from_hex(const char *hex, char *out, int cap) {
    int n = 0;
    for (const char *p = hex; p[0] && p[1] && n < cap; p += 2) {
        int hi = p[0] <= '9' ? p[0] - '0' : (p[0] | 32) - 'a' + 10;
        int lo = p[1] <= '9' ? p[1] - '0' : (p[1] | 32) - 'a' + 10;
        out[n++] = (char)((hi << 4) | lo);
    }
    return n;
}

static int cmd_tokenize(const char *binpath, const char *text) {
    IroTokenizer t;
    if (iro_tok_load(binpath, &t) != 0) return 1;
    int32_t ids[IRO_TOK_MAX_IDS];
    int n = iro_tok_encode(&t, text, 1, ids, IRO_TOK_MAX_IDS);
    if (n < 0) { fprintf(stderr, "encode gagal\n"); iro_tok_free(&t); return 1; }
    for (int i = 0; i < n; i++) {
        printf("%d", ids[i]);
        if (i + 1 < n) printf(" ");
    }
    printf("\n");
    iro_tok_free(&t);
    return 0;
}

static int cmd_test_tokenizer(const char *binpath, const char *vecpath) {
    IroTokenizer t;
    if (iro_tok_load(binpath, &t) != 0) return 1;

    FILE *f = fopen(vecpath, "r");
    if (!f) { perror("fopen vectors"); iro_tok_free(&t); return 1; }

    char line[32768];
    int ntests = 0, npass = 0;
    while (fgets(line, sizeof(line), f)) {
        /* format: <ids spasi>\t<hex normalized>\t<hex raw> */
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = '\0';
        char *hex_norm = tab + 1;
        char *tab2 = strchr(hex_norm, '\t');
        if (!tab2) continue;
        *tab2 = '\0';
        char *hex_raw = tab2 + 1;
        hex_raw[strcspn(hex_raw, "\r\n")] = '\0';

        /* expected ids */
        int32_t exp[IRO_TOK_MAX_IDS];
        int nexp = 0;
        for (char *tok = strtok(line, " "); tok && nexp < IRO_TOK_MAX_IDS; tok = strtok(NULL, " "))
            exp[nexp++] = (int32_t)atoi(tok);

        char raw[8192], norm_exp[8192];
        int rlen = from_hex(hex_raw, raw, (int)sizeof(raw) - 1);
        raw[rlen] = '\0';
        int nlen_exp = from_hex(hex_norm, norm_exp, (int)sizeof(norm_exp) - 1);
        norm_exp[nlen_exp] = '\0';

        /* pipeline C: normalize -> encode */
        char *norm = iro_normalize(raw);
        int32_t got[IRO_TOK_MAX_IDS];
        int ngot = norm ? iro_tok_encode(&t, norm, 1, got, IRO_TOK_MAX_IDS) : -1;

        int match = norm && (ngot == nexp) && (strcmp(norm, norm_exp) == 0);
        for (int i = 0; match && i < ngot; i++)
            if (got[i] != exp[i]) match = 0;

        ntests++;
        if (match) {
            npass++;
            printf("PASS [%d ids] %.*s\n", ngot, rlen > 60 ? 60 : rlen, raw);
        } else {
            printf("FAIL %.*s\n  raw: %.*s\n", rlen > 50 ? 50 : rlen, raw, rlen > 50 ? 50 : rlen, raw);
            if (!norm) printf("  normalize GAGAL\n");
            else if (strcmp(norm, norm_exp) != 0)
                printf("  norm beda: got [%s] exp [%s]\n", norm, norm_exp);
            printf("  exp (%d):", nexp);
            for (int i = 0; i < nexp; i++) printf(" %d", exp[i]);
            printf("\n  got (%d):", ngot);
            for (int i = 0; i < ngot; i++) printf(" %d", got[i]);
            printf("\n");
        }
        free(norm);
    }
    fclose(f);
    iro_tok_free(&t);
    printf("\ngate normalize+tokenizer: %d/%d PASS\n", npass, ntests);
    return npass == ntests ? 0 : 1;
}

/* baca seluruh file ke buffer malloc */
static void *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n <= 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    void *buf = malloc((size_t)n);
    if (buf && fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); buf = NULL; }
    fclose(f);
    if (out_len) *out_len = (size_t)n;
    return buf;
}

static int compare_golden_f32(const char *goldendir, const char *name,
                              const float *data, size_t n, double *out_maxerr) {
    char path[512];
    int nw = snprintf(path, sizeof(path), "%s/%s.f32", goldendir, name);
    if (nw < 0 || (size_t)nw >= sizeof(path)) return -1;
    size_t ref_len = 0;
    float *ref = read_file(path, &ref_len);
    if (!ref) {
        fprintf(stderr, "golden tidak ada: %s\n", path);
        return -1;
    }
    if (ref_len != n * sizeof(float)) {
        fprintf(stderr, "shape golden salah: %s (%zu byte, expected %zu)\n",
                path, ref_len, n * sizeof(float));
        free(ref);
        return -1;
    }
    double maxerr = 0.0;
    for (size_t i = 0; i < n; i++) {
        double err = fabs((double)data[i] - (double)ref[i]);
        if (err > maxerr) maxerr = err;
    }
    free(ref);
    *out_maxerr = maxerr;
    return 0;
}

typedef struct {
    const char *goldendir;
    int valid_rows;
    double maxerr;
    double finalerr;
    int compared;
    int errors;
} BackboneTrace;

static int trace_backbone(void *user, const char *name, const float *data,
                          int rows, int cols) {
    BackboneTrace *trace = user;
    char path[512];
    int nw = snprintf(path, sizeof(path), "%s/%s.f32", trace->goldendir, name);
    if (nw < 0 || (size_t)nw >= sizeof(path)) {
        fprintf(stderr, "path golden terlalu panjang: %s\n", name);
        trace->errors++;
        return -1;
    }

    size_t ref_len = 0;
    float *ref = read_file(path, &ref_len);
    if (!ref) {
        /* Intermediate traces are optional for old fixtures; final output is not. */
        if (!strcmp(name, "backbone_out")) {
            fprintf(stderr, "golden wajib tidak ada: %s\n", path);
            trace->errors++;
            return -1;
        }
        return 0;
    }

    size_t n = (size_t)rows * cols;
    if (ref_len != n * sizeof(float)) {
        fprintf(stderr, "shape golden salah: %s (%zu byte, expected %zu)\n",
                path, ref_len, n * sizeof(float));
        free(ref);
        trace->errors++;
        return -1;
    }

    double maxerr = 0.0, valid_maxerr = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < n; i++) {
        double err = fabs((double)data[i] - (double)ref[i]);
        if (err > maxerr) { maxerr = err; worst = i; }
        if (i < (size_t)trace->valid_rows * cols && err > valid_maxerr)
            valid_maxerr = err;
    }
    int is_final = !strcmp(name, "backbone_out");
    printf("  %-14s valid %.6g | all %.6g @ [%zu,%zu] (C %.6g, ref %.6g)  %s\n",
           name, valid_maxerr, maxerr, worst / (size_t)cols, worst % (size_t)cols,
           data[worst], ref[worst],
           is_final ? (maxerr < 1e-3 ? "PASS" : "FAIL") : "TRACE");
    if (maxerr > trace->maxerr) trace->maxerr = maxerr;
    if (is_final) trace->finalerr = maxerr;
    trace->compared++;
    free(ref);
    return 0;
}

static int cmd_test_backbone(const char *stpath, const char *goldendir) {
    IroSafetensors st;
    if (iro_st_load(stpath, &st) != 0) return 1;
    IroBackbone bb;
    if (iro_backbone_init(&bb, &st, 256) != 0) {
        iro_st_free(&st);
        return 1;
    }
    int result = 1;
    int64_t *ids64 = NULL;
    float *out = NULL;

    char path[512];
    snprintf(path, sizeof(path), "%s/token_ids.i64", goldendir);
    size_t ids_len = 0;
    ids64 = read_file(path, &ids_len);
    int n_ids = (int)(ids_len / 8);
    if (!ids64 || ids_len % sizeof(int64_t) != 0 || n_ids <= 0 || n_ids > 256) {
        fprintf(stderr, "token_ids tidak ada atau invalid: %s\n", path);
        goto cleanup;
    }

    const int MAXLEN = 256;
    int32_t ids[512];
    uint8_t mask[512];
    memset(mask, 0, sizeof(mask));
    for (int i = 0; i < MAXLEN; i++) {
        ids[i] = i < n_ids ? (int32_t)ids64[i] : bb.pad_id;
        mask[i] = i < n_ids ? 1 : 0;
    }

    out = malloc(sizeof(float) * (size_t)MAXLEN * bb.hidden);
    if (!out) { fprintf(stderr, "alokasi output backbone gagal\n"); goto cleanup; }

    BackboneTrace trace = {
        .goldendir = goldendir,
        .valid_rows = n_ids,
        .maxerr = 0.0,
        .finalerr = INFINITY,
        .compared = 0,
        .errors = 0,
    };

    /* timing */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int r = iro_backbone_forward_trace(
        &bb, ids, mask, MAXLEN, out, trace_backbone, &trace
    );
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    if (r != 0 || trace.errors || trace.compared == 0) {
        fprintf(stderr, "forward/golden trace gagal\n");
        goto cleanup;
    }

    printf("backbone: %d token, %d checkpoint dibandingkan\n", n_ids, trace.compared);
    printf("final max abs err: %.6g | trace max: %.6g | waktu: %.2f s | gate < 1e-3: %s\n",
           trace.finalerr, trace.maxerr, secs,
           trace.finalerr < 1e-3 ? "PASS" : "FAIL");
    result = trace.finalerr < 1e-3 ? 0 : 1;

cleanup:
    free(ids64);
    free(out);
    iro_backbone_free(&bb);
    iro_st_free(&st);
    return result;
}

static int cmd_test_projector(const char *stpath, const char *goldendir) {
    enum { MAXLEN = 256, BACKBONE_DIM = 768, CONDITION_DIM = 512 };
    int result = 1;
    IroSafetensors st;
    if (iro_st_load(stpath, &st) != 0) return 1;
    IroTextProjector projector;
    if (iro_text_projector_init(&projector, &st) != 0) {
        iro_st_free(&st);
        return 1;
    }

    char path[512];
    size_t ids_len = 0, backbone_len = 0;
    int64_t *ids = NULL;
    float *backbone = NULL, *projected = NULL, *text_state = NULL;
    snprintf(path, sizeof(path), "%s/token_ids.i64", goldendir);
    ids = read_file(path, &ids_len);
    int n_ids = (int)(ids_len / sizeof(int64_t));
    if (!ids || ids_len % sizeof(int64_t) != 0 || n_ids <= 0 || n_ids > MAXLEN) {
        fprintf(stderr, "token_ids tidak ada atau invalid: %s\n", path);
        goto cleanup;
    }

    snprintf(path, sizeof(path), "%s/backbone_out.f32", goldendir);
    backbone = read_file(path, &backbone_len);
    if (!backbone || backbone_len !=
        sizeof(float) * (size_t)MAXLEN * BACKBONE_DIM) {
        fprintf(stderr, "backbone golden tidak ada atau invalid: %s\n", path);
        goto cleanup;
    }

    projected = malloc(sizeof(float) * (size_t)MAXLEN * CONDITION_DIM);
    text_state = malloc(sizeof(float) * (size_t)MAXLEN * CONDITION_DIM);
    if (!projected || !text_state) goto cleanup;
    uint8_t mask[MAXLEN];
    for (int i = 0; i < MAXLEN; i++) mask[i] = i < n_ids;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (iro_text_projector_forward(&projector, backbone, mask, MAXLEN,
                                   projected, text_state) != 0) {
        fprintf(stderr, "forward text projector gagal\n");
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) +
                  1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);

    double projected_err = 0.0, state_err = 0.0;
    size_t n = (size_t)MAXLEN * CONDITION_DIM;
    if (compare_golden_f32(goldendir, "text_projector_out", projected, n,
                           &projected_err) != 0 ||
        compare_golden_f32(goldendir, "text_state", text_state, n,
                           &state_err) != 0)
        goto cleanup;

    double maxerr = projected_err > state_err ? projected_err : state_err;
    printf("text projector: %d token | projected err %.6g | norm err %.6g\n",
           n_ids, projected_err, state_err);
    printf("waktu: %.2f s | gate < 1e-4: %s\n", secs,
           maxerr < 1e-4 ? "PASS" : "FAIL");
    result = maxerr < 1e-4 ? 0 : 1;

cleanup:
    free(ids);
    free(backbone);
    free(projected);
    free(text_state);
    iro_st_free(&st);
    return result;
}

static int cmd_test_duration(const char *stpath, const char *goldendir) {
    enum { MAXLEN = 256, CONDITION_DIM = 512 };
    int result = 1;
    IroSafetensors st;
    if (iro_st_load(stpath, &st) != 0) return 1;
    IroDurationPredictor duration;
    if (iro_duration_init(&duration, &st) != 0) {
        iro_st_free(&st);
        return 1;
    }

    char path[512];
    size_t ids_len = 0, state_len = 0, ref_len = 0;
    int64_t *ids = NULL;
    float *text_state = NULL, *ref = NULL;
    snprintf(path, sizeof(path), "%s/token_ids.i64", goldendir);
    ids = read_file(path, &ids_len);
    int n_ids = (int)(ids_len / sizeof(int64_t));
    if (!ids || ids_len % sizeof(int64_t) != 0 || n_ids <= 0 || n_ids > MAXLEN) {
        fprintf(stderr, "token_ids tidak ada atau invalid: %s\n", path);
        goto cleanup;
    }
    snprintf(path, sizeof(path), "%s/text_state.f32", goldendir);
    text_state = read_file(path, &state_len);
    if (!text_state || state_len != sizeof(float) * (size_t)MAXLEN * CONDITION_DIM) {
        fprintf(stderr, "text_state golden tidak ada atau invalid: %s\n", path);
        goto cleanup;
    }
    snprintf(path, sizeof(path), "%s/duration_out.f32", goldendir);
    ref = read_file(path, &ref_len);
    if (!ref || ref_len != sizeof(float)) {
        fprintf(stderr, "duration golden tidak ada atau invalid: %s\n", path);
        goto cleanup;
    }

    uint8_t mask[MAXLEN];
    memset(mask, 1, (size_t)n_ids);
    float got = 0.0f;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (iro_duration_forward_text(&duration, text_state, mask, n_ids, &got) != 0) {
        fprintf(stderr, "forward duration gagal\n");
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) +
                  1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    double err = fabs((double)got - ref[0]);
    int got_frames = iro_duration_frame_count(got, 1.0f, 0.5f, 30.0f, 48000, 1920);
    int ref_frames = iro_duration_frame_count(ref[0], 1.0f, 0.5f, 30.0f, 48000, 1920);
    int pass = err < 1e-4 && got_frames == ref_frames;
    printf("duration: log_frames C %.7g | ref %.7g | err %.6g\n",
           got, ref[0], err);
    printf("frames C %d | ref %d | waktu %.2f s | gate: %s\n",
           got_frames, ref_frames, secs, pass ? "PASS" : "FAIL");
    result = pass ? 0 : 1;

cleanup:
    free(ids);
    free(text_state);
    free(ref);
    iro_st_free(&st);
    return result;
}

static int cmd_test_duration_vectors(const char *stpath, const char *vecpath) {
    int result = 1;
    IroSafetensors st;
    if (iro_st_load(stpath, &st) != 0) return 1;
    IroDurationPredictor duration;
    if (iro_duration_init(&duration, &st) != 0) {
        iro_st_free(&st);
        return 1;
    }
    FILE *f = fopen(vecpath, "rb");
    if (!f) {
        perror("duration vectors");
        iro_st_free(&st);
        return 1;
    }

    char magic[8];
    uint32_t cases = 0, dim = 0;
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        memcmp(magic, "IRODUR1\0", sizeof(magic)) != 0 ||
        fread(&cases, sizeof(cases), 1, f) != 1 ||
        fread(&dim, sizeof(dim), 1, f) != 1 ||
        cases == 0 || cases > 10000 || dim != 512) {
        fprintf(stderr, "header duration vectors invalid: %s\n", vecpath);
        goto cleanup;
    }

    double worst = 0.0;
    int passed = 0;
    for (uint32_t c = 0; c < cases; c++) {
        uint32_t tokens = 0;
        float expected = 0.0f;
        if (fread(&tokens, sizeof(tokens), 1, f) != 1 ||
            fread(&expected, sizeof(expected), 1, f) != 1 ||
            tokens == 0 || tokens > 256) {
            fprintf(stderr, "case duration %u invalid\n", c);
            goto cleanup;
        }
        size_t n = (size_t)tokens * dim;
        float *state = malloc(sizeof(float) * n);
        uint8_t *mask = malloc(tokens);
        if (!state || !mask || fread(state, sizeof(float), n, f) != n) {
            free(state);
            free(mask);
            fprintf(stderr, "data duration case %u terpotong\n", c);
            goto cleanup;
        }
        memset(mask, 1, tokens);
        float got = 0.0f;
        int ok = iro_duration_forward_text(&duration, state, mask, (int)tokens, &got) == 0;
        free(state);
        free(mask);
        if (!ok) {
            fprintf(stderr, "forward duration case %u gagal\n", c);
            goto cleanup;
        }
        double err = fabs((double)got - expected);
        int got_frames = iro_duration_frame_count(got, 1.0f, 0.5f, 30.0f, 48000, 1920);
        int expected_frames = iro_duration_frame_count(
            expected, 1.0f, 0.5f, 30.0f, 48000, 1920
        );
        int pass = err < 1e-4 && got_frames == expected_frames;
        printf("%s duration[%02u]: tokens=%u err=%.6g frames=%d/%d\n",
               pass ? "PASS" : "FAIL", c, tokens, err, got_frames, expected_frames);
        if (err > worst) worst = err;
        if (pass) passed++;
    }
    if (fgetc(f) != EOF) {
        fprintf(stderr, "duration vectors punya trailing data\n");
        goto cleanup;
    }
    printf("duration gate: %d/%u PASS | worst log_frames err %.6g\n",
           passed, cases, worst);
    result = passed == (int)cases ? 0 : 1;

cleanup:
    fclose(f);
    iro_st_free(&st);
    return result;
}

static int cmd_test_dit_prepare(const char *stpath, const char *goldendir) {
    enum { COND_DIM = 3840, LATENT_DIM = 32, MODEL_DIM = 1280 };
    int result = 1;
    IroSafetensors st;
    if (iro_st_load(stpath, &st) != 0) return 1;
    IroDiT dit;
    if (iro_dit_init(&dit, &st) != 0) {
        iro_st_free(&st);
        return 1;
    }

    char path[512];
    size_t cond_ref_len = 0, x_len = 0, input_ref_len = 0;
    float *cond_ref = NULL, *x = NULL, *input_ref = NULL;
    float *t = NULL, *cond = NULL, *input = NULL;
    snprintf(path, sizeof(path), "%s/cond_embed_step000.f32", goldendir);
    cond_ref = read_file(path, &cond_ref_len);
    if (!cond_ref || cond_ref_len % (sizeof(float) * COND_DIM) != 0) {
        fprintf(stderr, "cond golden tidak ada atau invalid: %s\n", path);
        goto cleanup;
    }
    int batch = (int)(cond_ref_len / (sizeof(float) * COND_DIM));
    if (batch <= 0) goto cleanup;

    snprintf(path, sizeof(path), "%s/x_t_step000.f32", goldendir);
    x = read_file(path, &x_len);
    if (!x || x_len % (sizeof(float) * (size_t)batch * LATENT_DIM) != 0) {
        fprintf(stderr, "x_t golden tidak ada atau invalid: %s\n", path);
        goto cleanup;
    }
    int seq = (int)(x_len / (sizeof(float) * (size_t)batch * LATENT_DIM));
    if (seq <= 0) goto cleanup;
    snprintf(path, sizeof(path), "%s/in_proj_out_step000.f32", goldendir);
    input_ref = read_file(path, &input_ref_len);
    size_t expected_input_len =
        sizeof(float) * (size_t)batch * seq * MODEL_DIM;
    if (!input_ref || input_ref_len != expected_input_len) {
        fprintf(stderr, "in_proj golden tidak ada atau invalid: %s\n", path);
        goto cleanup;
    }

    t = malloc(sizeof(float) * (size_t)batch);
    cond = malloc(cond_ref_len);
    input = malloc(input_ref_len);
    if (!t || !cond || !input) goto cleanup;
    /* sample_euler_rf_cfg starts at init_scale=0.999, not exactly 1.0. */
    for (int b = 0; b < batch; b++) t[b] = 0.999f;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (iro_dit_condition(&dit, t, batch, cond) != 0 ||
        iro_dit_input(&dit, x, batch, seq, input) != 0) {
        fprintf(stderr, "DiT prepare forward gagal\n");
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) +
                  1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    double cond_err = 0.0, input_err = 0.0;
    if (compare_golden_f32(goldendir, "cond_embed_step000", cond,
                           (size_t)batch * COND_DIM, &cond_err) != 0 ||
        compare_golden_f32(goldendir, "in_proj_out_step000", input,
                           (size_t)batch * seq * MODEL_DIM, &input_err) != 0)
        goto cleanup;
    double worst = cond_err > input_err ? cond_err : input_err;
    printf("DiT prepare: batch=%d seq=%d | cond err %.6g | in_proj err %.6g\n",
           batch, seq, cond_err, input_err);
    printf("waktu: %.2f s | gate < 1e-4: %s\n", secs,
           worst < 1e-4 ? "PASS" : "FAIL");
    result = worst < 1e-4 ? 0 : 1;

cleanup:
    free(cond_ref); free(x); free(input_ref);
    free(t); free(cond); free(input);
    iro_st_free(&st);
    return result;
}

static int cmd_test_dit_adaln(const char *stpath, const char *goldendir) {
    enum { COND_DIM = 3840, MODEL_DIM = 1280 };
    int result = 1;
    IroSafetensors st;
    if (iro_st_load(stpath, &st) != 0) return 1;
    IroDiT dit;
    if (iro_dit_init(&dit, &st) != 0) {
        iro_st_free(&st);
        return 1;
    }

    char path[512];
    size_t cond_len = 0, x_len = 0, h_ref_len = 0, gate_ref_len = 0;
    float *cond = NULL, *x = NULL, *h_ref = NULL, *gate_ref = NULL;
    float *h = NULL, *gate = NULL;

    snprintf(path, sizeof(path), "%s/cond_embed_step000.f32", goldendir);
    cond = read_file(path, &cond_len);
    if (!cond || cond_len % (sizeof(float) * COND_DIM) != 0) {
        fprintf(stderr, "cond golden tidak ada atau invalid: %s\n", path);
        goto cleanup;
    }
    int batch = (int)(cond_len / (sizeof(float) * COND_DIM));
    if (batch <= 0) goto cleanup;

    snprintf(path, sizeof(path), "%s/in_proj_out_step000.f32", goldendir);
    x = read_file(path, &x_len);
    if (!x || x_len % (sizeof(float) * (size_t)batch * MODEL_DIM) != 0) {
        fprintf(stderr, "in_proj golden tidak ada atau invalid: %s\n", path);
        goto cleanup;
    }
    int seq = (int)(x_len / (sizeof(float) * (size_t)batch * MODEL_DIM));
    if (seq <= 0) goto cleanup;

    snprintf(path, sizeof(path), "%s/dit_b0_attn_adaln_h.f32", goldendir);
    h_ref = read_file(path, &h_ref_len);
    snprintf(path, sizeof(path), "%s/dit_b0_attn_adaln_gate.f32", goldendir);
    gate_ref = read_file(path, &gate_ref_len);
    size_t expected_h_len = sizeof(float) * (size_t)batch * seq * MODEL_DIM;
    size_t expected_gate_len = sizeof(float) * (size_t)batch * MODEL_DIM;
    if (!h_ref || h_ref_len != expected_h_len ||
        !gate_ref || gate_ref_len != expected_gate_len) {
        fprintf(stderr, "AdaLN golden tidak ada atau ukurannya invalid\n");
        goto cleanup;
    }

    h = malloc(expected_h_len);
    gate = malloc(expected_gate_len);
    if (!h || !gate) goto cleanup;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (iro_dit_adaln(&dit.attention_adaln[0], dit.norm_eps,
                      x, cond, batch, seq, h, gate) != 0) {
        fprintf(stderr, "DiT AdaLN forward gagal\n");
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) +
                  1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    double h_err = 0.0, gate_err = 0.0;
    if (compare_golden_f32(goldendir, "dit_b0_attn_adaln_h", h,
                           (size_t)batch * seq * MODEL_DIM, &h_err) != 0 ||
        compare_golden_f32(goldendir, "dit_b0_attn_adaln_gate", gate,
                           (size_t)batch * MODEL_DIM, &gate_err) != 0)
        goto cleanup;
    double worst = h_err > gate_err ? h_err : gate_err;
    printf("DiT block 0 attention AdaLN: batch=%d seq=%d | h err %.6g | "
           "gate err %.6g\n", batch, seq, h_err, gate_err);
    printf("waktu: %.2f s | gate < 1e-4: %s\n", secs,
           worst < 1e-4 ? "PASS" : "FAIL");
    result = worst < 1e-4 ? 0 : 1;

cleanup:
    free(cond); free(x); free(h_ref); free(gate_ref);
    free(h); free(gate);
    iro_st_free(&st);
    return result;
}

static int cmd_test_dit_attention(const char *stpath, const char *goldendir) {
    enum { COND_DIM = 3840, MODEL_DIM = 1280, TEXT_DIM = 512 };
    int result = 1;
    IroSafetensors st;
    if (iro_st_load(stpath, &st) != 0) return 1;
    IroDiT dit;
    if (iro_dit_init(&dit, &st) != 0) {
        iro_st_free(&st);
        return 1;
    }

    char path[512];
    size_t cond_len = 0, x_len = 0, text_one_len = 0, mask_len = 0;
    size_t ref_k_len = 0, ref_v_len = 0, ref_out_len = 0;
    float *cond = NULL, *x = NULL, *text_one = NULL;
    uint8_t *mask = NULL;
    float *ref_k = NULL, *ref_v = NULL, *ref_out = NULL;
    float *h = NULL, *adaln_gate = NULL, *text = NULL;
    float *text_k = NULL, *text_v = NULL, *attention_out = NULL;

    snprintf(path, sizeof(path), "%s/cond_embed_step000.f32", goldendir);
    cond = read_file(path, &cond_len);
    if (!cond || cond_len % (sizeof(float) * COND_DIM) != 0) goto invalid;
    int batch = (int)(cond_len / (sizeof(float) * COND_DIM));

    snprintf(path, sizeof(path), "%s/in_proj_out_step000.f32", goldendir);
    x = read_file(path, &x_len);
    if (!x || batch <= 0 ||
        x_len % (sizeof(float) * (size_t)batch * MODEL_DIM) != 0) goto invalid;
    int seq = (int)(x_len / (sizeof(float) * (size_t)batch * MODEL_DIM));

    snprintf(path, sizeof(path), "%s/text_state.f32", goldendir);
    text_one = read_file(path, &text_one_len);
    if (!text_one || text_one_len % (sizeof(float) * TEXT_DIM) != 0) goto invalid;
    int text_seq = (int)(text_one_len / (sizeof(float) * TEXT_DIM));

    snprintf(path, sizeof(path), "%s/dit_b0_text_mask.u8", goldendir);
    mask = read_file(path, &mask_len);
    snprintf(path, sizeof(path), "%s/dit_b0_text_k.f32", goldendir);
    ref_k = read_file(path, &ref_k_len);
    snprintf(path, sizeof(path), "%s/dit_b0_text_v.f32", goldendir);
    ref_v = read_file(path, &ref_v_len);
    snprintf(path, sizeof(path), "%s/dit_b0_attn_out.f32", goldendir);
    ref_out = read_file(path, &ref_out_len);
    size_t context_n = (size_t)batch * text_seq * MODEL_DIM;
    size_t latent_n = (size_t)batch * seq * MODEL_DIM;
    if (!mask || mask_len != (size_t)batch * text_seq ||
        !ref_k || ref_k_len != sizeof(float) * context_n ||
        !ref_v || ref_v_len != sizeof(float) * context_n ||
        !ref_out || ref_out_len != sizeof(float) * latent_n)
        goto invalid;

    h = malloc(sizeof(float) * latent_n);
    adaln_gate = malloc(sizeof(float) * (size_t)batch * MODEL_DIM);
    text = calloc((size_t)batch * text_seq * TEXT_DIM, sizeof(float));
    text_k = malloc(sizeof(float) * context_n);
    text_v = malloc(sizeof(float) * context_n);
    attention_out = malloc(sizeof(float) * latent_n);
    if (!h || !adaln_gate || !text || !text_k || !text_v || !attention_out)
        goto cleanup;
    memcpy(text, text_one, text_one_len);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (iro_dit_adaln(&dit.attention_adaln[0], dit.norm_eps,
                      x, cond, batch, seq, h, adaln_gate) != 0 ||
        iro_dit_text_kv(&dit.attention[0], dit.norm_eps,
                        text, batch, text_seq, text_k, text_v) != 0 ||
        iro_dit_attention(&dit.attention[0], dit.norm_eps, h,
                          text_k, text_v, mask,
                          batch, seq, text_seq, attention_out) != 0) {
        fprintf(stderr, "DiT joint attention forward gagal\n");
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) +
                  1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    double k_err = 0.0, v_err = 0.0, out_err = 0.0;
    if (compare_golden_f32(goldendir, "dit_b0_text_k", text_k,
                           context_n, &k_err) != 0 ||
        compare_golden_f32(goldendir, "dit_b0_text_v", text_v,
                           context_n, &v_err) != 0 ||
        compare_golden_f32(goldendir, "dit_b0_attn_out", attention_out,
                           latent_n, &out_err) != 0)
        goto cleanup;
    double worst = k_err > v_err ? k_err : v_err;
    if (out_err > worst) worst = out_err;
    printf("DiT block 0 joint attention: batch=%d latent=%d text=%d | "
           "text K err %.6g | V err %.6g | out err %.6g\n",
           batch, seq, text_seq, k_err, v_err, out_err);
    printf("waktu %s termasuk text KV: %.2f s | gate < 1e-3: %s\n",
           iro_ops_backend_name(), secs, worst < 1e-3 ? "PASS" : "FAIL");
    result = worst < 1e-3 ? 0 : 1;
    goto cleanup;

invalid:
    fprintf(stderr, "fixture DiT attention tidak ada atau ukurannya invalid\n");
cleanup:
    free(cond); free(x); free(text_one); free(mask);
    free(ref_k); free(ref_v); free(ref_out);
    free(h); free(adaln_gate); free(text);
    free(text_k); free(text_v); free(attention_out);
    iro_st_free(&st);
    return result;
}

static int cmd_test_dit_mlp(const char *stpath, const char *goldendir) {
    enum { COND_DIM = 3840, MODEL_DIM = 1280 };
    int result = 1;
    IroSafetensors st;
    if (iro_st_load(stpath, &st) != 0) return 1;
    IroDiT dit;
    if (iro_dit_init(&dit, &st) != 0) {
        iro_st_free(&st);
        return 1;
    }

    char path[512];
    size_t cond_len = 0, x_len = 0, ref_len = 0;
    float *cond = NULL, *x = NULL, *ref = NULL;
    float *h = NULL, *gate = NULL, *mlp_out = NULL, *block_out = NULL;
    snprintf(path, sizeof(path), "%s/cond_embed_step000.f32", goldendir);
    cond = read_file(path, &cond_len);
    if (!cond || cond_len % (sizeof(float) * COND_DIM) != 0) goto invalid;
    int batch = (int)(cond_len / (sizeof(float) * COND_DIM));

    snprintf(path, sizeof(path), "%s/dit_b0_x_after_attention.f32", goldendir);
    x = read_file(path, &x_len);
    if (!x || batch <= 0 ||
        x_len % (sizeof(float) * (size_t)batch * MODEL_DIM) != 0) goto invalid;
    int seq = (int)(x_len / (sizeof(float) * (size_t)batch * MODEL_DIM));
    size_t latent_n = (size_t)batch * seq * MODEL_DIM;
    snprintf(path, sizeof(path), "%s/dit_b0_out.f32", goldendir);
    ref = read_file(path, &ref_len);
    if (!ref || ref_len != sizeof(float) * latent_n) goto invalid;

    h = malloc(sizeof(float) * latent_n);
    gate = malloc(sizeof(float) * (size_t)batch * MODEL_DIM);
    mlp_out = malloc(sizeof(float) * latent_n);
    block_out = malloc(sizeof(float) * latent_n);
    if (!h || !gate || !mlp_out || !block_out) goto cleanup;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (iro_dit_adaln(&dit.mlp_adaln[0], dit.norm_eps,
                      x, cond, batch, seq, h, gate) != 0 ||
        iro_dit_swiglu(&dit.mlp[0], h, batch, seq, mlp_out) != 0) {
        fprintf(stderr, "DiT SwiGLU forward gagal\n");
        goto cleanup;
    }
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq; s++) {
            size_t row = ((size_t)b * seq + s) * MODEL_DIM;
            const float *gate_row = gate + (size_t)b * MODEL_DIM;
            for (int d = 0; d < MODEL_DIM; d++)
                block_out[row + d] = x[row + d] + gate_row[d] * mlp_out[row + d];
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) +
                  1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);

    double h_err = 0.0, gate_err = 0.0, mlp_err = 0.0, block_err = 0.0;
    if (compare_golden_f32(goldendir, "dit_b0_mlp_adaln_h", h,
                           latent_n, &h_err) != 0 ||
        compare_golden_f32(goldendir, "dit_b0_mlp_adaln_gate", gate,
                           (size_t)batch * MODEL_DIM, &gate_err) != 0 ||
        compare_golden_f32(goldendir, "dit_b0_mlp_out", mlp_out,
                           latent_n, &mlp_err) != 0 ||
        compare_golden_f32(goldendir, "dit_b0_out", block_out,
                           latent_n, &block_err) != 0)
        goto cleanup;
    double worst = h_err > gate_err ? h_err : gate_err;
    if (mlp_err > worst) worst = mlp_err;
    if (block_err > worst) worst = block_err;
    printf("DiT block 0 MLP: batch=%d seq=%d | AdaLN h %.6g gate %.6g | "
           "SwiGLU %.6g | block %.6g\n",
           batch, seq, h_err, gate_err, mlp_err, block_err);
    printf("waktu %s: %.2f s | gate < 1e-3: %s\n",
           iro_ops_backend_name(), secs, worst < 1e-3 ? "PASS" : "FAIL");
    result = worst < 1e-3 ? 0 : 1;
    goto cleanup;

invalid:
    fprintf(stderr, "fixture DiT MLP tidak ada atau ukurannya invalid\n");
cleanup:
    free(cond); free(x); free(ref);
    free(h); free(gate); free(mlp_out); free(block_out);
    iro_st_free(&st);
    return result;
}

static int cmd_test_dit_forward(const char *stpath, const char *goldendir) {
    enum { COND_DIM = 3840, MODEL_DIM = 1280, TEXT_DIM = 512, LATENT_DIM = 32 };
    int result = 1;
    IroSafetensors st;
    if (iro_st_load(stpath, &st) != 0) return 1;
    IroDiT dit;
    if (iro_dit_init(&dit, &st) != 0) {
        iro_st_free(&st);
        return 1;
    }

    char path[512];
    size_t text_one_len = 0, mask_cfg_len = 0;
    float *text_one = NULL, *text_cfg = NULL;
    uint8_t *mask_cfg = NULL;
    float *cache_cfg_k = NULL, *cache_cfg_v = NULL;
    float *cache_cond_k = NULL, *cache_cond_v = NULL;
    float *cond = NULL, *x = NULL, *velocity_ref = NULL;
    float *state = NULL, *velocity = NULL;

    snprintf(path, sizeof(path), "%s/text_state.f32", goldendir);
    text_one = read_file(path, &text_one_len);
    if (!text_one || text_one_len % (sizeof(float) * TEXT_DIM) != 0) goto invalid;
    int text_seq = (int)(text_one_len / (sizeof(float) * TEXT_DIM));
    snprintf(path, sizeof(path), "%s/dit_b0_text_mask.u8", goldendir);
    mask_cfg = read_file(path, &mask_cfg_len);
    if (!mask_cfg || mask_cfg_len != (size_t)2 * text_seq) goto invalid;

    size_t context_cfg_n = (size_t)2 * text_seq * MODEL_DIM;
    size_t context_cond_n = (size_t)text_seq * MODEL_DIM;
    text_cfg = calloc((size_t)2 * text_seq * TEXT_DIM, sizeof(float));
    cache_cfg_k = malloc(sizeof(float) * IRO_DIT_LAYERS * context_cfg_n);
    cache_cfg_v = malloc(sizeof(float) * IRO_DIT_LAYERS * context_cfg_n);
    cache_cond_k = malloc(sizeof(float) * IRO_DIT_LAYERS * context_cond_n);
    cache_cond_v = malloc(sizeof(float) * IRO_DIT_LAYERS * context_cond_n);
    if (!text_cfg || !cache_cfg_k || !cache_cfg_v ||
        !cache_cond_k || !cache_cond_v)
        goto cleanup;
    memcpy(text_cfg, text_one, text_one_len);

    struct timespec cache_t0, cache_t1;
    clock_gettime(CLOCK_MONOTONIC, &cache_t0);
    for (int layer = 0; layer < IRO_DIT_LAYERS; layer++) {
        if (iro_dit_text_kv(&dit.attention[layer], dit.norm_eps,
                            text_cfg, 2, text_seq,
                            cache_cfg_k + (size_t)layer * context_cfg_n,
                            cache_cfg_v + (size_t)layer * context_cfg_n) != 0 ||
            iro_dit_text_kv(&dit.attention[layer], dit.norm_eps,
                            text_one, 1, text_seq,
                            cache_cond_k + (size_t)layer * context_cond_n,
                            cache_cond_v + (size_t)layer * context_cond_n) != 0) {
            fprintf(stderr, "DiT text KV layer %d gagal\n", layer);
            goto cleanup;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &cache_t1);

    /* Warm one CFG-shaped pass and validate every intermediate block outside
       the benchmark timer.  The matched PyTorch harness does the same warmup. */
    size_t warm_cond_len = 0, warm_x_len = 0;
    snprintf(path, sizeof(path), "%s/cond_embed_step000.f32", goldendir);
    cond = read_file(path, &warm_cond_len);
    snprintf(path, sizeof(path), "%s/in_proj_out_step000.f32", goldendir);
    x = read_file(path, &warm_x_len);
    if (!cond || warm_cond_len != sizeof(float) * 2 * COND_DIM ||
        !x || warm_x_len % (sizeof(float) * 2 * MODEL_DIM) != 0)
        goto invalid;
    int warm_seq = (int)(warm_x_len / (sizeof(float) * 2 * MODEL_DIM));
    size_t warm_latent_n = (size_t)2 * warm_seq * MODEL_DIM;
    state = malloc(sizeof(float) * warm_latent_n);
    velocity = malloc(sizeof(float) * (size_t)2 * warm_seq * LATENT_DIM);
    if (!state || !velocity) goto cleanup;
    memcpy(state, x, sizeof(float) * warm_latent_n);
    double worst_layer = 0.0;
    for (int layer = 0; layer < IRO_DIT_LAYERS; layer++) {
        if (iro_dit_block(&dit, layer, state, cond,
                          cache_cfg_k + (size_t)layer * context_cfg_n,
                          cache_cfg_v + (size_t)layer * context_cfg_n,
                          mask_cfg, 2, warm_seq, text_seq) != 0) {
            fprintf(stderr, "DiT warmup block %d gagal\n", layer);
            goto cleanup;
        }
        char golden_name[64];
        snprintf(golden_name, sizeof(golden_name), "dit_b%02d_out", layer);
        double layer_err = 0.0;
        if (compare_golden_f32(goldendir, golden_name, state,
                               warm_latent_n, &layer_err) != 0)
            goto cleanup;
        if (layer_err > worst_layer) worst_layer = layer_err;
        printf("  block %02d diagnostic max abs err %.6g\n", layer, layer_err);
    }
    if (iro_dit_output(&dit, state, 2, warm_seq, velocity) != 0) {
        fprintf(stderr, "DiT warmup output projection gagal\n");
        goto cleanup;
    }
    free(cond); cond = NULL;
    free(x); x = NULL;
    free(state); state = NULL;
    free(velocity); velocity = NULL;

    double total_forward_secs = 0.0;
    double worst_velocity = 0.0;
    for (int step = 0; step < 8; step++) {
        size_t cond_len = 0, x_len = 0, velocity_ref_len = 0;
        snprintf(path, sizeof(path), "%s/cond_embed_step%03d.f32", goldendir, step);
        cond = read_file(path, &cond_len);
        if (!cond || cond_len % (sizeof(float) * COND_DIM) != 0) goto invalid;
        int batch = (int)(cond_len / (sizeof(float) * COND_DIM));
        if (batch != 1 && batch != 2) goto invalid;
        snprintf(path, sizeof(path), "%s/in_proj_out_step%03d.f32", goldendir, step);
        x = read_file(path, &x_len);
        if (!x || x_len % (sizeof(float) * (size_t)batch * MODEL_DIM) != 0)
            goto invalid;
        int seq = (int)(x_len / (sizeof(float) * (size_t)batch * MODEL_DIM));
        size_t latent_n = (size_t)batch * seq * MODEL_DIM;
        size_t velocity_n = (size_t)batch * seq * LATENT_DIM;
        snprintf(path, sizeof(path), "%s/velocity_step%03d.f32", goldendir, step);
        velocity_ref = read_file(path, &velocity_ref_len);
        if (!velocity_ref || velocity_ref_len != sizeof(float) * velocity_n)
            goto invalid;

        state = malloc(sizeof(float) * latent_n);
        velocity = malloc(sizeof(float) * velocity_n);
        if (!state || !velocity) goto cleanup;
        memcpy(state, x, sizeof(float) * latent_n);
        size_t context_n = batch == 2 ? context_cfg_n : context_cond_n;
        const float *cache_k = batch == 2 ? cache_cfg_k : cache_cond_k;
        const float *cache_v = batch == 2 ? cache_cfg_v : cache_cond_v;

        struct timespec forward_t0, forward_t1;
        clock_gettime(CLOCK_MONOTONIC, &forward_t0);
        for (int layer = 0; layer < IRO_DIT_LAYERS; layer++) {
            if (iro_dit_block(&dit, layer, state, cond,
                              cache_k + (size_t)layer * context_n,
                              cache_v + (size_t)layer * context_n,
                              mask_cfg, batch, seq, text_seq) != 0) {
                fprintf(stderr, "DiT step %d block %d gagal\n", step, layer);
                goto cleanup;
            }
        }
        if (iro_dit_output(&dit, state, batch, seq, velocity) != 0) {
            fprintf(stderr, "DiT output projection step %d gagal\n", step);
            goto cleanup;
        }
        clock_gettime(CLOCK_MONOTONIC, &forward_t1);
        double forward_secs = (double)(forward_t1.tv_sec - forward_t0.tv_sec) +
                              1e-9 * (double)(forward_t1.tv_nsec - forward_t0.tv_nsec);
        total_forward_secs += forward_secs;

        char velocity_name[64];
        snprintf(velocity_name, sizeof(velocity_name), "velocity_step%03d", step);
        double velocity_err = 0.0;
        if (compare_golden_f32(goldendir, velocity_name, velocity,
                               velocity_n, &velocity_err) != 0)
            goto cleanup;
        if (velocity_err > worst_velocity) worst_velocity = velocity_err;
        printf("  step %03d batch=%d velocity err %.6g | %.3f s\n",
               step, batch, velocity_err, forward_secs);

        free(cond); cond = NULL;
        free(x); x = NULL;
        free(velocity_ref); velocity_ref = NULL;
        free(state); state = NULL;
        free(velocity); velocity = NULL;
    }

    double cache_secs = (double)(cache_t1.tv_sec - cache_t0.tv_sec) +
                        1e-9 * (double)(cache_t1.tv_nsec - cache_t0.tv_nsec);
    printf("DiT full %s: CFG+cond caches %.3f s | 8 forwards %.3f s | "
           "worst velocity err %.6g | layer abs diagnostic %.6g\n",
           iro_ops_backend_name(), cache_secs, total_forward_secs,
           worst_velocity, worst_layer);
    printf("velocity gate < 1e-3: %s\n",
           worst_velocity < 1e-3 ? "PASS" : "FAIL");
    result = worst_velocity < 1e-3 ? 0 : 1;
    goto cleanup;

invalid:
    fprintf(stderr, "fixture DiT forward tidak ada atau ukurannya invalid\n");
cleanup:
    free(text_one); free(text_cfg); free(mask_cfg);
    free(cache_cfg_k); free(cache_cfg_v);
    free(cache_cond_k); free(cache_cond_v);
    free(cond); free(x); free(velocity_ref); free(state); free(velocity);
    iro_st_free(&st);
    return result;
}

typedef struct {
    const char *goldendir;
    int sequence_length;
    int steps;
    int seen;
    double worst_state;
    double worst_velocity;
} EulerGoldenTrace;

static int euler_golden_trace(void *user, int step, float timestep, int batch,
                              const float *x_t, const float *velocity) {
    EulerGoldenTrace *trace = user;
    int expected_batch = step < trace->steps / 2 ? 2 : 1;
    float expected_t = (1.0f - (float)step / (float)trace->steps) * 0.999f;
    if (step != trace->seen || batch != expected_batch ||
        fabsf(timestep - expected_t) > 1e-7f) {
        fprintf(stderr, "Euler trace step/timestep/batch tidak cocok\n");
        return -1;
    }
    size_t n = (size_t)batch * trace->sequence_length * 32;
    char name[64];
    double state_err = 0.0, velocity_err = 0.0;
    snprintf(name, sizeof(name), "x_t_step%03d", step);
    if (compare_golden_f32(trace->goldendir, name, x_t, n, &state_err) != 0)
        return -1;
    snprintf(name, sizeof(name), "velocity_step%03d", step);
    if (compare_golden_f32(trace->goldendir, name, velocity,
                           n, &velocity_err) != 0)
        return -1;
    if (state_err > trace->worst_state) trace->worst_state = state_err;
    if (velocity_err > trace->worst_velocity)
        trace->worst_velocity = velocity_err;
    trace->seen++;
    printf("  step %03d t=%.6f batch=%d | state err %.6g | velocity err %.6g\n",
           step, timestep, batch, state_err, velocity_err);
    return 0;
}

static int cmd_test_euler(const char *stpath, const char *goldendir) {
    enum { TEXT_DIM = 512, LATENT_DIM = 32, STEPS = 8 };
    int result = 1;
    IroSafetensors st;
    if (iro_st_load(stpath, &st) != 0) return 1;
    IroDiT dit;
    if (iro_dit_init(&dit, &st) != 0) {
        iro_st_free(&st);
        return 1;
    }

    char path[512];
    size_t text_len = 0, mask_len = 0, x0_len = 0;
    size_t final_x_len = 0, final_v_len = 0;
    float *text = NULL, *x0 = NULL, *final_x = NULL, *final_v = NULL;
    uint8_t *mask = NULL;
    float *output = NULL, *final_ref = NULL;

    snprintf(path, sizeof(path), "%s/text_state.f32", goldendir);
    text = read_file(path, &text_len);
    if (!text || text_len % (sizeof(float) * TEXT_DIM) != 0) goto invalid;
    int text_tokens = (int)(text_len / (sizeof(float) * TEXT_DIM));
    snprintf(path, sizeof(path), "%s/dit_b0_text_mask.u8", goldendir);
    mask = read_file(path, &mask_len);
    if (!mask || mask_len != (size_t)2 * text_tokens) goto invalid;
    snprintf(path, sizeof(path), "%s/x_t_step000.f32", goldendir);
    x0 = read_file(path, &x0_len);
    if (!x0 || x0_len % (sizeof(float) * 2 * LATENT_DIM) != 0) goto invalid;
    int sequence_length = (int)(x0_len / (sizeof(float) * 2 * LATENT_DIM));
    size_t latent_n = (size_t)sequence_length * LATENT_DIM;
    output = malloc(sizeof(float) * latent_n);
    final_ref = malloc(sizeof(float) * latent_n);
    if (!output || !final_ref) goto cleanup;

    EulerGoldenTrace trace = {
        .goldendir = goldendir,
        .sequence_length = sequence_length,
        .steps = STEPS,
    };
    IroEulerConfig config = {
        .steps = STEPS,
        .init_scale = 0.999f,
        .cfg_scale_text = 3.0f,
        .cfg_min_t = 0.5f,
        .cfg_max_t = 1.0f,
    };
    IroEulerStats stats = {0};
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (iro_sample_euler_text(&dit, text, mask, text_tokens, x0,
                              sequence_length, &config, output,
                              euler_golden_trace, &trace, &stats) != 0) {
        fprintf(stderr, "Euler text sampling gagal\n");
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    snprintf(path, sizeof(path), "%s/x_t_step007.f32", goldendir);
    final_x = read_file(path, &final_x_len);
    snprintf(path, sizeof(path), "%s/velocity_step007.f32", goldendir);
    final_v = read_file(path, &final_v_len);
    if (!final_x || !final_v || final_x_len != sizeof(float) * latent_n ||
        final_v_len != sizeof(float) * latent_n)
        goto invalid;
    float final_dt = -config.init_scale / (float)config.steps;
    double final_err = 0.0;
    for (size_t i = 0; i < latent_n; i++) {
        final_ref[i] = final_x[i] + final_v[i] * final_dt;
        double err = fabs((double)output[i] - (double)final_ref[i]);
        if (err > final_err) final_err = err;
    }
    double secs = (double)(t1.tv_sec - t0.tv_sec) +
                  1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    double cache_mib = (double)stats.context_cache_bytes / (1024.0 * 1024.0);
    /* Per-step state/velocity are diagnostics: an autoregressive state delta
       can be amplified by the DiT Jacobian.  P1's product gate is the final
       latent after all Euler updates. */
    int pass = trace.seen == STEPS && final_err < 1e-2;
    printf("Euler full %s: %.3f s | compact text %d/%d token | cache %.2f MiB\n",
           iro_ops_backend_name(), secs, stats.context_tokens, text_tokens,
           cache_mib);
    printf("worst state %.6g | worst velocity %.6g | final latent %.6g | "
           "gate: %s\n", trace.worst_state, trace.worst_velocity, final_err,
           pass ? "PASS" : "FAIL");
    result = pass ? 0 : 1;
    goto cleanup;

invalid:
    fprintf(stderr, "fixture Euler tidak ada atau ukurannya invalid\n");
cleanup:
    free(text); free(mask); free(x0); free(final_x); free(final_v);
    free(output); free(final_ref);
    iro_st_free(&st);
    return result;
}

typedef struct {
    const char *goldendir;
    int sequence_length;
    int seen;
    double worst_state;
    double worst_velocity;
} SpeakerEulerTrace;

static int speaker_euler_trace(void *user, int step, float timestep, int batch,
                               const float *x_t, const float *velocity) {
    SpeakerEulerTrace *trace = user;
    int expected_batch = step == 0 ? 3 : 1;
    float expected_t = (1.0f - (float)step / 2.0f) * 0.999f;
    if (step != trace->seen || batch != expected_batch ||
        fabsf(timestep - expected_t) > 1e-7f)
        return -1;
    size_t n = (size_t)batch * trace->sequence_length * 32;
    char name[80];
    double state_err = 0.0, velocity_err = 0.0;
    snprintf(name, sizeof(name), "speaker_euler_x_step%03d", step);
    if (compare_golden_f32(trace->goldendir, name, x_t, n, &state_err) != 0)
        return -1;
    snprintf(name, sizeof(name), "speaker_euler_velocity_step%03d", step);
    if (compare_golden_f32(trace->goldendir, name, velocity, n,
                           &velocity_err) != 0)
        return -1;
    if (state_err > trace->worst_state) trace->worst_state = state_err;
    if (velocity_err > trace->worst_velocity)
        trace->worst_velocity = velocity_err;
    trace->seen++;
    printf("  speaker step %d t=%.6f batch=%d state %.6g velocity %.6g\n",
           step, timestep, batch, state_err, velocity_err);
    return 0;
}

static int cmd_test_euler_speaker(const char *stpath, const char *goldendir) {
    int result = 1;
    IroSafetensors st;
    if (iro_st_load(stpath, &st) != 0) return 1;
    IroDiT dit;
    if (iro_dit_init(&dit, &st) != 0) {
        iro_st_free(&st);
        return 1;
    }
    char path[512];
    size_t text_bytes = 0, speaker_bytes = 0, noise_bytes = 0, final_bytes = 0;
    snprintf(path, sizeof(path), "%s/duration_text.f32", goldendir);
    float *text = read_file(path, &text_bytes);
    snprintf(path, sizeof(path), "%s/speaker_state.f32", goldendir);
    float *speaker = read_file(path, &speaker_bytes);
    snprintf(path, sizeof(path), "%s/speaker_euler_noise.f32", goldendir);
    float *noise = read_file(path, &noise_bytes);
    snprintf(path, sizeof(path), "%s/speaker_euler_final.f32", goldendir);
    float *expected = read_file(path, &final_bytes);
    if (!text || !speaker || !noise || !expected || text_bytes == 0 ||
        text_bytes % (512 * sizeof(float)) || speaker_bytes == 0 ||
        speaker_bytes % (768 * sizeof(float)) || noise_bytes == 0 ||
        noise_bytes % (32 * sizeof(float)) || final_bytes != noise_bytes) {
        fprintf(stderr, "fixture speaker Euler tidak ada atau invalid\n");
        goto cleanup_speaker_euler;
    }
    int text_tokens = (int)(text_bytes / (512 * sizeof(float)));
    int speaker_tokens = (int)(speaker_bytes / (768 * sizeof(float)));
    int sequence_length = (int)(noise_bytes / (32 * sizeof(float)));
    uint8_t *text_mask = malloc((size_t)text_tokens);
    float *output = malloc(noise_bytes);
    if (!text_mask || !output) {
        free(text_mask); free(output);
        goto cleanup_speaker_euler;
    }
    memset(text_mask, 1, (size_t)text_tokens);
    IroEulerConfig config = {
        .steps = 2,
        .init_scale = 0.999f,
        .cfg_scale_text = 3.0f,
        .cfg_scale_speaker = 5.0f,
        .cfg_min_t = 0.5f,
        .cfg_max_t = 1.0f,
    };
    SpeakerEulerTrace trace = {
        .goldendir = goldendir,
        .sequence_length = sequence_length,
    };
    IroEulerStats stats = {0};
    if (iro_sample_euler_speaker(&dit, text, text_mask, text_tokens,
                                 speaker, speaker_tokens, noise,
                                 sequence_length, &config, output,
                                 speaker_euler_trace, &trace, &stats) != 0) {
        fprintf(stderr, "Euler text+speaker sampling gagal\n");
        free(text_mask); free(output);
        goto cleanup_speaker_euler;
    }
    double final_err = 0.0;
    for (size_t i = 0; i < noise_bytes / sizeof(float); i++) {
        double error = fabs((double)output[i] - expected[i]);
        if (error > final_err) final_err = error;
    }
    double cache_mib = (double)stats.context_cache_bytes / (1024.0 * 1024.0);
    int pass = trace.seen == 2 && trace.worst_velocity < 1e-2 &&
               final_err < 1e-2;
    printf("speaker Euler: context %d token | cache %.2f MiB | worst velocity "
           "%.6g | final %.6g | %s\n", stats.context_tokens, cache_mib,
           trace.worst_velocity, final_err, pass ? "PASS" : "FAIL");
    result = pass ? 0 : 1;
    free(text_mask);
    free(output);

cleanup_speaker_euler:
    free(text); free(speaker); free(noise); free(expected);
    iro_st_free(&st);
    return result;
}

typedef struct {
    const char *goldendir;
    double worst;
    int compared;
} CodecGoldenTrace;

typedef struct {
    struct timespec previous;
    struct timespec begin;
} CodecTimingTrace;

static int codec_timing_trace(void *user, const char *name,
                              const float *data, int frames, int channels) {
    (void)data;
    CodecTimingTrace *trace = user;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double stage = (double)(now.tv_sec - trace->previous.tv_sec) +
                   1e-9 * (double)(now.tv_nsec - trace->previous.tv_nsec);
    double total = (double)(now.tv_sec - trace->begin.tv_sec) +
                   1e-9 * (double)(now.tv_nsec - trace->begin.tv_nsec);
    printf("  profile %-24s [%d,%d] stage %.3f s | total %.3f s\n",
           name, frames, channels, stage, total);
    trace->previous = now;
    return 0;
}

static int codec_golden_trace(void *user, const char *name,
                              const float *data, int frames, int channels) {
    CodecGoldenTrace *trace = user;
    double error = 0.0;
    if (compare_golden_f32(trace->goldendir, name, data,
                           (size_t)frames * channels, &error) != 0)
        return -1;
    if (error > trace->worst) trace->worst = error;
    trace->compared++;
    printf("  %-24s [%d,%d] max abs err %.6g\n",
           name, frames, channels, error);
    return 0;
}

static int cmd_test_encoder(const char *stpath, const char *goldendir) {
    int result = 1;
    IroSafetensors st;
    if (iro_st_load(stpath, &st) != 0) return 1;
    IroDACVAEEncoder encoder;
    if (iro_dacvae_encoder_init(&encoder, &st) != 0) {
        iro_st_free(&st);
        return 1;
    }
    char path[512];
    size_t waveform_bytes = 0;
    snprintf(path, sizeof(path), "%s/waveform.f32", goldendir);
    float *waveform = read_file(path, &waveform_bytes);
    if (!waveform || waveform_bytes == 0 || waveform_bytes % sizeof(float)) {
        fprintf(stderr, "fixture waveform encoder tidak ada atau invalid: %s\n", path);
        free(waveform);
        iro_st_free(&st);
        return 1;
    }
    size_t sample_count = waveform_bytes / sizeof(float);
    float *latent = NULL;
    int frames = 0;
    CodecGoldenTrace trace = {.goldendir = goldendir};
    if (iro_dacvae_encode_mean(&encoder, waveform, sample_count,
                               &latent, &frames,
                               codec_golden_trace, &trace) != 0)
        goto cleanup_encoder;
    free(latent);
    latent = NULL;
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    if (iro_dacvae_encode_mean(&encoder, waveform, sample_count,
                               &latent, &frames, NULL, NULL) != 0)
        goto cleanup_encoder;
    clock_gettime(CLOCK_MONOTONIC, &end);
    double secs = (double)(end.tv_sec - begin.tv_sec) +
                  1e-9 * (double)(end.tv_nsec - begin.tv_nsec);
    int pass = trace.compared == 8 && trace.worst < 1e-3 &&
               frames == (int)((sample_count + 1919u) / 1920u);
    printf("DACVAE encoder %s: %zu sample -> %d latent frame | %.3f s | "
           "worst %.6g | gate < 1e-3: %s\n",
           iro_ops_backend_name(), sample_count, frames, secs,
           trace.worst, pass ? "PASS" : "FAIL");
    result = pass ? 0 : 1;

cleanup_encoder:
    free(latent);
    free(waveform);
    iro_st_free(&st);
    return result;
}

typedef struct {
    const char *goldendir;
    double worst;
    int compared;
} SpeakerGoldenTrace;

static int speaker_golden_trace(void *user, const char *name,
                                const float *data, int frames, int channels) {
    SpeakerGoldenTrace *trace = user;
    double error = 0.0;
    if (compare_golden_f32(trace->goldendir, name, data,
                           (size_t)frames * channels, &error) != 0)
        return -1;
    if (error > trace->worst) trace->worst = error;
    trace->compared++;
    printf("  %-20s [%d,%d] max abs err %.6g\n",
           name, frames, channels, error);
    return 0;
}

static int cmd_test_speaker(const char *stpath, const char *goldendir) {
    int result = 1;
    IroSafetensors st;
    if (iro_st_load(stpath, &st) != 0) return 1;
    IroSpeakerEncoder encoder;
    if (iro_speaker_init(&encoder, &st) != 0) {
        iro_st_free(&st);
        return 1;
    }
    char path[512];
    size_t latent_bytes = 0;
    snprintf(path, sizeof(path), "%s/latent.f32", goldendir);
    float *latent = read_file(path, &latent_bytes);
    if (!latent || latent_bytes == 0 ||
        latent_bytes % (32 * sizeof(float)) != 0) {
        fprintf(stderr, "fixture latent speaker tidak ada atau invalid: %s\n", path);
        free(latent);
        iro_st_free(&st);
        return 1;
    }
    int latent_frames = (int)(latent_bytes / (32 * sizeof(float)));
    float *state = NULL;
    int state_frames = 0;
    SpeakerGoldenTrace trace = {.goldendir = goldendir};
    if (iro_speaker_encode(&encoder, latent, latent_frames,
                           &state, &state_frames,
                           speaker_golden_trace, &trace) != 0)
        goto cleanup_speaker;
    IroDurationPredictor duration;
    if (iro_duration_init(&duration, &st) != 0) goto cleanup_speaker;
    size_t text_bytes = 0, expected_bytes = 0;
    snprintf(path, sizeof(path), "%s/duration_text.f32", goldendir);
    float *duration_text = read_file(path, &text_bytes);
    snprintf(path, sizeof(path), "%s/duration_speaker_out.f32", goldendir);
    float *duration_expected = read_file(path, &expected_bytes);
    if (!duration_text || text_bytes == 0 ||
        text_bytes % (512 * sizeof(float)) != 0 ||
        !duration_expected || expected_bytes != sizeof(float)) {
        fprintf(stderr, "fixture duration speaker tidak ada atau invalid\n");
        free(duration_text);
        free(duration_expected);
        goto cleanup_speaker;
    }
    int text_frames = (int)(text_bytes / (512 * sizeof(float)));
    uint8_t *text_mask = malloc((size_t)text_frames);
    float duration_got = 0.0f;
    if (!text_mask) {
        free(duration_text);
        free(duration_expected);
        goto cleanup_speaker;
    }
    memset(text_mask, 1, (size_t)text_frames);
    int duration_ok = iro_duration_forward(
        &duration, duration_text, text_mask, text_frames,
        state, NULL, &duration_got) == 0;
    double duration_err = duration_ok
        ? fabs((double)duration_got - duration_expected[0]) : INFINITY;
    int duration_frames = iro_duration_frame_count(
        duration_got, 1.0f, 0.5f, 30.0f, 48000, 1920);
    int expected_duration_frames = iro_duration_frame_count(
        duration_expected[0], 1.0f, 0.5f, 30.0f, 48000, 1920);
    printf("  duration+speaker     log %.7g/%.7g err %.6g frames %d/%d\n",
           duration_got, duration_expected[0], duration_err,
           duration_frames, expected_duration_frames);
    free(text_mask);
    free(duration_text);
    free(duration_expected);
    IroDiT dit;
    size_t speaker_kv_n = (size_t)state_frames * IRO_DIT_MODEL_DIM;
    float *speaker_k = malloc(sizeof(float) * speaker_kv_n);
    float *speaker_v = malloc(sizeof(float) * speaker_kv_n);
    double speaker_k_err = INFINITY, speaker_v_err = INFINITY;
    if (!speaker_k || !speaker_v || iro_dit_init(&dit, &st) != 0 ||
        iro_dit_speaker_kv(&dit.attention[0], dit.norm_eps,
                           state, 1, state_frames, speaker_k, speaker_v) != 0 ||
        compare_golden_f32(goldendir, "dit_speaker_k", speaker_k,
                           speaker_kv_n, &speaker_k_err) != 0 ||
        compare_golden_f32(goldendir, "dit_speaker_v", speaker_v,
                           speaker_kv_n, &speaker_v_err) != 0) {
        free(speaker_k);
        free(speaker_v);
        goto cleanup_speaker;
    }
    printf("  DiT speaker KV      [%d,%d] err K %.6g V %.6g\n",
           state_frames, IRO_DIT_MODEL_DIM, speaker_k_err, speaker_v_err);
    free(speaker_k);
    free(speaker_v);
    free(state);
    state = NULL;
    double timings[5];
    for (int repeat = 0; repeat < 5; repeat++) {
        struct timespec begin, end;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        if (iro_speaker_encode(&encoder, latent, latent_frames,
                               &state, &state_frames, NULL, NULL) != 0)
            goto cleanup_speaker;
        clock_gettime(CLOCK_MONOTONIC, &end);
        timings[repeat] = (double)(end.tv_sec - begin.tv_sec) +
                          1e-9 * (double)(end.tv_nsec - begin.tv_nsec);
        free(state);
        state = NULL;
    }
    for (int i = 0; i < 5; i++)
        for (int j = i + 1; j < 5; j++)
            if (timings[j] < timings[i]) {
                double swap = timings[i]; timings[i] = timings[j]; timings[j] = swap;
            }
    double secs = timings[2];
    int expected_frames = latent_frames / IRO_SPEAKER_PATCH + 1;
    int pass = trace.compared == IRO_SPEAKER_LAYERS + 3 &&
               trace.worst < 1e-3 && state_frames == expected_frames &&
               duration_err < 1e-4 && duration_frames == expected_duration_frames &&
               speaker_k_err < 1e-4 && speaker_v_err < 1e-4;
    printf("speaker encoder %s: %d latent -> %d context frame | median %.3f s | "
           "worst %.6g | gate < 1e-3: %s\n",
           iro_ops_backend_name(), latent_frames, state_frames, secs,
           trace.worst, pass ? "PASS" : "FAIL");
    result = pass ? 0 : 1;

cleanup_speaker:
    free(state);
    free(latent);
    iro_st_free(&st);
    return result;
}

static int cmd_encode_reference(const char *stpath, const char *wavpath,
                                const char *output_path) {
    IroAudio audio;
    float loudness = 0.0f, gain = 0.0f;
    if (iro_prepare_reference_wav(wavpath, -16.0f, &audio,
                                  &loudness, &gain) != 0) {
        fprintf(stderr, "gagal memuat/preprocess reference WAV: %s\n", wavpath);
        return 1;
    }
    IroSafetensors st;
    if (iro_st_load(stpath, &st) != 0) {
        iro_audio_free(&audio);
        return 1;
    }
    IroDACVAEEncoder encoder;
    if (iro_dacvae_encoder_init(&encoder, &st) != 0) {
        iro_st_free(&st);
        iro_audio_free(&audio);
        return 1;
    }
    float *latent = NULL;
    int frames = 0;
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    int failed = iro_dacvae_encode_mean(&encoder, audio.samples,
                                         audio.sample_count,
                                         &latent, &frames, NULL, NULL) != 0;
    clock_gettime(CLOCK_MONOTONIC, &end);
    FILE *output = NULL;
    if (!failed) output = fopen(output_path, "wb");
    size_t values = (size_t)frames * 32u;
    if (!output || fwrite(latent, sizeof(float), values, output) != values)
        failed = 1;
    if (output && fclose(output) != 0) failed = 1;
    double secs = (double)(end.tv_sec - begin.tv_sec) +
                  1e-9 * (double)(end.tv_nsec - begin.tv_nsec);
    if (!failed)
        printf("Reference encoded: %zu sample @ 48 kHz -> %d x 32 latent | "
               "LUFS %.3f gain %.5f | %.3f s | %s\n",
               audio.sample_count, frames, loudness, gain, secs, output_path);
    else
        fprintf(stderr, "gagal encode/menulis reference latent\n");
    free(latent);
    iro_st_free(&st);
    iro_audio_free(&audio);
    return failed ? 1 : 0;
}

static int cmd_test_codec(const char *stpath, const char *goldendir) {
    enum { CODEBOOK_DIM = 32, DEFAULT_REPEATS = 3, MAX_REPEATS = 9 };
    int result = 1;
    IroSafetensors st;
    if (iro_st_load(stpath, &st) != 0) return 1;
    IroDACVAEDecoder decoder;
    if (iro_dacvae_init(&decoder, &st) != 0) {
        iro_st_free(&st);
        return 1;
    }

    char path[512];
    size_t latent_len = 0;
    snprintf(path, sizeof(path), "%s/codec_latent_final.f32", goldendir);
    float *latent = read_file(path, &latent_len);
    if (!latent || latent_len % (sizeof(float) * CODEBOOK_DIM) != 0) {
        fprintf(stderr, "fixture latent codec tidak ada atau invalid: %s\n", path);
        free(latent);
        iro_st_free(&st);
        return 1;
    }
    int latent_frames = (int)(latent_len / (sizeof(float) * CODEBOOK_DIM));
    int output_frames = iro_dacvae_output_length(latent_frames);
    float *waveform = malloc(sizeof(float) * (size_t)output_frames);
    if (!waveform) {
        free(latent);
        iro_st_free(&st);
        return 1;
    }

    CodecGoldenTrace trace = {.goldendir = goldendir};
    if (iro_dacvae_decode(&decoder, latent, latent_frames, waveform,
                          codec_golden_trace, &trace) != 0) {
        fprintf(stderr, "DACVAE decode gagal\n");
        goto cleanup_codec;
    }
    /* The golden pass warms weights and validates every stage.  Keep its file
       I/O and comparisons outside the matched decoder latency. */
    int repeats = DEFAULT_REPEATS;
    const char *repeats_env = getenv("IRO_CODEC_REPEATS");
    if (repeats_env) {
        char *end = NULL;
        long parsed = strtol(repeats_env, &end, 10);
        if (end != repeats_env && *end == '\0' && parsed >= 1 &&
            parsed <= MAX_REPEATS)
            repeats = (int)parsed;
    }
    double samples[MAX_REPEATS];
    for (int run = 0; run < repeats; run++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (iro_dacvae_decode(&decoder, latent, latent_frames, waveform,
                              NULL, NULL) != 0) {
            fprintf(stderr, "DACVAE timed decode gagal\n");
            goto cleanup_codec;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        samples[run] = (double)(t1.tv_sec - t0.tv_sec) +
                       1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    }
    for (int i = 1; i < repeats; i++) {
        double sample = samples[i];
        int j = i;
        while (j > 0 && samples[j - 1] > sample) {
            samples[j] = samples[j - 1];
            j--;
        }
        samples[j] = sample;
    }
    double secs = samples[repeats / 2];
    int pass = trace.compared == 7 && trace.worst < 1e-3;
    int backend_threads = iro_ops_get_threads();
    printf("DACVAE %s threads=%d: %d latent frame -> %d sample | median %.3f s "
           "best %.3f s | "
           "worst %.6g | gate < 1e-3: %s\n",
           iro_ops_backend_name(), backend_threads > 0 ? backend_threads : 1,
           latent_frames, output_frames, secs, samples[0],
           trace.worst, pass ? "PASS" : "FAIL");
    const char *profile_env = getenv("IRO_CODEC_PROFILE");
    if (profile_env && strcmp(profile_env, "0")) {
        CodecTimingTrace timing;
        clock_gettime(CLOCK_MONOTONIC, &timing.begin);
        timing.previous = timing.begin;
        if (iro_dacvae_decode(&decoder, latent, latent_frames, waveform,
                              codec_timing_trace, &timing) != 0) {
            fprintf(stderr, "DACVAE profiled decode gagal\n");
            goto cleanup_codec;
        }
    }
    result = pass ? 0 : 1;

cleanup_codec:
    free(latent); free(waveform);
    iro_st_free(&st);
    return result;
}

static uint16_t read_u16le(const unsigned char *bytes) {
    return (uint16_t)bytes[0] | (uint16_t)((uint16_t)bytes[1] << 8);
}

static uint32_t read_u32le(const unsigned char *bytes) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static int cmd_test_wav(const char *goldendir, const char *output_path) {
    enum { LATENT_DIM = 32, HOP_LENGTH = 1920, SAMPLE_RATE = 48000 };
    char path[512];
    size_t latent_bytes = 0, waveform_bytes = 0;
    snprintf(path, sizeof(path), "%s/codec_latent_final.f32", goldendir);
    float *latent = read_file(path, &latent_bytes);
    snprintf(path, sizeof(path), "%s/waveform.f32", goldendir);
    float *waveform = read_file(path, &waveform_bytes);
    if (!latent || !waveform || latent_bytes % (sizeof(float) * LATENT_DIM) ||
        waveform_bytes % sizeof(float)) {
        fprintf(stderr, "fixture WAV tidak ada atau invalid\n");
        free(latent); free(waveform);
        return 1;
    }
    int frames = (int)(latent_bytes / (sizeof(float) * LATENT_DIM));
    size_t target_samples = waveform_bytes / sizeof(float);
    int flatten = iro_find_flattening_point(latent, frames, LATENT_DIM, 0.0f,
                                             20, 0.05f, 0.1f);
    size_t trimmed = iro_trimmed_sample_count(latent, frames, LATENT_DIM,
                                               target_samples, HOP_LENGTH,
                                               20, 0.05f, 0.1f);
    if (trimmed > target_samples ||
        iro_write_wav_pcm16(output_path, waveform, trimmed, SAMPLE_RATE) != 0) {
        fprintf(stderr, "gagal menulis WAV: %s\n", output_path);
        free(latent); free(waveform);
        return 1;
    }

    size_t wav_bytes = 0;
    unsigned char *wav = read_file(output_path, &wav_bytes);
    int valid = wav && wav_bytes == 44u + trimmed * 2u &&
                !memcmp(wav, "RIFF", 4) && !memcmp(wav + 8, "WAVEfmt ", 8) &&
                read_u32le(wav + 24) == SAMPLE_RATE &&
                read_u16le(wav + 34) == 16 && !memcmp(wav + 36, "data", 4) &&
                read_u32le(wav + 40) == trimmed * 2u;
    for (size_t i = 0; valid && i < trimmed; i++) {
        float sample = waveform[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        int16_t expected = (int16_t)lrintf(sample * 32767.0f);
        if ((int16_t)read_u16le(wav + 44u + i * 2u) != expected) valid = 0;
    }
    /* Synthetic padded tail makes sure the trim branch itself is exercised. */
    float tail_probe[5 * 2] = {1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
    int probe_flatten = iro_find_flattening_point(tail_probe, 5, 2, 0.0f,
                                                   2, 0.05f, 0.1f);
    valid = valid && flatten == frames && trimmed == target_samples &&
            probe_flatten == 3;
    printf("WAV PCM16: flatten %d/%d latent frame | %zu sample @ %d Hz | "
           "tail probe %d | gate: %s\n",
           flatten, frames, trimmed, SAMPLE_RATE, probe_flatten,
           valid ? "PASS" : "FAIL");
    free(wav); free(latent); free(waveform);
    return valid ? 0 : 1;
}

static double elapsed_seconds(struct timespec begin, struct timespec end) {
    return (double)(end.tv_sec - begin.tv_sec) +
           1e-9 * (double)(end.tv_nsec - begin.tv_nsec);
}

static int cmd_test_pipeline(const char *model_path, const char *decoder_path,
                             const char *goldendir, const char *output_path) {
    enum {
        TEXT_DIM = 512,
        LATENT_DIM = 32,
        STEPS = 8,
        HOP_LENGTH = 1920,
        SAMPLE_RATE = 48000,
    };
    int result = 1;
    char path[512];
    size_t text_bytes = 0, mask_bytes = 0, noise_bytes = 0;
    float *text = NULL, *noise = NULL, *latent = NULL;
    uint8_t *mask = NULL;
    float *latent_ref = NULL, *waveform_ref = NULL, *waveform = NULL;
    size_t latent_ref_bytes = 0, waveform_ref_bytes = 0;
    IroSafetensors model_st = {.fd = -1}, decoder_st = {.fd = -1};
    struct timespec sample_begin, sample_end, decode_begin, decode_end;

    snprintf(path, sizeof(path), "%s/text_state.f32", goldendir);
    text = read_file(path, &text_bytes);
    snprintf(path, sizeof(path), "%s/dit_b0_text_mask.u8", goldendir);
    mask = read_file(path, &mask_bytes);
    snprintf(path, sizeof(path), "%s/x_t_step000.f32", goldendir);
    noise = read_file(path, &noise_bytes);
    snprintf(path, sizeof(path), "%s/codec_latent_final.f32", goldendir);
    latent_ref = read_file(path, &latent_ref_bytes);
    snprintf(path, sizeof(path), "%s/waveform.f32", goldendir);
    waveform_ref = read_file(path, &waveform_ref_bytes);
    if (!text || text_bytes % (sizeof(float) * TEXT_DIM) || !mask || !noise ||
        noise_bytes % (2u * sizeof(float) * LATENT_DIM) || !latent_ref ||
        latent_ref_bytes % (sizeof(float) * LATENT_DIM) || !waveform_ref ||
        waveform_ref_bytes % sizeof(float)) {
        fprintf(stderr, "fixture pipeline tidak ada atau invalid\n");
        goto cleanup_pipeline;
    }
    int text_tokens = (int)(text_bytes / (sizeof(float) * TEXT_DIM));
    int latent_frames = (int)(latent_ref_bytes /
                              (sizeof(float) * LATENT_DIM));
    size_t latent_n = (size_t)latent_frames * LATENT_DIM;
    if (mask_bytes != (size_t)2 * text_tokens ||
        noise_bytes != 2u * sizeof(float) * latent_n)
        goto invalid_pipeline;
    latent = malloc(sizeof(float) * latent_n);
    if (!latent) goto cleanup_pipeline;

    if (iro_st_load(model_path, &model_st) != 0) goto cleanup_pipeline;
    IroDiT dit;
    if (iro_dit_init(&dit, &model_st) != 0) goto cleanup_pipeline;
    IroEulerConfig config = {
        .steps = STEPS,
        .init_scale = 0.999f,
        .cfg_scale_text = 3.0f,
        .cfg_min_t = 0.5f,
        .cfg_max_t = 1.0f,
    };
    IroEulerStats stats = {0};
    clock_gettime(CLOCK_MONOTONIC, &sample_begin);
    if (iro_sample_euler_text(&dit, text, mask, text_tokens, noise,
                              latent_frames, &config, latent, NULL, NULL,
                              &stats) != 0) {
        fprintf(stderr, "pipeline: Euler sampling gagal\n");
        goto cleanup_pipeline;
    }
    clock_gettime(CLOCK_MONOTONIC, &sample_end);
    iro_st_free(&model_st);
    memset(&model_st, 0, sizeof(model_st));
    model_st.fd = -1;

    double latent_error = 0.0;
    for (size_t i = 0; i < latent_n; i++) {
        double error = fabs((double)latent[i] - latent_ref[i]);
        if (error > latent_error) latent_error = error;
    }

    if (iro_st_load(decoder_path, &decoder_st) != 0) goto cleanup_pipeline;
    IroDACVAEDecoder decoder;
    if (iro_dacvae_init(&decoder, &decoder_st) != 0) goto cleanup_pipeline;
    int decoded_samples = iro_dacvae_output_length(latent_frames);
    waveform = malloc(sizeof(float) * (size_t)decoded_samples);
    if (!waveform) goto cleanup_pipeline;
    clock_gettime(CLOCK_MONOTONIC, &decode_begin);
    if (iro_dacvae_decode(&decoder, latent, latent_frames, waveform,
                          NULL, NULL) != 0) {
        fprintf(stderr, "pipeline: DACVAE decode gagal\n");
        goto cleanup_pipeline;
    }
    clock_gettime(CLOCK_MONOTONIC, &decode_end);

    size_t target_samples = waveform_ref_bytes / sizeof(float);
    if (target_samples > (size_t)decoded_samples)
        target_samples = (size_t)decoded_samples;
    size_t output_samples = iro_trimmed_sample_count(
        latent, latent_frames, LATENT_DIM, target_samples, HOP_LENGTH,
        20, 0.05f, 0.1f);
    double waveform_max = 0.0, waveform_abs = 0.0;
    double dot = 0.0, norm = 0.0, ref_norm = 0.0;
    for (size_t i = 0; i < output_samples; i++) {
        double got = waveform[i], expected = waveform_ref[i];
        double error = fabs(got - expected);
        if (error > waveform_max) waveform_max = error;
        waveform_abs += error;
        dot += got * expected;
        norm += got * got;
        ref_norm += expected * expected;
    }
    double waveform_mae = output_samples
                              ? waveform_abs / (double)output_samples
                              : INFINITY;
    double cosine = norm > 0.0 && ref_norm > 0.0
                        ? dot / sqrt(norm * ref_norm)
                        : 0.0;
    if (iro_write_wav_pcm16(output_path, waveform, output_samples,
                            SAMPLE_RATE) != 0) {
        fprintf(stderr, "pipeline: gagal menulis %s\n", output_path);
        goto cleanup_pipeline;
    }
    double sample_secs = elapsed_seconds(sample_begin, sample_end);
    double decode_secs = elapsed_seconds(decode_begin, decode_end);
    double audio_secs = (double)output_samples / SAMPLE_RATE;
    double rtf = audio_secs > 0.0 ? (sample_secs + decode_secs) / audio_secs
                                  : INFINITY;
    int pass = latent_error < 1e-2 && waveform_mae < 2e-3 && cosine > 0.99;
    int backend_threads = iro_ops_get_threads();
    printf("Pipeline %s threads=%d: Euler %.3f s + decode %.3f s = %.3f s | "
           "audio %.3f s | RTF %.3f\n",
           iro_ops_backend_name(), backend_threads > 0 ? backend_threads : 1,
           sample_secs, decode_secs,
           sample_secs + decode_secs, audio_secs, rtf);
    printf("latent max %.6g | waveform max %.6g MAE %.6g cosine %.8f | "
           "%zu sample | gate: %s\n",
           latent_error, waveform_max, waveform_mae, cosine, output_samples,
           pass ? "PASS" : "FAIL");
    result = pass ? 0 : 1;
    goto cleanup_pipeline;

invalid_pipeline:
    fprintf(stderr, "shape fixture pipeline tidak cocok\n");
cleanup_pipeline:
    iro_st_free(&model_st);
    iro_st_free(&decoder_st);
    free(text); free(mask); free(noise); free(latent); free(latent_ref);
    free(waveform_ref); free(waveform);
    return result;
}

static int cmd_generate_text(int argc, char **argv) {
    if (argc < 3) return 1;
    const char *model = getenv("IRO_MODEL");
    const char *tokenizer = getenv("IRO_TOKENIZER");
    const char *decoder = getenv("IRO_DECODER");
    const char *encoder = getenv("IRO_ENCODER");
    IroGenerateConfig config = {
        .model_path = model ? model : "weights/model.safetensors",
        .tokenizer_path = tokenizer ? tokenizer : "weights/tokenizer.bin",
        .decoder_path = decoder ? decoder : "weights/dacvae_decoder.safetensors",
        .encoder_path = encoder ? encoder : "weights/dacvae_encoder.safetensors",
        .text = argv[2],
        .output_path = "out.wav",
        .seed = 42,
        .steps = 40,
    };
    for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc) {
            fprintf(stderr, "generate: nilai tidak ada untuk %s\n", argv[i]);
            return 1;
        }
        const char *option = argv[i], *value = argv[i + 1];
        if (!strcmp(option, "--model")) config.model_path = value;
        else if (!strcmp(option, "--tokenizer")) config.tokenizer_path = value;
        else if (!strcmp(option, "--decoder")) config.decoder_path = value;
        else if (!strcmp(option, "--encoder")) config.encoder_path = value;
        else if (!strcmp(option, "--ref")) config.reference_path = value;
        else if (!strcmp(option, "--caption")) config.caption = value;
        else if (!strcmp(option, "--out")) config.output_path = value;
        else if (!strcmp(option, "--noise")) config.noise_path = value;
        else if (!strcmp(option, "--dump-dir")) config.dump_dir = value;
        else if (!strcmp(option, "--seed")) {
            char *end = NULL;
            unsigned long long seed = strtoull(value, &end, 10);
            if (end == value || *end) {
                fprintf(stderr, "generate: seed tidak valid: %s\n", value);
                return 1;
            }
            config.seed = (uint64_t)seed;
        } else if (!strcmp(option, "--steps")) {
            char *end = NULL;
            long steps = strtol(value, &end, 10);
            if (end == value || *end || steps <= 0 || steps > 1000) {
                fprintf(stderr, "generate: steps harus integer 1..1000\n");
                return 1;
            }
            config.steps = (int)steps;
        } else {
            fprintf(stderr, "generate: opsi tidak dikenal: %s\n", option);
            return 1;
        }
    }
    IroGenerateStats stats = {0};
    if (iro_generate_text(&config, &stats) != 0) {
        fprintf(stderr, "generate text-to-audio gagal\n");
        return 1;
    }
    double audio_seconds = (double)stats.output_samples / 48000.0;
    double total = stats.encode_seconds + stats.sample_seconds +
                   stats.decode_seconds;
    printf("Generated %s: %d token%s%s -> %d latent frame -> %.3f s audio\n",
           config.output_path, stats.tokens,
           config.reference_path ? " + speaker reference" : "",
           config.caption && config.caption[0] ? " + caption" : "",
           stats.latent_frames, audio_seconds);
    printf("encode %.3f s | Euler %.3f s | decode %.3f s | total %.3f s | "
           "RTF %.3f | seed %llu | steps %d\n",
           stats.encode_seconds, stats.sample_seconds, stats.decode_seconds,
           total, audio_seconds > 0.0 ? total / audio_seconds : INFINITY,
           (unsigned long long)config.seed, config.steps);
    printf("mmap front-end dilepas sebelum DiT: %.1f MiB\n",
           (double)stats.evicted_model_bytes / (1024.0 * 1024.0));
    return 0;
}

int main(int argc, char **argv) {
    const char *threads_env = getenv("IRO_NUM_THREADS");
    if (threads_env) {
        char *end = NULL;
        long threads = strtol(threads_env, &end, 10);
        if (end == threads_env || *end || threads <= 0 || threads > 1024) {
            fprintf(stderr, "IRO_NUM_THREADS harus integer 1..1024\n");
            return 1;
        }
        iro_ops_set_threads((int)threads);
    }
    if (argc >= 2 && !strcmp(argv[1], "--text")) {
        int result = cmd_generate_text(argc, argv);
        if (result && argc < 3) usage();
        return result;
    }
    if (argc == 2 && !strcmp(argv[1], "--bench-linear"))
        return cmd_bench_linear();
    if (argc == 4 && !strcmp(argv[1], "--tokenize")) return cmd_tokenize(argv[2], argv[3]);
    if (argc == 4 && !strcmp(argv[1], "--test-tokenizer"))
        return cmd_test_tokenizer(argv[2], argv[3]);
    if (argc == 4 && !strcmp(argv[1], "--test-backbone"))
        return cmd_test_backbone(argv[2], argv[3]);
    if (argc == 4 && !strcmp(argv[1], "--test-projector"))
        return cmd_test_projector(argv[2], argv[3]);
    if (argc == 4 && !strcmp(argv[1], "--test-duration"))
        return cmd_test_duration(argv[2], argv[3]);
    if (argc == 4 && !strcmp(argv[1], "--test-duration-vectors"))
        return cmd_test_duration_vectors(argv[2], argv[3]);
    if (argc == 4 && !strcmp(argv[1], "--test-dit-prepare"))
        return cmd_test_dit_prepare(argv[2], argv[3]);
    if (argc == 4 && !strcmp(argv[1], "--test-dit-adaln"))
        return cmd_test_dit_adaln(argv[2], argv[3]);
    if (argc == 4 && !strcmp(argv[1], "--test-dit-attention"))
        return cmd_test_dit_attention(argv[2], argv[3]);
    if (argc == 4 && !strcmp(argv[1], "--test-dit-mlp"))
        return cmd_test_dit_mlp(argv[2], argv[3]);
    if (argc == 4 && !strcmp(argv[1], "--test-dit-forward"))
        return cmd_test_dit_forward(argv[2], argv[3]);
    if (argc == 4 && !strcmp(argv[1], "--test-euler"))
        return cmd_test_euler(argv[2], argv[3]);
    if (argc == 4 && !strcmp(argv[1], "--test-euler-speaker"))
        return cmd_test_euler_speaker(argv[2], argv[3]);
    if (argc == 4 && !strcmp(argv[1], "--test-codec"))
        return cmd_test_codec(argv[2], argv[3]);
    if (argc == 4 && !strcmp(argv[1], "--test-encoder"))
        return cmd_test_encoder(argv[2], argv[3]);
    if (argc == 4 && !strcmp(argv[1], "--test-speaker"))
        return cmd_test_speaker(argv[2], argv[3]);
    if (argc == 5 && !strcmp(argv[1], "--encode-reference"))
        return cmd_encode_reference(argv[2], argv[3], argv[4]);
    if (argc == 4 && !strcmp(argv[1], "--test-wav"))
        return cmd_test_wav(argv[2], argv[3]);
    if (argc == 6 && !strcmp(argv[1], "--test-pipeline"))
        return cmd_test_pipeline(argv[2], argv[3], argv[4], argv[5]);
    if (argc != 3) { usage(); return 1; }
    if (!strcmp(argv[1], "--list-tensors")) return cmd_list(argv[2]);
    if (!strcmp(argv[1], "--info")) return cmd_info(argv[2]);
    if (!strcmp(argv[1], "--check")) return cmd_check(argv[2]);
    usage();
    return 1;
}
