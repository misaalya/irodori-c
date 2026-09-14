#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "irodori.h"
#include "ops.h"

enum {
    MODEL_DIM = 1280,
    QKV_DIM = 3840,
    QKVG_DIM = 5120,
};

typedef struct {
    double max_abs;
    double mae;
} ErrorStats;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int parse_positive_int(const char *text, int *out) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || !end || *end != '\0' || value <= 0 || value > 1000000)
        return -1;
    *out = (int)value;
    return 0;
}

static float *load_activation_rows(const char *path, int *rows_out) {
    struct stat info;
    if (stat(path, &info) != 0 || info.st_size <= 0 ||
        (uint64_t)info.st_size % (sizeof(float) * MODEL_DIM) != 0) {
        fprintf(stderr, "bench_fused_qkvg: ukuran aktivasi tidak valid: %s\n",
                path);
        return NULL;
    }
    uint64_t rows64 = (uint64_t)info.st_size / (sizeof(float) * MODEL_DIM);
    if (rows64 > 1000000) {
        fprintf(stderr, "bench_fused_qkvg: aktivasi terlalu besar\n");
        return NULL;
    }
    size_t count = (size_t)rows64 * MODEL_DIM;
    float *dst = malloc(sizeof(float) * count);
    if (!dst) return NULL;
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "bench_fused_qkvg: gagal membuka %s\n", path);
        free(dst);
        return NULL;
    }
    size_t got = fread(dst, sizeof(float), count, fp);
    int extra = fgetc(fp);
    int failed = got != count || extra != EOF || ferror(fp);
    fclose(fp);
    if (failed) {
        fprintf(stderr, "bench_fused_qkvg: gagal membaca aktivasi lengkap\n");
        free(dst);
        return NULL;
    }
    *rows_out = (int)rows64;
    return dst;
}

static const float *require_weight(const IroSafetensors *st, const char *name) {
    const IroTensor *tensor = iro_st_get(st, name);
    if (!tensor || tensor->dtype != IRO_ST_F32 || tensor->ndims != 2 ||
        tensor->shape[0] != MODEL_DIM || tensor->shape[1] != MODEL_DIM) {
        fprintf(stderr, "bench_fused_qkvg: tensor tidak cocok: %s\n", name);
        return NULL;
    }
    return (const float *)tensor->data;
}

static void fill_rows(float *dst, int rows, const float *seed, int seed_rows) {
    for (int row = 0; row < rows; row++) {
        memcpy(dst + (size_t)row * MODEL_DIM,
               seed + (size_t)(row % seed_rows) * MODEL_DIM,
               sizeof(float) * MODEL_DIM);
    }
}

static void deinterleave_qkvg(const float *src, float *q, float *k, float *v,
                              float *gate, int rows) {
    for (int row = 0; row < rows; row++) {
        const float *s = src + (size_t)row * QKVG_DIM;
        size_t off = (size_t)row * MODEL_DIM;
        memcpy(q + off, s, sizeof(float) * MODEL_DIM);
        memcpy(k + off, s + MODEL_DIM, sizeof(float) * MODEL_DIM);
        memcpy(v + off, s + 2 * MODEL_DIM, sizeof(float) * MODEL_DIM);
        memcpy(gate + off, s + 3 * MODEL_DIM, sizeof(float) * MODEL_DIM);
    }
}

static void deinterleave_qkv(const float *src, float *q, float *k, float *v,
                             int rows) {
    for (int row = 0; row < rows; row++) {
        const float *s = src + (size_t)row * QKV_DIM;
        size_t off = (size_t)row * MODEL_DIM;
        memcpy(q + off, s, sizeof(float) * MODEL_DIM);
        memcpy(k + off, s + MODEL_DIM, sizeof(float) * MODEL_DIM);
        memcpy(v + off, s + 2 * MODEL_DIM, sizeof(float) * MODEL_DIM);
    }
}

static void run_separate(const float *x,
                         const float *wq, const float *wk,
                         const float *wv, const float *gate_w,
                         float *q, float *k, float *v, float *gate,
                         int rows) {
    iro_linear(x, wq, NULL, q, rows, MODEL_DIM, MODEL_DIM);
    iro_linear(x, wk, NULL, k, rows, MODEL_DIM, MODEL_DIM);
    iro_linear(x, wv, NULL, v, rows, MODEL_DIM, MODEL_DIM);
    iro_linear(x, gate_w, NULL, gate, rows, MODEL_DIM, MODEL_DIM);
}

static void run_qkvg(const float *x, const float *packed,
                     float *fused, float *q, float *k, float *v, float *gate,
                     int rows) {
    iro_linear(x, packed, NULL, fused, rows, MODEL_DIM, QKVG_DIM);
    deinterleave_qkvg(fused, q, k, v, gate, rows);
}

static void run_qkv_gate(const float *x, const float *packed,
                         const float *gate_w, float *fused,
                         float *q, float *k, float *v, float *gate,
                         int rows) {
    iro_linear(x, packed, NULL, fused, rows, MODEL_DIM, QKV_DIM);
    iro_linear(x, gate_w, NULL, gate, rows, MODEL_DIM, MODEL_DIM);
    deinterleave_qkv(fused, q, k, v, rows);
}

static int cmp_double(const void *lhs, const void *rhs) {
    double a = *(const double *)lhs;
    double b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static double median_copy(const double *values, int count) {
    double *copy = malloc(sizeof(double) * (size_t)count);
    if (!copy) return NAN;
    memcpy(copy, values, sizeof(double) * (size_t)count);
    qsort(copy, (size_t)count, sizeof(double), cmp_double);
    double result = count & 1 ? copy[count / 2]
                              : 0.5 * (copy[count / 2 - 1] + copy[count / 2]);
    free(copy);
    return result;
}

static double checksum4(const float *q, const float *k, const float *v,
                        const float *gate, size_t count) {
    const float *sets[4] = {q, k, v, gate};
    double sum = 0.0;
    size_t stride = count / 4096 + 1;
    for (int set = 0; set < 4; set++) {
        for (size_t i = 0; i < count; i += stride)
            sum += (double)sets[set][i] * (double)(1 + ((i + set) % 17));
    }
    return sum;
}

static ErrorStats error4(const float *bq, const float *bk,
                         const float *bv, const float *bg,
                         const float *cq, const float *ck,
                         const float *cv, const float *cg,
                         size_t count) {
    const float *base[4] = {bq, bk, bv, bg};
    const float *cand[4] = {cq, ck, cv, cg};
    ErrorStats stats = {0.0, 0.0};
    for (int set = 0; set < 4; set++) {
        for (size_t i = 0; i < count; i++) {
            double err = fabs((double)base[set][i] - (double)cand[set][i]);
            if (err > stats.max_abs) stats.max_abs = err;
            stats.mae += err;
        }
    }
    stats.mae /= (double)(count * 4);
    return stats;
}

static int bench_shape(const float *seed, int seed_rows,
                       const float *wq, const float *wk, const float *wv,
                       const float *gate_w, const float *packed,
                       int rows, int pairs) {
    const size_t count = (size_t)rows * MODEL_DIM;
    const size_t qkvg_count = (size_t)rows * QKVG_DIM;
    float *x = malloc(sizeof(float) * count);
    float *bq = malloc(sizeof(float) * count);
    float *bk = malloc(sizeof(float) * count);
    float *bv = malloc(sizeof(float) * count);
    float *bg = malloc(sizeof(float) * count);
    float *cq = malloc(sizeof(float) * count);
    float *ck = malloc(sizeof(float) * count);
    float *cv = malloc(sizeof(float) * count);
    float *cg = malloc(sizeof(float) * count);
    float *fused = malloc(sizeof(float) * qkvg_count);
    double *base_times = malloc(sizeof(double) * (size_t)pairs);
    double *candidate_times = malloc(sizeof(double) * (size_t)pairs);
    if (!x || !bq || !bk || !bv || !bg || !cq || !ck || !cv || !cg ||
        !fused || !base_times || !candidate_times) {
        fprintf(stderr, "bench_fused_qkvg: alokasi gagal untuk M=%d\n", rows);
        free(x); free(bq); free(bk); free(bv); free(bg);
        free(cq); free(ck); free(cv); free(cg); free(fused);
        free(base_times); free(candidate_times);
        return -1;
    }

    fill_rows(x, rows, seed, seed_rows);
    run_separate(x, wq, wk, wv, gate_w, bq, bk, bv, bg, rows);
    run_qkvg(x, packed, fused, cq, ck, cv, cg, rows);
    ErrorStats qkvg_error = error4(bq, bk, bv, bg, cq, ck, cv, cg, count);
    double qkvg_checksum = checksum4(cq, ck, cv, cg, count);

    for (int pair = 0; pair < pairs; pair++) {
        double start;
        if ((pair & 1) == 0) {
            start = now_seconds();
            run_separate(x, wq, wk, wv, gate_w, bq, bk, bv, bg, rows);
            base_times[pair] = now_seconds() - start;
            start = now_seconds();
            run_qkvg(x, packed, fused, cq, ck, cv, cg, rows);
            candidate_times[pair] = now_seconds() - start;
        } else {
            start = now_seconds();
            run_qkvg(x, packed, fused, cq, ck, cv, cg, rows);
            candidate_times[pair] = now_seconds() - start;
            start = now_seconds();
            run_separate(x, wq, wk, wv, gate_w, bq, bk, bv, bg, rows);
            base_times[pair] = now_seconds() - start;
        }
        printf("{\"type\":\"qkvg_pair\",\"backend\":\"%s\","
               "\"M\":%d,\"pair\":%d,\"separate_seconds\":%.9f,"
               "\"candidate_seconds\":%.9f,\"speedup\":%.6f}\n",
               iro_ops_backend_name(), rows, pair + 1,
               base_times[pair], candidate_times[pair],
               base_times[pair] / candidate_times[pair]);
    }

    double base_median = median_copy(base_times, pairs);
    double candidate_median = median_copy(candidate_times, pairs);
    printf("{\"type\":\"qkvg_summary\",\"backend\":\"%s\","
           "\"M\":%d,\"pairs\":%d,\"separate_median_seconds\":%.9f,"
           "\"candidate_median_seconds\":%.9f,\"speedup\":%.6f,"
           "\"max_abs\":%.9g,\"mae\":%.9g,\"checksum\":%.17g}\n",
           iro_ops_backend_name(), rows, pairs, base_median, candidate_median,
           base_median / candidate_median, qkvg_error.max_abs,
           qkvg_error.mae, qkvg_checksum);

    run_separate(x, wq, wk, wv, gate_w, bq, bk, bv, bg, rows);
    run_qkv_gate(x, packed, gate_w, fused, cq, ck, cv, cg, rows);
    ErrorStats qkv_error = error4(bq, bk, bv, bg, cq, ck, cv, cg, count);
    double qkv_checksum = checksum4(cq, ck, cv, cg, count);

    for (int pair = 0; pair < pairs; pair++) {
        double start;
        if ((pair & 1) == 0) {
            start = now_seconds();
            run_separate(x, wq, wk, wv, gate_w, bq, bk, bv, bg, rows);
            base_times[pair] = now_seconds() - start;
            start = now_seconds();
            run_qkv_gate(x, packed, gate_w, fused, cq, ck, cv, cg, rows);
            candidate_times[pair] = now_seconds() - start;
        } else {
            start = now_seconds();
            run_qkv_gate(x, packed, gate_w, fused, cq, ck, cv, cg, rows);
            candidate_times[pair] = now_seconds() - start;
            start = now_seconds();
            run_separate(x, wq, wk, wv, gate_w, bq, bk, bv, bg, rows);
            base_times[pair] = now_seconds() - start;
        }
        printf("{\"type\":\"qkv_gate_pair\",\"backend\":\"%s\","
               "\"M\":%d,\"pair\":%d,\"separate_seconds\":%.9f,"
               "\"candidate_seconds\":%.9f,\"speedup\":%.6f}\n",
               iro_ops_backend_name(), rows, pair + 1,
               base_times[pair], candidate_times[pair],
               base_times[pair] / candidate_times[pair]);
    }

    base_median = median_copy(base_times, pairs);
    candidate_median = median_copy(candidate_times, pairs);
    printf("{\"type\":\"qkv_gate_summary\",\"backend\":\"%s\","
           "\"M\":%d,\"pairs\":%d,\"separate_median_seconds\":%.9f,"
           "\"candidate_median_seconds\":%.9f,\"speedup\":%.6f,"
           "\"max_abs\":%.9g,\"mae\":%.9g,\"checksum\":%.17g}\n",
           iro_ops_backend_name(), rows, pairs, base_median, candidate_median,
           base_median / candidate_median, qkv_error.max_abs,
           qkv_error.mae, qkv_checksum);

    free(x); free(bq); free(bk); free(bv); free(bg);
    free(cq); free(ck); free(cv); free(cg); free(fused);
    free(base_times); free(candidate_times);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr,
                "usage: %s MODEL.safetensors dit_b0_attn_adaln_h.f32 [pairs]\n",
                argv[0]);
        return 2;
    }
    int pairs = 5;
    if (argc == 4 && parse_positive_int(argv[3], &pairs) != 0) {
        fprintf(stderr, "bench_fused_qkvg: pairs harus integer positif\n");
        return 2;
    }
    if (pairs < 3) {
        fprintf(stderr, "bench_fused_qkvg: minimal 3 pasangan A/B\n");
        return 2;
    }

    int threads = 2;
    const char *threads_env = getenv("IRO_NUM_THREADS");
    if (threads_env && parse_positive_int(threads_env, &threads) != 0) {
        fprintf(stderr, "bench_fused_qkvg: IRO_NUM_THREADS tidak valid\n");
        return 2;
    }
    iro_ops_set_threads(threads);
    fprintf(stderr, "bench_fused_qkvg: backend=%s threads=%d active=%d\n",
            iro_ops_backend_name(), threads, iro_ops_get_threads());

    IroSafetensors st = {.fd = -1};
    if (iro_st_load(argv[1], &st) != 0) return 1;
    const float *wq = require_weight(&st, "blocks.0.attention.wq.weight");
    const float *wk = require_weight(&st, "blocks.0.attention.wk.weight");
    const float *wv = require_weight(&st, "blocks.0.attention.wv.weight");
    const float *gate_w = require_weight(&st, "blocks.0.attention.gate.weight");
    if (!wq || !wk || !wv || !gate_w) {
        iro_st_free(&st);
        return 1;
    }

    int seed_rows = 0;
    float *seed = load_activation_rows(argv[2], &seed_rows);
    float *packed = malloc(sizeof(float) * (size_t)QKVG_DIM * MODEL_DIM);
    if (!seed || !packed) {
        fprintf(stderr, "bench_fused_qkvg: alokasi seed/packed gagal\n");
        free(seed); free(packed); iro_st_free(&st);
        return 1;
    }

    double pack_start = now_seconds();
    memcpy(packed, wq, sizeof(float) * (size_t)MODEL_DIM * MODEL_DIM);
    memcpy(packed + (size_t)MODEL_DIM * MODEL_DIM, wk,
           sizeof(float) * (size_t)MODEL_DIM * MODEL_DIM);
    memcpy(packed + (size_t)2 * MODEL_DIM * MODEL_DIM, wv,
           sizeof(float) * (size_t)MODEL_DIM * MODEL_DIM);
    memcpy(packed + (size_t)3 * MODEL_DIM * MODEL_DIM, gate_w,
           sizeof(float) * (size_t)MODEL_DIM * MODEL_DIM);
    double pack_seconds = now_seconds() - pack_start;
    printf("{\"type\":\"qkvg_pack\",\"backend\":\"%s\","
           "\"bytes\":%zu,\"seconds\":%.9f}\n",
           iro_ops_backend_name(),
           sizeof(float) * (size_t)QKVG_DIM * MODEL_DIM, pack_seconds);

    const int shapes[] = {112, 116, 224, 232, 448, 464};
    int failed = 0;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
        failed |= bench_shape(seed, seed_rows, wq, wk, wv, gate_w, packed,
                              shapes[i], pairs) != 0;

    free(seed);
    free(packed);
    iro_st_free(&st);
    return failed ? 1 : 0;
}
