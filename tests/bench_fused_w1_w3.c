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
    MLP_DIM = 3680,
    FUSED_DIM = 7360,
};

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
        fprintf(stderr, "bench_fused_w1_w3: ukuran aktivasi tidak valid: %s\n",
                path);
        return NULL;
    }
    uint64_t rows64 = (uint64_t)info.st_size / (sizeof(float) * MODEL_DIM);
    if (rows64 > 1000000) {
        fprintf(stderr, "bench_fused_w1_w3: aktivasi terlalu besar\n");
        return NULL;
    }
    size_t count = (size_t)rows64 * MODEL_DIM;
    float *dst = malloc(sizeof(float) * count);
    if (!dst) return NULL;
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "bench_fused_w1_w3: gagal membuka %s\n", path);
        free(dst);
        return NULL;
    }
    size_t got = fread(dst, sizeof(float), count, fp);
    int extra = fgetc(fp);
    int failed = got != count || extra != EOF || ferror(fp);
    fclose(fp);
    if (failed) {
        fprintf(stderr, "bench_fused_w1_w3: gagal membaca aktivasi lengkap\n");
        free(dst);
        return NULL;
    }
    *rows_out = (int)rows64;
    return dst;
}

static const float *require_weight(const IroSafetensors *st, const char *name) {
    const IroTensor *tensor = iro_st_get(st, name);
    if (!tensor || tensor->dtype != IRO_ST_F32 || tensor->ndims != 2 ||
        tensor->shape[0] != MLP_DIM || tensor->shape[1] != MODEL_DIM) {
        fprintf(stderr, "bench_fused_w1_w3: tensor tidak cocok: %s\n", name);
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

static void fuse_silu_gate_strided(const float *fused, float *hidden, int rows) {
    for (int row = 0; row < rows; row++) {
        const float *src = fused + (size_t)row * FUSED_DIM;
        float *dst = hidden + (size_t)row * MLP_DIM;
        for (int col = 0; col < MLP_DIM; col++)
            dst[col] = iro_silu(src[col]) * src[MLP_DIM + col];
    }
}

static void run_separate(const float *x, const float *w1, const float *w3,
                         float *a, float *b, int rows) {
    iro_linear(x, w1, NULL, a, rows, MODEL_DIM, MLP_DIM);
    iro_linear(x, w3, NULL, b, rows, MODEL_DIM, MLP_DIM);
    iro_silu_mul_inplace(a, b, (size_t)rows * MLP_DIM);
}

static void run_fused(const float *x, const float *packed, float *fused,
                      float *hidden, int rows) {
    iro_linear(x, packed, NULL, fused, rows, MODEL_DIM, FUSED_DIM);
    fuse_silu_gate_strided(fused, hidden, rows);
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

static double checksum(const float *values, size_t count) {
    double sum = 0.0;
    size_t stride = count / 4096 + 1;
    for (size_t i = 0; i < count; i += stride)
        sum += (double)values[i] * (double)(1 + (i % 17));
    return sum;
}

static int bench_shape(const float *seed, int seed_rows,
                       const float *w1, const float *w3, const float *packed,
                       int rows, int pairs) {
    const size_t x_count = (size_t)rows * MODEL_DIM;
    const size_t hidden_count = (size_t)rows * MLP_DIM;
    const size_t fused_count = (size_t)rows * FUSED_DIM;
    float *x = malloc(sizeof(float) * x_count);
    float *base = malloc(sizeof(float) * hidden_count);
    float *gate = malloc(sizeof(float) * hidden_count);
    float *fused = malloc(sizeof(float) * fused_count);
    float *candidate = malloc(sizeof(float) * hidden_count);
    double *base_times = malloc(sizeof(double) * (size_t)pairs);
    double *candidate_times = malloc(sizeof(double) * (size_t)pairs);
    if (!x || !base || !gate || !fused || !candidate ||
        !base_times || !candidate_times) {
        fprintf(stderr, "bench_fused_w1_w3: alokasi gagal untuk M=%d\n", rows);
        free(x); free(base); free(gate); free(fused); free(candidate);
        free(base_times); free(candidate_times);
        return -1;
    }

    fill_rows(x, rows, seed, seed_rows);
    run_separate(x, w1, w3, base, gate, rows);
    run_fused(x, packed, fused, candidate, rows);

    double max_abs = 0.0;
    double mae = 0.0;
    for (size_t i = 0; i < hidden_count; i++) {
        double err = fabs((double)base[i] - (double)candidate[i]);
        if (err > max_abs) max_abs = err;
        mae += err;
    }
    mae /= (double)hidden_count;

    for (int pair = 0; pair < pairs; pair++) {
        double start;
        if ((pair & 1) == 0) {
            start = now_seconds();
            run_separate(x, w1, w3, base, gate, rows);
            base_times[pair] = now_seconds() - start;
            start = now_seconds();
            run_fused(x, packed, fused, candidate, rows);
            candidate_times[pair] = now_seconds() - start;
        } else {
            start = now_seconds();
            run_fused(x, packed, fused, candidate, rows);
            candidate_times[pair] = now_seconds() - start;
            start = now_seconds();
            run_separate(x, w1, w3, base, gate, rows);
            base_times[pair] = now_seconds() - start;
        }
        printf("{\"type\":\"w1_w3_pair\",\"backend\":\"%s\","
               "\"M\":%d,\"pair\":%d,\"separate_seconds\":%.9f,"
               "\"fused_seconds\":%.9f,\"speedup\":%.6f}\n",
               iro_ops_backend_name(), rows, pair + 1,
               base_times[pair], candidate_times[pair],
               base_times[pair] / candidate_times[pair]);
    }

    double base_median = median_copy(base_times, pairs);
    double candidate_median = median_copy(candidate_times, pairs);
    printf("{\"type\":\"w1_w3_summary\",\"backend\":\"%s\","
           "\"M\":%d,\"pairs\":%d,\"separate_median_seconds\":%.9f,"
           "\"fused_median_seconds\":%.9f,\"speedup\":%.6f,"
           "\"max_abs\":%.9g,\"mae\":%.9g,\"checksum\":%.17g}\n",
           iro_ops_backend_name(), rows, pairs, base_median, candidate_median,
           base_median / candidate_median, max_abs, mae,
           checksum(candidate, hidden_count));

    free(x); free(base); free(gate); free(fused); free(candidate);
    free(base_times); free(candidate_times);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr,
                "usage: %s MODEL.safetensors dit_b0_mlp_adaln_h.f32 [pairs]\n",
                argv[0]);
        return 2;
    }
    int pairs = 5;
    if (argc == 4 && parse_positive_int(argv[3], &pairs) != 0) {
        fprintf(stderr, "bench_fused_w1_w3: pairs harus integer positif\n");
        return 2;
    }
    if (pairs < 3) {
        fprintf(stderr, "bench_fused_w1_w3: minimal 3 pasangan A/B\n");
        return 2;
    }

    int threads = 2;
    const char *threads_env = getenv("IRO_NUM_THREADS");
    if (threads_env && parse_positive_int(threads_env, &threads) != 0) {
        fprintf(stderr, "bench_fused_w1_w3: IRO_NUM_THREADS tidak valid\n");
        return 2;
    }
    iro_ops_set_threads(threads);
    fprintf(stderr, "bench_fused_w1_w3: backend=%s threads=%d active=%d\n",
            iro_ops_backend_name(), threads, iro_ops_get_threads());

    IroSafetensors st = {.fd = -1};
    if (iro_st_load(argv[1], &st) != 0) return 1;
    const float *w1 = require_weight(&st, "blocks.0.mlp.w1.weight");
    const float *w3 = require_weight(&st, "blocks.0.mlp.w3.weight");
    if (!w1 || !w3) {
        iro_st_free(&st);
        return 1;
    }

    int seed_rows = 0;
    float *seed = load_activation_rows(argv[2], &seed_rows);
    float *packed = malloc(sizeof(float) * (size_t)FUSED_DIM * MODEL_DIM);
    if (!seed || !packed) {
        fprintf(stderr, "bench_fused_w1_w3: alokasi seed/packed gagal\n");
        free(seed); free(packed); iro_st_free(&st);
        return 1;
    }

    double pack_start = now_seconds();
    memcpy(packed, w1, sizeof(float) * (size_t)MLP_DIM * MODEL_DIM);
    memcpy(packed + (size_t)MLP_DIM * MODEL_DIM, w3,
           sizeof(float) * (size_t)MLP_DIM * MODEL_DIM);
    double pack_seconds = now_seconds() - pack_start;
    printf("{\"type\":\"w1_w3_pack\",\"backend\":\"%s\","
           "\"bytes\":%zu,\"seconds\":%.9f}\n",
           iro_ops_backend_name(),
           sizeof(float) * (size_t)FUSED_DIM * MODEL_DIM, pack_seconds);

    const int shapes[] = {112, 116, 224, 232, 448, 464};
    int failed = 0;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
        failed |= bench_shape(seed, seed_rows, w1, w3, packed,
                              shapes[i], pairs) != 0;

    free(seed);
    free(packed);
    iro_st_free(&st);
    return failed ? 1 : 0;
}
