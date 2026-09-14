#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mkl.h>
#include "irodori.h"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Standalone public-API screen. Packing is separately timed and costed;
   this is kernel evidence only, never an E2E acceptance benchmark. */
int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL dit_b0_mlp_adaln_h.f32\n", argv[0]);
        return 2;
    }
    mkl_set_num_threads(2);
    mkl_set_dynamic(0);
    MKLVersion version;
    mkl_get_version(&version);
    fprintf(stderr, "oneMKL %d.%d.%d threads=%d\n", version.MajorVersion,
            version.MinorVersion, version.UpdateVersion, mkl_get_max_threads());
    IroSafetensors st = {.fd = -1};
    if (iro_st_load(argv[1], &st)) return 1;
    const IroTensor *tensor = iro_st_get(&st, "blocks.0.mlp.w1.weight");
    if (!tensor || tensor->dtype != IRO_ST_F32 || tensor->ndims != 2 ||
        tensor->shape[0] != 3680 || tensor->shape[1] != 1280) {
        iro_st_free(&st);
        return 1;
    }
    const float *w = (const float *)tensor->data;
    const int n = 3680, k = 1280, seed_rows = 112;
    float *seed = malloc((size_t)seed_rows * k * sizeof(float));
    FILE *f = fopen(argv[2], "rb");
    if (!seed || !f) { free(seed); if (f) fclose(f); iro_st_free(&st); return 1; }
    size_t got = fread(seed, sizeof(float), (size_t)seed_rows * k, f);
    int bad = ferror(f);
    fclose(f);
    if (bad || got != (size_t)seed_rows * k) { free(seed); iro_st_free(&st); return 1; }
    const int shapes[] = {30, 112, 224, 345, 448, 464};
    int status = 0;
    for (size_t shape = 0; shape < sizeof(shapes)/sizeof(shapes[0]); shape++) {
        int m = shapes[shape];
        size_t bytes = cblas_sgemm_pack_get_size(CblasBMatrix, m, n, k);
        float *packed = mkl_malloc(bytes, 64);
        float *x = mkl_malloc((size_t)m * k * sizeof(float), 64);
        float *a = mkl_malloc((size_t)m * n * sizeof(float), 64);
        float *b = mkl_malloc((size_t)m * n * sizeof(float), 64);
        if (!packed || !x || !a || !b) {
            mkl_free(packed); mkl_free(x); mkl_free(a); mkl_free(b); status = 1; break;
        }
        for (int row = 0; row < m; row++)
            memcpy(x + (size_t)row*k, seed + (size_t)(row%seed_rows)*k, sizeof(float)*k);
        double start = now();
        cblas_sgemm_pack(CblasRowMajor, CblasBMatrix, CblasTrans,
                         m, n, k, 1.0f, w, k, packed);
        double packing = now() - start;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, n, k, 1.0f, x, k, w, k, 0.0f, a, n);
        cblas_sgemm_compute(CblasRowMajor, CblasNoTrans, CblasPacked,
                            m, n, k, x, k, packed, k, 0.0f, b, n);
        double max = 0.0, mae = 0.0;
        for (size_t i = 0; i < (size_t)m*n; i++) {
            if (!isfinite(a[i]) || !isfinite(b[i])) { status = 1; break; }
            double e = fabs((double)a[i]-b[i]);
            if (e > max) max = e;
            mae += e;
        }
        if (max > 1e-4) status = 1;
        printf("{\"type\":\"packed_quality\",\"M\":%d,\"bytes\":%zu,"
               "\"pack_seconds\":%.9f,\"max_abs\":%.9g,\"mae\":%.9g,\"pass\":%s}\n",
               m, bytes, packing, max, mae/((size_t)m*n), status ? "false" : "true");
        if (!status) for (int pair = 0; pair < 5; pair++) {
            double times[2];
            for (int order = 0; order < 2; order++) {
                int use_packed = (pair + order) & 1;
                start = now();
                for (int repeat = 0; repeat < 5; repeat++) {
                    if (use_packed)
                        cblas_sgemm_compute(CblasRowMajor, CblasNoTrans, CblasPacked,
                                            m, n, k, x, k, packed, k, 0.0f, b, n);
                    else
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                    m, n, k, 1.0f, x, k, w, k, 0.0f, a, n);
                }
                times[use_packed] = (now()-start)/5;
            }
            printf("{\"type\":\"packed_pair\",\"M\":%d,\"pair\":%d,"
                   "\"baseline\":%.9f,\"candidate\":%.9f,\"speedup\":%.6f}\n",
                   m, pair+1, times[0], times[1], times[0]/times[1]);
            fflush(stdout);
        }
        mkl_free(packed); mkl_free(x); mkl_free(a); mkl_free(b);
        if (status) break;
    }
    free(seed); iro_st_free(&st);
    return status;
}
