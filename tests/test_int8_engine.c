/* test_int8_engine.c — W8A8 engine lifecycle: invalid precision is rejected,
   int8 init reports its payload, two warm requests are byte-identical, an
   FP32 engine on the same inputs differs (the path is really taken), and
   both engines free cleanly.  Run under ASan/UBSan when possible. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "generate.h"
#include "ops.h"

static long file_bytes(const char *path, unsigned char **data) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    *data = malloc((size_t)n);
    if (!*data || fread(*data, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(*data); return -1; }
    fclose(f);
    return n;
}

static int same_file(const char *a, const char *b, int *equal) {
    unsigned char *da = NULL, *db = NULL;
    long na = file_bytes(a, &da), nb = file_bytes(b, &db);
    if (na < 0 || nb < 0) { free(da); free(db); return -1; }
    *equal = na == nb && memcmp(da, db, (size_t)na) == 0;
    free(da); free(db);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s MODEL TOKENIZER DECODER NOISE\n", argv[0]);
        return 2;
    }
    int failures = 0;
    IroEngine engine = {0};
    IroEngineConfig bad = { .model_path = argv[1], .tokenizer_path = argv[2],
                            .decoder_path = argv[3], .dit_precision = 7 };
    if (iro_engine_init(&engine, &bad) == 0) {
        fprintf(stderr, "FAIL: invalid dit_precision accepted\n");
        iro_engine_free(&engine);
        failures++;
    }

    IroEngineConfig cfg = { .model_path = argv[1], .tokenizer_path = argv[2],
                            .decoder_path = argv[3],
                            .dit_precision = IRO_DIT_PRECISION_INT8 };
    if (iro_engine_init(&engine, &cfg) != 0) {
        fprintf(stderr, "FAIL: int8 engine init\n");
        return 1;
    }
    IroGenerateConfig request = {
        .text = "こんにちは、色とりどりの世界へようこそ。",
        .noise_path = argv[4],
        .seed = 42,
        .steps = 2,
        .output_path = "/tmp/irodori-int8-a.wav",
    };
    IroGenerateStats stats = {0};
    if (iro_engine_generate(&engine, &request, &stats) != 0) { fprintf(stderr, "FAIL: generate 1\n"); return 1; }
    if (stats.dit_precision != IRO_DIT_PRECISION_INT8 || stats.dit_int8_bytes == 0) {
        fprintf(stderr, "FAIL: stats precision=%d bytes=%zu\n", stats.dit_precision, stats.dit_int8_bytes);
        failures++;
    }
    printf("int8 payload %.1f MiB, sample %.3f s (backend %s, fast int8 %d)\n",
           (double)stats.dit_int8_bytes / (1024.0 * 1024.0), stats.sample_seconds,
           iro_ops_backend_name(), iro_int8_fast_available());
    request.output_path = "/tmp/irodori-int8-b.wav";
    if (iro_engine_generate(&engine, &request, &stats) != 0) { fprintf(stderr, "FAIL: generate 2\n"); return 1; }
    int equal = 0;
    if (same_file("/tmp/irodori-int8-a.wav", "/tmp/irodori-int8-b.wav", &equal) != 0 || !equal) {
        fprintf(stderr, "FAIL: warm int8 requests differ\n");
        failures++;
    }
    iro_engine_free(&engine);
    if (engine.impl) { fprintf(stderr, "FAIL: engine not cleared\n"); failures++; }

    IroEngineConfig fp32 = { .model_path = argv[1], .tokenizer_path = argv[2],
                             .decoder_path = argv[3] };
    if (iro_engine_init(&engine, &fp32) != 0) { fprintf(stderr, "FAIL: fp32 init\n"); return 1; }
    request.output_path = "/tmp/irodori-int8-fp32.wav";
    if (iro_engine_generate(&engine, &request, &stats) != 0) { fprintf(stderr, "FAIL: fp32 generate\n"); return 1; }
    if (stats.dit_precision != IRO_DIT_PRECISION_FP32 || stats.dit_int8_bytes != 0) {
        fprintf(stderr, "FAIL: fp32 stats leak int8 fields\n");
        failures++;
    }
    iro_engine_free(&engine);
    if (same_file("/tmp/irodori-int8-a.wav", "/tmp/irodori-int8-fp32.wav", &equal) != 0 || equal) {
        fprintf(stderr, "FAIL: int8 output identical to fp32 (path not taken?)\n");
        failures++;
    }
    /* Codec int8 on an FP32 DiT: identical latent, so the waveform must stay
       within a few tens of dB of the FP32 decoder and be deterministic. */
    IroEngineConfig codec_cfg = { .model_path = argv[1], .tokenizer_path = argv[2],
                                  .decoder_path = argv[3],
                                  .codec_precision = IRO_DIT_PRECISION_INT8 };
    if (iro_engine_init(&engine, &codec_cfg) != 0) { fprintf(stderr, "FAIL: codec int8 init\n"); return 1; }
    request.output_path = "/tmp/irodori-int8-codec-a.wav";
    if (iro_engine_generate(&engine, &request, &stats) != 0) { fprintf(stderr, "FAIL: codec generate 1\n"); return 1; }
    if (stats.codec_precision != IRO_DIT_PRECISION_INT8 || stats.codec_int8_bytes == 0 ||
        stats.dit_precision != IRO_DIT_PRECISION_FP32) {
        fprintf(stderr, "FAIL: codec stats precision=%d bytes=%zu\n", stats.codec_precision, stats.codec_int8_bytes);
        failures++;
    }
    printf("codec int8 payload %.1f MiB, decode %.3f s\n",
           (double)stats.codec_int8_bytes / (1024.0 * 1024.0), stats.decode_seconds);
    request.output_path = "/tmp/irodori-int8-codec-b.wav";
    if (iro_engine_generate(&engine, &request, &stats) != 0) { fprintf(stderr, "FAIL: codec generate 2\n"); return 1; }
    iro_engine_free(&engine);
    if (same_file("/tmp/irodori-int8-codec-a.wav", "/tmp/irodori-int8-codec-b.wav", &equal) != 0 || !equal) {
        fprintf(stderr, "FAIL: warm codec int8 requests differ\n");
        failures++;
    }
    {
        unsigned char *pa = NULL, *pb = NULL;
        long na = file_bytes("/tmp/irodori-int8-fp32.wav", &pa);
        long nb = file_bytes("/tmp/irodori-int8-codec-a.wav", &pb);
        if (na < 48 || nb != na) {
            fprintf(stderr, "FAIL: codec int8 WAV length differs from fp32 (%ld vs %ld)\n", na, nb);
            failures++;
        } else {
            double sig = 0.0, err = 0.0;
            for (long i = 44; i + 1 < na; i += 2) {
                int16_t a = (int16_t)(pa[i] | (pa[i + 1] << 8));
                int16_t b = (int16_t)(pb[i] | (pb[i + 1] << 8));
                sig += (double)a * a;
                err += (double)(a - b) * (a - b);
            }
            double snr = err > 0 ? 10.0 * log10(sig / err) : 99.0;
            printf("codec int8 vs fp32 waveform SNR %.1f dB\n", snr);
            if (snr < 20.0) { fprintf(stderr, "FAIL: codec int8 SNR below 20 dB\n"); failures++; }
        }
        free(pa); free(pb);
    }
    printf("int8 engine: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
