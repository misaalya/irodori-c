#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "generate.h"

static int files_equal(const char *left_path, const char *right_path) {
    FILE *left = fopen(left_path, "rb");
    FILE *right = fopen(right_path, "rb");
    if (!left || !right) {
        if (left) fclose(left);
        if (right) fclose(right);
        return 0;
    }
    int equal = 1;
    unsigned char left_buf[16384], right_buf[16384];
    for (;;) {
        size_t left_n = fread(left_buf, 1, sizeof(left_buf), left);
        size_t right_n = fread(right_buf, 1, sizeof(right_buf), right);
        if (left_n != right_n || memcmp(left_buf, right_buf, left_n) != 0) {
            equal = 0;
            break;
        }
        if (left_n == 0) {
            if (ferror(left) || ferror(right)) equal = 0;
            break;
        }
    }
    fclose(left);
    fclose(right);
    return equal;
}

static double total_seconds(const IroGenerateStats *stats) {
    return stats->encode_seconds + stats->sample_seconds + stats->decode_seconds;
}

int main(int argc, char **argv) {
    if (argc != 7 && argc != 8) {
        fprintf(stderr,
                "usage: %s MODEL TOKENIZER DECODER ENCODER REF NOISE [pairs]\n",
                argv[0]);
        return 2;
    }
    int pairs = argc == 8 ? atoi(argv[7]) : 3;
    if (pairs < 3 || pairs > 100) {
        fprintf(stderr, "prepared-reference: pairs harus 3..100\n");
        return 2;
    }

    IroEngine engine = {0};
    IroEngineConfig engine_config = {
        .model_path = argv[1],
        .tokenizer_path = argv[2],
        .decoder_path = argv[3],
        .encoder_path = argv[4],
    };
    if (iro_engine_init(&engine, &engine_config) != 0) {
        fprintf(stderr, "prepared-reference: engine init failed\n");
        return 1;
    }

    IroPreparedReference prepared = {0};
    IroPreparedReferenceStats prepare_stats = {0};
    if (iro_engine_prepare_reference(&engine, argv[5], &prepared,
                                     &prepare_stats) != 0 ||
        !prepared.impl || prepare_stats.reference_frames <= 0 ||
        prepare_stats.speaker_tokens <= 0 || prepare_stats.bytes == 0) {
        fprintf(stderr, "prepared-reference: preparation failed\n");
        iro_prepared_reference_free(&prepared);
        iro_engine_free(&engine);
        return 1;
    }

    IroGenerateConfig request = {
        .text = "こんにちは、色とりどりの世界へようこそ。",
        .noise_path = argv[6],
        .seed = 42,
        .steps = 1,
    };
    int result = 0;
    for (int pair = 0; pair < pairs && result == 0; pair++) {
        char uncached_path[128], cached_path[128];
        snprintf(uncached_path, sizeof(uncached_path),
                 "/tmp/irodori-reference-uncached-%d.wav", pair);
        snprintf(cached_path, sizeof(cached_path),
                 "/tmp/irodori-reference-cached-%d.wav", pair);
        IroGenerateStats uncached = {0}, cached = {0};

        for (int run = 0; run < 2 && result == 0; run++) {
            int cached_first = pair & 1;
            int use_cached = run == (cached_first ? 0 : 1);
            request.reference_path = use_cached ? NULL : argv[5];
            request.prepared_reference = use_cached ? &prepared : NULL;
            request.output_path = use_cached ? cached_path : uncached_path;
            result = iro_engine_generate(
                &engine, &request, use_cached ? &cached : &uncached);
        }

        if (result == 0 &&
            (!files_equal(uncached_path, cached_path) ||
             uncached.prepared_reference_used ||
             !cached.prepared_reference_used ||
             cached.prepared_reference_bytes != prepare_stats.bytes ||
             cached.speaker_tokens != uncached.speaker_tokens)) {
            fprintf(stderr, "prepared-reference: parity/stats mismatch pair %d\n",
                    pair + 1);
            result = -1;
        }
        if (result == 0) {
            printf("{\"type\":\"prepared_reference_pair\",\"pair\":%d,"
                   "\"first\":\"%s\",\"uncached_encode_seconds\":%.6f,"
                   "\"cached_encode_seconds\":%.6f,"
                   "\"uncached_total_seconds\":%.6f,"
                   "\"cached_total_seconds\":%.6f,"
                   "\"total_speedup\":%.6f,\"wav_exact\":true}\n",
                   pair + 1, (pair & 1) ? "cached" : "uncached",
                   uncached.encode_seconds, cached.encode_seconds,
                   total_seconds(&uncached), total_seconds(&cached),
                   total_seconds(&uncached) / total_seconds(&cached));
            fflush(stdout);
        }
    }

    request.reference_path = argv[5];
    request.prepared_reference = &prepared;
    request.output_path = "/tmp/irodori-reference-conflict.wav";
    if (result == 0 && iro_engine_generate(&engine, &request, NULL) == 0) {
        fprintf(stderr, "prepared-reference: accepted conflicting inputs\n");
        result = -1;
    }

    iro_prepared_reference_free(&prepared);
    iro_prepared_reference_free(&prepared);
    iro_engine_free(&engine);
    if (result != 0) return 1;
    printf("PASS prepared reference: %d pairs, %.3f s prepare, %zu bytes\n",
           pairs, prepare_stats.prepare_seconds, prepare_stats.bytes);
    return 0;
}
