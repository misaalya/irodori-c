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

typedef struct { int count; int bad; } Observer;
static void observe(void *user, const char *stage) {
    Observer *o = user;
    const char *expected[] = {"conditions", "sampling", "decode", "cleanup"};
    if (strcmp(stage, expected[o->count % 4])) o->bad = 1;
    o->count++;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s MODEL TOKENIZER DECODER NOISE\n", argv[0]);
        return 2;
    }

    IroEngine engine = {0};
    IroEngineConfig engine_config = {
        .model_path = argv[1],
        .tokenizer_path = argv[2],
        .decoder_path = argv[3],
    };
    if (iro_engine_init(&engine, &engine_config) != 0) {
        fprintf(stderr, "engine init failed\n");
        return 1;
    }

    Observer observer = {0};
    IroGenerateConfig request = {
        .text = "こんにちは、色とりどりの世界へようこそ。",
        .noise_path = argv[4],
        .seed = 42,
        .steps = 1,
        .stage_observer = observe,
        .stage_observer_user = &observer,
    };
    IroGenerateStats first = {0}, second = {0};
    request.output_path = "/tmp/irodori-engine-reuse-a.wav";
    int result = iro_engine_generate(&engine, &request, &first);
    request.output_path = "/tmp/irodori-engine-reuse-b.wav";
    if (result == 0) result = iro_engine_generate(&engine, &request, &second);
    iro_engine_free(&engine);

    if (result != 0 || !files_equal("/tmp/irodori-engine-reuse-a.wav",
                                    "/tmp/irodori-engine-reuse-b.wav")) {
        fprintf(stderr, "FAIL reusable engine determinism\n");
        return 1;
    }
    if (observer.bad || observer.count != 8 || first.tokens != second.tokens ||
        first.latent_frames != second.latent_frames ||
        first.output_samples != second.output_samples) {
        fprintf(stderr, "FAIL reusable engine stats mismatch\n");
        return 1;
    }

    printf("PASS reusable engine: %d tokens, %d frames, %d samples; "
           "warm request %.3f s\n",
           second.tokens, second.latent_frames, second.output_samples,
           second.encode_seconds + second.sample_seconds + second.decode_seconds);
    return 0;
}
