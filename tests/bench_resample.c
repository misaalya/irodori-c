#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "audio.h"

static double now_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

int main(void) {
    enum { SOURCE_RATE = 44100, TARGET_RATE = 48000, SECONDS = 10, RUNS = 3 };
    size_t input_count = SOURCE_RATE * SECONDS;
    float *input = malloc(sizeof(float) * input_count);
    if (!input) return 1;
    for (size_t i = 0; i < input_count; i++) {
        float t = (float)i / SOURCE_RATE;
        input[i] = 0.6f * sinf(2.0f * 3.14159265358979323846f * 440.0f * t) +
                   0.2f * sinf(2.0f * 3.14159265358979323846f * 7013.0f * t);
    }
    double elapsed[RUNS];
    float checksum = 0.0f;
    for (int run = 0; run < RUNS; run++) {
        float *output = NULL;
        size_t output_count = 0;
        double start = now_seconds();
        if (iro_resample_sinc(input, input_count, SOURCE_RATE, TARGET_RATE,
                              &output, &output_count) != 0) {
            free(input);
            return 1;
        }
        elapsed[run] = now_seconds() - start;
        checksum += output[output_count / 2];
        free(output);
    }
    for (int i = 1; i < RUNS; i++) {
        double value = elapsed[i];
        int j = i;
        while (j > 0 && elapsed[j - 1] > value) {
            elapsed[j] = elapsed[j - 1];
            j--;
        }
        elapsed[j] = value;
    }
    printf("sinc 44.1k->48k: %d s audio | median %.4f s | %.1fx realtime | checksum %.7g\n",
           SECONDS, elapsed[RUNS / 2], SECONDS / elapsed[RUNS / 2], checksum);
    free(input);
    return 0;
}
