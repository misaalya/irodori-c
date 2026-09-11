#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "generate.h"

typedef struct {
    const char *model;
    const char *tokenizer;
    const char *decoder;
    const char *encoder;
    const char *reference;
    const char *caption;
    const char *text;
    const char *noise;
    uint64_t seed;
    int steps;
} WorkerConfig;

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s --model PATH --tokenizer PATH --decoder PATH "
            "--text TEXT --noise PATH --seed N --steps N "
            "[--encoder PATH --ref WAV] [--caption TEXT]\n",
            program);
}

static int parse_u64(const char *text, uint64_t *value) {
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || !end || *end) return -1;
    *value = (uint64_t)parsed;
    return 0;
}

static int parse_int(const char *text, int *value) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (errno || !end || *end || parsed <= 0 || parsed > 1000000) return -1;
    *value = (int)parsed;
    return 0;
}

static int parse_args(int argc, char **argv, WorkerConfig *config) {
    memset(config, 0, sizeof(*config));
    config->seed = 42;
    for (int i = 1; i < argc; i++) {
        if (i + 1 >= argc) return -1;
        const char *value = argv[++i];
        if (strcmp(argv[i - 1], "--model") == 0) config->model = value;
        else if (strcmp(argv[i - 1], "--tokenizer") == 0) config->tokenizer = value;
        else if (strcmp(argv[i - 1], "--decoder") == 0) config->decoder = value;
        else if (strcmp(argv[i - 1], "--encoder") == 0) config->encoder = value;
        else if (strcmp(argv[i - 1], "--ref") == 0) config->reference = value;
        else if (strcmp(argv[i - 1], "--caption") == 0) config->caption = value;
        else if (strcmp(argv[i - 1], "--text") == 0) config->text = value;
        else if (strcmp(argv[i - 1], "--noise") == 0) config->noise = value;
        else if (strcmp(argv[i - 1], "--seed") == 0) {
            if (parse_u64(value, &config->seed) != 0) return -1;
        } else if (strcmp(argv[i - 1], "--steps") == 0) {
            if (parse_int(value, &config->steps) != 0) return -1;
        } else {
            return -1;
        }
    }
    if (!config->model || !config->tokenizer || !config->decoder ||
        !config->text || !config->noise || config->steps <= 0)
        return -1;
    if (config->reference && !config->encoder) return -1;
    return 0;
}

static double elapsed_seconds(const struct timespec *begin,
                              const struct timespec *end) {
    return (double)(end->tv_sec - begin->tv_sec) +
           1e-9 * (double)(end->tv_nsec - begin->tv_nsec);
}

static long peak_rss_kib(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) return -1;
#ifdef __APPLE__
    return (long)(usage.ru_maxrss / 1024);
#else
    return (long)usage.ru_maxrss;
#endif
}

static int run_request(IroEngine *engine, const WorkerConfig *worker,
                       const char *output_path, const char *kind) {
    IroGenerateConfig request = {
        .reference_path = worker->reference,
        .text = worker->text,
        .caption = worker->caption,
        .output_path = output_path,
        .noise_path = worker->noise,
        .seed = worker->seed,
        .steps = worker->steps,
    };
    IroGenerateStats stats = {0};
    struct timespec begin, end;
    if (clock_gettime(CLOCK_MONOTONIC, &begin) != 0) return -1;
    int result = iro_engine_generate(engine, &request, &stats);
    if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) return -1;
    if (result != 0) return -1;

    printf("__IRO_RESULT__={\"kind\":\"%s\",\"elapsed_seconds\":%.9f,"
           "\"encode_seconds\":%.9f,\"sample_seconds\":%.9f,"
           "\"decode_seconds\":%.9f,\"tokens\":%d,\"latent_frames\":%d,"
           "\"output_samples\":%d,\"peak_rss_kib\":%ld}\n",
           kind, elapsed_seconds(&begin, &end), stats.encode_seconds,
           stats.sample_seconds, stats.decode_seconds, stats.tokens,
           stats.latent_frames, stats.output_samples, peak_rss_kib());
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    WorkerConfig worker;
    if (parse_args(argc, argv, &worker) != 0) {
        usage(argv[0]);
        return 2;
    }

    IroEngineConfig engine_config = {
        .model_path = worker.model,
        .tokenizer_path = worker.tokenizer,
        .decoder_path = worker.decoder,
        .encoder_path = worker.encoder,
    };
    IroEngine engine = {0};
    if (iro_engine_init(&engine, &engine_config) != 0) {
        fprintf(stderr, "bench worker: engine init failed\n");
        return 1;
    }
    printf("__IRO_READY__\n");
    fflush(stdout);

    char *line = NULL;
    size_t capacity = 0;
    int status = 0;
    while (getline(&line, &capacity, stdin) >= 0) {
        size_t length = strlen(line);
        while (length && (line[length - 1] == '\n' || line[length - 1] == '\r'))
            line[--length] = '\0';
        if (strcmp(line, "QUIT") == 0) break;
        char *tab = strchr(line, '\t');
        if (!tab) {
            fprintf(stderr, "bench worker: expected KIND<TAB>OUTPUT\n");
            status = 1;
            break;
        }
        *tab = '\0';
        const char *kind = line;
        const char *output = tab + 1;
        if ((strcmp(kind, "WARM") != 0 && strcmp(kind, "RUN") != 0) || !*output) {
            fprintf(stderr, "bench worker: invalid command\n");
            status = 1;
            break;
        }
        if (run_request(&engine, &worker, output, kind) != 0) {
            fprintf(stderr, "bench worker: generation failed\n");
            status = 1;
            break;
        }
    }
    free(line);
    iro_engine_free(&engine);
    return status;
}
