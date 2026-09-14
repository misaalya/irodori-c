#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "generate.h"
#include "ops.h"

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
    int retain_frontend;
    int memory_profile;
    int prepared;
    int packed_mib;
    int dit_precision;
    int codec_precision;
    const char *dump_dir;
} WorkerConfig;

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s --model PATH --tokenizer PATH --decoder PATH "
            "--text TEXT [--noise PATH] --seed N --steps N "
            "[--encoder PATH --ref WAV] [--caption TEXT] "
            "[--retain-frontend 0|1] [--memory-profile 0|1] "
            "[--prepared-reference 0|1] [--packed-mib 0..2048] "
            "[--dit-precision fp32|int8] [--dump-dir PATH]\n",
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
    /* Harnesses that can only pass environment to the worker select the
       precision here; an explicit --dit-precision still wins. */
    const char *precision_env = getenv("IRO_DIT_PRECISION");
    if (precision_env && *precision_env) {
        if (!strcmp(precision_env, "int8")) config->dit_precision = IRO_DIT_PRECISION_INT8;
        else if (strcmp(precision_env, "fp32")) return -1;
    }
    const char *codec_env = getenv("IRO_CODEC_PRECISION");
    if (codec_env && *codec_env) {
        if (!strcmp(codec_env, "int8")) config->codec_precision = IRO_DIT_PRECISION_INT8;
        else if (strcmp(codec_env, "fp32")) return -1;
    }
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
        else if (strcmp(argv[i - 1], "--retain-frontend") == 0 ||
                 strcmp(argv[i - 1], "--memory-profile") == 0 ||
                 strcmp(argv[i - 1], "--prepared-reference") == 0) {
            if (strcmp(value, "0") && strcmp(value, "1")) return -1;
            int enabled = strcmp(value, "1") == 0;
            if (!strcmp(argv[i - 1], "--retain-frontend")) config->retain_frontend = enabled;
            else if (!strcmp(argv[i - 1], "--memory-profile")) config->memory_profile = enabled;
            else config->prepared = enabled;
        }
        else if (strcmp(argv[i - 1], "--dump-dir") == 0) config->dump_dir = value;
        else if (strcmp(argv[i - 1], "--dit-precision") == 0) {
            if (!strcmp(value, "fp32")) config->dit_precision = IRO_DIT_PRECISION_FP32;
            else if (!strcmp(value, "int8")) config->dit_precision = IRO_DIT_PRECISION_INT8;
            else return -1;
        }
        else if (strcmp(argv[i - 1], "--codec-precision") == 0) {
            if (!strcmp(value, "fp32")) config->codec_precision = IRO_DIT_PRECISION_FP32;
            else if (!strcmp(value, "int8")) config->codec_precision = IRO_DIT_PRECISION_INT8;
            else return -1;
        }
        else if (strcmp(argv[i - 1], "--packed-mib") == 0) {
            if (!strcmp(value, "0")) config->packed_mib = 0;
            else if (parse_int(value, &config->packed_mib) || config->packed_mib > 2048) return -1;
        }
        else if (strcmp(argv[i - 1], "--seed") == 0) {
            if (parse_u64(value, &config->seed) != 0) return -1;
        } else if (strcmp(argv[i - 1], "--steps") == 0) {
            if (parse_int(value, &config->steps) != 0) return -1;
        } else {
            return -1;
        }
    }
    if (!config->model || !config->tokenizer || !config->decoder ||
        !config->text || config->steps <= 0)
        return -1;
    if (config->reference && !config->encoder) return -1;
    if (config->prepared && !config->reference) return -1;
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

static const char *env_or_empty(const char *name) {
    const char *value = getenv(name);
    return value ? value : "";
}

static void memory_snapshot(void *user, const char *stage) {
    (void)user;
    FILE *file = fopen("/proc/self/smaps_rollup", "r");
    if (!file) return;
    long rss = 0, pss = 0, clean = 0, dirty = 0, swap = 0, huge = 0;
    char line[256], key[64];
    long value;
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "%63s %ld", key, &value) != 2) continue;
        if (!strcmp(key, "Rss:")) rss = value;
        else if (!strcmp(key, "Pss:")) pss = value;
        else if (!strcmp(key, "Private_Clean:")) clean = value;
        else if (!strcmp(key, "Private_Dirty:")) dirty = value;
        else if (!strcmp(key, "Swap:")) swap = value;
        else if (!strcmp(key, "AnonHugePages:")) huge = value;
    }
    fclose(file);
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    fprintf(stderr, "__IRO_MEMORY__={\"stage\":\"%s\",\"time\":%.9f,"
            "\"rss_kib\":%ld,\"pss_kib\":%ld,\"uss_kib\":%ld,"
            "\"swap_kib\":%ld,\"anon_huge_kib\":%ld}\n",
            stage, now.tv_sec + now.tv_nsec * 1e-9,
            rss, pss, clean + dirty, swap, huge);
}

static int run_request(IroEngine *engine, const WorkerConfig *worker,
                       const IroPreparedReference *prepared,
                       const char *output_path, const char *kind) {
    IroGenerateConfig request = {
        .reference_path = prepared ? NULL : worker->reference,
        .prepared_reference = prepared,
        .text = worker->text,
        .caption = worker->caption,
        .output_path = output_path,
        .noise_path = worker->noise,
        .dump_dir = worker->dump_dir,
        .seed = worker->seed,
        .steps = worker->steps,
        .stage_observer = worker->memory_profile ? memory_snapshot : NULL,
    };
    IroGenerateStats stats = {0};
    struct timespec begin, end;
    struct rusage before = {0}, after = {0};
    getrusage(RUSAGE_SELF, &before);
    if (clock_gettime(CLOCK_MONOTONIC, &begin) != 0) return -1;
    int result = iro_engine_generate(engine, &request, &stats);
    if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) return -1;
    getrusage(RUSAGE_SELF, &after);
    if (result != 0) return -1;

    printf("__IRO_RESULT__={\"kind\":\"%s\",\"elapsed_seconds\":%.9f,"
           "\"encode_seconds\":%.9f,\"sample_seconds\":%.9f,"
           "\"decode_seconds\":%.9f,\"tokens\":%d,\"latent_frames\":%d,"
           "\"output_samples\":%d,\"peak_rss_kib\":%ld,"
           "\"minor_faults\":%ld,\"major_faults\":%ld,"
           "\"evicted_model_bytes\":%" PRIu64 ",\"prepared_reference_used\":%d,\"packed_cache_bytes\":%zu,"
           "\"dit_precision\":\"%s\",\"dit_int8_bytes\":%zu,"
           "\"codec_precision\":\"%s\",\"codec_int8_bytes\":%zu}\n",
           kind, elapsed_seconds(&begin, &end), stats.encode_seconds,
           stats.sample_seconds, stats.decode_seconds, stats.tokens,
           stats.latent_frames, stats.output_samples, peak_rss_kib(),
           after.ru_minflt - before.ru_minflt, after.ru_majflt - before.ru_majflt,
           stats.evicted_model_bytes, stats.prepared_reference_used, stats.packed_cache_bytes,
           stats.dit_precision == IRO_DIT_PRECISION_INT8 ? "int8" : "fp32", stats.dit_int8_bytes,
           stats.codec_precision == IRO_DIT_PRECISION_INT8 ? "int8" : "fp32", stats.codec_int8_bytes);
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    WorkerConfig worker;
    if (parse_args(argc, argv, &worker) != 0) {
        usage(argv[0]);
        return 2;
    }

    int requested_threads = 0;
    const char *threads_env = getenv("IRO_NUM_THREADS");
    if (threads_env) {
        if (parse_int(threads_env, &requested_threads) != 0 || requested_threads > 1024) {
            fprintf(stderr, "bench worker: IRO_NUM_THREADS must be an integer 1..1024\n");
            return 2;
        }
        int active_threads = iro_ops_set_threads(requested_threads);
        if (active_threads > 0 && active_threads != requested_threads) {
            fprintf(stderr,
                    "bench worker: backend thread mismatch requested=%d active=%d backend=%s\n",
                    requested_threads, active_threads, iro_ops_backend_name());
            return 2;
        }
    }

    IroEngineConfig engine_config = {
        .model_path = worker.model,
        .tokenizer_path = worker.tokenizer,
        .decoder_path = worker.decoder,
        .encoder_path = worker.encoder,
        .retain_frontend_weights = worker.retain_frontend,
        .packed_cache_budget = (size_t)worker.packed_mib * 1024 * 1024,
        .dit_precision = worker.dit_precision,
        .codec_precision = worker.codec_precision,
    };
    int any_int8 = worker.dit_precision == IRO_DIT_PRECISION_INT8 ||
                   worker.codec_precision == IRO_DIT_PRECISION_INT8;
    if (any_int8 && !iro_int8_fast_available()) {
        fprintf(stderr, "bench worker: backend %s has no vectorized int8 GEMM\n",
                iro_ops_backend_name());
        return 2;
    }
    if (any_int8) {
        (void)iro_int8_prepare_backend();
        if (iro_int8_selfcheck() != 0) {
            fprintf(stderr, "bench worker: int8 backend saturates (non-VNNI MKL branch?)\n");
            return 2;
        }
    }
    IroEngine engine = {0};
    struct timespec init_begin, init_end;
    clock_gettime(CLOCK_MONOTONIC, &init_begin);
    if (iro_engine_init(&engine, &engine_config) != 0) {
        fprintf(stderr, "bench worker: engine init failed\n");
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &init_end);
    double init_seconds = (double)(init_end.tv_sec - init_begin.tv_sec) +
                          1e-9 * (double)(init_end.tv_nsec - init_begin.tv_nsec);
    IroPreparedReference prepared = {0};
    IroPreparedReferenceStats prep_stats = {0};
    if (worker.prepared && iro_engine_prepare_reference(
            &engine, worker.reference, &prepared, &prep_stats) != 0) {
        iro_engine_free(&engine);
        return 1;
    }
    if (worker.memory_profile) memory_snapshot(NULL, "ready");
    printf("__IRO_READY__={\"backend\":\"%s\",\"requested_threads\":%d,"
           "\"backend_threads\":%d,\"MKL_CBWR\":\"%s\","
           "\"MKL_NUM_THREADS\":\"%s\",\"OPENBLAS_NUM_THREADS\":\"%s\","
           "\"OMP_NUM_THREADS\":\"%s\",\"retain_frontend\":%d,"
           "\"memory_profile\":%d,\"prepared_reference\":%d,"
           "\"prepare_seconds\":%.9f,\"prepared_bytes\":%zu,\"packed_mib\":%d,"
           "\"dit_precision\":\"%s\",\"init_seconds\":%.9f,\"mkl_cbwr_effective\":\"%s\","
           "\"codec_precision\":\"%s\"}\n",
           iro_ops_backend_name(), requested_threads, iro_ops_get_threads(),
           env_or_empty("MKL_CBWR"), env_or_empty("MKL_NUM_THREADS"),
           env_or_empty("OPENBLAS_NUM_THREADS"), env_or_empty("OMP_NUM_THREADS"), worker.retain_frontend,
           worker.memory_profile, worker.prepared, prep_stats.prepare_seconds, prep_stats.bytes, worker.packed_mib,
           worker.dit_precision == IRO_DIT_PRECISION_INT8 ? "int8" : "fp32", init_seconds,
           any_int8 && iro_int8_prepare_backend()
               ? iro_int8_prepare_backend() : env_or_empty("MKL_CBWR"),
           worker.codec_precision == IRO_DIT_PRECISION_INT8 ? "int8" : "fp32");
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
        if (run_request(&engine, &worker, worker.prepared ? &prepared : NULL, output, kind) != 0) {
            fprintf(stderr, "bench worker: generation failed\n");
            status = 1;
            break;
        }
    }
    free(line);
    iro_prepared_reference_free(&prepared);
    iro_engine_free(&engine);
    if (worker.memory_profile) memory_snapshot(NULL, "engine_free");
    return status;
}
