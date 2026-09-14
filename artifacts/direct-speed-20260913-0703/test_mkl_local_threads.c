#include <pthread.h>
#include <stdio.h>

extern int mkl_get_max_threads(void);
extern int MKL_Set_Num_Threads_Local(int threads);

static void *worker(void *opaque) {
    long id = (long)opaque;
    fprintf(stderr, "worker%ld before=%d\n", id, mkl_get_max_threads());
    return NULL;
}

int main(void) {
    pthread_t threads[2];
    fprintf(stderr, "main=%d\n", mkl_get_max_threads());
    int previous = MKL_Set_Num_Threads_Local(1);
    fprintf(stderr, "main previous=%d local=%d\n", previous,
            mkl_get_max_threads());
    for (long i = 0; i < 2; i++)
        if (pthread_create(&threads[i], NULL, worker, (void *)i) != 0) return 1;
    for (int i = 0; i < 2; i++) pthread_join(threads[i], NULL);
    MKL_Set_Num_Threads_Local(previous);
    fprintf(stderr, "main-after=%d\n", mkl_get_max_threads());
    return 0;
}
