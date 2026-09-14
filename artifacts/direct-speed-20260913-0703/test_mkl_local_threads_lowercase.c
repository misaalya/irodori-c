#include <stdio.h>

extern int mkl_get_max_threads(void);
extern int mkl_set_num_threads_local(int threads);

int main(void) {
    fprintf(stderr, "before=%d\n", mkl_get_max_threads());
    int previous = mkl_set_num_threads_local(1);
    fprintf(stderr, "previous=%d current=%d\n", previous,
            mkl_get_max_threads());
    return 0;
}
