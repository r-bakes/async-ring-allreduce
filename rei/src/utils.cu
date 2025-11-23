// utils.cu

#include <stdio.h>
#include <sys/time.h>

#include "interface.h"

__global__ void init_input_kernel(float* buf, int rank, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size) buf[idx] = 100.0f * rank + idx * 100.0f / input_size;
}

bool check_correctness(float* h_res, int rank, int n_ranks, int input_size, float atol) {
    int sum_ranks = n_ranks * (n_ranks - 1) * 50;
    for (int i = 0; i < input_size; i++) {
        float expected = (float)sum_ranks + (float)n_ranks * 100.0f * i / input_size;
        float got = h_res[i];
        float diff = fabsf(got - expected);
        if (diff > atol) {
            fprintf(
                stderr,
                "Rank %d: verification FAILED, mismatch at idx %d: got %f expected %f (diff %f)\n",
                rank,
                i,
                got,
                expected,
                diff
            );
            return false;
        }
    }
    return true;
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}
