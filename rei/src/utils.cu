// utils.cu

#include <stdio.h>
#include <sys/time.h>

#include "interface.h"

__global__ void init_input_kernel(float* buf, int rank, long input_size) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size) buf[idx] = 100.0f * rank + idx * 100.0f / input_size;
}

bool check_correctness(float* h_res, int rank, int n_ranks, long input_size, float atol) {
    int sum_ranks = n_ranks * (n_ranks - 1) * 50;
    for (long i = 0; i < input_size; i++) {
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
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

void analyze_runtime(RunArgs* args, double* deltas) {
    const int n_iters = args->n_iters;

    double sum_latency = 0.0;
    double min_latency = deltas[0];
    double max_latency = deltas[0];
    for (int i = 0; i < n_iters; i++) {
        double t = deltas[i];
        sum_latency += t;
        if (t < min_latency) min_latency = t;
        if (t > max_latency) max_latency = t;
    }

    double avg_latency = sum_latency / n_iters;
    double sum_std = 0.0;
    for (int i = 0; i < n_iters; i++)
        sum_std += (deltas[i] - avg_latency) * (deltas[i] - avg_latency);

    *(args->avg_latency) = avg_latency;
    *(args->std_latency) = sqrt(sum_std / n_iters);
    *(args->min_latency) = min_latency;
    *(args->max_latency) = max_latency;
}
