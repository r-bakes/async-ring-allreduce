// utils.cu

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "interface.h"

__constant__ unsigned int c_reduce_ns = 5000;
/** 0: legacy proportional (buf_sz >> 8) ns; 1: fixed microseconds from ALLREDUCE_INTER_US */
__constant__ unsigned int c_inter_mode = 0;
__constant__ unsigned long long c_inter_fixed_ns = 0;

static unsigned int g_inter_mode_host = 0;
static unsigned long long g_inter_fixed_ns_host = 0;

void init_benchmark_knob_from_env(void) {
    unsigned int rn = 5000u;
    const char* comp = getenv("ALLREDUCE_COMPUTE_NS");
    if (!comp || comp[0] == '\0') comp = getenv("ALLREDUCE_REDUCE_NS");
    if (comp && comp[0] != '\0') rn = (unsigned int)strtoul(comp, NULL, 10);
    CUDA_CALL(cudaMemcpyToSymbol(c_reduce_ns, &rn, sizeof(rn)));

    unsigned int mode = 0u;
    unsigned long long fixed_ns = 0ull;
    const char* inter = getenv("ALLREDUCE_INTER_US");
    if (inter && inter[0] != '\0') {
        mode = 1u;
        fixed_ns = strtoull(inter, NULL, 10) * 1000ULL;
    }
    CUDA_CALL(cudaMemcpyToSymbol(c_inter_mode, &mode, sizeof(mode)));
    CUDA_CALL(cudaMemcpyToSymbol(c_inter_fixed_ns, &fixed_ns, sizeof(fixed_ns)));
    g_inter_mode_host = mode;
    g_inter_fixed_ns_host = fixed_ns;
}

__global__ void init_input_kernel(float* buf, int rank, long input_size) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size) buf[idx] = 100.0f * rank + idx * 100.0f / input_size;
}

__global__ void add_kernel(float* dest, const float* src, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    __nanosleep(c_reduce_ns);
    if (idx < n) dest[idx] += src[idx];
}

__global__ void sim_latency_kernel(size_t buf_sz) {
    unsigned long long total =
        (c_inter_mode == 0u) ? ((unsigned long long)buf_sz >> 8u) : c_inter_fixed_ns;
    unsigned int sleep_ns = (total > 0xffffffffULL) ? 0xffffffffu : (unsigned int)total;
    __nanosleep(sleep_ns);
}

void ncclSendRecv(
    float* send_buf,
    float* recv_buf,
    size_t buf_sz,
    int rank,
    int send_rank,
    int recv_rank,
    ncclComm_t comm,
    cudaStream_t stream
) {
    NCCL_CALL(ncclGroupStart());
    NCCL_CALL(ncclSend(send_buf, buf_sz, ncclFloat, send_rank, comm, stream));
    NCCL_CALL(ncclRecv(recv_buf, buf_sz, ncclFloat, recv_rank, comm, stream));
    NCCL_CALL(ncclGroupEnd());

    static constexpr int group_size = 2;
    const int group = rank / group_size;
    const int send_group = send_rank / group_size;
    const int recv_group = recv_rank / group_size;

    if (group != send_group || group != recv_group) {
        if (g_inter_mode_host != 1u || g_inter_fixed_ns_host != 0ull)
            sim_latency_kernel<<<1, 32, 0, stream>>>(buf_sz);
    }
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
