// benchmark.cu
// Calls the ring-allreduce implementation with various buffer sizes and measures the latency

#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <nccl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// ! NOTE: use the same function name and interface for all implementations
extern "C" void ring_allreduce(
    float* inout_buf, int input_size, ncclComm_t comm, int rank, int n_ranks, cudaStream_t stream
);



// define macros to call cuda commands with error checking
#define CUDA_CALL(cmd)                                                                       \
    do {                                                                                     \
        cudaError_t e = cmd;                                                                 \
        if (e != cudaSuccess) {                                                              \
            fprintf(stderr, "CUDA:%s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                              \
        }                                                                                    \
    } while (0)

#define NCCL_CALL(cmd)                                                                       \
    do {                                                                                     \
        ncclResult_t r = cmd;                                                                \
        if (r != ncclSuccess) {                                                              \
            fprintf(stderr, "NCCL:%s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE);                                                              \
        }                                                                                    \
    } while (0)



struct ThreadArgs {
    int rank;
    int n_ranks;
    int input_size;
    int* devs;
    ncclComm_t* comms;
    int iters;
    int warmup;
    double* out_time;
};

// initialize input kernel: buf[i] = 100*rank + i
__global__ void init_input_kernel(float* buf, int count, int rank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) buf[idx] = 100.0f * rank + (float)idx;
}

// get current time
static double now_sec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

// thread function to call ring-allreduce on a rank and benchmark
void* thread_fn(void* arg) {
    ThreadArgs* a = (ThreadArgs*)arg;
    int rank = a->rank;
    int n_ranks = a->n_ranks;
    int input_size = a->input_size;
    int dev = a->devs[rank];
    ncclComm_t comm = a->comms[rank];

    CUDA_CALL(cudaSetDevice(dev));
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));


    // initialize input for this rank at d_buf
    float* d_buf = nullptr;
    CUDA_CALL(cudaMalloc(&d_buf, input_size * sizeof(float)));

    const int threads = 256;
    int blocks = (input_size + threads - 1) / threads;
    init_input_kernel<<<blocks, threads, 0, stream>>>(d_buf, input_size, rank);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaStreamSynchronize(stream));


    // warmup
    for (int i = 0; i < a->warmup; i++)
        ring_allreduce(d_buf, input_size, comm, rank, n_ranks, stream);
    CUDA_CALL(cudaStreamSynchronize(stream));


    // benchmark
    double t0 = now_sec();
    for (int i = 0; i < a->iters; i++)
        ring_allreduce(d_buf, input_size, comm, rank, n_ranks, stream);
    CUDA_CALL(cudaStreamSynchronize(stream));
    double t1 = now_sec();

    if (rank == 0) *(a->out_time) = (t1 - t0) * 1e6 / (double)a->iters;  // µs per iter

    CUDA_CALL(cudaFree(d_buf));
    CUDA_CALL(cudaStreamDestroy(stream));
    return nullptr;
}



// Usage: ./correctness <n_devices>
int main(int argc, char** argv) {
    // parse CLI
    if (argc < 2) {
        printf("Usage: %s <n_devices>(example: %s 4)\n", argv[0], argv[0]);
        return 1;
    }
    int n_devices = atoi(argv[1]);
    if (n_devices != 2 && n_devices != 4) {
        fprintf(stderr, "n_devices must be 2 or 4\n");
        return 1;
    }

    // device list (device i = rank i)
    int devs[n_devices];
    for (int i = 0; i < n_devices; i++) devs[i] = i;

    // create NCCL communicators
    ncclComm_t comms[n_devices];
    NCCL_CALL(ncclCommInitAll(comms, n_devices, devs));

    // create output file
    FILE* f = fopen("results.csv", "w");
    fprintf(f, "input_size,input_bytes,avg_us\n");
    fflush(f);

    const int warmup = 20;
    const int iters = 200;
    const int min_sz = 256;         // 1KB
    const int max_sz = 1073741824;  // 4GB
    assert(max_sz <= 1073741824);   // otherwise won't fit, we use long below as *2 could overflow

    for (long input_size = min_sz; input_size <= max_sz; input_size *= 2) {
        assert(input_size % n_devices == 0);
        size_t bytes = (size_t)input_size * sizeof(float);
        printf("\n=== Benchmarking input_size = %d floats (%zu bytes) ===\n", input_size, bytes);

        pthread_t threads[n_devices];
        ThreadArgs args[n_devices];

        double t_avg = 0.0;
        for (int r = 0; r < n_devices; r++) {
            args[r].rank = r;
            args[r].n_ranks = n_devices;
            args[r].input_size = input_size;
            args[r].devs = devs;
            args[r].comms = comms;
            args[r].iters = iters;
            args[r].warmup = warmup;
            args[r].out_time = &t_avg;

            int rc = pthread_create(&threads[r], nullptr, thread_fn, &args[r]);
            if (rc) {
                fprintf(stderr, "Failed to create thread %d\n", r);
                exit(1);
            }
        }

        for (int r = 0; r < n_devices; r++) pthread_join(threads[r], nullptr);

        fprintf(f, "%d,%zu,%.3f\n", input_size, bytes, t_avg);
        fflush(f);
        printf("Average latency: %.3f µs\n", t_avg);
    }

    // cleanup
    for (int r = 0; r < n_devices; r++) ncclCommDestroy(comms[r]);
    fclose(f);
    printf("\nAll done. Results written to results.csv\n");

    return 0;
}
