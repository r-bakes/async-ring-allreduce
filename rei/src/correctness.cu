// correctness.cu
// Calls the ring-allreduce implementation and checks the output for correctness

#include <assert.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void ring_allreduce_naive(
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
};

// initialize input kernel: buf[i] = 100*rank + i
__global__ void init_input_kernel(float* buf, int count, int rank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) buf[idx] = 100.0f * rank + (float)idx;
}

// thread function to call ring-allreduce on a rank and check its output
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


    // call ring all-reduce (in-place)
    ring_allreduce_naive(d_buf, input_size, comm, rank, n_ranks, stream);


    // copy back result to host and verify: output[i] = 100*0+i + 100*1+i + ... 100*(nranks-1)+i
    float* h_res = (float*)malloc(input_size * sizeof(float));
    CUDA_CALL(cudaMemcpy(h_res, d_buf, input_size * sizeof(float), cudaMemcpyDeviceToHost));

    const float atol = 1e-3f;
    int sum_ranks = n_ranks * (n_ranks - 1) * 50;
    bool ok = true;
    for (int i = 0; i < input_size; i++) {
        float expected = (float)sum_ranks + (float)n_ranks * (float)i;
        float got = h_res[i];
        float diff = fabsf(got - expected);
        if (diff > atol) {
            fprintf(
                stderr,
                "Rank %d mismatch at idx %d: got %f expected %f (diff %f)\n",
                rank,
                i,
                got,
                expected,
                diff
            );
            ok = false;
            break;
        }
    }
    if (ok)
        printf("Rank %d: verification PASSED (count=%d)\n", rank, input_size);
    else
        printf("Rank %d: verification FAILED\n", rank);

    free(h_res);
    CUDA_CALL(cudaFree(d_buf));
    CUDA_CALL(cudaStreamDestroy(stream));
    return nullptr;
}



// Usage: ./correctness <n_devices> <input_size>
int main(int argc, char** argv) {
    // parse CLI
    if (argc < 3) {
        printf("Usage: %s <n_devices> <inputsz> (example: %s 4 1048576)\n", argv[0], argv[0]);
        return 1;
    }
    int n_devices = atoi(argv[1]);
    if (n_devices < 2 or n_devices > 4) {
        fprintf(stderr, "n_devices must be between 2 and 4\n");
        return 1;
    }
    int input_size = atoi(argv[2]);
    if (input_size <= 0) {
        fprintf(stderr, "input_size must be > 0\n");
        return 1;
    }
    if (input_size % n_devices) {
        fprintf(stderr, "input_size must be a multiple of n_devices\n");
        return 1;
    }

    // device list (device i = rank i)
    int devs[n_devices];
    for (int i = 0; i < n_devices; i++) devs[i] = i;

    // create NCCL communicators
    ncclComm_t comms[n_devices];
    NCCL_CALL(ncclCommInitAll(comms, n_devices, devs));

    // make threads, one per rank
    pthread_t threads[n_devices];
    ThreadArgs args[n_devices];
    for (int r = 0; r < n_devices; r++) {
        args[r].rank = r;
        args[r].n_ranks = n_devices;
        args[r].input_size = input_size;
        args[r].devs = devs;
        args[r].comms = comms;
        int rc = pthread_create(&threads[r], nullptr, thread_fn, &args[r]);
        if (rc) {
            fprintf(stderr, "Failed to create thread %d\n", r);
            exit(1);
        }
    }

    // wait + cleanup
    for (int r = 0; r < n_devices; r++) pthread_join(threads[r], nullptr);
    for (int r = 0; r < n_devices; r++) ncclCommDestroy(comms[r]);
    printf("All done.\n");

    return 0;
}
