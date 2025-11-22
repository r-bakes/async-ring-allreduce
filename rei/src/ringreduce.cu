// ringreduce.cu
// Implements naive ring all-reduce using RS + AG with ncclSend/ncclRecv.

#include <assert.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdint.h>
#include <stdio.h>



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



// element-wise add kernel: dest[i + offset] += src[i]
__global__ void add_kernel(float* dest, const float* src, int offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dest[offset + idx] += src[idx];
}

// ring all-reduce using RS + AG
extern "C" void ring_allreduce(
    float* inout_buf, int input_size, ncclComm_t comm, int rank, int n_ranks, cudaStream_t stream
) {
    // compute chunk size and allocate temporary receive buffer
    assert(input_size % n_ranks == 0);
    int chunk_size = input_size / n_ranks;
    float* temp_buf = nullptr;
    CUDA_CALL(cudaMalloc(&temp_buf, chunk_size * sizeof(float)));

    // --- REDUCE-SCATTER ---
    for (int step = 0; step < n_ranks - 1; step++) {
        int send_chunk = (n_ranks - 1 + rank - step) % n_ranks;
        int recv_chunk = (n_ranks - 2 + rank - step) % n_ranks;

        int next_rank = (rank + 1) % n_ranks;
        int prev_rank = (rank - 1 + n_ranks) % n_ranks;

        int send_off = send_chunk * chunk_size;
        int recv_off = recv_chunk * chunk_size;

        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(ncclSend(inout_buf + send_off, chunk_size, ncclFloat, next_rank, comm, stream));
        NCCL_CALL(ncclRecv(temp_buf, chunk_size, ncclFloat, prev_rank, comm, stream));
        NCCL_CALL(ncclGroupEnd());

        CUDA_CALL(cudaStreamSynchronize(stream));

        // reduce
        const int threads = 256;
        int blocks = (chunk_size + threads - 1) / threads;
        add_kernel<<<blocks, threads, 0, stream>>>(inout_buf, temp_buf, recv_off, chunk_size);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaStreamSynchronize(stream));
    }

    // --- ALL-GATHER ---
    for (int step = 0; step < n_ranks - 1; step++) {
        int send_chunk = (n_ranks + rank - step) % n_ranks;
        int recv_chunk = (n_ranks + rank - step - 1) % n_ranks;

        int next_rank = (rank + 1) % n_ranks;
        int prev_rank = (rank - 1 + n_ranks) % n_ranks;

        int send_off = send_chunk * chunk_size;
        int recv_off = recv_chunk * chunk_size;

        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(ncclSend(inout_buf + send_off, chunk_size, ncclFloat, next_rank, comm, stream));
        NCCL_CALL(ncclRecv(inout_buf + recv_off, chunk_size, ncclFloat, prev_rank, comm, stream));
        NCCL_CALL(ncclGroupEnd());

        CUDA_CALL(cudaStreamSynchronize(stream));
    }

    CUDA_CALL(cudaFree(temp_buf));
}
