// naive_ringreduce.cu
// Implements naive ring all-reduce using RS + AG with ncclSend/ncclRecv.

#include <assert.h>
#include <stdio.h>

#include <utility>

#include "interface.h"



// helper functions to get send and recv chunk offsets
static std::pair<long, long> get_offset(int step, int rank, int n_ranks, long chunk_size) {
    assert(step >= 0 && step < 2 * (n_ranks - 1));
    long send_chunk = (2 * n_ranks - 1 + rank - step) % n_ranks;
    long recv_chunk = (2 * n_ranks - 2 + rank - step) % n_ranks;
    return {send_chunk * chunk_size, recv_chunk * chunk_size};
}

// element-wise add kernel: dest[i + offset] += src[i]
static __global__ void add_kernel(float* dest, const float* src, long offset, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dest[offset + idx] += src[idx];
}

// ring all-reduce using RS + AG
static void ring_allreduce(
    const float* d_inbuf, float* d_outbuf, long input_size, ncclComm_t comm, cudaStream_t stream
) {
    // get rank and number of ranks
    int rank, n_ranks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &n_ranks);

    // copy input buffer to output buffer
    if (d_inbuf != d_outbuf)
        CUDA_CALL(cudaMemcpyAsync(
            d_outbuf, d_inbuf, input_size * sizeof(float), cudaMemcpyDeviceToDevice, stream
        ));

    // compute chunk size and allocate temporary receive buffer
    assert(input_size % n_ranks == 0);
    long chunk_size = input_size / n_ranks;
    float* temp_buf = nullptr;
    CUDA_CALL(cudaMalloc(&temp_buf, chunk_size * sizeof(float)));

    // --- REDUCE-SCATTER ---
    int next_rank = (rank + 1) % n_ranks;
    int prev_rank = (rank - 1 + n_ranks) % n_ranks;
    for (int step = 0; step < n_ranks - 1; step++) {
        auto [send_off, recv_off] = get_offset(step, rank, n_ranks, chunk_size);
        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(ncclSend(d_outbuf + send_off, chunk_size, ncclFloat, next_rank, comm, stream));
        NCCL_CALL(ncclRecv(temp_buf, chunk_size, ncclFloat, prev_rank, comm, stream));
        NCCL_CALL(ncclGroupEnd());

        // reduce
        const int threads = 256;
        long blocks = (chunk_size + threads - 1) / threads;
        add_kernel<<<blocks, threads, 0, stream>>>(d_outbuf, temp_buf, recv_off, chunk_size);
        CUDA_CALL(cudaGetLastError());
    }

    // --- ALL-GATHER ---
    for (int step = n_ranks - 1; step < 2 * (n_ranks - 1); step++) {
        auto [send_off, recv_off] = get_offset(step, rank, n_ranks, chunk_size);
        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(ncclSend(d_outbuf + send_off, chunk_size, ncclFloat, next_rank, comm, stream));
        NCCL_CALL(ncclRecv(d_outbuf + recv_off, chunk_size, ncclFloat, prev_rank, comm, stream));
        NCCL_CALL(ncclGroupEnd());
    }

    CUDA_CALL(cudaStreamSynchronize(stream));
    CUDA_CALL(cudaFree(temp_buf));
}



// interface function, runs for each rank
void* ring_naive(RunArgs* args) {
    long input_size = args->input_size;
    ncclComm_t comm = args->comm;
    int rank, n_ranks, device;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &n_ranks);
    ncclCommCuDevice(comm, &device);


    // initialize CUDA stream
    CUDA_CALL(cudaSetDevice(device));
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));


    // initialize input and output
    float* d_inbuf = nullptr;
    CUDA_CALL(cudaMalloc(&d_inbuf, input_size * sizeof(float)));

    const int threads = 256;
    long blocks = (input_size + threads - 1) / threads;
    init_input_kernel<<<blocks, threads, 0, stream>>>(d_inbuf, rank, input_size);
    CUDA_CALL(cudaGetLastError());

    float* d_outbuf = nullptr;
    CUDA_CALL(cudaMalloc(&d_outbuf, input_size * sizeof(float)));


    // call ring all-reduce
    ring_allreduce(d_inbuf, d_outbuf, input_size, comm, stream);


    // copy back result to host and verify output, short circuit if incorrect
    float* h_res = (float*)malloc(input_size * sizeof(float));
    CUDA_CALL(cudaMemcpy(h_res, d_outbuf, input_size * sizeof(float), cudaMemcpyDeviceToHost));
    *(args->correct) = check_correctness(h_res, rank, n_ranks, input_size, args->atol);
    free(h_res);

    if (!*(args->correct)) {
        CUDA_CALL(cudaFree(d_inbuf));
        CUDA_CALL(cudaFree(d_outbuf));
        CUDA_CALL(cudaStreamDestroy(stream));
        return nullptr;
    }


    // warmup
    for (int i = 0; i < args->n_warmup; i++)
        ring_allreduce(d_inbuf, d_outbuf, input_size, comm, stream);


    // benchmark
    double* deltas = (double*)malloc(args->n_iters * sizeof(double));
    for (int i = 0; i < args->n_iters; i++) {
        double t0 = get_time();
        ring_allreduce(d_inbuf, d_outbuf, input_size, comm, stream);
        double t1 = get_time();
        deltas[i] = t1 - t0;
    }
    analyze_runtime(args, deltas);
    free(deltas);


    // cleanup
    CUDA_CALL(cudaFree(d_inbuf));
    CUDA_CALL(cudaFree(d_outbuf));
    CUDA_CALL(cudaStreamDestroy(stream));
    return nullptr;
}
