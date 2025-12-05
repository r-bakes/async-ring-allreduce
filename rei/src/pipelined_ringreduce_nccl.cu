// pipelined_ringreduce_nccl.cu
// Implements ring all-reduce using pipelined RS + AG with ncclSend/ncclRecv.

#include <assert.h>
#include <stdio.h>

#include <tuple>

#include "interface.h"



// helper functions to get send and recv chunk offsets
static std::pair<long, long> get_offset(
    int step, int rank, int n_chunks, int n_batches, long chunk_size
) {
    assert(step >= 0 && step < 2 * (n_chunks - n_batches));
    long send_chunk = (2 * n_chunks - 1 + rank * n_batches - step) % n_chunks;
    long recv_chunk = (2 * n_chunks - 1 - n_batches + rank * n_batches - step) % n_chunks;
    return {send_chunk * chunk_size, recv_chunk * chunk_size};
}

// element-wise add kernel: dest[i + offset] += src[i]
static __global__ void add_kernel(float* dest, const float* src, long offset, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dest[offset + idx] += src[idx];
}

// ring all-reduce using RS + AG
static void ring_allreduce(
    const float* d_inbuf, float* d_outbuf, long input_size, ncclComm_t comm, cudaStream_t streams[2]
) {
    // get rank and number of ranks
    int rank, n_ranks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &n_ranks);

    // copy input buffer to output buffer
    if (d_inbuf != d_outbuf)
        CUDA_CALL(cudaMemcpyAsync(
            d_outbuf, d_inbuf, input_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]
        ));

    // compute chunk size and allocate temporary receive buffers
    const int n_batches = 2;
    int n_chunks = n_ranks * n_batches;
    assert(n_batches > 1);
    assert(input_size >= n_chunks);
    assert(input_size % n_chunks == 0);
    long chunk_size = input_size / n_chunks;
    float* temp_bufs[2];
    CUDA_CALL(cudaMalloc(&temp_bufs[0], chunk_size * sizeof(float)));
    CUDA_CALL(cudaMalloc(&temp_bufs[1], chunk_size * sizeof(float)));

    // --- REDUCE-SCATTER ---
    int next_rank = (rank + 1) % n_ranks;
    int prev_rank = (rank - 1 + n_ranks) % n_ranks;

    auto [send_off, recv_off] = get_offset(0, rank, n_chunks, n_batches, chunk_size);
    NCCL_CALL(ncclGroupStart());
    NCCL_CALL(ncclSend(d_outbuf + send_off, chunk_size, ncclFloat, next_rank, comm, streams[0]));
    NCCL_CALL(ncclRecv(temp_bufs[0], chunk_size, ncclFloat, prev_rank, comm, streams[0]));
    NCCL_CALL(ncclGroupEnd());

    for (int step = 1; step < n_chunks - n_batches; step++) {
        // reduce
        const int threads = 256;
        long blocks = (chunk_size + threads - 1) / threads;
        add_kernel<<<blocks, threads, 0, streams[(step + 1) % 2]>>>(
            d_outbuf, temp_bufs[(step + 1) % 2], recv_off, chunk_size
        );
        CUDA_CALL(cudaGetLastError());

        std::tie(send_off, recv_off) = get_offset(step, rank, n_chunks, n_batches, chunk_size);
        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(
            ncclSend(d_outbuf + send_off, chunk_size, ncclFloat, next_rank, comm, streams[step % 2])
        );
        NCCL_CALL(
            ncclRecv(temp_bufs[step % 2], chunk_size, ncclFloat, prev_rank, comm, streams[step % 2])
        );
        NCCL_CALL(ncclGroupEnd());

        // CUDA_CALL(cudaStreamSynchronize(streams[(step + 1) % 2]));
    }

    // reduce (happens concurrently with first all gather)
    const int threads = 256;
    long blocks = (chunk_size + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, streams[1]>>>(d_outbuf, temp_bufs[1], recv_off, chunk_size);
    CUDA_CALL(cudaGetLastError());

    // --- ALL-GATHER ---
    std::tie(send_off, recv_off)
        = get_offset(n_chunks - n_batches, rank, n_chunks, n_batches, chunk_size);
    NCCL_CALL(ncclGroupStart());
    NCCL_CALL(ncclSend(d_outbuf + send_off, chunk_size, ncclFloat, next_rank, comm, streams[0]));
    NCCL_CALL(ncclRecv(d_outbuf + recv_off, chunk_size, ncclFloat, prev_rank, comm, streams[0]));
    NCCL_CALL(ncclGroupEnd());

    // CUDA_CALL(cudaStreamSynchronize(streams[1]));

    for (int step = n_chunks - n_batches + 1; step < 2 * (n_chunks - n_batches); step++) {
        std::tie(send_off, recv_off) = get_offset(step, rank, n_chunks, n_batches, chunk_size);
        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(
            ncclSend(d_outbuf + send_off, chunk_size, ncclFloat, next_rank, comm, streams[0])
        );
        NCCL_CALL(
            ncclRecv(d_outbuf + recv_off, chunk_size, ncclFloat, prev_rank, comm, streams[0])
        );
        NCCL_CALL(ncclGroupEnd());
    }

    CUDA_CALL(cudaStreamSynchronize(streams[0]));
    CUDA_CALL(cudaFree(temp_bufs[0]));
    CUDA_CALL(cudaFree(temp_bufs[1]));
}



// interface function, runs for each rank
void* ring_pipelined_nccl(RunArgs* args) {
    long input_size = args->input_size;
    ncclComm_t comm = args->comm;
    int rank, n_ranks, device;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &n_ranks);
    ncclCommCuDevice(comm, &device);


    // initialize CUDA streams
    CUDA_CALL(cudaSetDevice(device));
    cudaStream_t streams[2];
    CUDA_CALL(cudaStreamCreate(&streams[0]));
    CUDA_CALL(cudaStreamCreate(&streams[1]));


    // initialize input and output
    float* d_inbuf = nullptr;
    CUDA_CALL(cudaMalloc(&d_inbuf, input_size * sizeof(float)));

    const int threads = 256;
    long blocks = (input_size + threads - 1) / threads;
    init_input_kernel<<<blocks, threads, 0, streams[0]>>>(d_inbuf, rank, input_size);
    CUDA_CALL(cudaGetLastError());

    float* d_outbuf = nullptr;
    CUDA_CALL(cudaMalloc(&d_outbuf, input_size * sizeof(float)));


    // call ring all-reduce
    ring_allreduce(d_inbuf, d_outbuf, input_size, comm, streams);


    // copy back result to host and verify output, short circuit if incorrect
    float* h_res = (float*)malloc(input_size * sizeof(float));
    CUDA_CALL(cudaMemcpy(h_res, d_outbuf, input_size * sizeof(float), cudaMemcpyDeviceToHost));
    *(args->correct) = check_correctness(h_res, rank, n_ranks, input_size, args->atol);
    free(h_res);

    if (!*(args->correct)) {
        CUDA_CALL(cudaFree(d_inbuf));
        CUDA_CALL(cudaFree(d_outbuf));
        CUDA_CALL(cudaStreamDestroy(streams[0]));
        CUDA_CALL(cudaStreamDestroy(streams[1]));
        return nullptr;
    }


    // warmup
    for (int i = 0; i < args->n_warmup; i++)
        ring_allreduce(d_inbuf, d_outbuf, input_size, comm, streams);


    // benchmark
    double* deltas = (double*)malloc(args->n_iters * sizeof(double));
    for (int i = 0; i < args->n_iters; i++) {
        double t0 = get_time();
        ring_allreduce(d_inbuf, d_outbuf, input_size, comm, streams);
        double t1 = get_time();
        deltas[i] = t1 - t0;
    }
    analyze_runtime(args, deltas);
    free(deltas);


    // cleanup
    CUDA_CALL(cudaFree(d_inbuf));
    CUDA_CALL(cudaFree(d_outbuf));
    CUDA_CALL(cudaStreamDestroy(streams[0]));
    CUDA_CALL(cudaStreamDestroy(streams[1]));
    return nullptr;
}
