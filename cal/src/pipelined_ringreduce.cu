// pipelined_ringreduce.cu
// Implements naive ring all-reduce using pipelined RS + AG with ncclSend/ncclRecv.

#include <assert.h>
#include <stdio.h>

#include <tuple>

#include "interface.h"

// helper functions to get send and recv chunk offsets
static std::pair<int, int> get_offset(
    int step, int rank, int n_chunks, int n_batches, int chunk_size
) {
    assert(step >= 0 && step < 2 * (n_chunks - n_batches));
    int send_chunk = (2 * n_chunks - 1 + rank * n_batches - step) % n_chunks;
    int recv_chunk = (2 * n_chunks - 1 - n_batches + rank * n_batches - step) % n_chunks;
    return {send_chunk * chunk_size, recv_chunk * chunk_size};
}

// element-wise add kernel: dest[i + offset] += src[i]
static __global__ void add_kernel(float* dest, const float* src, int offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dest[offset + idx] += src[idx];
}

// ring all-reduce using RS + AG
static void ring_allreduce(
    const float* d_inbuf, float* d_outbuf, int input_size, ncclComm_t comm, cudaStream_t streams[2], cudaEvent_t events[2]
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
    const int n_batches = 8;
    int n_chunks = n_ranks * n_batches;
    assert(n_batches > 1);
    assert(input_size >= n_chunks);
    assert(input_size % n_chunks == 0);
    int chunk_size = input_size / n_chunks;
    float* temp_bufs[2];
    CUDA_CALL(cudaMalloc(&temp_bufs[0], chunk_size * sizeof(float)));
    CUDA_CALL(cudaMalloc(&temp_bufs[1], chunk_size * sizeof(float)));

    // --- REDUCE-SCATTER ---
    int next_rank = (rank + 1) % n_ranks;
    int prev_rank = (rank - 1 + n_ranks) % n_ranks;

    // Prologue: Start first Recv on Stream 0
    auto [send_off, recv_off] = get_offset(0, rank, n_chunks, n_batches, chunk_size);
    NCCL_CALL(ncclGroupStart());
    NCCL_CALL(ncclSend(d_outbuf + send_off, chunk_size, ncclFloat, next_rank, comm, streams[0]));
    NCCL_CALL(ncclRecv(temp_bufs[0], chunk_size, ncclFloat, prev_rank, comm, streams[0]));
    NCCL_CALL(ncclGroupEnd());
    
    // [FIX]: Record Event 0 to signal "Recv 0 done"
    CUDA_CALL(cudaEventRecord(events[0], streams[0]));

    for (int step = 1; step < n_chunks - n_batches; step++) {
        int compute_stream_idx = (step + 1) % 2; // Stream for add_kernel
        int comm_stream_idx = step % 2;          // Stream for NCCL
        
        // [FIX]: 1. Wait for Recv to finish before Computing
        // The add_kernel needs data from temp_bufs[compute_stream_idx]. 
        // This was filled by the previous step's Recv on streams[compute_stream_idx].
        // We make the compute stream wait for the event recorded after that Recv.
        CUDA_CALL(cudaStreamWaitEvent(streams[compute_stream_idx], events[compute_stream_idx], 0));

        // reduce
        const int threads = 256;
        int blocks = (chunk_size + threads - 1) / threads;
        add_kernel<<<blocks, threads, 0, streams[compute_stream_idx]>>>(
            d_outbuf, temp_bufs[compute_stream_idx], recv_off, chunk_size
        );
        CUDA_CALL(cudaGetLastError());

        // [FIX]: 2. Record "Compute Done" 
        // (Optional for strict correctness if ping-pong is perfect, but safer to add if we reuse buffers heavily)
        // Here we rely on the fact that we don't overwrite temp_buf until the next comm cycle.

        // Get offsets for the *current* step's communication
        std::tie(send_off, recv_off) = get_offset(step, rank, n_chunks, n_batches, chunk_size);
        
        // [FIX]: 3. Wait for Compute to finish before Overwriting?
        // We are about to Recv into temp_bufs[comm_stream_idx]. 
        // We must ensure the *previous* compute using this buffer is done.
        // Since the previous compute on this buffer happened on this SAME stream (comm_stream_idx) 
        // two steps ago, implicit stream ordering handles this. No WaitEvent needed here.

        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(
            ncclSend(d_outbuf + send_off, chunk_size, ncclFloat, next_rank, comm, streams[comm_stream_idx])
        );
        NCCL_CALL(
            ncclRecv(temp_bufs[comm_stream_idx], chunk_size, ncclFloat, prev_rank, comm, streams[comm_stream_idx])
        );
        NCCL_CALL(ncclGroupEnd());
        
        // [FIX]: 4. Record Event to signal "Recv done" for this step
        CUDA_CALL(cudaEventRecord(events[comm_stream_idx], streams[comm_stream_idx]));
    }

    // Epilogue: Final Reduce (happens concurrently with first all gather)
    // Needs to wait for the last Recv (which was on Stream 1)
    CUDA_CALL(cudaStreamWaitEvent(streams[1], events[1], 0));
    
    const int threads = 256;
    int blocks = (chunk_size + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, streams[1]>>>(d_outbuf, temp_bufs[1], recv_off, chunk_size);
    CUDA_CALL(cudaGetLastError());

    // --- ALL-GATHER ---
    std::tie(send_off, recv_off)
        = get_offset(n_chunks - n_batches, rank, n_chunks, n_batches, chunk_size);
    NCCL_CALL(ncclGroupStart());
    NCCL_CALL(ncclSend(d_outbuf + send_off, chunk_size, ncclFloat, next_rank, comm, streams[0]));
    NCCL_CALL(ncclRecv(d_outbuf + recv_off, chunk_size, ncclFloat, prev_rank, comm, streams[0]));
    NCCL_CALL(ncclGroupEnd());

    // [FIX]: Ensure the final compute on Stream 1 finishes before we depend on it in later AG steps
    // (Though AG uses Stream 0, it might read data computed on Stream 1)
    CUDA_CALL(cudaEventRecord(events[1], streams[1]));
    CUDA_CALL(cudaStreamWaitEvent(streams[0], events[1], 0));

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
    CUDA_CALL(cudaStreamSynchronize(streams[1])); // Sync both
    CUDA_CALL(cudaFree(temp_bufs[0]));
    CUDA_CALL(cudaFree(temp_bufs[1]));
}


// interface function, runs for each rank
void* ring_pipelined(RunArgs* args) {
    int input_size = args->input_size;
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
    
    // [FIX]: Create Events
    cudaEvent_t events[2];
    CUDA_CALL(cudaEventCreate(&events[0]));
    CUDA_CALL(cudaEventCreate(&events[1]));


    // initialize input and output
    float* d_inbuf = nullptr;
    CUDA_CALL(cudaMalloc(&d_inbuf, input_size * sizeof(float)));

    const int threads = 256;
    int blocks = (input_size + threads - 1) / threads;
    init_input_kernel<<<blocks, threads, 0, streams[0]>>>(d_inbuf, rank, input_size);
    CUDA_CALL(cudaGetLastError());

    float* d_outbuf = nullptr;
    CUDA_CALL(cudaMalloc(&d_outbuf, input_size * sizeof(float)));


    // call ring all-reduce
    ring_allreduce(d_inbuf, d_outbuf, input_size, comm, streams, events);


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
        ring_allreduce(d_inbuf, d_outbuf, input_size, comm, streams, events);


    // benchmark
    double t0 = get_time();
    for (int i = 0; i < args->n_iters; i++)
        ring_allreduce(d_inbuf, d_outbuf, input_size, comm, streams, events);
    double t1 = get_time();
    if (rank == 0) *(args->avg_latency) = (t1 - t0) * 1e6 / (double)args->n_iters;  // µs per iter


    // cleanup
    CUDA_CALL(cudaFree(d_inbuf));
    CUDA_CALL(cudaFree(d_outbuf));
    CUDA_CALL(cudaStreamDestroy(streams[0]));
    CUDA_CALL(cudaStreamDestroy(streams[1]));
    CUDA_CALL(cudaEventDestroy(events[0])); // [FIX]: Clean up
    CUDA_CALL(cudaEventDestroy(events[1]));
    return nullptr;
}