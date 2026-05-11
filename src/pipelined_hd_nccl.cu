// pipelined_halving_doubling_nccl.cu
// Pipelined Rabenseifner halving-doubling all-reduce with ncclSend/ncclRecv.
// Each of the 2*log2(N) steps is split into n_batches sub-transfers; batches
// run on alternating streams so comm of batch b overlaps with add of batch 1-b.
// Cross-step dependencies are enforced via cudaEvents: each stream waits on
// the OTHER stream's prior-step completion before issuing the next step.

#include <assert.h>
#include <stdio.h>

#include "interface.h"



static int ilog2_exact(int x) {
    int r = 0;
    while ((1 << r) < x) r++;
    return r;
}

// pipelined halving-doubling all-reduce
static void hd_pipelined_impl(
    const float* d_inbuf,
    float* d_outbuf,
    long input_size,
    ncclComm_t comm,
    cudaStream_t streams[2],
    int n_batches
) {
    int rank, n_ranks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &n_ranks);
    assert((n_ranks & (n_ranks - 1)) == 0);
    assert(input_size % n_ranks == 0);

    const int S = ilog2_exact(n_ranks);
    const long chunk_size = input_size / n_ranks;
    // smallest sub-half at d=S-1 is chunk_size/n_batches — must be integer
    assert(chunk_size % n_batches == 0);

    // copy input buffer to output buffer on stream[0]
    if (d_inbuf != d_outbuf)
        CUDA_CALL(cudaMemcpyAsync(
            d_outbuf, d_inbuf, input_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]
        ));

    // events for cross-stream step-to-step synchronization
    cudaEvent_t events[2];
    CUDA_CALL(cudaEventCreateWithFlags(&events[0], cudaEventDisableTiming));
    CUDA_CALL(cudaEventCreateWithFlags(&events[1], cudaEventDisableTiming));
    // make the initial copy visible to stream[1]
    CUDA_CALL(cudaEventRecord(events[0], streams[0]));
    CUDA_CALL(cudaStreamWaitEvent(streams[1], events[0], 0));

    // temp receive buffers for RS — sized to largest sub-half (step d=0)
    float* temp_bufs[2] = {nullptr, nullptr};
    long max_sub_half = (S > 0) ? ((input_size / 2) / n_batches) : 0;
    if (S > 0) {
        CUDA_CALL(cudaMalloc(&temp_bufs[0], max_sub_half * sizeof(float)));
        CUDA_CALL(cudaMalloc(&temp_bufs[1], max_sub_half * sizeof(float)));
    }

    const int threads = 256;

    // --- REDUCE-SCATTER: recursive halving (pipelined) ---
    for (int d = 0; d < S; d++) {
        int h = S - 1 - d;
        int mask_bit = 1 << h;
        int partner = rank ^ mask_bit;
        long half_size = (long)mask_bit * chunk_size;
        long kept_chunk = (long)(rank & ~(mask_bit - 1));
        long sent_chunk = kept_chunk ^ mask_bit;
        long kept_off = kept_chunk * chunk_size;
        long sent_off = sent_chunk * chunk_size;
        long sub_half = half_size / n_batches;

        // each stream waits on the OTHER stream's prior-step completion
        if (d > 0) {
            CUDA_CALL(cudaStreamWaitEvent(streams[0], events[1], 0));
            CUDA_CALL(cudaStreamWaitEvent(streams[1], events[0], 0));
        }

        // issue batches on alternating streams (intra-step pipelining)
        for (int b = 0; b < n_batches; b++) {
            long sub_sent = sent_off + (long)b * sub_half;
            long sub_kept = kept_off + (long)b * sub_half;

            ncclSendRecv(
                d_outbuf + sub_sent,
                temp_bufs[b],
                sub_half,
                rank,
                partner,
                partner,
                comm,
                streams[b]
            );

            long blocks = (sub_half + threads - 1) / threads;
            add_kernel<<<blocks, threads, 0, streams[b]>>>(
                d_outbuf + sub_kept, temp_bufs[b], sub_half
            );
            CUDA_CALL(cudaGetLastError());
        }

        CUDA_CALL(cudaEventRecord(events[0], streams[0]));
        CUDA_CALL(cudaEventRecord(events[1], streams[1]));
    }

    // --- ALL-GATHER: recursive doubling (pipelined) ---
    for (int d = 0; d < S; d++) {
        int mask_bit = 1 << d;
        int partner = rank ^ mask_bit;
        long block_size = (long)mask_bit * chunk_size;
        long my_off = (long)(rank & ~(mask_bit - 1)) * chunk_size;
        long partner_off = (long)(partner & ~(mask_bit - 1)) * chunk_size;
        long sub_block = block_size / n_batches;

        // cross-stream sync against prior step (RS tail or AG tail)
        CUDA_CALL(cudaStreamWaitEvent(streams[0], events[1], 0));
        CUDA_CALL(cudaStreamWaitEvent(streams[1], events[0], 0));

        for (int b = 0; b < n_batches; b++) {
            long my_b = my_off + (long)b * sub_block;
            long partner_b = partner_off + (long)b * sub_block;

            ncclSendRecv(
                d_outbuf + my_b,
                d_outbuf + partner_b,
                sub_block,
                rank,
                partner,
                partner,
                comm,
                streams[b]
            );
        }

        CUDA_CALL(cudaEventRecord(events[0], streams[0]));
        CUDA_CALL(cudaEventRecord(events[1], streams[1]));
    }

    CUDA_CALL(cudaStreamSynchronize(streams[0]));
    CUDA_CALL(cudaStreamSynchronize(streams[1]));
    if (S > 0) {
        CUDA_CALL(cudaFree(temp_bufs[0]));
        CUDA_CALL(cudaFree(temp_bufs[1]));
    }
    CUDA_CALL(cudaEventDestroy(events[0]));
    CUDA_CALL(cudaEventDestroy(events[1]));
}



// interface function, runs for each MPI rank
void halving_doubling_pipelined(RunArgs* args) {
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


    // call pipelined halving-doubling
    hd_pipelined_impl(d_inbuf, d_outbuf, input_size, comm, streams, args->n_batches);


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
        return;
    }


    // warmup
    for (int i = 0; i < args->n_warmup; i++)
        hd_pipelined_impl(d_inbuf, d_outbuf, input_size, comm, streams, args->n_batches);


    // benchmark
    double* deltas = (double*)malloc(args->n_iters * sizeof(double));
    for (int i = 0; i < args->n_iters; i++) {
        double t0 = get_time();
        hd_pipelined_impl(d_inbuf, d_outbuf, input_size, comm, streams, args->n_batches);
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
}
