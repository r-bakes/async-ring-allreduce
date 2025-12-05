// nccl_ringreduce.cu
// Calls the NCCL ring all-reduce implementation

#include <assert.h>
#include <stdio.h>

#include "interface.h"



// interface function, runs for each rank
void* ring_nccl(RunArgs* args) {
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


    // call nccl all-reduce
    NCCL_CALL(ncclAllReduce(d_inbuf, d_outbuf, input_size, ncclFloat, ncclSum, comm, stream));
    CUDA_CALL(cudaStreamSynchronize(stream));


    // copy back result to host and verify output, short circuit if incorrect
    float* h_res = (float*)malloc(input_size * sizeof(float));
    CUDA_CALL(cudaMemcpy(h_res, d_outbuf, input_size * sizeof(float), cudaMemcpyDeviceToHost));
    *(args->correct) = check_correctness(h_res, rank, n_ranks, input_size, args->atol);
    free(h_res);
    assert(*(args->correct));  // should be correct since we're using nccl's All Reduce


    // warmup
    for (int i = 0; i < args->n_warmup; i++)
        NCCL_CALL(ncclAllReduce(d_inbuf, d_outbuf, input_size, ncclFloat, ncclSum, comm, stream));


    // benchmark
    double* deltas = (double*)malloc(args->n_iters * sizeof(double));
    for (int i = 0; i < args->n_iters; i++) {
        double t0 = get_time();
        NCCL_CALL(ncclAllReduce(d_inbuf, d_outbuf, input_size, ncclFloat, ncclSum, comm, stream));
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
