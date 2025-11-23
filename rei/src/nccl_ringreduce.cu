// nccl_ringreduce.cu
// Calls the NCCL ring all-reduce implementation

#include <assert.h>
#include <stdio.h>

#include "interface.h"



// interface function, runs for each rank
void* ring_nccl(RunArgs* args) {
    int input_size = args->input_size;
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
    int blocks = (input_size + threads - 1) / threads;
    init_input_kernel<<<blocks, threads, 0, stream>>>(d_inbuf, rank, input_size);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaStreamSynchronize(stream));

    float* d_outbuf = nullptr;
    CUDA_CALL(cudaMalloc(&d_outbuf, input_size * sizeof(float)));


    // call nccl all-reduce
    ncclAllReduce(d_inbuf, d_outbuf, input_size, ncclFloat, ncclSum, comm, stream);
    cudaStreamSynchronize(stream);


    // copy back result to host and verify output, short circuit if incorrect
    float* h_res = (float*)malloc(input_size * sizeof(float));
    CUDA_CALL(cudaMemcpy(h_res, d_outbuf, input_size * sizeof(float), cudaMemcpyDeviceToHost));
    *(args->correct) = check_correctness(h_res, rank, n_ranks, input_size, args->atol);
    free(h_res);
    assert(*(args->correct));  // should be correct since we're using nccl's All Reduce


    // warmup
    for (int i = 0; i < args->n_warmup; i++)
        ncclAllReduce(d_inbuf, d_outbuf, input_size, ncclFloat, ncclSum, comm, stream);
    CUDA_CALL(cudaStreamSynchronize(stream));


    // benchmark
    double t0 = get_time();
    for (int i = 0; i < args->n_iters; i++)
        ncclAllReduce(d_inbuf, d_outbuf, input_size, ncclFloat, ncclSum, comm, stream);
    CUDA_CALL(cudaStreamSynchronize(stream));
    double t1 = get_time();
    if (rank == 0) *(args->avg_latency) = (t1 - t0) * 1e6 / (double)args->n_iters;  // µs per iter


    // cleanup
    CUDA_CALL(cudaFree(d_inbuf));
    CUDA_CALL(cudaFree(d_outbuf));
    CUDA_CALL(cudaStreamDestroy(stream));
    return nullptr;
}
