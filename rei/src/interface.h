// interface.h
// A common interface for all implementations of the ring algorithm

#pragma once

#include <cuda_runtime.h>
#include <nccl.h>



// define macros for running cuda and nccl commands with error checking
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


static constexpr int MAX_RANKS = 4;

typedef struct {
    // all-reduce arguments
    long input_size;
    ncclComm_t comm;
    // benchmark & correctness arguments
    int n_warmup;
    int n_iters;
    float atol;
    bool* correct;
    double* avg_latency;
    double* std_latency;
    double* min_latency;
    double* max_latency;
} RunArgs;



// initialize buf[i] = 100*rank + 100*i/input_size
__global__ void init_input_kernel(float* buf, int rank, long input_size);

// verify output[i] = 100*0 + 100*1 + ... 100*(n_ranks-1) + n_ranks*100*i/input_size
bool check_correctness(float* h_res, int rank, int n_ranks, long input_size, float atol);

// get current time in µs
double get_time();

// compute and record average, std, min, and max latency in µs
void analyze_runtime(RunArgs* args, double* deltas);



/** A common interface for the thread function that runs the ring algorithm for a rank.
 *
 * Every implementation must have the following signature and behavior:
 * - it should initialize the input using the init_input_kernel
 * - it should run the ring implementation and set correct using check_correctness
 * - it should run the ring implementation n_warmup times
 * - it should run the ring implementation n_iters more times and set avg_latency using get_time
 */
typedef void* (*RingRunFunc)(RunArgs* args);



// TODO: add new implementations here
void* ring_nccl(RunArgs* args);
void* ring_naive(RunArgs* args);
void* ring_pipelined_nccl(RunArgs* args);
void* ring_pipelined_async(RunArgs* args);