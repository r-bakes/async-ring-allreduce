// p2p_ringreduce.cu
// Implements Ring All-Reduce using direct cudaMemcpyAsync (P2P) + CUDA Events.
// FIX: Added cross-rank cudaStreamWaitEvent to prevent race conditions.

#include <assert.h>
#include <stdio.h>
#include <pthread.h>
#include <tuple>

#include "interface.h"

// Global registry for P2P pointers and events
static float* g_buffers[16]; 
static cudaEvent_t g_events[16]; // Events to signal "Data Ready"
static pthread_barrier_t g_barrier;
static bool g_barrier_init = false;

// Helper to handle one-time barrier initialization
static void sync_threads(int n_ranks) {
    static pthread_mutex_t init_lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&init_lock);
    if (!g_barrier_init) {
        pthread_barrier_init(&g_barrier, NULL, n_ranks);
        g_barrier_init = true;
    }
    pthread_mutex_unlock(&init_lock);
    
    pthread_barrier_wait(&g_barrier);
}

static std::pair<int, int> get_offset(int step, int rank, int n_ranks, int chunk_size) {
    int send_chunk = (2 * n_ranks - 1 + rank - step) % n_ranks;
    int recv_chunk = (2 * n_ranks - 2 + rank - step) % n_ranks;
    return {send_chunk * chunk_size, recv_chunk * chunk_size};
}

static __global__ void add_kernel(float* dest, const float* src, int offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dest[offset + idx] += src[idx];
}

static void ring_allreduce_p2p(
    float* d_buf, 
    int input_size, 
    int rank, 
    int n_ranks, 
    cudaStream_t stream
) {
    // 1. Peer Access Setup
    for (int i = 0; i < n_ranks; i++) {
        if (i != rank) {
            cudaDeviceEnablePeerAccess(i, 0);
        }
    }

    // 2. Share Pointers & Create Events
    g_buffers[rank] = d_buf;
    // Create event with DisableTiming (performance) and Interprocess (if needed, though unnecessary for threads)
    CUDA_CALL(cudaEventCreateWithFlags(&g_events[rank], cudaEventDisableTiming));
    
    // Initial Sync: Ensure everyone has allocated and registered
    sync_threads(n_ranks); 

    int prev_rank = (rank - 1 + n_ranks) % n_ranks;
    float* prev_rank_buf = g_buffers[prev_rank];

    assert(input_size % n_ranks == 0);
    int chunk_size = input_size / n_ranks;
    float* scratch_buf = nullptr;
    CUDA_CALL(cudaMalloc(&scratch_buf, chunk_size * sizeof(float)));

    // [CRITICAL FIX]: Record initial event "My Initial Data is Ready"
    // Since we just copied input -> d_buf before calling this function,
    // we must signal that this data is safe to read.
    CUDA_CALL(cudaEventRecord(g_events[rank], stream));

    // --- REDUCE-SCATTER ---
    for (int step = 0; step < n_ranks - 1; step++) {
        auto [send_off, recv_off] = get_offset(step, rank, n_ranks, chunk_size);
        
        // 1. CPU Barrier: Ensure neighbor has recorded their event
        sync_threads(n_ranks); 

        // 2. GPU Wait: stream must wait for Neighbor's stream to finish writing the data it need
        CUDA_CALL(cudaStreamWaitEvent(stream, g_events[prev_rank], 0));

        // Pull Data Safe now
        CUDA_CALL(cudaMemcpyAsync(
            scratch_buf, 
            prev_rank_buf + recv_off, 
            chunk_size * sizeof(float), 
            cudaMemcpyDeviceToDevice, 
            stream
        ));
        
        // Compute
        const int threads = 256;
        int blocks = (chunk_size + threads - 1) / threads;
        add_kernel<<<blocks, threads, 0, stream>>>(d_buf, scratch_buf, recv_off, chunk_size);
        
        // 5. Record Event ready buffer.
        CUDA_CALL(cudaEventRecord(g_events[rank], stream));
    }

    // --- ALL-GATHER ---
    for (int step = n_ranks - 1; step < 2 * (n_ranks - 1); step++) {
        auto [send_off, recv_off] = get_offset(step, rank, n_ranks, chunk_size);
        
        sync_threads(n_ranks);
        CUDA_CALL(cudaStreamWaitEvent(stream, g_events[prev_rank], 0));

        CUDA_CALL(cudaMemcpyAsync(
            d_buf + recv_off, 
            prev_rank_buf + recv_off, 
            chunk_size * sizeof(float), 
            cudaMemcpyDeviceToDevice, 
            stream
        ));
        
        CUDA_CALL(cudaEventRecord(g_events[rank], stream));
    }

    CUDA_CALL(cudaStreamSynchronize(stream));
    CUDA_CALL(cudaFree(scratch_buf));
    CUDA_CALL(cudaEventDestroy(g_events[rank]));
}


// interface function
void* ring_p2p(RunArgs* args) {
    int input_size = args->input_size;
    ncclComm_t comm = args->comm;
    int rank, n_ranks, device;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &n_ranks);
    ncclCommCuDevice(comm, &device);

    CUDA_CALL(cudaSetDevice(device));
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));

    float* d_buf = nullptr;
    CUDA_CALL(cudaMalloc(&d_buf, input_size * sizeof(float)));

    const int threads = 256;
    int blocks = (input_size + threads - 1) / threads;
    init_input_kernel<<<blocks, threads, 0, stream>>>(d_buf, rank, input_size);
    
    // Setup d_outbuf
    float* d_outbuf = nullptr;
    CUDA_CALL(cudaMalloc(&d_outbuf, input_size * sizeof(float)));
    CUDA_CALL(cudaMemcpyAsync(d_outbuf, d_buf, input_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    // Run P2P Ring
    ring_allreduce_p2p(d_outbuf, input_size, rank, n_ranks, stream);

    // Verify
    float* h_res = (float*)malloc(input_size * sizeof(float));
    CUDA_CALL(cudaMemcpy(h_res, d_outbuf, input_size * sizeof(float), cudaMemcpyDeviceToHost));
    *(args->correct) = check_correctness(h_res, rank, n_ranks, input_size, args->atol);
    free(h_res);

    if (!*(args->correct)) {
        CUDA_CALL(cudaFree(d_buf));
        CUDA_CALL(cudaFree(d_outbuf));
        CUDA_CALL(cudaStreamDestroy(stream));
        return nullptr;
    }

    // Warmup
    for (int i = 0; i < args->n_warmup; i++) {
         // Re-init inputs to prevent NaN/Inf blowup over many iterations
         init_input_kernel<<<blocks, threads, 0, stream>>>(d_outbuf, rank, input_size);
         ring_allreduce_p2p(d_outbuf, input_size, rank, n_ranks, stream);
    }

    // Benchmark
    double t0 = get_time();
    for (int i = 0; i < args->n_iters; i++)
        ring_allreduce_p2p(d_outbuf, input_size, rank, n_ranks, stream);
    CUDA_CALL(cudaStreamSynchronize(stream)); 
    double t1 = get_time();
    
    if (rank == 0) *(args->avg_latency) = (t1 - t0) * 1e6 / (double)args->n_iters;

    CUDA_CALL(cudaFree(d_buf));
    CUDA_CALL(cudaFree(d_outbuf));
    CUDA_CALL(cudaStreamDestroy(stream));
    return nullptr;
}