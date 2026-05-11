// benchmark.cu
// Calls the various ring all reduce implementations with various buffer sizes and measures the
// latency while checking for correctness

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>

#include "interface.h"


// TODO: add new implementations here
static RingRunFunc impls[] = {
    paard_pipelined_nccl,
    paard_nccl,
    ring_pipelined_nccl,
    ring_naive,
    halving_doubling_pipelined,
    halving_doubling_allreduce,
};

static const char* impl_names[] = {
    "Pipelined Paard",
    "Classic Paard",
    "Pipelined Ring",
    "Classic Ring",
    "Pipelined HD",
    "Classic HD",
};



// Usage: ./benchmark
int main(int argc, char** argv) {
    // initialize MPI
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // get n_ranks
    if (argc != 2) {
        if (world_rank == 0) printf("missing required positional argument <n_ranks>\n");
        MPI_Finalize();
        return 1;
    }
    int n_ranks = atoi(argv[1]);
    if (n_ranks > world_size) {
        if (world_rank == 0)
            printf("n_ranks %d is larger than the world size %d\n", n_ranks, world_size);
        MPI_Finalize();
        return 1;
    }
    if (n_ranks < 2) {
        if (world_rank == 0) printf("n_ranks must be at least 2, got %d\n", n_ranks);
        MPI_Finalize();
        return 1;
    }

    // create communicator between active ranks and drop unparticipating ranks
    MPI_Comm active_comm;
    MPI_Comm_split(
        MPI_COMM_WORLD, (world_rank < n_ranks) ? 1 : MPI_UNDEFINED, world_rank, &active_comm
    );
    if (active_comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    int rank;
    MPI_Comm_rank(active_comm, &rank);

    // disable certain algorithms based on n_ranks
    bool use_paard = true;
    bool use_tree = true;
    if (n_ranks != 6) {
        if (rank == 0) printf("PAARD will be skipped as n_ranks != 6\n");
        use_paard = false;
    }
    if ((n_ranks & (n_ranks - 1)) != 0) {
        if (rank == 0) printf("HD will be skipped as n_ranks is not a power of 2\n");
        use_tree = false;
    }

    // set up GPU for this process
    int devices_per_node;
    CUDA_CALL(cudaGetDeviceCount(&devices_per_node));
    if (use_paard && devices_per_node != 2) {
        if (rank == 0)
            printf("the number of GPUs/node must equal 2 for PAARD, got %d\n", devices_per_node);
        MPI_Comm_free(&active_comm);
        MPI_Finalize();
        return 1;
    }
    int local_rank = rank % devices_per_node;
    CUDA_CALL(cudaSetDevice(local_rank));
    init_benchmark_knob_from_env();

    int sweep_n_batches = 2;
    const char* be = getenv("ALLREDUCE_B");
    if (!be || be[0] == '\0') be = getenv("ALLREDUCE_N_BATCHES");
    if (be && be[0] != '\0') {
        sweep_n_batches = atoi(be);
        if (sweep_n_batches < 2) sweep_n_batches = 2;
    }

    // get NCCL Unique ID from rank 0
    ncclUniqueId id;
    if (rank == 0) NCCL_CALL(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, active_comm);

    // initialize NCCL communicator
    ncclComm_t comm;
    NCCL_CALL(ncclCommInitRank(&comm, n_ranks, id, rank));

    if (rank == 0)
        printf(
            "impl,input_size,input_bytes,avg_latency,std_latency,min_latency,max_latency,"
            "throughput\n"
        );

    const int n_warmup = 200;
    const int n_iters = 200;
    const float atol = 1e-3f;
    const long min_sz = 384;         // 1.5KB
    const long max_sz = 2147483648;  // 8GB

    constexpr int n_impl = sizeof(impls) / sizeof(impls[0]);
    for (int i = 0; i < n_impl; i++) {
        const auto impl = impls[i];
        const auto& impl_name = impl_names[i];

        if (!use_paard && (impl == paard_nccl || impl == paard_pipelined_nccl)) continue;
        if (!use_tree && (impl == halving_doubling_allreduce || impl == halving_doubling_pipelined))
            continue;

        for (long input_size = min_sz; input_size <= max_sz; input_size *= 2) {
            size_t n_bytes = (size_t)input_size * sizeof(float);

            double local_avg = 1.0;
            double local_std = 0.0;
            double local_min = 0.0;
            double local_max = 0.0;
            bool local_correct = 0;

            RunArgs args;
            args.input_size = input_size;
            args.comm = comm;
            args.n_batches = sweep_n_batches;
            args.n_warmup = n_warmup;
            args.n_iters = n_iters;
            args.atol = atol;
            args.correct = &local_correct;
            args.avg_latency = &local_avg;
            args.std_latency = &local_std;
            args.min_latency = &local_min;
            args.max_latency = &local_max;

            // run the impl
            impl(&args);

            // run correctness check
            int local_correct_int = int(local_correct);
            int global_correct = 0;
            MPI_Allreduce(&local_correct_int, &global_correct, 1, MPI_INT, MPI_MIN, active_comm);
            if (global_correct != 1) {
                if (rank == 0) printf("%s FAILED, stopping\n", impl_name);
                break;
            }

            // get global metrics (max to get slowest out of all ranks)
            double global_avg;
            double global_std;
            double global_min;
            double global_max;
            MPI_Reduce(&local_avg, &global_avg, 1, MPI_DOUBLE, MPI_MAX, 0, active_comm);
            MPI_Reduce(&local_std, &global_std, 1, MPI_DOUBLE, MPI_MAX, 0, active_comm);
            MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MAX, 0, active_comm);
            MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, active_comm);

            if (rank == 0) {
                double throughput = n_bytes / global_avg;
                printf(
                    "%s,%lu,%zu,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                    impl_name,
                    input_size,
                    n_bytes,
                    global_avg,
                    global_std,
                    global_min,
                    global_max,
                    throughput
                );
            }
        }
    }

    // cleanup
    NCCL_CALL(ncclCommDestroy(comm));
    MPI_Comm_free(&active_comm);
    MPI_Finalize();

    return 0;
}
