// benchmark.cu
// Calls the various ring all reduce implementations with various buffer sizes and measures the
// latency while checking for correctness

#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <string>

#include "interface.h"


// TODO: add new implementations here
static RingRunFunc impls[] = {
    ring_pipelined_async,
    ring_nccl,
    ring_naive,
};

static const char* impl_names[] = {
    "Async",
    "NCCL",
    "Classic",
};



// Usage: ./benchmark <n_devices> <filename>
int main(int argc, char** argv) {
    // parse CLI
    if (argc < 3) {
        printf("Usage: %s <n_devices> <filename> (example: %s 4 result.csv)\n", argv[0], argv[0]);
        return 1;
    }
    int n_devices = atoi(argv[1]);
    if (n_devices != 2 && n_devices != 4) {
        fprintf(stderr, "n_devices must be 2 or 4\n");
        return 1;
    }
    std::string filename = "results/" + std::string(argv[2]);

    // initialize device ranks (device[i] = rank[i])
    int devices[n_devices];
    for (int i = 0; i < n_devices; i++) devices[i] = i;

    // create NCCL communicators
    ncclComm_t comms[n_devices];
    NCCL_CALL(ncclCommInitAll(comms, n_devices, devices));

    // create output file
    FILE* f = fopen(filename.c_str(), "w");
    fprintf(
        f,
        "impl,input_size,input_bytes,avg_latency,std_latency,min_latency,max_latency,throughput\n"
    );
    fflush(f);

    const int n_warmup = 200;
    const int n_iters = 200;
    const float atol = 1e-3f;
    const long min_sz = 256;         // 1KB
    const long max_sz = 2147483648;  // 8GB

    const int n_impl = sizeof(impls) / sizeof(impls[0]);
    for (int i = 0; i < n_impl; i++) {
        printf("\n=== Running Implementation: %s ===\n", impl_names[i]);

        // use long as 1073741824*2 could overflow, meaning our for loop won't terminate correctly
        for (long input_size = min_sz; input_size <= max_sz; input_size *= 2) {
            size_t n_bytes = (size_t)input_size * sizeof(float);
            printf("input_size: %lu (%zu bytes), ", input_size, n_bytes);

            // start thread for each GPU
            auto impl = (void* (*)(void*))impls[i];
            double avg_latency = 1.0;
            double std_latency = 0.0;
            double min_latency = 0.0;
            double max_latency = 0.0;

            pthread_t threads[n_devices];
            RunArgs args[n_devices];
            bool corrects[n_devices];
            for (int r = 0; r < n_devices; r++) {
                args[r].input_size = input_size;
                args[r].comm = comms[r];
                args[r].n_warmup = n_warmup;
                args[r].n_iters = n_iters;
                args[r].atol = atol;
                args[r].correct = corrects + r;
                args[r].avg_latency = &avg_latency;
                args[r].std_latency = &std_latency;
                args[r].min_latency = &min_latency;
                args[r].max_latency = &max_latency;

                int rc = pthread_create(&threads[r], nullptr, impl, &args[r]);
                if (rc) {
                    fprintf(stderr, "Failed to create thread %d\n", r);
                    exit(1);
                }
            }

            for (int r = 0; r < n_devices; r++) pthread_join(threads[r], nullptr);

            bool all_correct = true;
            for (int r = 0; r < n_devices; r++) all_correct &= corrects[r];
            if (all_correct) {
                double throughput = n_bytes / avg_latency;
                fprintf(
                    f,
                    "%s,%lu,%zu,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                    impl_names[i],
                    input_size,
                    n_bytes,
                    avg_latency,
                    std_latency,
                    min_latency,
                    max_latency,
                    throughput
                );
                fflush(f);
                printf(
                    "average latency: %.3fµs, throughput: %.3fbytes/µs\n", avg_latency, throughput
                );
            } else {
                printf("FAILED, stopping");
                break;
            }
        }
    }

    // cleanup
    for (int r = 0; r < n_devices; r++) ncclCommDestroy(comms[r]);
    fclose(f);
    printf("\nAll done. Results written to %s\n", filename.c_str());

    return 0;
}
