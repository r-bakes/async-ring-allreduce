# Copied Rei's playground

## How to run

First-time:

```shell
conda create --prefix $PSCRATCH/project -c nvidia nccl
```

Every-time:

```shell
# start up GPUs & navigate to directory
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account m4999_g
cd ~/CS5470/async-ring-allreduce/rei

conda activate $PSCRATCH/project

# compile (TODO: add new implementations here to be compiled)
# NOTE: for the real benchmarking, compile with -DNDEBUG and -O2
nvcc -o benchmark \
    src/benchmark.cu src/utils.cu \
    src/nccl_ringreduce.cu src/naive_ringreduce.cu src/pipelined_ringreduce.cu \
    -I$PSCRATCH/project/include \
    -L$PSCRATCH/project/lib \
    -lnccl -lpthread

# Benchmark

nvcc -O3 -DNDEBUG -o benchmark \
    src/benchmark.cu src/utils.cu \
    src/nccl_ringreduce.cu src/naive_ringreduce.cu src/pipelined_ringreduce.cu \
    src/p2p_ringreduce.cu \
    -I$PSCRATCH/project/include \
    -L$PSCRATCH/project/lib \
    -lnccl -lpthread


# run for 4 gpus
LD_LIBRARY_PATH=$PSCRATCH/project/lib NCCL_DEBUG=WARN ./benchmark 4 output.csv

# run for 2 gpus
LD_LIBRARY_PATH=$PSCRATCH/project/lib NCCL_DEBUG=WARN ./benchmark 2 output.csv
```

