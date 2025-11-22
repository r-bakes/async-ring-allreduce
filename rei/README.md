# Rei's playground

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

# compile
nvcc -o correctness \
    src/correctness.cu src/ringreduce.cu \
    -I$PSCRATCH/project/include \
    -L$PSCRATCH/project/lib \
    -lnccl -lpthread
nvcc -o benchmark \
    src/benchmark.cu src/ringreduce.cu \
    -I$PSCRATCH/project/include \
    -L$PSCRATCH/project/lib \
    -lnccl -lpthread \
    -O3

# run
LD_LIBRARY_PATH=$PSCRATCH/project/lib NCCL_DEBUG=WARN ./correctness 4 1048576
LD_LIBRARY_PATH=$PSCRATCH/project/lib NCCL_DEBUG=WARN ./benchmark 4
```