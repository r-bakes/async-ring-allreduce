# AsyncAllReduce

## How to run

```shell
cd $PSCRATCH/async-ring-allreduce/
./build.sh       # compile, optionally pass -r to build in release mode
sbatch ./run.sh  # run, optionally pass -r to run in release mode, and -n=N_RANKS to run with N_RANKS ranks
```

## Experiment knobs (3 env vars)

Set before `sbatch` / `srun`; read once per rank after `cudaSetDevice` (`init_benchmark_knob_from_env()` in `src/utils.cu`).

| Variable | Meaning | Default |
|----------|---------|---------|
| `ALLREDUCE_B` | Micro-batches (`b`) for **Pipelined Ring** and **Pipelined HD** only (≥ 2). PAARD ignores this. | `2` |
| `ALLREDUCE_COMPUTE_NS` | `__nanosleep` inside `add_kernel` (simulate reduction cost). | `5000` |
| `ALLREDUCE_INTER_US` | Synthetic delay after each **cross-group** `ncclSend/Recv`: **omit** unset → legacy size-proportional `(float_count >> 8)` ns; **set** to `N` → fixed **`N` microseconds** per step (`N=0` = hardware only). | _(unset,_ legacy prop.) |

Sweep example:

```shell
ALLREDUCE_B=4 ALLREDUCE_COMPUTE_NS=8000 ALLREDUCE_INTER_US=50 sbatch run.sh -r
```

Older names still work as fallbacks in code only: `ALLREDUCE_N_BATCHES`, `ALLREDUCE_REDUCE_NS`.

## Contributing

To add a new implementation, you will have to modify these files
- `src/your-impl.cu` containing the implementation, refer to `src/interface.h`
- `src/interface.h` containing the function signature for your implementation
- `src/benchmark.cu` with `impls` and `impl_names` updated accordingly
- `bench.sh` to compile with the newly created `your-impl.cu`