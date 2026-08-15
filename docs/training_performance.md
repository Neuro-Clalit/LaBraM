# Training performance timing

Each training engine now records timing metrics at two levels:

- Per batch: `data_time_sec`, `step_time_sec`, and `host_compute_time_sec`.
- Per epoch: train, validation, test, checkpoint, complete-epoch, and
  elapsed-run time; phases with known sample counts also report samples/sec.

The values are written to the console, the epoch `log.txt` record, and the
configured TensorBoard/ClearML writer under the `timing` series. `step_time_sec`
is the low-overhead host wall-clock measurement. To add precise GPU kernel time,
set `logging.precise_cuda_timing=true`; CUDA events then add
`gpu_compute_time_sec` without synchronizing after every batch.

## Current evidence

The checked-in local debug logs are small synthetic runs, not production
benchmarks. They nevertheless show that data wait was a small fraction of the
observed iteration time:

| Phase | Training step | Data wait | Validation step | Data wait |
| --- | ---: | ---: | ---: | ---: |
| VQNSP | 0.42–0.55 s | 0.01–0.03 s | 0.22–0.26 s | 0.01–0.014 s |
| Pre-training | 0.44–0.51 s | 0.006–0.014 s | — | — |
| Fine-tuning | 0.04–0.06 s | below 0.003 s | 0.009–0.010 s | below 0.002 s |

Use a representative target-GPU run before drawing production conclusions.
The epoch-level timings identify whether data loading, GPU work, checkpointing,
or evaluation is the largest cost.

## Recommendations

1. Keep per-step CUDA synchronization disabled in normal training. The previous
   unconditional synchronization in every engine serialized CPU/GPU work and
   reduced throughput. Enable precise CUDA timing only while profiling.
2. In pre-training, measure the frozen VQNSP tokenizer separately before
   optimizing it. It performs a second GPU model pass for every batch; cached
   token targets may help only if preprocessing/storage and augmentation
   semantics permit it.
3. For fine-tuning, evaluate the cost of detailed metrics, curve artifacts, and
   running the test split every epoch. Routine experiments usually need
   validation each epoch and test evaluation only for selected/final checkpoints.
4. Tune `num_workers`, persistent workers, prefetching, and pin-memory only when
   `data_time_sec` is a material share of the target workload.
