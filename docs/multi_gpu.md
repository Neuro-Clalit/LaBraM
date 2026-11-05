# Multi-GPU training and inference (DDP)

Every trainer (VQNSP, pre-training, fine-tuning, cross-validation) and the
offline scorer `scripts/eval_age_by_cohort.py` run on several GPUs with PyTorch
**DistributedDataParallel** (DDP): one process per GPU, each with a full model
replica and its own shard of every batch. After each backward pass the
gradients are averaged across processes, so all replicas take the same
optimizer step. One config switch turns it on; no `torchrun` command line is needed.

## 1. Turning it on

```bash
# All visible GPUs of this machine (local run):
python -m labram.runs.run_finetune --config labram/configs/defaults/finetune_tuab_age.json \
  --set distributed.nproc_per_node=-1 trainer.global_batch_size=128

# Same thing, written out by hand (still supported):
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 -m labram.runs.run_finetune \
  --config labram/configs/defaults/finetune_tuab_age.json --set trainer.global_batch_size=128

# SageMaker, one 4-GPU instance:
python -m labram.runs.submit_sagemaker --config ... --set sagemaker.enabled=true \
  sagemaker.instance_type=ml.g5.12xlarge distributed.nproc_per_node=-1 trainer.global_batch_size=128

# Offline scoring on all GPUs:
python scripts/eval_age_by_cohort.py --data-path ... --run E1=... --nproc-per-node -1
```

### Table 1.1. Options

| Option | Default | Meaning |
|---|---|---|
| `distributed.nproc_per_node` | `1` | processes per machine, one per GPU; `-1` = every visible GPU. Above 1 the entry point replaces itself with `python -m torch.distributed.run` (`labram/utils/launch.py`) |
| `trainer.global_batch_size` | `0` (off) | fine-tune only: samples per optimizer step $B$ of Eq. (2.1), kept the same for any number of GPUs; `trainer.batch_size` becomes the per-GPU cap |
| `distributed.backend` | `auto` | `nccl` on CUDA, `gloo` on CPU (tests, CPU-only machines) |
| `distributed.find_unused_parameters` | `true` | DDP tolerates parameters without a gradient in a step; `false` is faster when every trainable parameter is used |
| `distributed.dist_eval` | `false` | implied under DDP for fine-tuning: val, test and train_eval are sharded across ranks and gathered back |
| `--nproc-per-node` (`eval_age_by_cohort.py`) | `1` | GPUs used for prediction |

How the relaunch works: the process checks whether it is already a torchrun
worker (`LOCAL_RANK` / `RANK` set). If it is not and more than one process is
wanted, it `exec`s torchrun, which starts the workers as
`python -m labram.utils.launch`. That trampoline reads the real entry point and
its arguments from `$LABRAM_LAUNCH_TARGET` (JSON), so torchrun never parses
them; otherwise it would take an option such as `--run` for its own `--run-path`.
`OMP_NUM_THREADS` defaults to 1 per worker. In a multi-instance SageMaker job,
`--nnodes`, `--node_rank` and `--master_addr` come from `SM_HOSTS` /
`SM_CURRENT_HOST`.

## 2. What DDP computes

With $W$ processes (GPUs), a per-GPU micro-batch of $b$ samples and $k$
gradient-accumulation micro-steps (`trainer.update_freq`), one optimizer step
sees

$$
B = W \cdot k \cdot b \tag{2.1}
$$

samples, where $B$ is the effective (global) batch, $W$ the world size, $k$ the
number of micro-steps and $b$ the per-GPU batch (`trainer.batch_size`). The
gradient applied to the shared weights $\theta$ is the average of the per-rank
gradients, each the sum over its $k$ micro-steps of the mean loss $L$ divided
by $k$ (`train_finetune.train_one_epoch` divides the loss by `update_freq`):

$$
g = \frac{1}{W} \sum_{r=1}^{W} \sum_{j=1}^{k} \frac{1}{k}\, \nabla_\theta
    L\!\left(x^{(r,j)}_{1..b}\right) \tag{2.2}
$$

where $x^{(r,j)}_{1..b}$ is micro-batch $j$ on rank $r$. That is the gradient
of the mean loss over all $B$ samples, i.e. the same update one GPU would make
on a batch of $B$. DDP performs the $\tfrac{1}{W}\sum_r$ as bucketed all-reduces
that overlap with the backward pass.

### 2.1 Keeping the recipe when changing the GPU count

The fine-tune LR is not scaled with the batch, so a recipe tuned at one $B$
should keep that $B$. With `trainer.global_batch_size` = $B$ and per-GPU cap
$b_{\max}$ (`trainer.batch_size`), `runs/common.py::resolve_global_batch` picks

$$
k = \left\lceil \frac{B}{W\, b_{\max}} \right\rceil, \qquad b = \frac{B}{W\,k} \tag{2.3}
$$

and refuses a $B$ that does not split into whole micro-batches. Examples:

### Table 2.1. Global batch 128 on 1, 4 and 8 GPUs (cap 128)

| GPUs $W$ | per-GPU $b$ | micro-steps $k$ | steps per epoch |
|---|---|---|---|
| 1 | 128 | 1 | $N/128$ |
| 4 | 32 | 1 | $N/128$ |
| 8 | 16 | 1 | $N/128$ |

The number of optimizer steps, the LR schedule and the warm-up are unchanged;
each step costs about $1/W$ of the single-GPU time. Without
`global_batch_size`, $B$ grows with $W$ by Eq. (2.1) and the LR usually needs
scaling, e.g. linearly, $\mathrm{lr}' = \mathrm{lr}\cdot B'/B$. VQNSP and
pre-training already apply the linear rule (`lr x B / 128`).

## 3. Best practices applied

- **No gradient sync on accumulation micro-steps.** Micro-steps before an
  update run under `DDP.no_sync()` (forward and backward). Only the last one
  all-reduces the accumulated gradients, which cuts communication by a factor $k$
  (`train_finetune.train_one_epoch`; pre-training did this already).
- **Every rank owns one GPU.** `torch.cuda.set_device(LOCAL_RANK)`, NCCL bound
  to that device, `device_ids=[LOCAL_RANK]`; `gradient_as_bucket_view=True` (one gradient copy less).
- **Identical start, different data.** DDP broadcasts rank 0's weights at
  construction. The train `DistributedSampler` (or `RecordingSubsetSampler` /
  age-bin sampler) gives each rank a disjoint shard, reshuffled each epoch with
  `set_epoch`. Augmentation seeds differ per rank (`seed + rank`).
- **Sharded, exact evaluation.** Each rank scores every $W$-th window; the shards
  are all-gathered back in dataset order and the sampler's padding duplicates
  are dropped (`utils.gather_sharded_eval`), so per-recording pooling, per-cohort
  and per-dataset metrics match a single-GPU evaluation (checked in
  `tests/test_multi_gpu.py`: same checkpoint, 1 vs 2 processes, identical metrics).
- **Rank 0 only** writes checkpoints, logs, TensorBoard/ClearML and the SageMaker
  `model.tar.gz` contents. The first rank of each machine stages data onto that
  machine's NVMe while the others wait at a barrier.
- **Clean shutdown.** After a successful run every rank leaves the process group
  together (`utils.cleanup_distributed`, from `exit_guard.run_and_exit`).
- **CPU fallback.** `gloo` when there is no CUDA, so the whole path is testable on CPU.

## 4. Practical notes

- **Data loading** is per process: `data.num_workers` workers per GPU. On a
  48-vCPU g5.12xlarge with 4 GPUs, 6–10 workers each is a sensible range.
- **SageMaker.** Multi-GPU instances: ml.g5.12xlarge / g5.24xlarge (4 × A10G),
  g5.48xlarge (8 × A10G), p4d.24xlarge (8 × A100). The account needs a training
  quota for the instance type (Service Quotas, "ml.g5.12xlarge for training job usage").
- **Small eval sets.** With $W$ ranks the last eval shard may hold up to $W-1$
  padding samples; they are dropped after the gather, so the metrics are unaffected.
- **BatchNorm.** LaBraM uses LayerNorm/GroupNorm only, so no `SyncBatchNorm` is
  needed. A future model with BatchNorm should convert it
  (`torch.nn.SyncBatchNorm.convert_sync_batchnorm`) before DDP-wrapping.
