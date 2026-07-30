# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Distributed / multi-process helpers.
# ---------------------------------------------------------

import os

import torch
import torch.distributed as dist


def setup_for_distributed(is_master):
    """Disable printing when not in the master process."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False):
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=op, async_op=async_op)
    return tensor


def all_gather_batch(tensors):
    """Performs all_gather operation on the provided tensors."""
    world_size = get_world_size()
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_all, tensor, async_op=False)
        tensor_list.append(tensor_all)
    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


def gather_sharded_eval(pred, true, groups, total=None):
    """All-gather one rank's eval predictions and rebuild the dataset order.

    Intended for an eval loader sharded by :class:`DistributedSampler`, where
    rank ``r`` of ``W`` holds the samples at original positions
    ``r, r + W, r + 2W, ...``. Interleaving the per-rank arrays in that order
    reconstructs the sampler's (padded) sequence; ``total`` — the underlying
    dataset length — then truncates the duplicate entries DistributedSampler
    appends to equalize shard sizes.

    Without this, each rank computes metrics over its own ~1/W shard: rate
    metrics get noisy, confusion-matrix counts are a fraction of the true
    totals, and per-case window aggregation pools only the windows of a case
    that happened to land on the local rank.

    Returns ``(pred, true, groups)`` unchanged outside distributed mode.
    """
    import numpy as np

    world_size = get_world_size()
    if world_size == 1:
        return pred, true, groups

    buf = [None] * world_size
    dist.all_gather_object(buf, (np.asarray(pred), np.asarray(true), list(groups)))
    return interleave_shards([b[0] for b in buf], [b[1] for b in buf],
                             [b[2] for b in buf], total=total)


def interleave_shards(preds, trues, grps, total=None):
    """Rebuild dataset order from per-rank :class:`DistributedSampler` shards.

    Rank ``r`` of ``W`` holds original positions ``r, r + W, r + 2W, ...``, so
    taking one element from each rank in turn restores the sampler's sequence;
    ``total`` truncates the equalizing duplicates the sampler appends. Split out
    of :func:`gather_sharded_eval` so the reordering is testable without a
    process group.
    """
    import numpy as np

    world_size = len(preds)
    order = [(r, i) for i in range(max(len(p) for p in preds))
             for r in range(world_size) if i < len(preds[r])]
    if total is not None:
        order = order[:total]

    pred_out = np.concatenate([np.asarray(preds[r])[i:i + 1] for r, i in order], axis=0)
    true_out = np.concatenate([np.asarray(trues[r])[i:i + 1] for r, i in order], axis=0)
    # Case ids are absent unless every rank reported them (the dataset yields
    # 2-tuples); an empty list keeps evaluate() on the window-level path.
    groups_out = [grps[r][i] for r, i in order] if all(grps) else []
    return pred_out, true_out, groups_out


class GatherLayer(torch.autograd.Function):
    """all_gather with backward support (gradients are not cut, unlike dist.all_gather)."""

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """all_gather that keeps the graph connected for backward."""
    world_size = get_world_size()
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)
    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


def _get_rank_env():
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return int(os.environ['OMPI_COMM_WORLD_RANK'])


def _get_local_rank_env():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])


def _get_world_size_env():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return int(os.environ['OMPI_COMM_WORLD_SIZE'])


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = _get_rank_env()
        args.world_size = _get_world_size_env()
        args.gpu = _get_local_rank_env()
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
