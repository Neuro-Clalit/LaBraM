# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Setup helpers (debug overrides, device resolution, dataloader construction,
# checkpoint loading) for run_class_finetuning.
# ---------------------------------------------------------

from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.utils.data

import labram.utils as utils

logger = utils.get_logger(__name__)


def apply_debug_overrides(args) -> None:
    """In-place: shrink training schedule for fast smoke runs."""
    if not args.debug:
        return
    logger.info("[DEBUG MODE] Overriding training args for fast iteration")
    args.epochs = max(1, min(args.epochs, 2))
    args.batch_size = min(args.batch_size, 4)
    args.num_workers = 0
    args.warmup_epochs = 0
    args.save_ckpt = False
    args.dist_eval = False
    if args.output_dir:
        args.log_dir = args.log_dir or args.output_dir


def resolve_device(requested: str) -> torch.device:
    """Map 'auto' to the best available device; otherwise honor the request."""
    if requested == 'auto':
        if torch.cuda.is_available():
            requested = 'cuda'
        elif torch.backends.mps.is_available():
            requested = 'mps'
        else:
            requested = 'cpu'
    return torch.device(requested)


def enable_window_ids(dataset, group_by: str = 'recording', *, warn: bool = True) -> int:
    """In-place: make the leaf datasets under ``dataset`` yield a per-window case
    id, so evaluation can aggregate window predictions per recording/subject.

    Walks the whole wrapper tree — ``list``, :class:`~torch.utils.data.Subset`
    (``.dataset``) and :class:`~torch.utils.data.ConcatDataset` (``.datasets``),
    nested arbitrarily — and flips ``return_id`` on every leaf that supports it.
    Recursing into ``ConcatDataset`` matters for cross-validation: a fold's split
    concatenates one loader per source (train + val roots), so without it no leaf
    is ever reached and per-case aggregation silently degrades to window-level.

    Returns the number of leaf datasets that accepted ids. When that is zero and
    ``warn`` is set, logs a warning — an aggregation mode was configured but the
    data cannot support it (e.g. a non-TUH loader), which would otherwise fail
    silently inside ``evaluate``."""
    if dataset is None:
        return 0
    n = _enable_window_ids(dataset, group_by)
    if n == 0 and warn:
        logger.warning(
            "Per-case window aggregation is configured but no dataset in %s "
            "supports per-window case ids; evaluation will report window-level "
            "metrics only.", type(dataset).__name__)
    return n


def _enable_window_ids(dataset, group_by: str) -> int:
    """Recursive worker for :func:`enable_window_ids`; returns the leaf count."""
    if dataset is None:
        return 0
    if isinstance(dataset, (list, tuple)):
        return sum(_enable_window_ids(d, group_by) for d in dataset)
    if isinstance(dataset, torch.utils.data.Subset):
        return _enable_window_ids(dataset.dataset, group_by)
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        return sum(_enable_window_ids(d, group_by) for d in dataset.datasets)
    if hasattr(dataset, 'return_id'):
        dataset.return_id = True
        dataset.group_by = group_by
        return 1
    return 0


def subset_for_debug(dataset, n: int):
    """None passthrough; for a single Dataset return a torch.utils.data.Subset
    of the first min(n, len) items; for a list of Datasets apply recursively."""
    if dataset is None:
        return None
    if isinstance(dataset, list):
        return [subset_for_debug(d, n) for d in dataset]
    return torch.utils.data.Subset(dataset, list(range(min(n, len(dataset)))))


@dataclass
class DataLoaders:
    train: torch.utils.data.DataLoader
    val: Optional[torch.utils.data.DataLoader]
    test: Optional[Union[torch.utils.data.DataLoader, List[torch.utils.data.DataLoader]]]
    # A fixed sample of training recordings scored like val/test each epoch
    # (``evaluation.train_eval_recordings``); None when off.
    train_eval: Optional[torch.utils.data.DataLoader] = None


def build_train_eval_loader(dataset_train, eval_cfg, data_cfg, batch_size: int, num_workers: int,
                            pin_memory: bool, seed: int = 0, distributed: bool = False):
    """The ``train_eval`` loader: up to ``eval_cfg.train_eval_recordings``
    training recordings per source dataset, each with its first
    ``data_cfg.eval_minutes`` of listed windows, scored sequentially with
    per-recording ids. None when disabled or nothing can be selected."""
    from labram.data.dataset_groups import train_eval_subset
    n = int(getattr(eval_cfg, 'train_eval_recordings', 0) or 0)
    if n <= 0 or dataset_train is None:
        return None
    windows = max(1, int(getattr(data_cfg, 'eval_minutes', 5.0) * 60
                         // max(getattr(data_cfg, 'window_sec', 10), 1)))
    subset = train_eval_subset(dataset_train, n, windows, seed=seed)
    if subset is None:
        return None
    if getattr(eval_cfg, 'agg_windows', 'none') != 'none':
        enable_window_ids(subset, eval_cfg.agg_case_by)
    logger.info("train_eval: %d windows from up to %d training recordings per dataset",
                len(subset), n)
    # Under DDP each rank scores a shard; evaluate() gathers them back.
    sampler = (torch.utils.data.DistributedSampler(subset, shuffle=False) if distributed
               else torch.utils.data.SequentialSampler(subset))
    return torch.utils.data.DataLoader(
        subset, sampler=sampler,
        batch_size=int(getattr(eval_cfg, 'eval_batch_size', 0) or 0) or int(1.5 * batch_size),
        num_workers=num_workers,
        pin_memory=pin_memory, drop_last=False)


def build_samplers(dataset_train, dataset_val, dataset_test, num_tasks: int, global_rank: int, dist_eval: bool):
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True,
    )
    logger.info("Sampler_train = %s", str(sampler_train))

    if dist_eval:
        # Each rank scores every num_tasks-th sample; evaluate() all-gathers the
        # shards back into dataset order and drops the padding duplicates the
        # sampler adds to equalise shard sizes, so metrics are exact.
        def shard(d):
            return (torch.utils.data.DistributedSampler(
                d, num_replicas=num_tasks, rank=global_rank, shuffle=False)
                if d is not None else None)
        sampler_val = shard(dataset_val)
        sampler_test = ([shard(d) for d in dataset_test] if isinstance(dataset_test, list)
                        else shard(dataset_test))
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val is not None else None
        sampler_test = torch.utils.data.SequentialSampler(dataset_test) if dataset_test is not None else None

    return sampler_train, sampler_val, sampler_test


def build_dataloaders(dataset_train, dataset_val, dataset_test,
                      sampler_train, sampler_val, sampler_test,
                      batch_size: int, num_workers: int, pin_memory: bool,
                      eval_batch_size: int = 0, collate_fn=None) -> DataLoaders:
    eval_batch_size = int(eval_batch_size) or int(1.5 * batch_size)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
    )

    if dataset_val is None:
        return DataLoaders(train=train_loader, val=None, test=None)

    val_loader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )

    if dataset_test is None:          # e.g. a cohort without a test split
        test_loader = None
    elif isinstance(dataset_test, list):
        test_loader = [torch.utils.data.DataLoader(
            d, sampler=s,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_fn,
        ) for d, s in zip(dataset_test, sampler_test)]
    else:
        test_loader = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_fn,
        )

    return DataLoaders(train=train_loader, val=val_loader, test=test_loader)


def freeze_except(model: torch.nn.Module, prefixes) -> int:
    """Freeze every parameter whose name does not start with one of
    ``prefixes``; returns the number of parameters left trainable. Raises when
    no parameter matches, which would otherwise train nothing."""
    prefixes = tuple(prefixes)
    trainable = 0
    for name, param in model.named_parameters():
        keep = name.startswith(prefixes)
        param.requires_grad_(keep)
        trainable += param.numel() if keep else 0
    if trainable == 0:
        raise ValueError(f"model.trainable_prefixes={list(prefixes)} matches no parameter")
    return trainable


def required_time_patches(data_cfg) -> int:
    """Time-embedding rows a fine-tune model needs: one per 1 s patch of the
    ``data.window_sec`` input, and never fewer than the pretrained 16."""
    return max(16, int(getattr(data_cfg, 'window_sec', 10) or 10))


def resize_time_embed(time_embed: torch.Tensor, n_patches: int) -> torch.Tensor:
    """Linearly interpolate a ``(1, T, D)`` time embedding to ``n_patches`` rows
    (the usual ViT position-embedding resize), so a 16-patch pretrained
    embedding can drive a longer input."""
    if time_embed.shape[1] == n_patches:
        return time_embed
    resized = torch.nn.functional.interpolate(
        time_embed.transpose(1, 2).float(), size=n_patches,
        mode='linear', align_corners=True)
    return resized.transpose(1, 2).to(time_embed.dtype)


def load_finetune_checkpoint(model: torch.nn.Module, ckpt_cfg) -> None:
    """Load weights from ckpt_cfg.finetune into model.

    Accepts a FinetuneCheckpointConfig or any object with finetune, model_key,
    model_filter_name, and model_prefix attributes.
    """
    if not ckpt_cfg.finetune:
        return

    if ckpt_cfg.finetune.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            ckpt_cfg.finetune, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(ckpt_cfg.finetune, map_location='cpu', weights_only=False)

    logger.info("Load ckpt from %s", ckpt_cfg.finetune)

    checkpoint_model = None
    for model_key in ckpt_cfg.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            logger.info("Load state_dict by model_key = %s", model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    if ckpt_cfg.model_filter_name != '':
        new_dict = OrderedDict()
        for key, value in checkpoint_model.items():
            if key.startswith('student.'):
                new_dict[key[len('student.'):]] = value
        if new_dict:
            checkpoint_model = new_dict
        else:
            # A fine-tuned checkpoint (e.g. a warm start from a previous run)
            # has plain keys; filtering it would load nothing at all.
            logger.info("No 'student.' keys in %s: loading its weights unfiltered", ckpt_cfg.finetune)

    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        # Absent from the model when head_type=attention (head.q/kv/proj/mlp/fc).
        if k in checkpoint_model and (k not in state_dict
                                      or checkpoint_model[k].shape != state_dict[k].shape):
            logger.info("Removing key %s from pretrained checkpoint", k)
            del checkpoint_model[k]

    for key in list(checkpoint_model.keys()):
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

    if 'time_embed' in checkpoint_model and 'time_embed' in state_dict:
        n_patches = state_dict['time_embed'].shape[1]
        if checkpoint_model['time_embed'].shape[1] != n_patches:
            logger.info("Resizing time_embed %d -> %d patches",
                        checkpoint_model['time_embed'].shape[1], n_patches)
            checkpoint_model['time_embed'] = resize_time_embed(
                checkpoint_model['time_embed'], n_patches)

    utils.load_state_dict(model, checkpoint_model, prefix=ckpt_cfg.model_prefix)
