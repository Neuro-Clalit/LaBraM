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


def build_samplers(dataset_train, dataset_val, dataset_test, num_tasks: int, global_rank: int, dist_eval: bool):
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True,
    )
    logger.info("Sampler_train = %s", str(sampler_train))

    if dist_eval:
        if dataset_val is not None and len(dataset_val) % num_tasks != 0:
            logger.warning(
                'Enabling distributed evaluation with an eval dataset not divisible by process number. '
                'This will slightly alter validation results as extra duplicate entries are added to achieve '
                'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        if isinstance(dataset_test, list):
            sampler_test = [torch.utils.data.DistributedSampler(
                d, num_replicas=num_tasks, rank=global_rank, shuffle=False) for d in dataset_test]
        else:
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val is not None else None
        sampler_test = torch.utils.data.SequentialSampler(dataset_test) if dataset_test is not None else None

    return sampler_train, sampler_val, sampler_test


def build_dataloaders(dataset_train, dataset_val, dataset_test,
                      sampler_train, sampler_val, sampler_test,
                      batch_size: int, num_workers: int, pin_memory: bool) -> DataLoaders:
    eval_batch_size = int(1.5 * batch_size)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    if dataset_val is None:
        return DataLoaders(train=train_loader, val=None, test=None)

    val_loader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    if isinstance(dataset_test, list):
        test_loader = [torch.utils.data.DataLoader(
            d, sampler=s,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ) for d, s in zip(dataset_test, sampler_test)]
    else:
        test_loader = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return DataLoaders(train=train_loader, val=val_loader, test=test_loader)


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
        checkpoint_model = new_dict

    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            logger.info("Removing key %s from pretrained checkpoint", k)
            del checkpoint_model[k]

    for key in list(checkpoint_model.keys()):
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

    utils.load_state_dict(model, checkpoint_model, prefix=ckpt_cfg.model_prefix)
