"""Tests for finetune_setup helpers (carved out of run_class_finetuning in PR #3)."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from types import SimpleNamespace

import torch
import torch.utils.data

from finetune_setup import (
    DataLoaders,
    apply_debug_overrides,
    build_dataloaders,
    build_samplers,
    load_finetune_checkpoint,
    resolve_device,
    subset_for_debug,
)


class _SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, n=100):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.zeros(2), torch.tensor(i % 2, dtype=torch.long)


def _base_args(**overrides):
    defaults = dict(
        debug=False, epochs=30, batch_size=64, num_workers=10,
        warmup_epochs=5, save_ckpt=True, dist_eval=False,
        output_dir="", log_dir=None, debug_samples=16,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestApplyDebugOverrides:
    def test_no_op_when_debug_disabled(self):
        args = _base_args(debug=False)
        apply_debug_overrides(args)
        assert args.epochs == 30
        assert args.batch_size == 64
        assert args.num_workers == 10

    def test_shrinks_schedule_in_debug(self):
        args = _base_args(debug=True)
        apply_debug_overrides(args)
        assert args.epochs == 2  # max(1, min(30, 2)) -> 2
        assert args.batch_size == 4
        assert args.num_workers == 0
        assert args.warmup_epochs == 0
        assert args.save_ckpt is False
        assert args.dist_eval is False

    def test_log_dir_falls_back_to_output_dir(self, tmp_path):
        args = _base_args(debug=True, output_dir=str(tmp_path), log_dir=None)
        apply_debug_overrides(args)
        assert args.log_dir == str(tmp_path)

    def test_existing_log_dir_kept(self, tmp_path):
        args = _base_args(debug=True, output_dir=str(tmp_path), log_dir="/elsewhere")
        apply_debug_overrides(args)
        assert args.log_dir == "/elsewhere"

    def test_clamps_epochs_at_least_one(self):
        args = _base_args(debug=True, epochs=0)
        apply_debug_overrides(args)
        assert args.epochs == 1


class TestResolveDevice:
    def test_explicit_cpu(self):
        assert resolve_device("cpu").type == "cpu"

    def test_auto_returns_a_device(self):
        # Just confirm it returns a torch.device without raising
        dev = resolve_device("auto")
        assert isinstance(dev, torch.device)
        assert dev.type in {"cpu", "cuda", "mps"}


class TestSubsetForDebug:
    def test_none_passes_through(self):
        assert subset_for_debug(None, 5) is None

    def test_subset_clamps_to_len(self):
        ds = _SyntheticDataset(n=20)
        sub = subset_for_debug(ds, 100)
        assert len(sub) == 20  # min(100, 20)

    def test_subset_takes_first_n(self):
        ds = _SyntheticDataset(n=20)
        sub = subset_for_debug(ds, 5)
        assert len(sub) == 5

    def test_subsets_each_in_list(self):
        ds = [_SyntheticDataset(n=20), _SyntheticDataset(n=10)]
        out = subset_for_debug(ds, 6)
        assert isinstance(out, list)
        assert [len(d) for d in out] == [6, 6]


class TestBuildSamplers:
    def test_distributed_train_sampler_used(self):
        train, val, test = _SyntheticDataset(), _SyntheticDataset(), _SyntheticDataset()
        st, sv, ste = build_samplers(train, val, test, num_tasks=1, global_rank=0, dist_eval=False)
        assert isinstance(st, torch.utils.data.DistributedSampler)
        # With dist_eval=False, val/test samplers are SequentialSampler
        assert isinstance(sv, torch.utils.data.SequentialSampler)
        assert isinstance(ste, torch.utils.data.SequentialSampler)

    def test_dist_eval_yields_distributed_eval_samplers(self):
        train, val, test = _SyntheticDataset(), _SyntheticDataset(), _SyntheticDataset()
        st, sv, ste = build_samplers(train, val, test, num_tasks=1, global_rank=0, dist_eval=True)
        assert isinstance(sv, torch.utils.data.DistributedSampler)
        assert isinstance(ste, torch.utils.data.DistributedSampler)

    def test_handles_none_eval_datasets(self):
        train = _SyntheticDataset()
        st, sv, ste = build_samplers(train, None, None, num_tasks=1, global_rank=0, dist_eval=False)
        assert sv is None
        assert ste is None

    def test_dataset_test_as_list_yields_list_of_samplers(self):
        train, val = _SyntheticDataset(), _SyntheticDataset()
        test_list = [_SyntheticDataset(), _SyntheticDataset()]
        st, sv, ste = build_samplers(train, val, test_list, num_tasks=1, global_rank=0, dist_eval=True)
        assert isinstance(ste, list) and len(ste) == 2


class TestBuildDataloaders:
    def _build(self, train, val, test):
        st, sv, ste = build_samplers(
            train, val, test, num_tasks=1, global_rank=0, dist_eval=False,
        )
        return build_dataloaders(
            train, val, test, st, sv, ste,
            batch_size=4, num_workers=0, pin_memory=False,
        )

    def test_full_eval_path(self):
        train, val, test = _SyntheticDataset(), _SyntheticDataset(), _SyntheticDataset()
        loaders = self._build(train, val, test)
        assert isinstance(loaders, DataLoaders)
        assert loaders.train is not None
        assert loaders.val is not None
        assert loaders.test is not None

    def test_no_eval_path(self):
        train = _SyntheticDataset()
        loaders = self._build(train, None, None)
        assert loaders.train is not None
        assert loaders.val is None
        assert loaders.test is None

    def test_test_as_list_produces_list_of_loaders(self):
        train, val = _SyntheticDataset(), _SyntheticDataset()
        test_list = [_SyntheticDataset(), _SyntheticDataset()]
        st, sv, ste = build_samplers(
            train, val, test_list, num_tasks=1, global_rank=0, dist_eval=True,
        )
        loaders = build_dataloaders(
            train, val, test_list, st, sv, ste,
            batch_size=4, num_workers=0, pin_memory=False,
        )
        assert isinstance(loaders.test, list) and len(loaders.test) == 2


class TestLoadFinetuneCheckpoint:
    def test_no_op_when_finetune_empty(self):
        # Must not touch model when args.finetune is empty
        args = SimpleNamespace(finetune="", model_key="model", model_filter_name="", model_prefix="")
        load_finetune_checkpoint(model=None, args=args)  # must not raise
