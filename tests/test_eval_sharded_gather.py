"""Distributed eval gathering: with a DistributedSampler on the eval loader each
rank holds only its ~1/W shard, so metrics (and per-case window pooling above
all) must be computed on the gathered, dataset-ordered predictions."""

import pickle

import numpy as np
import pytest
import torch

from labram.configs.train_config import EvaluationConfig
from labram.data.tuh_datasets import TUABLoader
from labram.runs.finetune_setup import enable_window_ids
from labram.train.train_finetune import (
    _is_sharded_loader,
    _loader_dataset_len,
    evaluate,
)
from labram.utils import gather_sharded_eval, interleave_shards


class _DummyModel(torch.nn.Module):
    def forward(self, x, channel_indices=None, classify_only=True):
        b = x.shape[0]
        return x.reshape(b, -1).mean(dim=1, keepdim=True)


@pytest.fixture
def tuab_dataset(tmp_path):
    root = tmp_path / "tuab"
    root.mkdir()
    files = []
    # 2 recordings x 3 windows.
    for rec, (label, base) in {"aaaa_s1_t0": (1, 1.0), "bbbb_s2_t0": (0, -1.0)}.items():
        for w in range(3):
            name = f"{rec}_{w}.pkl"
            files.append(name)
            with open(root / name, "wb") as f:
                pickle.dump({"X": (np.ones((1, 200)) * base * 100).astype("f4"),
                             "y": label}, f)
    ds = TUABLoader(str(root), files)
    enable_window_ids(ds, group_by="recording")
    return ds


# ------------------------------------------------------------------ reordering


def test_interleave_shards_restores_dataset_order():
    # World of 2: rank 0 held positions 0, 2, 4; rank 1 held 1, 3, 5.
    preds = [np.array([[0.0], [2.0], [4.0]]), np.array([[1.0], [3.0], [5.0]])]
    trues = [np.array([[0], [0], [0]]), np.array([[1], [1], [1]])]
    grps = [["a", "a", "b"], ["a", "b", "b"]]

    pred, true, groups = interleave_shards(preds, trues, grps)

    assert pred.ravel().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert true.ravel().tolist() == [0, 1, 0, 1, 0, 1]
    assert groups == ["a", "a", "a", "b", "b", "b"]


def test_interleave_shards_truncates_sampler_padding():
    """DistributedSampler pads to equal shard sizes; ``total`` drops the
    duplicates so confusion-matrix counts match the real dataset size."""
    # 5 samples over 2 ranks -> the sampler pads to 6 (position 0 repeated).
    preds = [np.array([[0.0], [2.0], [4.0]]), np.array([[1.0], [3.0], [0.0]])]
    trues = [np.array([[0], [0], [0]]), np.array([[1], [1], [0]])]
    grps = [["a", "a", "b"], ["a", "b", "a"]]

    pred, true, groups = interleave_shards(preds, trues, grps, total=5)

    assert pred.ravel().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert len(true) == 5 and len(groups) == 5


def test_interleave_shards_without_ids_yields_no_groups():
    preds = [np.array([[0.0]]), np.array([[1.0]])]
    trues = [np.array([[0]]), np.array([[1]])]
    pred, true, groups = interleave_shards(preds, trues, [[], []])
    assert groups == []
    assert len(pred) == 2


def test_gather_sharded_eval_is_noop_outside_distributed():
    pred, true = np.array([[1.0], [2.0]]), np.array([[1], [0]])
    out = gather_sharded_eval(pred, true, ["a", "b"])
    assert out[0] is pred and out[1] is true and out[2] == ["a", "b"]


# ------------------------------------------------------------------ detection


def test_sharded_loader_detection(tuab_dataset):
    plain = torch.utils.data.DataLoader(tuab_dataset, batch_size=2)
    assert _is_sharded_loader(plain) is False
    assert _loader_dataset_len(plain) == 6

    sampler = torch.utils.data.DistributedSampler(
        tuab_dataset, num_replicas=2, rank=0, shuffle=False)
    sharded = torch.utils.data.DataLoader(tuab_dataset, sampler=sampler, batch_size=2)
    assert _is_sharded_loader(sharded) is True
    assert _loader_dataset_len(sharded) == 6


# ------------------------------------------------------------------ end to end


def test_evaluate_on_unsharded_loader_sees_every_window(tuab_dataset):
    """Baseline for the sharding test below: without a DistributedSampler every
    window and both cases are scored."""
    loader = torch.utils.data.DataLoader(tuab_dataset, batch_size=2)
    eval_cfg = EvaluationConfig(agg_windows="mean", detailed_metrics=True,
                                log_confusion_matrix=False, log_curves=False)
    stats = evaluate(loader, _DummyModel(), torch.device("cpu"),
                     metrics=["accuracy"], is_binary=True, nb_classes=1,
                     eval_cfg=eval_cfg)
    cells = ["cm_tp", "cm_tn", "cm_fp", "cm_fn"]
    assert sum(stats[c] for c in cells) == 2                    # 2 cases
    assert sum(stats[f"window_{c}"] for c in cells) == 6        # 6 windows


def test_evaluate_on_a_shard_covers_only_the_local_slice(tuab_dataset):
    """Documents what the gather exists to fix: a rank's DistributedSampler
    shard holds a fraction of each case's windows. Outside a process group no
    gather is possible, so this measures the un-gathered shard directly."""
    sampler = torch.utils.data.DistributedSampler(
        tuab_dataset, num_replicas=2, rank=0, shuffle=False)
    loader = torch.utils.data.DataLoader(tuab_dataset, sampler=sampler, batch_size=2)
    eval_cfg = EvaluationConfig(agg_windows="mean", detailed_metrics=True,
                                log_confusion_matrix=False, log_curves=False)
    stats = evaluate(loader, _DummyModel(), torch.device("cpu"),
                     metrics=["accuracy"], is_binary=True, nb_classes=1,
                     eval_cfg=eval_cfg)
    # Rank 0 of 2 sees 3 of the 6 windows; both cases still appear, but each is
    # pooled from half its windows. The gather in evaluate() restores all 6.
    cells = ["cm_tp", "cm_tn", "cm_fp", "cm_fn"]
    assert sum(stats[f"window_{c}"] for c in cells) == 3
    assert sum(stats[c] for c in cells) == 2
