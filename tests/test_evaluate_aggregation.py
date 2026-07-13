"""Integration test: evaluate() consumes a loader that yields per-window case
ids (3-tuples) and aggregates window predictions per case (Task 5), and returns
the detailed metric set (Task 3)."""

import pickle

import numpy as np
import pytest
import torch

from labram.configs.train_config import EvaluationConfig
from labram.data.tuh_datasets import TUABLoader
from labram.runs.finetune_setup import enable_window_ids
from labram.train.train_finetune import evaluate


class _DummyModel(torch.nn.Module):
    """Maps each window to a positive-class logit correlated with its label,
    which is embedded as a constant offset in the synthetic signal."""

    def forward(self, x, channel_indices=None, classify_only=True):
        # x: [B, N, A, T]; use the per-sample mean as the logit.
        b = x.shape[0]
        return x.reshape(b, -1).mean(dim=1, keepdim=True)


@pytest.fixture
def tuab_loader(tmp_path):
    root = tmp_path / "tuab"
    root.mkdir()
    files = []
    # 2 recordings x 3 windows. Recording A is label 1 (high signal), B label 0.
    for rec, (label, base) in {"aaaa_s1_t0": (1, 1.0), "bbbb_s2_t0": (0, -1.0)}.items():
        for w in range(3):
            name = f"{rec}_{w}.pkl"
            files.append(name)
            with open(root / name, "wb") as f:
                pickle.dump({"X": (np.ones((1, 200)) * base * 100).astype("f4"),
                             "y": label}, f)
    ds = TUABLoader(str(root), files)
    enable_window_ids(ds, group_by="recording")
    return torch.utils.data.DataLoader(ds, batch_size=2)


def test_evaluate_aggregates_windows_into_cases(tuab_loader):
    eval_cfg = EvaluationConfig(agg_windows="mean", detailed_metrics=True,
                                log_confusion_matrix=False, log_curves=False)
    stats = evaluate(tuab_loader, _DummyModel(), torch.device("cpu"),
                     metrics=["accuracy", "balanced_accuracy"], is_binary=True,
                     nb_classes=1, eval_cfg=eval_cfg)
    # Two cases, cleanly separable -> perfect metrics after aggregation.
    assert stats["accuracy"] == pytest.approx(1.0)
    assert stats["cm_tp"] + stats["cm_tn"] == 2  # exactly 2 cases classified
    assert "sensitivity" in stats and "specificity" in stats


def test_evaluate_without_aggregation_is_window_level(tuab_loader):
    eval_cfg = EvaluationConfig(agg_windows="none")
    stats = evaluate(tuab_loader, _DummyModel(), torch.device("cpu"),
                     metrics=["accuracy"], is_binary=True, nb_classes=1,
                     eval_cfg=eval_cfg)
    # 6 windows scored individually.
    assert stats["cm_tp"] + stats["cm_tn"] + stats["cm_fp"] + stats["cm_fn"] == 6
