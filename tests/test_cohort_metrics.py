"""Evaluation reports normal / abnormal recordings separately (case level)."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from labram.configs.train_config import EvaluationConfig
from labram.train import train_finetune as tf
from test_finetune import N_CHANNELS, T_PATCH, _make_model

LABELS = {"rec_a": "normal", "rec_b": "normal", "rec_c": "abnormal", "rec_d": "abnormal",
          "rec_e": "abnormal"}


def _loader():
    ids = [r for r in LABELS for _ in range(2)]
    y = torch.tensor([20.0 + 10 * i for i, r in enumerate(LABELS) for _ in range(2)])
    X = torch.randn(len(ids), N_CHANNELS, T_PATCH)

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(ids)

        def __getitem__(self, i):
            return X[i], y[i], ids[i]
    return DataLoader(_DS(), batch_size=4)


def _evaluate(monkeypatch, labels):
    monkeypatch.setattr(tf, "_case_labels", lambda loader: labels)
    return tf.evaluate(_loader(), _make_model(num_classes=1), torch.device("cpu"),
                       metrics=["mae", "r2"], is_binary=False, nb_classes=1, task="regression",
                       eval_cfg=EvaluationConfig(agg_windows="mean", detailed_metrics=False))


def test_each_cohort_gets_case_level_metrics(monkeypatch):
    ret = _evaluate(monkeypatch, LABELS)
    assert ret["normal_n_cases"] == 2 and ret["abnormal_n_cases"] == 3
    assert {"normal_mae", "normal_r2", "abnormal_mae", "abnormal_r2"} <= set(ret)
    # The case-weighted cohort MAEs recombine into the overall case MAE.
    combined = (2 * ret["normal_mae"] + 3 * ret["abnormal_mae"]) / 5
    assert np.isclose(combined, ret["mae"], rtol=1e-5)


def test_no_labels_means_no_cohort_keys(monkeypatch):
    ret = _evaluate(monkeypatch, None)
    assert not any(k.startswith(("normal_", "abnormal_")) for k in ret)


def test_summary_tables_show_the_cohort_columns():
    from labram.runs.common import build_summary_tables
    stats = {"loss": 0.2, "mae": 8.0, "normal_mae": 6.5, "abnormal_mae": 9.9, "normal_n_cases": 3}
    header = build_summary_tables({"best_epoch": 1, "last_epoch": 2, "best_val_stats": stats,
                                   "last_val_stats": stats})["val"][0]
    assert header[-3:] == ["normal_mae", "normal_n_cases", "abnormal_mae"]
