"""Tests for the offline fine-tune evaluation toolkit (labram.eval) and the
entropy / median aggregation / threshold-sweep helpers in eval_metrics."""

import pickle

import numpy as np
import pytest
import torch

from labram.data.tuh_datasets import TUABLoader, TUEVLoader
from labram.eval import (
    aggregate_result,
    build_case_loader,
    collect_predictions,
    compare_aggregations,
    comparison_table,
    case_window_predictions,
    entropy_accuracy_curve,
    rank_cases,
    selective_accuracy,
)
from labram.eval.inference import PredictionResult, ordered_files
from labram.utils import aggregate_windows, prediction_entropy, threshold_sweep


# ---------------------------------------------------------------------------
# eval_metrics helpers
# ---------------------------------------------------------------------------

def test_prediction_entropy_bounds_binary():
    h = prediction_entropy(np.array([0.5, 0.99, 0.01, 0.0, 1.0]), is_binary=True)
    assert h[0] == pytest.approx(1.0, abs=1e-6)      # maximally uncertain
    assert h[1] < 0.15 and h[2] < 0.15               # confident -> low
    assert np.all((h >= 0) & (h <= 1.0 + 1e-9))


def test_prediction_entropy_multiclass_normalized():
    probs = np.array([[0.25, 0.25, 0.25, 0.25], [0.97, 0.01, 0.01, 0.01]])
    h = prediction_entropy(probs, is_binary=False)
    assert h[0] == pytest.approx(1.0, abs=1e-6)      # uniform over 4 classes
    assert h[1] < 0.3


def test_aggregate_windows_median_and_entropy_binary():
    out = np.array([0.9, 0.85, 0.05])   # one outlier low-confidence flip
    tgt = np.array([1, 1, 1])
    grp = ['c', 'c', 'c']
    o_med, _ = aggregate_windows(out, tgt, grp, 'median', is_binary=True)
    assert float(o_med.ravel()[0]) == pytest.approx(0.85)
    o_mean, _ = aggregate_windows(out, tgt, grp, 'mean', is_binary=True)
    o_ent, _ = aggregate_windows(out, tgt, grp, 'entropy', is_binary=True)
    # entropy weighting keeps the two confident windows dominant; both agree the
    # value differs from the plain mean.
    assert float(o_ent.ravel()[0]) != pytest.approx(float(o_mean.ravel()[0]))


def test_aggregate_windows_median_multiclass():
    logits = np.log(np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]))
    tgt = np.array([0, 0, 0])
    grp = ['x', 'x', 'x']
    o, t = aggregate_windows(logits, tgt, grp, 'median', is_binary=False)
    assert o.shape == (1, 3)
    assert int(np.argmax(o.ravel())) == 0            # class 0 wins the median


def test_aggregate_windows_rejects_unknown_mode():
    with pytest.raises(ValueError):
        aggregate_windows(np.array([0.5]), np.array([1]), ['a'], 'bogus', True)


def test_threshold_sweep_shapes_and_values():
    y_score = np.array([0.1, 0.4, 0.6, 0.9])
    y_true = np.array([0, 0, 1, 1])
    sweep = threshold_sweep(y_score, y_true, thresholds=[0.0, 0.5, 1.0])
    assert sweep['accuracy'][1] == pytest.approx(1.0)   # 0.5 separates perfectly
    assert sweep['accuracy'][0] == pytest.approx(0.5)   # thr=0 -> all positive
    for k in ('f1', 'precision', 'recall', 'specificity', 'balanced_accuracy'):
        assert sweep[k].shape == (3,)


# ---------------------------------------------------------------------------
# Inference + analysis over a synthetic TUAB loader
# ---------------------------------------------------------------------------

class _DummyModel(torch.nn.Module):
    """Per-window logit = mean of the (already /100-scaled) signal."""

    def forward(self, x, channel_indices=None, classify_only=True):
        b = x.shape[0]
        return x.reshape(b, -1).mean(dim=1, keepdim=True)


@pytest.fixture
def tuab_dataset(tmp_path):
    root = tmp_path / "tuab"
    root.mkdir()
    files = []
    # 2 recordings x 4 windows; recording A label 1 (positive signal), B label 0.
    for rec, (label, base) in {"aaaa_s1_t0": (1, 1.0), "bbbb_s2_t0": (0, -1.0)}.items():
        for w in range(4):
            name = f"{rec}_{w}.pkl"
            files.append(name)
            with open(root / name, "wb") as f:
                pickle.dump({"X": (np.ones((2, 200)) * base * 100).astype("f4"),
                             "y": label}, f)
    return TUABLoader(str(root), files)


def test_collect_predictions_and_files(tuab_dataset):
    loader, files = build_case_loader(tuab_dataset, batch_size=3, agg_case_by="recording")
    assert files is not None and len(files) == 8
    result = collect_predictions(_DummyModel(), loader, torch.device("cpu"),
                                 is_binary=True, nb_classes=1, files=files)
    assert len(result) == 8
    assert result.groups is not None and set(result.groups) == {"aaaa_s1_t0", "bbbb_s2_t0"}
    assert result.probs.shape == (8,)
    assert result.entropy.shape == (8,)


def test_compare_aggregations_perfect_separation(tuab_dataset):
    loader, files = build_case_loader(tuab_dataset, batch_size=3)
    result = collect_predictions(_DummyModel(), loader, torch.device("cpu"),
                                 is_binary=True, nb_classes=1, files=files)
    reports = compare_aggregations(result, modes=("mean", "median", "entropy"))
    assert set(reports) == {"window", "mean", "median", "entropy"}
    for mode in ("mean", "median", "entropy"):
        assert reports[mode].scalars["accuracy"] == pytest.approx(1.0)
    table = comparison_table(reports)
    assert {r["method"] for r in table} == set(reports)


def test_aggregate_result_needs_groups():
    result = PredictionResult(
        probs=np.array([0.6, 0.4]), logits=np.array([[0.4], [-0.4]]),
        targets=np.array([1, 0]), groups=None, files=None,
        is_binary=True, nb_classes=1)
    with pytest.raises(ValueError):
        aggregate_result(result, "mean")


def test_entropy_accuracy_and_selective_curves(tuab_dataset):
    loader, files = build_case_loader(tuab_dataset, batch_size=4)
    result = collect_predictions(_DummyModel(), loader, torch.device("cpu"),
                                 is_binary=True, nb_classes=1, files=files)
    ea = entropy_accuracy_curve(result, n_bins=4)
    assert ea.counts.sum() == len(result)
    sel = selective_accuracy(result, n_points=5)
    assert sel.coverage[-1] == pytest.approx(1.0)
    assert np.all((sel.accuracy >= 0) & (sel.accuracy <= 1))


def test_rank_and_case_windows(tuab_dataset):
    loader, files = build_case_loader(tuab_dataset, batch_size=4)
    result = collect_predictions(_DummyModel(), loader, torch.device("cpu"),
                                 is_binary=True, nb_classes=1, files=files)
    ranked = rank_cases(result, mode="mean")
    assert len(ranked) == 2
    # Both cases are cleanly correct here, so margins are positive.
    assert all(margin > 0 for _, _, _, margin in ranked)
    cw = case_window_predictions(result, ranked[0][0])
    assert cw["probs"].shape[0] == 4
    assert cw["files"] is not None and len(cw["files"]) == 4


def test_ordered_files_subset(tuab_dataset):
    subset = torch.utils.data.Subset(tuab_dataset, [0, 2, 4])
    files = ordered_files(subset)
    assert files == [tuab_dataset.files[i] for i in (0, 2, 4)]


# ---------------------------------------------------------------------------
# Model rebuild + checkpoint round-trip
# ---------------------------------------------------------------------------

def test_build_model_by_name_and_checkpoint_roundtrip(tmp_path):
    from labram.eval import build_model_by_name, load_checkpoint_weights
    model = build_model_by_name("labram_base_patch200_200", nb_classes=1)
    ckpt = tmp_path / "checkpoint-final.pth"
    torch.save({"model": model.state_dict()}, ckpt)
    reloaded = build_model_by_name("labram_base_patch200_200", nb_classes=1)
    load_checkpoint_weights(reloaded, str(ckpt))
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), reloaded.state_dict().items()):
        assert k1 == k2 and torch.allclose(v1, v2)


def test_find_experiment_assets(tmp_path):
    from labram.eval import find_experiment_assets
    (tmp_path / "checkpoint-best.pth").write_bytes(b"x")
    (tmp_path / "data_split.json").write_text("{}")
    assets = find_experiment_assets(str(tmp_path))
    assert assets.checkpoint.endswith("checkpoint-best.pth")
    assert assets.data_split.endswith("data_split.json")
