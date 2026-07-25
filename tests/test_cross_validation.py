"""Cross-validation fine-tuning: fold splitting, artifacts, naming, aggregation,
and an end-to-end debug run of the CV runner on synthetic TUAB data."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from labram.configs.cv_config import CrossValidationConfig
from labram.configs.run_configs import FinetuneRunConfig
from labram.data.bundles import DatasetBundle
from labram.data.tuh_datasets import TUABLoader
from labram.data import cross_validation as cv
from labram.eval import cv_aggregation as agg


# ------------------------------------------------------------------ fixtures


def _make_tuab_root(root: Path, subjects, wins=2):
    root.mkdir(parents=True, exist_ok=True)
    files = []
    for s in subjects:
        for w in range(wins):
            fn = f"{s}_s001_t000_{w}.pkl"
            with open(root / fn, "wb") as f:
                pickle.dump({"X": np.random.randn(23, 200).astype("f4"), "y": w % 2}, f)
            files.append(fn)
    return TUABLoader(str(root), files)


@pytest.fixture
def bundle(tmp_path):
    train = _make_tuab_root(tmp_path / "train", [f"s{i:02d}" for i in range(8)])
    val = _make_tuab_root(tmp_path / "val", [f"s{i:02d}" for i in range(8, 10)])
    test = _make_tuab_root(tmp_path / "test", [f"s{i:02d}" for i in range(10, 12)])
    return DatasetBundle(train=train, val=val, test=test,
                         ch_names=["A"], nb_classes=1, metrics=["accuracy"])


# ------------------------------------------------------------------ config


def test_cv_config_validation():
    CrossValidationConfig(enabled=True, n_folds=5, fold=2).validate()
    with pytest.raises(ValueError):
        CrossValidationConfig(enabled=True, n_folds=1).validate()
    with pytest.raises(ValueError):
        CrossValidationConfig(enabled=True, n_folds=5, fold=5).validate()
    with pytest.raises(ValueError):
        CrossValidationConfig(enabled=True, split_by="bogus").validate()
    # Disabled config never raises regardless of values.
    CrossValidationConfig(enabled=False, n_folds=1).validate()


def test_finetune_run_config_has_cv_and_sagemaker():
    c = FinetuneRunConfig()
    assert hasattr(c, "cross_validation")
    assert hasattr(c, "sagemaker")
    d = c.as_dict()
    assert "cross_validation" in d and "sagemaker" in d


# ------------------------------------------------------------------ folds


def test_grouped_folds_are_group_disjoint(bundle):
    cfg = CrossValidationConfig(enabled=True, n_folds=5, split_by="subject", seed=42)
    folds = cv.build_grouped_folds(bundle, cfg, "TUAB")
    assert folds.n_folds == 5
    assert cv.subject_overlap(folds) == {}
    # train+val pool = 10 subjects, split evenly.
    assert sum(len(g) for g in folds.fold_groups) == 10
    # Original test set kept out of the CV pool.
    assert folds.held_out_test["n_groups"] == 2


def test_materialize_fold_disjoint_and_covers(bundle):
    cfg = CrossValidationConfig(enabled=True, n_folds=5, split_by="subject", seed=1)
    folds = cv.build_grouped_folds(bundle, cfg, "TUAB")
    seen_test_subjects = set()
    for k in range(folds.n_folds):
        b = cv.materialize_fold(folds, k, bundle)
        assert len(b.train) > 0 and len(b.val) > 0 and len(b.test) > 0
        # Fold test subjects are unique across folds (each case tested once).
        test_files = _files_of(b.test)
        subs = {f.split("_")[0] for f in test_files}
        assert not (subs & seen_test_subjects)
        seen_test_subjects |= subs
    # Every pooled subject appears as a test subject exactly once.
    assert len(seen_test_subjects) == 10


def _files_of(dataset):
    import torch.utils.data
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        out = []
        for d in dataset.datasets:
            out += list(d.files)
        return out
    return list(dataset.files)


def test_cv_split_json_roundtrip(bundle, tmp_path):
    cfg = CrossValidationConfig(enabled=True, n_folds=5, split_by="subject", seed=7)
    folds = cv.build_grouped_folds(bundle, cfg, "TUAB")
    path = cv.save_cv_split(folds, str(tmp_path / "cvout"))
    assert Path(path).exists()
    doc = cv.load_cv_split_dict(path)
    assert doc["n_folds"] == 5 and len(doc["folds"]) == 5

    folds2 = cv.apply_cv_split(bundle, doc, "TUAB")
    for k in range(5):
        a = _files_of(cv.materialize_fold(folds, k, bundle).test)
        b = _files_of(cv.materialize_fold(folds2, k, bundle).test)
        assert sorted(a) == sorted(b)


def test_pool_all_includes_test(bundle):
    cfg = CrossValidationConfig(enabled=True, n_folds=3, split_by="subject", pool="all")
    folds = cv.build_grouped_folds(bundle, cfg, "TUAB")
    assert sum(len(g) for g in folds.fold_groups) == 12  # all subjects
    assert folds.held_out_test is None


def test_too_many_folds_raises(bundle):
    cfg = CrossValidationConfig(enabled=True, n_folds=50, split_by="subject")
    with pytest.raises(ValueError):
        cv.build_grouped_folds(bundle, cfg, "TUAB")


# ------------------------------------------------------------------ naming


def test_experiment_naming_embeds_fold_and_folder():
    from labram.runs import finetune_cv as fcv
    c = FinetuneRunConfig()
    c.cross_validation.enabled = True
    c.cross_validation.n_folds = 5
    c.output.output_dir = "./checkpoints/finetune_tuab"
    c.clearml.project_name = "LaBraM"

    exp = fcv.cv_experiment_name(c)
    assert exp.endswith("cv5")

    fold_cfg = fcv.derive_fold_config(c, 3)
    assert fold_cfg.cross_validation.fold == 3
    assert fold_cfg.output.output_dir.endswith("fold_3")
    assert exp in fold_cfg.clearml.project_name          # base folder in the name
    assert fold_cfg.clearml.task_name == "fold_3"        # fold number in the name
    assert "cross-validation" in fold_cfg.clearml.tags


# ------------------------------------------------------------------ aggregation


def test_aggregate_fold_metrics():
    records = [
        {"fold": 0, "val": {"accuracy": 80.0}, "test": {"accuracy": 70.0, "pr_auc": 0.6},
         "max_accuracy_test": 70.0},
        {"fold": 1, "val": {"accuracy": 90.0}, "test": {"accuracy": 80.0, "pr_auc": 0.8},
         "max_accuracy_test": 80.0},
    ]
    summary = agg.aggregate_fold_metrics(records, "exp")
    assert summary["n_folds_collected"] == 2
    assert summary["test"]["accuracy"]["mean"] == pytest.approx(75.0)
    assert summary["test"]["accuracy"]["std"] == pytest.approx(5.0)
    assert summary["test"]["pr_auc"]["per_fold"] == [0.6, 0.8]
    table = agg.format_summary_table(summary, "test")
    assert table[0] == ["metric", "mean", "std", "min", "max", "n"]


def test_collect_from_clearml_parses_timestamped_fold_names(monkeypatch):
    """clearml.append_timestamp makes task names ``fold_<k>_<ts>``; the fold
    number must still be recovered (from the fold token, not the last token)."""
    import sys
    import types

    class _FakeTask:
        def __init__(self, name, acc):
            self.name = name
            self._acc = acc

        def get_reported_scalars(self):
            return {'val': {'accuracy': {'y': [self._acc + 5]}},
                    'test': {'accuracy': {'y': [self._acc]}}}

    class _FakeTaskCls:
        @staticmethod
        def get_tasks(project_name=None, task_name=None):
            return [_FakeTask('fold_0_1699999999999', 70.0),
                    _FakeTask('fold_1_1700000000000', 80.0)]

    fake = types.ModuleType('clearml')
    fake.Task = _FakeTaskCls
    monkeypatch.setitem(sys.modules, 'clearml', fake)

    records = agg.collect_fold_metrics_from_clearml('LaBraM/finetune_tuab_cv5')
    assert [r['fold'] for r in records] == [0, 1]
    assert records[1]['test']['accuracy'] == 80.0
    summary = agg.aggregate_fold_metrics(records)
    assert summary['test']['accuracy']['mean'] == pytest.approx(75.0)


def test_aggregate_cv_dir(tmp_path):
    for k, acc in enumerate([70.0, 74.0, 78.0]):
        d = tmp_path / f"fold_{k}"
        d.mkdir()
        (d / "fold_metrics.json").write_text(json.dumps({
            "fold": k, "val": {"accuracy": acc + 5}, "test": {"accuracy": acc},
            "max_accuracy_test": acc}))
    summary = agg.aggregate_cv_dir(str(tmp_path))
    assert summary["n_folds_collected"] == 3
    assert summary["test"]["accuracy"]["mean"] == pytest.approx(74.0)
    assert (tmp_path / "cv_summary.json").exists()


# ------------------------------------------------------------------ end-to-end


@pytest.mark.parametrize("fold_arg", [-1, 1])
def test_cv_runner_end_to_end(tmp_path, fold_arg):
    """Full CV runner on synthetic TUAB in debug mode: fold dirs, cv_split.json,
    per-fold metrics, and (for the all-folds run) an aggregated summary."""
    from labram.runs import finetune_cv as fcv

    # 6 train + 2 val subjects -> 8-subject pool, 4 folds.
    _make_tuab_root(tmp_path / "data" / "train", [f"s{i:02d}" for i in range(6)])
    _make_tuab_root(tmp_path / "data" / "val", [f"s{i:02d}" for i in range(6, 8)])
    _make_tuab_root(tmp_path / "data" / "test", [f"s{i:02d}" for i in range(8, 10)])

    config = FinetuneRunConfig.load_config("labram/configs/defaults/finetune_tuab.json")
    config.update(**{
        "data.dataset": "TUAB",
        "data.data_path": str(tmp_path / "data"),
        "trainer.debug": True,
        "trainer.epochs": 1,
        "trainer.debug_samples": 4,
        "optimizer.warmup_epochs": 0,
        "distributed.device": "cpu",
        "output.output_dir": str(tmp_path / "checkpoints" / "finetune_tuab"),
        "model.model": "labram_base_patch200_200",
        "logging.log_model_graph": False,
        "cross_validation.enabled": True,
        "cross_validation.n_folds": 4,
        "cross_validation.fold": fold_arg,
        "cross_validation.split_by": "subject",
    })

    result = fcv.run_cross_validation(config)
    base = Path(fcv.cv_base_dir(config))

    assert (base / "cv_split.json").exists()
    if fold_arg == -1:
        assert set(result["folds"].keys()) == {0, 1, 2, 3}
        for k in range(4):
            assert (base / f"fold_{k}" / "fold_metrics.json").exists()
        assert (base / "cv_summary.json").exists()
        assert "summary" in result
        assert result["summary"]["n_folds_collected"] == 4
    else:
        assert set(result["folds"].keys()) == {1}
        assert (base / "fold_1" / "fold_metrics.json").exists()
