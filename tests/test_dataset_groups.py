"""Per-dataset evaluation: dataset families, train-eval sampling, and the
dataset x cohort metrics of fine-tune evaluation."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from labram.data import dataset_groups as dg
from labram.runs.common import build_summary_tables, flatten_summary_metrics
from labram.train import train_finetune as tf


class Leaf:
    """Minimal npy-loader leaf: a root with a manifest and a window file list."""

    def __init__(self, root, files):
        self.root, self.files = str(root), list(files)
        self.random_crop = True
        self._mmaps = {"open": object()}

    def __len__(self):
        return len(self.files)


@pytest.fixture
def root(tmp_path):
    rows = ([{"stem": f"sub-S0001{i}_ses-1_task-EEG", "corpus": "heedb"} for i in range(5)]
            + [{"stem": f"aaaa{i:04d}_s001_t000", "corpus": "tueg"} for i in range(3)]
            + [{"stem": "zzzz_s001_t000"}])
    (tmp_path / "manifest.json").write_text(json.dumps({"recordings": rows}))
    dg._CACHE.clear()
    return tmp_path


def _files(stems, n=4):
    return [f"train/{s}_{k}.pkl" for s in stems for k in range(n)]


def test_dataset_family():
    assert dg.dataset_family("heedb") == "harvard"
    assert dg.dataset_family("TUEG") == "temple" and dg.dataset_family("tuh") == "temple"
    assert dg.dataset_family(None) == "unknown" and dg.dataset_family("other") == "other"


def test_root_lookup_maps_every_stem(root):
    lookup = dg.root_dataset_lookup(str(root))
    assert lookup["sub-S00010_ses-1_task-EEG"] == "harvard"
    assert lookup["aaaa0000_s001_t000"] == "temple" and lookup["zzzz_s001_t000"] == "unknown"


def test_train_eval_subset_is_per_dataset_capped_and_deterministic(root):
    stems = list(dg.root_dataset_lookup(str(root)))
    leaf = Leaf(root, _files(stems))
    sub = dg.train_eval_subset(leaf, per_dataset=2, windows_per_recording=3, seed=1)
    picked = {f.split("/")[1].rsplit("_", 1)[0] for f in sub.files}
    lookup = dg.root_dataset_lookup(str(root))
    per_ds = {}
    for s in picked:
        per_ds[lookup[s]] = per_ds.get(lookup[s], 0) + 1
    assert per_ds == {"harvard": 2, "temple": 2, "unknown": 1}
    assert all(sum(f.startswith(f"train/{s}_") for f in sub.files) == 3 for s in picked)
    assert sub.files == dg.train_eval_subset(leaf, 2, 3, seed=1).files       # seeded
    assert sub.random_crop is False and sub._mmaps == {} and leaf.random_crop is True
    assert leaf.files == _files(stems)                                        # source untouched
    assert dg.train_eval_subset(leaf, 0, 3) is None


def test_cohort_metrics_by_dataset_and_cohort(root, monkeypatch):
    stems = list(dg.root_dataset_lookup(str(root)))
    labels = {s: ("normal" if i % 3 == 0 else "abnormal" if i % 3 == 1 else None)
              for i, s in enumerate(stems)}
    monkeypatch.setattr(tf, "_case_labels", lambda loader: labels)
    groups = [s for s in stems for _ in range(2)]                 # 2 windows per recording
    rng = np.random.default_rng(0)
    true = np.repeat(rng.uniform(20, 80, len(stems)), 2)[:, None]
    pred = true + rng.normal(0, 5, true.shape)
    loader = SimpleNamespace(dataset=Leaf(root, []))
    out = tf._cohort_metrics(loader, pred, true, groups, "mean", ["mae", "r2"], False, 1,
                             tf.TASK_REGRESSION)
    assert out["harvard_n_cases"] == 5 and out["temple_n_cases"] == 3
    assert {"normal_mae", "abnormal_mae", "unlabelled_mae", "harvard_mae", "temple_mae",
            "harvard_normal_mae", "harvard_abnormal_mae"} <= set(out)
    assert "unknown_mae" not in out                               # one case: skipped
    case_err = np.abs(pred[:, 0].reshape(-1, 2).mean(1) - true[::2, 0])   # mean-pooled per recording
    harvard = [i for i, s in enumerate(stems) if dg.root_dataset_lookup(str(root))[s] == "harvard"]
    assert out["harvard_mae"] == pytest.approx(case_err[harvard].mean())
    only_ds = tf._cohort_metrics(loader, pred, true, groups, "mean", ["mae"], False, 1,
                                 tf.TASK_REGRESSION, by_label=False)
    assert "harvard_mae" in only_ds and "normal_mae" not in only_ds
    assert tf._cohort_metrics(loader, pred, true, groups, "mean", ["mae"], False, 1,
                              tf.TASK_REGRESSION, per_dataset=False, by_label=False) == {}


def test_summary_includes_train_eval_and_dataset_groups():
    stats = {"mae": 7.0, "harvard_mae": 7.5, "temple_normal_mae": 5.2, "harvard_n_cases": 10}
    summary = {"best_epoch": 3, "last_epoch": 4, "best_val_stats": stats,
               "best_train_eval_stats": {"mae": 4.0, "harvard_mae": 4.1},
               "last_train_eval_stats": {"mae": 3.9}}
    flat = flatten_summary_metrics(summary)
    assert flat["train_eval_harvard_mae"] == 4.1 and flat["val_temple_normal_mae"] == 5.2
    tables = build_summary_tables(summary)
    assert "train_eval" in tables
    assert {"harvard_mae", "temple_normal_mae", "harvard_n_cases"} <= set(tables["val"][0])


def _mixed_corpus(tmp_path, windows=6):
    """npy age cohort with Harvard (heedb) and Temple (tuh) recordings in every split."""
    W = 2000
    root = tmp_path / "mixed"
    (root / "recordings").mkdir(parents=True)
    rng = np.random.default_rng(2)
    rows, meta, files = [], {}, {"train": [], "val": [], "test": []}
    i = 0
    for split, n in (("train", 3), ("val", 2), ("test", 2)):
        for corpus in ("heedb", "tuh"):
            for j in range(n):
                stem = (f"sub-S0001{i:03d}_ses-1_task-EEG" if corpus == "heedb"
                        else f"t{i:07d}_s001_t000")
                np.save(root / "recordings" / f"{stem}.npy",
                        rng.normal(scale=20, size=(windows * W, 23)).astype(np.float16))
                rows.append({"stem": stem, "n_windows": windows, "n_samples": windows * W,
                             "source_split": split, "channel_mask": [True] * 23, "corpus": corpus})
                meta[stem] = {"stem": stem, "subject": f"p{i}", "session": "1", "token": None,
                              "age": 25 + 6 * i, "sex": "F", "year": 2018, "raw_age": 25 + 6 * i,
                              "label": "normal" if j % 2 == 0 else "abnormal"}
                files[split] += [f"{split}/{stem}_{k}.pkl" for k in range(windows)]
                i += 1
    json.dump({"format_version": 2, "dtype": "float16", "recordings": rows},
              open(root / "manifest.json", "w"))
    json.dump({"version": 1, "recordings": meta}, open(root / "age_metadata.json", "w"))
    json.dump({"version": 1, "seed": 0, "files": files}, open(root / "age_split.json", "w"))
    return root


def test_run_finetune_reports_every_split_per_dataset_and_cohort(tmp_path):
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.runs import run_finetune
    dg._CACHE.clear()
    root = _mixed_corpus(tmp_path)
    out = tmp_path / "out"
    cfg = FinetuneRunConfig.load_config(
        "labram/configs/defaults/finetune_tuab_age.json",
        **{"data.data_path": str(root), "data.data_format": "npy",
           "data.trim_start_sec": 0.0, "data.trim_end_sec": 0.0, "data.eval_minutes": 0.5,
           "data.num_workers": 0, "data.stage_dir": "none", "trainer.batch_size": 4,
           "trainer.epochs": 1, "optimizer.warmup_epochs": 0,
           "evaluation.train_eval_recordings": 2,
           "distributed.device": "cpu", "finetune_checkpoint.finetune": "",
           "clearml.enabled": False, "output.output_dir": str(out), "output.log_dir": str(out),
           "logging.log_model_graph": False})
    summary = run_finetune.main(cfg)
    for split in ("best_val_stats", "best_test_stats", "best_train_eval_stats"):
        stats = summary[split]
        assert {"harvard_mae", "temple_mae", "harvard_n_cases", "temple_n_cases"} <= set(stats), split
    assert summary["best_val_stats"]["harvard_n_cases"] == 2
    assert summary["best_train_eval_stats"]["temple_n_cases"] == 2      # 2 of 3 per dataset
    assert "harvard_normal_mae" not in summary["best_val_stats"]       # 1 case per cohort
    assert {"normal_mae", "abnormal_mae"} <= set(summary["best_test_stats"])
