"""Evaluation-only cohorts: every recording in test, scored by a model trained elsewhere."""
import json
import os
import sys

import numpy as np
import torch

from labram.data.tuh_datasets import prepare_TUAB_age_dataset
from labram.data.window_selection import WindowSelection, apply_window_selection

W = 2000


def _test_only_cohort(tmp_path):
    from test_window_qc import _corpus
    root = _corpus(tmp_path)
    split = json.load(open(root / "age_split.json"))
    split["files"] = {"train": [], "val": [],
                      "test": [f.replace(f"{s}/", "test/", 1) for s in ("train", "val", "test")
                               for f in split["files"][s]]}
    json.dump(split, open(root / "age_split.json", "w"))
    m = json.load(open(root / "manifest.json"))
    for r in m["recordings"]:
        r["source_split"] = "test"
    json.dump(m, open(root / "manifest.json", "w"))
    return root


def test_test_only_cohort_keeps_years_and_survives_window_selection(tmp_path):
    from labram.data import get_dataset_bundle
    root = _test_only_cohort(tmp_path)
    train, test, val, stats = prepare_TUAB_age_dataset(str(root), data_format="npy")
    assert stats is None and len(train) == 0 and len(test) == 5 * 20
    bundle = get_dataset_bundle("TUAB_AGE", str(root), data_format="npy")
    bundle = apply_window_selection(bundle, WindowSelection(trim_start_sec=10.0, eval_minutes=1.0))
    assert len(bundle.train) == 0 and len(bundle.test) == 5 * 6      # 1 min per recording
    y = bundle.test[0][1]
    assert 30 <= float(y) <= 50                                        # ages in years


def test_eval_script_scores_a_bag_model_on_a_test_only_cohort(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import eval_age_by_cohort as ev
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.runs import run_finetune
    root = _test_only_cohort(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    cfg = FinetuneRunConfig.load_config(
        "labram/configs/defaults/finetune_tuab_age.json",
        **{"data.data_path": str(root), "data.data_format": "npy", "data.trim_start_sec": 0.0,
           "data.trim_end_sec": 0.0, "data.eval_minutes": 0.0, "finetune_checkpoint.finetune": "",
           "mil.enabled": True, "mil.pool": "gated", "mil.gated": False, "mil.score_input": "h",
           "model.nb_classes": 1, "model.task": "regression"})
    cfg.save_to(str(run / "run_config.yaml"))
    model = run_finetune.get_model(cfg)
    torch.save({"model": model.state_dict(), "epoch": 0}, run / "checkpoint-best.pth")
    res = ev.evaluate_run(str(run), str(root), "checkpoint-best.pth", ["test"], torch.device("cpu"),
                          batch_size=4, pred_stats=(50.0, 10.0), stage_dir="none", num_workers=0)
    test = res["splits"]["test"]
    assert test["all"]["n"] == 5 and len(test["predictions"]) == 5
    assert np.isfinite(test["all"]["mae"])
