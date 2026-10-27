"""Tests for scripts/eval_runs.py and scripts/submit_eval_runs.py (no model, no AWS)."""
import os
import sys
import tarfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import eval_runs  # noqa: E402
import submit_eval_runs  # noqa: E402


def _windows():
    names = [f"r1_{k}.pkl" for k in (6, 7, 8, 9)] + [f"r2_{k}.pkl" for k in (6, 7)]
    return {"names": names, "recs": ["r1"] * 4 + ["r2"] * 2, "pred": np.array([10, 20, 30, 40, 50, 70.0]),
            "true": np.array([25.0] * 4 + [60.0] * 2), "labels": ["normal"] * 4 + ["abnormal"] * 2}


def test_per_recording_and_eval_budget():
    w = _windows()
    full = eval_runs.per_recording(w)
    assert full.loc["r1", "pred"] == 25 and full.loc["r2", "pred"] == 60 and full.loc["r1", "windows"] == 4
    keep = eval_runs.eval_budget({"eval_minutes": 20 / 60, "window_sec": 10})    # 2 windows per recording
    sub = eval_runs.per_recording(w, keep)
    assert sub.loc["r1", "pred"] == 15 and sub.loc["r1", "windows"] == 2 and sub.loc["r2", "windows"] == 2
    assert eval_runs.eval_budget({"eval_minutes": 0, "window_sec": 10}) is None


def test_metrics():
    m = eval_runs.metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert m["MAE"] == 0 and m["R2"] == 1 and m["n"] == 3


def test_tar_run_dir_and_fetch(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for f in ("checkpoint-best.pth", "run_config.yaml", "data_split.json", "checkpoint.pth"):
        (run / f).write_text("x")
    tar = submit_eval_runs.tar_run_dir(str(run), str(tmp_path / "m.tar.gz"))
    with tarfile.open(tar) as t:
        assert sorted(t.getnames()) == ["finetune/checkpoint-best.pth", "finetune/data_split.json",
                                        "finetune/run_config.yaml"]
    work = tmp_path / "work"
    work.mkdir()
    got = eval_runs.fetch_run(tar, str(work))
    assert os.path.exists(os.path.join(got, "run_config.yaml"))


def test_shard_balances():
    s = submit_eval_runs.shard(list(range(7)), 3)
    assert [len(x) for x in s] == [3, 2, 2] and sorted(sum(s, [])) == list(range(7))
