"""MIL Stage 1 on a synthetic feature cache (labram/mil)."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from labram.layers.bag_pool import DescriptorNorm
from labram.mil import stage1 as s1
from labram.mil.cache import Cache

D, Q = 8, 4


def _write_cache(root, n_rec=60, k=10, seed=0):
    """Windows carry the age in h[:, 0]; 'bad' windows (descriptor 0 high) are noise."""
    rng = np.random.default_rng(seed)
    run = os.path.join(root, "R")
    os.makedirs(run)
    torch.save({"weight": torch.tensor([[1.0] + [0.0] * (D - 1)]), "bias": torch.zeros(1)},
               os.path.join(run, "head.pt"))
    json.dump({"pred_stats": [50.0, 10.0]}, open(os.path.join(run, "run.json"), "w"))
    for split, off in (("val", 0), ("test", 1000), ("train", 2000)):
        rows, h, desc = [], [], []
        for r in range(n_rec):
            stem = f"s{off + r:04d}_s001_t000" if r % 2 else f"sub-X{off + r}_ses-1_task-EEG"
            age = rng.uniform(20, 80)
            for w in range(k):
                bad = rng.random() < 0.3
                z = (age - 50) / 10 + (rng.normal(0, 3) if bad else rng.normal(0, 0.2))
                h.append([z] + list(rng.normal(0, 1, D - 1)))
                desc.append([5.0 if bad else 0.0] + list(rng.normal(0, 1, Q - 1)))
                rows.append({"recording": stem, "window": w + 6, "minute": w / 6, "age": age,
                             "cohort": "normal", "dataset": "harvard" if r % 2 else "temple"})
        d = os.path.join(run, split)
        os.makedirs(d)
        h = np.asarray(h, np.float16)
        np.save(os.path.join(d, "h.npy"), h)
        np.save(os.path.join(d, "pred.npy"), (h[:, 0].astype(np.float32) * 10 + 50))
        np.save(os.path.join(d, "desc.npy"), np.asarray(desc, np.float32))
        pd.DataFrame(rows).to_parquet(os.path.join(d, "meta.parquet"))
        json.dump([f"d{i}" for i in range(Q)], open(os.path.join(d, "desc_columns.json"), "w"))
    return Cache(root, "R")


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    cache = _write_cache(str(tmp_path_factory.mktemp("mil")))
    stats = tuple(cache.info["pred_stats"])
    bags = {s: s1.load_bags(cache, s, stats, torch.device("cpu")) for s in ("val", "test", "train")}
    norm = DescriptorNorm(bags["train"].q.shape[-1]).fit(bags["train"].q[bags["train"].mask])
    head = {k: v.float() for k, v in cache.head_state().items()}
    return bags, norm, head, stats


def test_load_bags(data):
    bags, *_ = data
    b = bags["val"]
    assert b.h.shape == (60, 10, D) and b.q.shape == (60, 10, Q + 1) and b.mask.all()
    assert b.desc_columns[-1] == "minute"


def test_m0_run_equals_mean_of_window_predictions(data):
    bags, norm, head, stats = data
    res = s1.run_variant(s1.Variant("M0-run", kind="mean", fit_head=False), bags, head, norm, stats)
    ref = np.nanmean(bags["test"].win_pred, 1)
    assert np.allclose(res["test"]["pred"], ref, atol=0.05)


@pytest.mark.parametrize("protocol", ["val_cf", "train"])
def test_attention_on_descriptors_beats_mean(data, protocol):
    bags, norm, head, stats = data
    m0 = s1.run_variant(s1.Variant("M0", kind="mean", epochs=15), bags, head, norm, stats, protocol)
    m1 = s1.run_variant(s1.Variant("M1-q", kind="gated", score_input="q", epochs=15, lr=3e-3),
                        bags, head, norm, stats, protocol)
    age = bags["test"].rec["age"].to_numpy()
    assert np.abs(m1["test"]["pred"] - age).mean() < np.abs(m0["test"]["pred"] - age).mean()
    win = s1.window_frame(bags["test"], m1["test"]["alpha"], m1["head_after"], stats)
    an = s1.attention_analytics(win, bags["test"].rec, m1["test"]["pred"])
    assert an["win_rho_within"] < 0                      # weight tracks error
    assert an["max_abs_weighted_sum_minus_pred"] < 1e-2  # linear head: weighted sum = prediction


def test_metrics_rows(data):
    bags, *_ = data
    rec = bags["test"].rec
    rows = s1.metrics_rows("x", "test", rec, rec["age"].to_numpy() + 1, rec["age"].to_numpy())
    assert {r["group"] for r in rows} >= {"all", "temple", "harvard"}
    assert all(abs(r["mae"] - 1) < 1e-9 and r["delta_vs_ref"] == pytest.approx(1) for r in rows)
