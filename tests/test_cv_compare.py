import numpy as np
import pandas as pd
import pytest

from labram.eval import cv_compare as cc


def _preds(n=50, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.uniform(20, 80, n)
    stems = [f"r{i:03d}" for i in range(n)]
    out = {}
    for exp, noise in (("a", 2.0), ("b", 6.0)):
        out[exp] = {}
        for k in range(3):
            held = {s: [float(t + rng.normal(0, noise)), float(t)] for s, t in zip(stems, true)}
            test = {s: [float(t + rng.normal(0, noise)), float(t)]
                    for i, (s, t) in enumerate(zip(stems, true)) if i % 3 == k}
            out[exp][str(k)] = {"heldout": held, "test": test}
    return out


def test_metrics_perfect_and_bias():
    m = cc.metrics([1, 2, 3], [1, 2, 3])
    assert m["MAE"] == 0 and m["R2"] == 1 and m["bias"] == 0
    m = cc.metrics([2, 3, 4], [1, 2, 3])
    assert m["MAE"] == pytest.approx(1) and m["bias"] == pytest.approx(1)


def test_long_frame_and_matrix():
    d = cc.long_frame(_preds())
    assert set(d.split) == {"heldout", "test"}
    np.testing.assert_allclose(d.err, d.pred - d.true)
    w = cc.model_matrix(d, "heldout")
    assert w.shape == (50, 6) and "a/f0" in w.columns


def test_ensemble_median_beats_single_models():
    d = cc.long_frame(_preds())
    ens = cc.ensemble(d, "heldout")
    assert (ens.n_models == 3).all()
    single = d[(d.split == "heldout") & (d.exp == "b")].groupby("fold").err.apply(lambda e: e.abs().mean())
    assert cc.metrics(ens[ens.exp == "b"].pred, ens[ens.exp == "b"].true)["MAE"] < single.mean()


def test_out_of_fold_unique_and_rejects_duplicates():
    p = _preds()
    oof = cc.out_of_fold(cc.long_frame(p))
    assert oof.groupby("exp").stem.nunique().tolist() == [50, 50]
    p["a"]["1"]["test"]["r000"] = [1.0, 1.0]            # r000 is in fold 0 already
    with pytest.raises(ValueError):
        cc.out_of_fold(cc.long_frame(p))


def test_pairwise_rates():
    idx = pd.Index(["x", "y", "z", "w"])
    true = pd.Series([10.0, 20, 30, 40], index=idx)
    pred = pd.DataFrame({"m1": [11.0, 19, 31, 39], "m2": [12.0, 18, 36, 39]}, index=idx)
    r = cc.pairwise(pred, true, taus=(2.0,)).iloc[0]
    assert r.agree_2y == pytest.approx(0.75)            # |diff| = 1, 1, 5, 0
    assert r.sign_agree == pytest.approx(1.0)
    assert r.dMAE == pytest.approx(1.0 - 2.75)          # |err| m2 = 2, 2, 6, 1


def test_bootstrap_and_bias_correction():
    rng = np.random.default_rng(1)
    ea, eb = rng.normal(0, 1, 300), rng.normal(0, 3, 300)
    b = cc.bootstrap_delta_mae(ea, eb, n_boot=500)
    assert b["lo"] < b["dMAE"] < b["hi"] < 0 and b["p_a_better"] > 0.99
    true = np.linspace(20, 80, 100)
    err = -0.3 * (true - 50) + rng.normal(0, 0.1, 100)
    corrected = cc.bias_correct(err, true)
    assert abs(np.corrcoef(corrected, true)[0, 1]) < 0.05
    ci = cc.mean_ci([1.0, 2, 3])
    assert ci["lo"] < 2 < ci["hi"]
