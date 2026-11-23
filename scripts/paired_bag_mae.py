#!/usr/bin/env python
"""Paired comparison of two MIL runs' recording-level MAE on the same split.

Reads the ``<split>_bag_weights_epochNN.parquet`` files a MIL run writes under
``<run>/analysis/`` (one row per window: ``recording``, ``alpha``,
``window_pred``, ``age``). With a linear head and pool weights that sum to one
per recording, the recording prediction is the weighted window prediction

    y_hat_r = sum_k alpha_rk * window_pred_rk                      (Eq. 2.1 of
    docs/codebook_aux_p4_results.md)

and the paired difference of B against A is the mean over recordings of
|e_r^B| - |e_r^A|, with a percentile bootstrap interval over recordings (Eq. 2.4).

    python scripts/paired_bag_mae.py A.parquet B.parquet --label "P4-R -> P4-RB"
"""
import argparse

import numpy as np
import pandas as pd


def recording_predictions(path: str) -> pd.DataFrame:
    d = pd.read_parquet(path)
    return d.assign(w=d.alpha * d.window_pred).groupby("recording").agg(pred=("w", "sum"), age=("age", "first"))


def paired_difference(a_path: str, b_path: str, n_boot: int = 2000, seed: int = 0) -> dict:
    a, b = recording_predictions(a_path), recording_predictions(b_path)
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    if not (j.age_a == j.age_b).all():
        raise ValueError("the two runs disagree on a recording's age: not the same split")
    ea, eb = (j.pred_a - j.age_a).abs().to_numpy(), (j.pred_b - j.age_b).abs().to_numpy()
    d = eb - ea
    boots = d[np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(d), "mae_a": ea.mean(), "mae_b": eb.mean(), "diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
            "p_b_better": float(np.mean(boots < 0)), "prediction_r": float(np.corrcoef(j.pred_a, j.pred_b)[0, 1])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label", default="A -> B")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    r = paired_difference(args.a, args.b, args.n_boot, args.seed)
    print(f"{args.label}: n {r['n']}  MAE {r['mae_a']:.3f} -> {r['mae_b']:.3f}  diff {r['diff']:+.3f} "
          f"[95% CI {r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  P(B better) {r['p_b_better']:.3f}  "
          f"prediction r {r['prediction_r']:.4f}")


if __name__ == "__main__":
    main()
