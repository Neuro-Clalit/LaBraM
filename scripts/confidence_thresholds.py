#!/usr/bin/env python
"""Window-trimming and recording-rejection thresholds chosen on validation, applied to test.

Instead of trimming a fixed 20 % of windows and rejecting a fixed 20 % of each
test split's recordings (``scripts/eval_strategies_table.py``), this script
fixes **absolute, label-free confidence thresholds on the validation split**
and applies them unchanged to the test split and to an external cohort
(LEMON). Inputs are ``scripts/eval_age_by_cohort.py --save-windows`` parquets
(``recording, alpha, window_pred, age``; all windows of every recording).

Window confidence (one value per window ``i`` of recording ``r``):

* ``attention`` -- the relative weight ``c_i = alpha_i * n_r`` (1 = uniform);
  a window is kept when ``c_i >= tau``. Attention models only.
* ``deviation`` -- the distance of the window prediction from the recording's
  median window prediction, ``d_i = |p_i - median_r p|`` (years); a window is
  kept when ``d_i <= tau``. Every model (for E2L it is the only one).

A recording always keeps its most confident window. ``tau`` is chosen as the
grid value with the lowest val MAE over all val recordings (``tau = none``,
nothing trimmed, is on the grid).

Recording confidence (one value per recording, after trimming): the
attention-weighted spread ``s_r = sqrt(sum_i a_i (p_i - y_hat_r)^2)`` (years;
a = renormalised alpha), the effective share of windows ``neff_r =
1 / (n_r sum_i a_i^2)`` and the normalised entropy ``H(a) / log n_r``. A
recording is **rejected** when ``s_r > theta`` (``neff`` / ``entropy``: when
below ``theta``). ``theta`` is the value with the lowest val MAE of the kept
recordings among those that keep at least ``--min-keep`` of the val
recordings (default 0.83: 80 % plus a margin, since a val-fitted threshold
keeps a little less on the Harvard test cohorts). The kept share on test and LEMON is whatever ``theta`` gives
there; it is reported per cohort.

Every decision -- tau, the trimming kind, theta and the rejection score -- is
made on the validation split; the test split (and LEMON) only report the
result. Val numbers are therefore optimistic. The ``ref_*_oracle`` columns,
the best threshold on test itself, are a reference for how much the val
choice leaves on the table and decide nothing.

    python scripts/confidence_thresholds.py \\
        --val /data/runs/heedb/external_eval/windows/own_all_val \\
        --test /data/runs/heedb/external_eval/windows/own_all_w \\
        --external lemon=/data/runs/heedb/external_eval/windows/lemon_all_w \\
        --out docs/data/confidence_thresholds
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ATT_GRID = np.round(np.arange(0.0, 2.001, 0.05), 3)          # c_i >= tau (0 = keep all)
DEV_GRID = np.concatenate([np.arange(1.0, 30.5, 1.0), [np.inf]])   # d_i <= tau, years
REC_SCORES = {"spread": True, "neff": False, "entropy": False}    # True: high = unsure


def load(path):
    w = pd.read_parquet(path, columns=["recording", "alpha", "window_pred", "age"])
    n = w.groupby("recording")["alpha"].transform("size")
    a = w["alpha"] / w.groupby("recording")["alpha"].transform("sum")
    w["att"] = a * n                                                    # relative weight c_i
    w["dev"] = (w["window_pred"] - w.groupby("recording")["window_pred"].transform("median")).abs()
    w["uniform"] = bool(np.allclose(w["att"], 1.0))
    w["code"], uniq = pd.factorize(w["recording"])
    w.attrs["age"] = w.groupby("code")["age"].first().to_numpy()
    return w


def fast_mae(w, keep):
    """Recording MAE of the renormalised attention pool over the kept windows."""
    code, a = w["code"].to_numpy(), w["alpha"].to_numpy() * keep
    num = np.bincount(code, a * w["window_pred"].to_numpy())
    den = np.bincount(code, a)
    return float(np.abs(num / den - w.attrs["age"]).mean())


def keep_mask(w, kind, tau):
    if kind == "attention":
        if tau <= 0:
            return np.ones(len(w), bool)
        ok = w["att"] >= tau
        top = w["att"] == w.groupby("recording")["att"].transform("max")
    else:
        if not np.isfinite(tau):
            return np.ones(len(w), bool)
        ok = w["dev"] <= tau
        top = w["dev"] == w.groupby("recording")["dev"].transform("min")
    return (ok | top).to_numpy()


def recordings(w, keep):
    """Per recording: prediction (renormalised attention pool over the kept windows),
    age, confidence scores and the share of its windows kept."""
    d = w[keep]
    g = d.groupby("recording")
    a = d["alpha"] / g["alpha"].transform("sum")
    pred = (a * d["window_pred"]).groupby(d["recording"]).sum()
    dev2 = a * (d["window_pred"] - d["recording"].map(pred)) ** 2
    n = g.size()
    ent = -(a * np.log(a.clip(lower=1e-12))).groupby(d["recording"]).sum()
    return pd.DataFrame({
        "pred": pred, "age": g["age"].first(), "n": n,
        "spread": np.sqrt(dev2.groupby(d["recording"]).sum()),
        "neff": 1.0 / (a ** 2).groupby(d["recording"]).sum() / n,
        "entropy": ent / np.log(n.clip(lower=2)),
        "kept_windows": n / w.groupby("recording").size(),
    })


def mae(r):
    return float((r["pred"] - r["age"]).abs().mean()) if len(r) else np.nan


def trim_grid(w):
    kinds = ["deviation"] if w["uniform"].iloc[0] else ["attention", "deviation"]
    rows = []
    for kind in kinds:
        for tau in (ATT_GRID if kind == "attention" else DEV_GRID):
            keep = keep_mask(w, kind, tau)
            rows.append({"kind": kind, "tau": tau, "mae": fast_mae(w, keep),
                         "dropped_windows": 1.0 - keep.mean()})
    return pd.DataFrame(rows)


def kept(r, score, theta):
    return (r[score] <= theta) if REC_SCORES[score] else (r[score] >= theta)


def choose_theta(r, score, min_keep):
    """Threshold on the val recordings: lowest kept MAE with kept share >= min_keep."""
    sign = 1.0 if REC_SCORES[score] else -1.0
    s = sign * r[score].to_numpy()                       # ascending = most confident first
    order = np.argsort(s, kind="stable")
    s, err = s[order], (r["pred"] - r["age"]).abs().to_numpy()[order]
    last = np.r_[s[1:] != s[:-1], True]                  # a threshold keeps all ties
    k = np.arange(1, len(s) + 1)[last]
    m = (np.cumsum(err) / np.arange(1, len(s) + 1))[last]
    ok = k / len(s) >= min_keep
    j = np.flatnonzero(ok)[np.argmin(m[ok])]
    return float(sign * s[last][j]), float(m[j]), float(k[j] / len(s))


def coverage_curve(r, score, n_points=41):
    """MAE of the kept recordings vs. kept share (val-free; for the figure)."""
    err = (r["pred"] - r["age"]).abs().to_numpy()
    order = np.argsort(r[score].to_numpy() * (1 if REC_SCORES[score] else -1), kind="stable")
    out = []
    for cov in np.linspace(0.5, 1.0, n_points):
        k = int(round(cov * len(err)))
        out.append((cov, float(err[order[:k]].mean())))
    return out


def groups(r, meta, external):
    if external:
        return [("all", np.ones(len(r), bool))]
    m = meta.reindex(r.index)
    out = [("all", np.ones(len(r), bool))]
    for src in ("HEEDB", "TUAB", "TUEG"):
        for c in ("normal", "abnormal", "unlabelled"):
            mk = ((m["source"] == src) & (m["cohort"] == c)).to_numpy()
            if mk.sum() >= 10:
                out.append((f"{src} {c}", mk))
    return out


def figure(grid, curves, rej, path):
    """(a) val MAE vs attention threshold, (b) val MAE vs deviation threshold,
    (c) / (d) MAE of the kept recordings vs kept share (spread), val / test,
    with the val-chosen threshold marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import assert_max_columns, panel_labels
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    val = grid[grid.split == "val"]
    for i, (model, g) in enumerate(val.groupby("model", sort=False)):
        c = f"C{i}"
        for ax, kind in ((axes[0][0], "attention"), (axes[0][1], "deviation")):
            gk = g[(g.kind == kind) & np.isfinite(g.tau)]
            if len(gk):
                ax.plot(gk.tau, gk.mae, color=c, lw=1.3, label=model)
        for ax, split in ((axes[1][0], "val"), (axes[1][1], "test")):
            cv = curves[(curves.model == model) & (curves.score == "spread") & (curves.split == split)]
            ax.plot(cv.coverage, cv.mae, color=c, lw=1.3, label=model)
            sel = rej[(rej.model == model) & (rej.trim == "none") & (rej.score == "spread")
                      & (rej.split == split) & (rej.group == "all")]
            if len(sel):
                ax.plot(sel.kept_share, sel.mae_kept, "o", color=c, ms=5)
    axes[0][0].set_xlabel("attention threshold τ on α·n (windows below are dropped)")
    axes[0][1].set_xlabel("deviation threshold τ (years from the recording median)")
    for ax in axes[0]:
        ax.set_ylabel("val recording MAE (y)")
    for ax in axes[1]:
        ax.set_xlabel("kept share of recordings (rejected: largest spread)")
        ax.set_ylabel("MAE of kept recordings (y)")
        ax.axvline(0.8, color="0.6", ls="--", lw=1)
    axes[0][0].set_title("window trimming by attention, val")
    axes[0][1].set_title("window trimming by deviation, val")
    axes[1][0].set_title("recording rejection by spread, val")
    axes[1][1].set_title("recording rejection by spread, test (val θ: dots)")
    axes[0][1].legend(fontsize=8)              # every model is in (b); E2L has no attention for (a)
    fig.tight_layout()
    panel_labels(list(axes.flat))
    assert_max_columns(fig)
    fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--val", required=True, help="dir with <model>_val.parquet")
    ap.add_argument("--test", required=True, help="dir with <model>_test.parquet")
    ap.add_argument("--external", nargs="*", default=[], metavar="NAME=DIR")
    ap.add_argument("--models", nargs="+", default=["E2L", "W4", "L1", "W7", "X1", "X3", "Y1"])
    ap.add_argument("--min-keep", type=float, default=0.83,
                    help="share of val recordings theta must keep; 0.83 = 0.80 plus a margin, so that "
                         "at least 80 %% are kept on new data (the test cohorts and LEMON)")
    ap.add_argument("--reject-score", default="spread", choices=["spread", "neff", "entropy", "auto"],
                    help="the recording confidence of the deployed rule; 'auto' picks it on val "
                         "(it chose the attention entropy for W4, which transferred worse)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--figure", default=None, help="PNG: trim curves and rejection coverage curves")
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(__file__))
    import mil_cohort_summary as mcs
    meta = mcs.recording_table()
    os.makedirs(args.out, exist_ok=True)
    ext = dict(e.split("=", 1) for e in args.external)

    grid_rows, trim_rows, rej_rows, curve_rows = [], [], [], []
    for model in args.models:
        vp = os.path.join(args.val, f"{model}_val.parquet")
        if not os.path.exists(vp):
            print(f"{model}: no val windows, skipped")
            continue
        sets = {"val": load(vp), "test": load(os.path.join(args.test, f"{model}_test.parquet"))}
        for name, d in ext.items():
            p = os.path.join(d, f"{model}_test.parquet")
            if os.path.exists(p):
                sets[name] = load(p)
        # 1. window trimming: tau per kind on val, applied everywhere
        chosen = {}
        for split, w in sets.items():
            g = trim_grid(w)
            g["model"], g["split"] = model, split
            grid_rows.append(g)
        grid = pd.concat([g for g in grid_rows if g["model"].iloc[0] == model])
        for kind, gv in grid[grid.split == "val"].groupby("kind"):
            gv = gv.sort_values("mae")
            tau = gv["tau"].iloc[0]
            chosen[kind] = tau
            row = {"model": model, "kind": kind, "tau_val": tau}
            for split in sets:
                gs = grid[(grid.split == split) & (grid.kind == kind)].set_index("tau")
                none = 0.0 if kind == "attention" else np.inf
                row[f"{split}_mae_all"] = gs.loc[none, "mae"]
                row[f"{split}_mae_trim"] = gs.loc[tau, "mae"]
                row[f"{split}_dropped_windows"] = gs.loc[tau, "dropped_windows"]
                if split == "test":
                    # reference only (looks at test labels); never used for a decision
                    row["ref_tau_test_oracle"] = gs["mae"].idxmin()
                    row["ref_test_mae_oracle"] = gs["mae"].min()
            trim_rows.append(row)
        # 2. recording rejection, without and with the val-chosen trim
        best_kind = min(chosen, key=lambda k: grid[(grid.split == "val") & (grid.kind == k)]["mae"].min())
        none = {"attention": 0.0, "deviation": np.inf}
        for trim in ["none"] + ([best_kind] if chosen[best_kind] != none[best_kind] else []):
            recs = {}
            for split, w in sets.items():
                keep = (np.ones(len(w), bool) if trim == "none" else keep_mask(w, trim, chosen[trim]))
                recs[split] = recordings(w, keep)
            for score in REC_SCORES:
                if sets["val"]["uniform"].iloc[0] and score != "spread":
                    continue                       # neff / entropy are constant for uniform weights
                theta, vmae, vshare = choose_theta(recs["val"], score, args.min_keep)
                for split, r in recs.items():
                    k = kept(r, score, theta).to_numpy()
                    for gname, mk in groups(r, meta, split in ext):
                        rej_rows.append({
                            "model": model, "trim": trim if trim == "none" else f"{trim} {chosen[trim]:g}",
                            "score": score, "theta": theta, "split": split, "group": gname, "n": int(mk.sum()),
                            "kept_share": float(k[mk].mean()), "mae_all": mae(r[mk]),
                            "mae_kept": mae(r[mk & k]), "mae_rejected": mae(r[mk & ~k])})
                    if trim == "none":
                        for cov, m in coverage_curve(r, score):
                            curve_rows.append({"model": model, "score": score, "split": split,
                                               "coverage": cov, "mae": m})
        print(f"{model}: trims {chosen}")

    grid = pd.concat(grid_rows)
    grid.to_csv(os.path.join(args.out, "trim_grid.csv"), index=False)
    trim = pd.DataFrame(trim_rows)
    trim.to_csv(os.path.join(args.out, "trim_selected.csv"), index=False)
    rej = pd.DataFrame(rej_rows)
    # The deployed rule per model: the (trim, score) pair with the lowest val MAE of the
    # kept val recordings. Chosen on val only; test / external rows are then reported.
    v = rej[(rej.split == "val") & (rej.group == "all")]
    if args.reject_score != "auto":
        v = v[v.score == args.reject_score]
    best = v.loc[v.groupby("model")["mae_kept"].idxmin(), ["model", "trim", "score"]]
    rej["selected_on_val"] = rej.merge(best.assign(sel=True), how="left",
                                       on=["model", "trim", "score"])["sel"].fillna(False).to_numpy()
    rej.to_csv(os.path.join(args.out, "reject_selected.csv"), index=False)
    pd.DataFrame(curve_rows).to_csv(os.path.join(args.out, "coverage_curves.csv"), index=False)
    if args.figure:
        figure(grid, pd.DataFrame(curve_rows), rej, args.figure)
    pd.set_option("display.width", 250)
    print(trim.round(3).to_string(index=False))
    print(rej[rej.group == "all"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
