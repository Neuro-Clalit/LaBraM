#!/usr/bin/env python
"""Window trimming by attention weight (technical report Section 6.3): inside every recording,
drop the 10 s windows the attention pool trusts least, then re-pool the rest.

**Window confidence.** For fold model ``k`` and a recording with ``n`` scored windows, the relative
attention weight of window ``i`` is ``w_i = n * alpha_i`` (1 = uniform weight).

**Thresholds (label-free, fixed once on validation).** ``tau`` = the 10th / 25th / 50th percentile of
``w`` over the validation windows of the fold models (each on its own validation fold at its best
epoch, adults, first 5 minutes): trimming ``low`` / ``medium`` / ``high``. The same three values are
applied to every cohort.

**Rule.** Each fold model keeps the windows with ``w_i >= tau`` (at least ``--min-windows``, the
highest-weighted, if fewer pass) and re-pools them exactly, ``yhat_k = sum_kept alpha_i p_i /
sum_kept alpha_i`` (exact for the softmax pool followed by the linear head). The ensemble is the
mean of its fold models; TDBRAIN and Dortmund sessions the mean of their files. Reference rows: no
trimming (the reported prediction) and the plain mean of the windows (no attention).

**Outputs.** ``window_trimming.csv`` (per set, model, rule: MAE with 95 % subject-bootstrap interval,
bias, paired change of the MAE against no trimming with its interval, share of windows trimmed,
share of units whose prediction changed), ``window_trimming_thresholds.csv`` and the figure.

    python scripts/window_trimming.py --set temple:<cohort>:<eval.json>:<windows>:temple ... \\
        --val-windows <parquet,...> --val-cohort <tuh_heedb_v1> --out-dir docs/data/external_eval/tech_report \\
        --figure docs/tech_report/fig_window_trimming.png --example lemon:sub-032490_lemon-rs
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from external_cohort_table import load_meta  # noqa: E402

LEVELS = [("low", 0.10), ("medium", 0.25), ("high", 0.50)]
N_BOOT = 2000


def pool(w, tau=None, min_windows=3, mean=False):
    """Per recording of one fold model's windows ``w`` (recording, alpha, window_pred): the pooled
    prediction after trimming at ``tau`` (None = no trimming; ``mean``: plain mean, no attention),
    the number of windows and the number trimmed."""
    w = w[["recording", "alpha", "window_pred"]].copy()
    g = w.groupby("recording", sort=False)
    n = g["alpha"].transform("size")
    w["rel"] = n * w["alpha"] / g["alpha"].transform("sum")
    if tau is None:
        w["keep"] = True
    else:
        rank = g["rel"].rank(ascending=False, method="first")
        w["keep"] = (w["rel"] >= tau) | (rank <= min_windows)
    k = w[w["keep"]]
    weight = pd.Series(1.0, index=k.index) if mean else k["alpha"]
    num = (weight * k["window_pred"]).groupby(k["recording"], sort=False).sum()
    den = weight.groupby(k["recording"], sort=False).sum()
    out = pd.DataFrame({"pred": num / den, "n": g.size(), "trimmed": (~w["keep"]).groupby(w["recording"], sort=False).sum()})
    return out


def boot_mae(err, groups, n_boot=N_BOOT, seed=0):
    """MAE and its 95 % interval over bootstrap resamples of subjects; ``err`` may be 2-D
    (columns = rules) to get paired resamples."""
    rng = np.random.default_rng(seed)
    err = np.abs(np.asarray(err, float))
    if err.ndim == 1:
        err = err[:, None]
    uniq, inv = np.unique(groups, return_inverse=True)
    s = np.zeros((len(uniq), err.shape[1]))
    np.add.at(s, inv, err)
    cnt = np.bincount(inv, minlength=len(uniq)).astype(float)
    boots = []
    for _ in range(n_boot):
        wgt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        boots.append((wgt[:, None] * s).sum(0) / (wgt * cnt).sum())
    return err.mean(0), np.array(boots)


def evaluate(units, rules, name, model, rows):
    """units: DataFrame with subject, age and one prediction column per rule (+ trimmed / n per rule)."""
    err = np.stack([units[f"pred_{r}"] - units["age"] for r in rules], axis=1)
    mae, boots = boot_mae(err, units["subject"].astype(str).to_numpy())
    base = rules.index("none")
    for j, r in enumerate(rules):
        d = boots[:, j] - boots[:, base]
        rows.append({"set": name, "model": model, "rule": r, "units": len(units), "subjects": units["subject"].nunique(),
                     "mae": mae[j], "mae_lo": np.percentile(boots[:, j], 2.5), "mae_hi": np.percentile(boots[:, j], 97.5),
                     "bias": err[:, j].mean(), "delta_mae": mae[j] - mae[base],
                     "delta_lo": np.percentile(d, 2.5), "delta_hi": np.percentile(d, 97.5),
                     "windows_trimmed_share": units.get(f"trim_{r}", pd.Series(0.0)).sum() / units["n_windows"].sum()
                     if f"trim_{r}" in units else 0.0,
                     "units_changed_share": float((np.abs(units[f"pred_{r}"] - units["pred_none"]) > 0.05).mean())})


def fold_table(w, taus, min_windows):
    t = {"none": pool(w)}
    t["mean"] = pool(w, mean=True)
    for lvl, tau in taus.items():
        t[lvl] = pool(w, tau, min_windows)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", required=True, metavar="NAME:COHORT:EVAL.json:WINDOWS_DIR[:FAMILY]")
    ap.add_argument("--model", default="P4ens")
    ap.add_argument("--val-windows", required=True, help="comma-separated validation window parquets of the fold models")
    ap.add_argument("--val-cohort", required=True, help="training cohort (corpus and subject of the validation recordings)")
    ap.add_argument("--min-windows", type=int, default=3)
    ap.add_argument("--min-age", type=float, default=18.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--figure", default=None)
    ap.add_argument("--example", default=None, metavar="SET:RECORDING")
    args = ap.parse_args()
    rules = ["none", "mean"] + [lvl for lvl, _ in LEVELS]

    # thresholds on validation
    val = [pd.read_parquet(p) for p in args.val_windows.split(",")]
    val = [v[v["age"] >= args.min_age] for v in val]
    rel = np.concatenate([(v.groupby("recording")["alpha"].transform("size") * v["alpha"]
                           / v.groupby("recording")["alpha"].transform("sum")).to_numpy() for v in val])
    taus = {lvl: float(np.quantile(rel, q)) for lvl, q in LEVELS}
    print("thresholds:", {k: round(v, 3) for k, v in taus.items()}, flush=True)

    rows, dists = [], {"validation": rel}
    # validation: each recording scored by the one fold model it was held out from
    coh = pd.read_parquet(os.path.join(args.val_cohort, "cohort.parquet")).set_index("stem")
    sub = np.where(coh["corpus"] == "heedb", "h" + coh["patient_id"].astype(str), coh["subject"].astype(str))
    coh = coh.assign(pat=sub)
    vunits = []
    for k, v in enumerate(val):
        t = fold_table(v, taus, args.min_windows)
        u = pd.DataFrame({f"pred_{r}": t[r]["pred"] for r in rules})
        for r in rules:
            u[f"trim_{r}"] = t[r]["trimmed"]
        u["n_windows"] = t["none"]["n"]
        u["age"] = v.groupby("recording")["age"].first()
        vunits.append(u)
    vu = pd.concat(vunits)
    vu = vu.join(coh[["corpus", "pat"]], how="left")
    for fam, corp in (("Temple validation", ("tuh",)), ("Harvard validation", ("heedb",))):
        g = vu[vu["corpus"].isin(corp)].rename(columns={"pat": "subject"})
        evaluate(g, rules, fam, "fold model", rows)
        print(f"{fam}: {len(g):,} recordings", flush=True)

    # test cohorts: ensemble of the fold models
    examples = {}
    for spec in args.set:
        name, cdir, evp, wdir, *family = spec.split(":")
        meta = load_meta(cdir, family[0] if family else None, args.min_age)
        ev = json.load(open(evp))
        members = ev[args.model].get("members", [args.model])
        per = []
        allrel = []
        for m in members:
            w = pd.read_parquet(os.path.join(wdir, f"{m}_test.parquet"), columns=["recording", "slot", "alpha", "window_pred"])
            w = w[w["recording"].isin(meta.index)]
            per.append(fold_table(w, taus, args.min_windows))
            allrel.append((w.groupby("recording")["alpha"].transform("size") * w["alpha"]
                           / w.groupby("recording")["alpha"].transform("sum")).to_numpy())
            if args.example and args.example.split(":")[0] == name:
                rec = args.example.split(":", 1)[1]
                examples.setdefault("windows", []).append(w[w["recording"] == rec].assign(fold=m))
        dists[name] = np.concatenate(allrel)
        idx = per[0]["none"].index
        u = pd.DataFrame(index=idx)
        for r in rules:
            u[f"pred_{r}"] = np.mean([p[r]["pred"].reindex(idx).to_numpy() for p in per], axis=0)
            u[f"trim_{r}"] = np.mean([p[r]["trimmed"].reindex(idx).to_numpy() for p in per], axis=0)
        u["n_windows"] = per[0]["none"]["n"].reindex(idx)
        cols = ["subject", "age"] + (["session"] if "task" in meta else [])
        u = meta[cols].join(u, how="inner")
        if "task" in meta:                                      # session = mean of its files
            u = u.groupby(["subject", "session"]).agg(
                {**{c: "mean" for c in u.columns if c.startswith("pred_") or c == "age"},
                 **{c: "sum" for c in u.columns if c.startswith("trim_") or c == "n_windows"}}).reset_index()
        evaluate(u, rules, name, args.model, rows)
        if args.example and args.example.split(":")[0] == name:
            examples["preds"] = u.loc[args.example.split(":", 1)[1]] if args.example.split(":", 1)[1] in u.index else None
        print(f"{name}: {len(u):,} units", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "window_trimming.csv"), index=False)
    pd.DataFrame([{"level": lvl, "validation_quantile": q, "tau": taus[lvl]} for lvl, q in LEVELS]).to_csv(
        os.path.join(args.out_dir, "window_trimming_thresholds.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(df.pivot_table(index="set", columns="rule", values="mae", sort=False).round(2).to_string())
        print(df.pivot_table(index="set", columns="rule", values="windows_trimmed_share", sort=False).round(3).to_string())
    if args.figure:
        figure(args.figure, df, taus, dists, examples)


def figure(path, df, taus, dists, examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import assert_max_columns, panel_labels
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 16})
    LC = {"low": "#f4b183", "medium": "#eb6834", "high": "#a33a0f"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 13.5))
    a = list(axes.flat)
    # (a) distribution of the relative attention weight
    edges = np.linspace(0, 3, 61)
    for name, col, lw in (("validation", "#1f1f1e", 2.4), ("temple", "#2a78d6", 1.4), ("harvard", "#6250d6", 1.4),
                          ("lemon", "#1baf7a", 1.4), ("chbp", "#d9480f", 1.4)):
        if name in dists:
            a[0].hist(np.clip(dists[name], 0, 3), bins=edges, density=True, histtype="step", lw=lw, color=col,
                      label={"validation": "validation (τ fixed here)", "temple": "Temple test", "harvard": "Harvard test", "lemon": "LEMON", "chbp": "CHBP"}[name])
    for lvl, tau in taus.items():
        a[0].axvline(tau, color=LC[lvl], lw=2, ls="--", label=f"{lvl}: τ = {tau:.2f}")
    a[0].set_xlabel("relative weight $w = n\\,\\alpha$ (1 = uniform; 3 = 3+)")
    a[0].set_ylabel("density of windows")
    a[0].set_title("window confidence and the three thresholds")
    a[0].legend(frameon=False, fontsize=12.5, loc="upper right")
    a[0].set_xlim(-0.05, 4.4)
    a[0].set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    # (b) example recording
    if examples.get("windows"):
        w = pd.concat(examples["windows"])
        n = w.groupby("fold")["alpha"].transform("size")
        w["rel"] = n * w["alpha"] / w.groupby("fold")["alpha"].transform("sum")
        e = w.groupby("slot").agg(rel=("rel", "mean"), p=("window_pred", "mean"))
        t = (e.index.to_numpy() + 0.5) / 6
        col = np.where(e["rel"] >= taus["high"], "#2a78d6",
                       np.where(e["rel"] >= taus["medium"], LC["high"], np.where(e["rel"] >= taus["low"], LC["medium"], LC["low"])))
        a[1].bar(t, e["rel"], width=1 / 6 * 0.9, color=col)
        for lvl, tau in taus.items():
            a[1].axhline(tau, color=LC[lvl], lw=1.6, ls="--")
        p = examples.get("preds")
        if p is not None:
            txt = " · ".join(f"{lbl} {p[f'pred_{r}']:.1f}"
                             for r, lbl in (("none", "none"), ("low", "low"), ("medium", "med."), ("high", "high")))
            a[1].set_title(f"one LEMON recording, aged {p['age'] - 2.5:.0f}–{p['age'] + 2.5:.0f}\npredicted {txt}", fontsize=14)
        a[1].set_xlabel("time in the scored recording (min)")
        a[1].set_ylabel("relative weight (fold mean)")
        from matplotlib.patches import Patch
        a[1].legend(handles=[Patch(color="#2a78d6", label="kept at all levels"), Patch(color=LC["high"], label="trimmed at high only"),
                             Patch(color=LC["medium"], label="trimmed at medium, high"), Patch(color=LC["low"], label="trimmed at all levels")],
                    frameon=False, fontsize=12, loc="upper left", ncol=2)
        a[1].set_ylim(0, max(3.4, e["rel"].max() * 1.6))
    # (c) share of windows trimmed, (d) change of the MAE
    sets = list(dict.fromkeys(df["set"]))
    y = np.arange(len(sets))[::-1]
    hgt = 0.26
    for j, (lvl, _) in enumerate(LEVELS):
        sub = df[df["rule"] == lvl].set_index("set").reindex(sets)
        a[2].barh(y + (1 - j) * hgt, 100 * sub["windows_trimmed_share"], hgt, color=LC[lvl], label=lvl)
        a[3].errorbar(sub["delta_mae"], y + (1 - j) * hgt, xerr=[sub["delta_mae"] - sub["delta_lo"], sub["delta_hi"] - sub["delta_mae"]],
                      fmt="o", color=LC[lvl], ms=6, capsize=3, label=lvl)
    sub = df[df["rule"] == "mean"].set_index("set").reindex(sets)
    a[3].errorbar(sub["delta_mae"], y - 2 * hgt, xerr=[sub["delta_mae"] - sub["delta_lo"], sub["delta_hi"] - sub["delta_mae"]],
                  fmt="s", color="#6b6a64", ms=5, capsize=3, label="plain mean (no attention)")
    labels = [s.replace("temple", "Temple test").replace("harvard", "Harvard test").replace("tdbrain", "TDBRAIN")
              .replace("lemon", "LEMON").replace("dortmund", "Dortmund").replace("srm", "SRM").replace("chbp", "CHBP") for s in sets]
    for ax in (a[2], a[3]):
        ax.set_yticks(y, labels)
    a[2].set_xlabel("windows trimmed (%)")
    a[2].set_title("share of windows trimmed")
    a[2].legend(frameon=False, fontsize=14, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.17))
    a[3].axvline(0, color="#1f1f1e", lw=1)
    a[3].set_xlabel("ΔMAE against no trimming (years)")
    a[3].set_title("effect on the MAE (95 % paired intervals)")
    a[3].legend(frameon=False, fontsize=13.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.17))
    for ax in a:
        ax.grid(color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    panel_labels(a)
    assert_max_columns(fig)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
