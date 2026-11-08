#!/usr/bin/env python
"""Tune A3 on val: spectrogram descriptors, window and record rejection, learned confidence.

Everything is fitted or selected on val (cross-fitted there, folds by
subject) and reported on test, which is never used for a choice. Per run:

1. **A3 descriptor sets** -- the 8 base descriptors, + STFT spectrogram
   descriptors, + Morlet-wavelet descriptors, + both
   (``labram/eval/window_descriptors.py``); each with its L2 strength chosen on
   val; the set with the lowest val MAE is selected.
2. **Window rejection** -- with the selected A3 weights, drop the windows whose
   relative weight ``w_i |W_c|`` is below a threshold, then pool the rest by
   their mean or by their A3 weights; threshold and pooling chosen on val.
3. **Learned recording confidence** -- a regressor of the A3-pooled recording's
   ``log(1 + |error|)`` from recording-level summaries (window spread, A3
   weight profile, TTA SD, descriptor means / SDs), with and without the
   spectrogram descriptors, ridge and gradient boosting; the variant with the
   best cross-fitted val Spearman is selected. Confidence = minus the
   prediction, so a good confidence is *negatively* correlated with the error.
4. **Record rejection** -- reject the least confident records at thresholds
   set as val quantiles (5-50 %), report realized test coverage and MAE.

    python scripts/window_confidence_tuning.py --in-dir /data/runs/heedb/window_conf \\
        --spectral /data/runs/heedb/window_conf/spectral_descriptors.parquet \\
        --run E2L --run E2 --out-dir /data/runs/heedb/window_conf/tuning --fig-dir docs/figures
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from labram.eval import window_confidence as wc

THRESHOLDS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25)
REFERENCE_SCORES = {"tta_sd_mean": "mean TTA SD", "wsd": "A3-weighted SD of windows",
                    "conc": "A3 weight concentration", "sd": "SD of windows"}
DATASETS = {"all": None, "TUH": "temple", "HEEDB": "harvard"}


def by_dataset(rec, value="abs_err"):
    out = {}
    for split, d in rec.groupby("split"):
        for name, ds in DATASETS.items():
            sub = d if ds is None else d[d["dataset"] == ds]
            out[(split, name)] = float(sub[value].mean())
    return out


def descriptor_sets(cols):
    stft = [c for c in cols if c.startswith("stft_")]
    cwt = [c for c in cols if c.startswith("cwt_")]
    return {"base": [], "base+stft": stft, "base+cwt": cwt, "base+stft+cwt": stft + cwt}


def a3_variants(df, sets):
    rows, keep = [], {}
    for name, extra in sets.items():
        w, theta, l2, scores = wc.tune_a3(df, extra=extra)
        rec = wc.pool(df, w)
        e = (df["pred"] - df["age"]).abs()
        t = df["split"] == "test"
        corr = wc.weight_error_correlation(df[t].assign(_w=w[t.to_numpy()], _e=e[t]), "_w", "_e")
        row = {"descriptors": name, "n_desc": len(wc.DESCRIPTORS) + len(extra), "l2": l2,
               "rho_within_test": corr["rho_within"], "frac_neg_test": corr["frac_neg"]}
        row.update({f"mae_{s}_{d}": v for (s, d), v in by_dataset(rec).items()})
        rows.append(row)
        keep[name] = (w, theta, extra)
        print(f"  A3[{name}] l2={l2} val {row['mae_val_all']:.3f} test {row['mae_test_all']:.3f} "
              f"(HEEDB {row['mae_test_HEEDB']:.3f}, TUH {row['mae_test_TUH']:.3f}) "
              f"rho_within {corr['rho_within']:.3f}", flush=True)
    table = pd.DataFrame(rows)
    best = table.loc[table["mae_val_all"].idxmin(), "descriptors"]
    return table, best, keep


def window_rejection(df, w):
    rows = []
    for pooled in ("mean", "weighted"):
        for t in THRESHOLDS:
            rec = wc.reject_windows(df, w, t, pooled=pooled)
            row = {"pooled": pooled, "threshold": t}
            row.update({f"mae_{s}_{d}": v for (s, d), v in by_dataset(rec).items()})
            row.update({f"kept_{s}_{d}": v for (s, d), v in by_dataset(rec, "frac_kept").items()})
            rows.append(row)
    table = pd.DataFrame(rows)
    sel = table.loc[table["mae_val_all"].idxmin()]
    return table, sel


def confidence_models(df, w, spectral_cols):
    rec = wc.pool(df, w)
    variants, feats = {}, {}
    for fset, extra in (("basic", []), ("spectral", spectral_cols)):
        feats[fset] = wc.recording_features(df, w, extra).merge(
            rec[wc.KEY + ["abs_err"]], on=wc.KEY, validate="one_to_one")
        for kind in ("ridge", "hgb"):
            variants[f"{kind}_{fset}"] = wc.fit_confidence(feats[fset], feats[fset]["abs_err"], kind)
    base = feats["basic"]
    scores = pd.DataFrame({**variants, **{k: base[k] for k in REFERENCE_SCORES}})
    out = base[list(wc.CONF_META[:6]) + ["abs_err", "pred_a3"]].join(scores.add_prefix("u_"))
    return out


def confidence_table(conf):
    rows = []
    for split, d in conf.groupby("split"):
        for name, ds in DATASETS.items():
            g = d if ds is None else d[d["dataset"] == ds]
            for col in [c for c in conf.columns if c.startswith("u_")]:
                u = g[col].to_numpy()
                rows.append({"split": split, "group": name, "score": col[2:], "n": len(g),
                             # confidence = -u: a useful one is negatively correlated with |error|
                             "rho_conf_err": -wc.spearman(u, g["abs_err"]),
                             "rho_conf_err_age": -wc.age_stratified_spearman(
                                 u, g["abs_err"].to_numpy(), g["age"].to_numpy()),
                             **wc.risk_coverage(u, g["abs_err"])})
    return pd.DataFrame(rows)


def record_rejection(conf, score):
    rows = []
    val, test = conf[conf.split == "val"], conf[conf.split == "test"]
    for name, ds in DATASETS.items():
        ref = val if ds is None else val[val.dataset == ds]
        t = test if ds is None else test[test.dataset == ds]
        # thresholds from all val recordings (one deployable rule), applied per dataset
        tab = wc.reject_records(val[f"u_{score}"], t[f"u_{score}"], t["abs_err"])
        rows.append(tab.assign(group=name, score=score, n=len(t), n_val=len(ref)))
    return pd.concat(rows, ignore_index=True)


LABELS = {"ridge_basic": "learned, ridge", "hgb_basic": "learned, boosting",
          "ridge_spectral": "learned, ridge + spectrogram", "hgb_spectral": "learned, boosting + spectrogram",
          **REFERENCE_SCORES}


def figures(results, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import assert_max_columns, panel_labels
    names = list(results)[:2]
    paths = {}

    fig, axes = plt.subplots(1, len(names), figsize=(14, 4.5), squeeze=False)
    for ax, name in zip(axes[0], names):
        r = results[name]
        tab = r["windows"][r["windows"].pooled == r["window_sel"]["pooled"]]
        for d, color in (("HEEDB", "C0"), ("TUH", "C1"), ("all", "0.3")):
            ax.plot(tab["threshold"], tab[f"mae_test_{d}"], "-o", color=color, ms=4, label=f"{d}, test")
            ax.plot(tab["threshold"], tab[f"mae_val_{d}"], "--", color=color, lw=1, label=f"{d}, val")
        ax.axvline(r["window_sel"]["threshold"], color="0.5", ls=":", lw=1)
        ax.set_xlabel("reject windows with relative A3 weight below")
        ax.set_ylabel("recording MAE (y)")
        ax.set_title(f"{name}, kept windows pooled by {r['window_sel']['pooled']}")
    axes[0][0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    panel_labels(axes[0])
    assert_max_columns(fig)
    paths["window_rejection"] = os.path.join(fig_dir, "window_confidence_window_rejection.png")
    fig.savefig(paths["window_rejection"], dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(14, 4.5), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        r = results[name]
        d = r["conf"][r["conf"].split == "test"]
        err = d["abs_err"].to_numpy()
        for col, ls in ((f"u_{r['conf_sel']}", "-"), ("u_tta_sd_mean", "-"), ("u_conc", "-"), ("abs_err", "--")):
            order = np.argsort(d[col].to_numpy(), kind="stable")
            risk = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
            label = "oracle (true error)" if col == "abs_err" else LABELS[col[2:]]
            ax.plot(np.arange(1, len(err) + 1) / len(err) * 100, risk, ls=ls, label=label)
        ax.axhline(err.mean(), color="0.5", lw=0.8, ls=":", label="random deferral")
        ax.set_xlim(10, 100)
        ax.set_ylim(0, err.mean() * 1.3)
        ax.set_xlabel("coverage (% of recordings kept)")
        ax.set_title(f"{name}, test, A3 pooling")
    axes[0][0].set_ylabel("MAE of kept recordings (y)")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    panel_labels(axes[0])
    paths["risk_coverage"] = os.path.join(fig_dir, "window_confidence_learned_risk_coverage.png")
    fig.savefig(paths["risk_coverage"], dpi=130)
    plt.close(fig)

    # per dataset x report cohort, val (cross-fitted) + test, first run
    name = names[0]
    r = results[name]
    d = r["conf"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 13.5), sharex=True)
    flat = []
    for ax, (ds, c) in zip(axes.ravel(), [(ds, c) for c in ("normal", "abnormal", "unlabelled")
                                         for ds in ("temple", "harvard")]):
        flat.append(ax)
        g = d[(d.dataset == ds) & (d.cohort == c)]
        for col in (f"u_{r['conf_sel']}", "u_tta_sd_mean", "u_conc"):
            b = pd.qcut(g[col].rank(method="first"), 5, labels=False) + 1
            m = g.groupby(b)["abs_err"].mean()
            ax.plot(m.index, m.values, "-o", ms=4, label=LABELS[col[2:]])
        ax.set_title(f"{'TUH' if ds == 'temple' else 'HEEDB'} {c}, n = {len(g)} recordings")
        ax.set_ylabel("recording MAE, A3 pooling (y)")
    for ax in axes[-1]:
        ax.set_xlabel("uncertainty quintile (5 = least confident)")
        ax.set_xticks(range(1, 6))
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    panel_labels(flat)
    assert_max_columns(fig)
    paths["cohorts"] = os.path.join(fig_dir, "window_confidence_learned_cohorts.png")
    fig.savefig(paths["cohorts"], dpi=110)
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--spectral", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fig-dir", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    spec = pd.read_parquet(args.spectral)
    spectral_cols = [c for c in spec.columns if c.startswith(("stft_", "cwt_"))]

    results, summary = {}, {}
    for name in args.run:
        df = pd.read_parquet(os.path.join(args.in_dir, f"{name}_windows.parquet"))
        df = df.merge(spec, on=["split", "recording", "window"], how="left", validate="one_to_one")
        missing = int(df[spectral_cols[0]].isna().sum())
        if missing:
            raise SystemExit(f"{name}: {missing} windows without spectral descriptors")
        df = wc.add_tta_stats(df.reset_index(drop=True))
        print(f"===== {name}: {len(df)} windows", flush=True)

        a3, best, keep = a3_variants(df, descriptor_sets(spectral_cols))
        w, theta, extra = keep[best]
        windows, wsel = window_rejection(df, w)
        conf = confidence_models(df, w, spectral_cols)
        ctab = confidence_table(conf)
        learned = [c for c in conf.columns if c.startswith("u_") and c[2:] in LABELS
                   and c[2:] not in REFERENCE_SCORES]
        val_rho = ctab[(ctab.split == "val") & (ctab.group == "all")].set_index("score")["rho_conf_err"]
        conf_sel = val_rho[[c[2:] for c in learned]].idxmin()            # most negative on val
        rej = pd.concat([record_rejection(conf, conf_sel), record_rejection(conf, "tta_sd_mean")],
                        ignore_index=True)

        for tab, fname in ((a3, "a3_descriptor_sets"), (windows, "window_rejection"),
                           (ctab, "learned_confidence"), (rej, "record_rejection")):
            tab.assign(run=name).to_csv(os.path.join(args.out_dir, f"{name}_{fname}.csv"), index=False)
        theta.rename("theta_std").to_csv(os.path.join(args.out_dir, f"{name}_a3_theta.csv"))
        conf.to_parquet(os.path.join(args.out_dir, f"{name}_learned_confidence.parquet"))
        results[name] = {"windows": windows, "window_sel": wsel, "conf": conf, "conf_sel": conf_sel}
        summary[name] = {"a3_selected": best, "a3_l2": float(a3.set_index("descriptors").loc[best, "l2"]),
                         "window_rejection": {"pooled": wsel["pooled"], "threshold": float(wsel["threshold"])},
                         "confidence_selected": conf_sel}
        with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 200):
            print(a3.round(3).to_string(index=False))
            print(theta.round(3).sort_values().to_string())
            print(windows.round(3).to_string(index=False))
            print(ctab[ctab.split == "test"].round(3).to_string(index=False))
            print(rej.round(3).to_string(index=False))
            print(summary[name])
    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    if args.fig_dir:
        print(figures(results, args.fig_dir))


if __name__ == "__main__":
    main()
