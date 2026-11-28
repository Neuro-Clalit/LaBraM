"""P4 (LaBraM) against the fine-tuned CBraMod and REVE models: data and statistics of
``notebooks/fm_encoder_summary.ipynb`` and ``docs/fm_encoder_summary.md``.

Models (all trained and selected on fold 0 of P4's 5-fold subject split):

* ``P4 (LaBraM)``  P4_L1 fold 0, best epoch 15;
* ``CBraMod``      CB5_L1 (CBraMod with its authors' fine-tuning settings), best epoch 12;
* ``REVE``         RV5_L1 (REVE-base with its authors' fine-tuning protocol), best epoch 10;
  (``SINGLE`` lists them in the report's order: P4 (LaBraM), CBraMod, REVE)
* ``P4ens (LaBraM)``         the mean of the 5 P4 fold models (test split and external cohorts only:
                             the other folds trained on fold 0's validation recordings);
* ``P4ens (LaBraM) + REVE``  that ensemble with REVE as a sixth member of equal weight
                             (``+ CBraMod`` likewise).

Sets: ``heldout`` = the adult recordings of the training cohort (Temple + Harvard) that no fold-0
model trained on, i.e. fold 0's validation fold and the test split; ``test`` = the test split
only; ``external`` = the adults of LEMON, Dortmund, TDBRAIN, SRM and CHBP (every window, no trims
for the short recordings, as ``scripts/external_runs/fm_external.sh``).

Data use: the Temple and Harvard recordings may only be shown as aggregates with small cells
suppressed (``MIN_N`` recordings, ``MIN_SUBJECTS`` subjects); per-recording points may only be
drawn for the open cohorts (``OPEN``, as ``scripts/rejection_scatter.py``).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COHORT = "/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1"
EXT_ROOT = "/data/datasets/EEG-public"
EVAL = "/data/runs/heedb/fm_encoders/eval"              # foreign-encoder scores of the external cohorts
P4_EXT = "/data/runs/heedb/external_eval"               # P4 fold models on the external cohorts
P4_CV = "/data/runs/heedb/mil_phased_cv/P4_L1"
RUNS = {
    "P4 (LaBraM)": ("/data/runs/fm_probe/p4_f0_val_rescore.json#P4_L1_f0", "/data/runs/fm_probe/p4_f0_rescore.json#P4_L1_f0"),
    "CBraMod": ("/data/runs/heedb/fm_encoders/CB5_L1/fold_0/analysis/val_bag_weights_epoch12.parquet",
                "/data/runs/heedb/fm_encoders/CB5_L1/fold_0/analysis/test_bag_weights_epoch12.parquet"),
    "REVE": ("/data/runs/heedb/fm_encoders/heedb_mil_RV5_L1_fold0_local_cv5/fold_0/analysis/val_bag_weights_epoch10.parquet",
             "/data/runs/heedb/fm_encoders/heedb_mil_RV5_L1_fold0_local_cv5/fold_0/analysis/test_bag_weights_epoch10.parquet"),
}
EXT_TAG = {"CBraMod": ("cb5", "CB5_L1_f0"), "REVE": ("rv5", "RV5_L1_f0")}
EXTERNAL = {"lemon": "LEMON", "dortmund": "DORTMUND", "tdbrain": "TDBRAIN", "srm": "SRM", "chbp": "CHBP"}
OPEN = ("lemon", "dortmund", "chbp")
NAME = {"lemon": "LEMON", "dortmund": "Dortmund", "tdbrain": "TDBRAIN", "srm": "SRM", "chbp": "CHBP"}

P4, CB, RV = "P4 (LaBraM)", "CBraMod", "REVE"
P4ENS, P4ENS_RV, P4ENS_CB = "P4ens (LaBraM)", "P4ens (LaBraM) + REVE", "P4ens (LaBraM) + CBraMod"
SINGLE = [P4, CB, RV]
ALL = [P4, CB, RV, P4ENS, P4ENS_RV]
# reference palette (docs: dataviz), categorical slots in fixed order; one colour per model everywhere
COLOR = {RV: "#257de0", CB: "#f47230", P4: "#00b894", P4ENS: "#f2a600", P4ENS_RV: "#ee7fac", P4ENS_CB: "#6a4ce2"}
STATUS_COLOR = {"normal": "#008a17", "unlabelled": "#6a4ce2", "abnormal": "#ec4647"}
GROUP_COLOR = {"reference": "#4d4d4d", "stroke": "#ee7fac", "epilepsy": "#008a17", "dementia": "#6a4ce2"}
INK, MUTED, GRID = "#191919", "#666666", "#e4e3df"
BANDS = [(18, 30, "18–29"), (30, 40, "30–39"), (40, 50, "40–49"), (50, 60, "50–59"), (60, 70, "60–69"),
         (70, 200, "70+")]          # as scripts/report_age_groups.py
MIN_N, MIN_SUBJECTS = 30, 10        # small-cell suppression (data use agreements)


# ------------------------------------------------------------------ data
def _scripts():
    path = os.path.join(REPO, "scripts")
    if path not in sys.path:
        sys.path.insert(0, path)


def bag_predictions(path):
    """recording -> predicted age of a MIL run's bag-weight file (attention-weighted window predictions)."""
    w = pd.read_parquet(path, columns=["recording", "alpha", "window_pred"])
    return (w["alpha"] * w["window_pred"]).groupby(w["recording"]).sum()


def read_predictions(path, split):
    """recording -> predicted age (years): a bag-weight parquet, or ``file.json#RUN`` of
    ``scripts/eval_age_by_cohort.py`` (its ``split``)."""
    if "#" in path:
        file, run = path.split("#", 1)
        return pd.Series(json.load(open(file))[run]["splits"][split]["predictions"], dtype=float)
    return bag_predictions(path)


def heldout(min_age=18.0):
    """The adult held-out recordings of fold 0 (validation fold + test split) with the single
    models' predictions, and the pathology group names. Columns of the cohort table
    (``scripts/pathology_brain_age.py::load_cohort``): age, sex, subject, family (harvard /
    temple), label (normal / abnormal / unlabelled), reference, one 0/1 column per group."""
    _scripts()
    from pathology_brain_age import load_cohort
    cohort, groups = load_cohort(COHORT, "multitask_targets_v2.npz", min_age)
    parts = []
    for i, split in enumerate(("val", "test")):
        cols = {m: read_predictions(paths[i], split) for m, paths in RUNS.items()}
        part = pd.DataFrame(cols).dropna()
        parts.append(part.assign(split=split))
    pred = pd.concat(parts)
    if pred.index.has_duplicates:
        raise ValueError("a recording is in the validation fold and in the test split")
    cohort = cohort.drop(columns=[c for c in ("split",) if c in cohort])   # the cohort's own split column
    return cohort.join(pred, how="inner"), groups


def _p4_fold_test(k):
    import contextlib
    import io
    _scripts()
    from fm_external_compare import _p4_test_file
    with contextlib.redirect_stdout(io.StringIO()):       # its note on fold 3 (epoch 15) is stated in the report
        path = _p4_test_file(os.path.join(P4_CV, f"fold_{k}", "cv", f"fold_{k}"))
    if path is None:
        raise FileNotFoundError(f"P4 fold {k}: no test predictions")
    return bag_predictions(path)


def test_split(min_age=18.0):
    """The adult test recordings with the single models and the two ensembles."""
    x, groups = heldout(min_age)
    x = x[x["split"] == "test"].copy()
    folds = pd.DataFrame({"f0": x[P4], **{f"f{k}": _p4_fold_test(k).reindex(x.index) for k in (1, 2, 3, 4)}})
    if folds.isna().any().any():
        raise ValueError("a P4 fold model lacks a test recording")
    x[P4ENS] = folds.mean(axis=1)
    x[P4ENS_RV] = (folds.sum(axis=1) + x[RV]) / (folds.shape[1] + 1)
    x[P4ENS_CB] = (folds.sum(axis=1) + x[CB]) / (folds.shape[1] + 1)
    return x, groups


def _json_predictions(path, prefix):
    if not os.path.exists(path):
        return {}
    return {k: pd.Series(v["splits"]["test"]["predictions"], dtype=float)
            for k, v in json.load(open(path)).items() if k.startswith(prefix)}


def external(min_age=18.0):
    """The adults of the five external cohorts with every model (one row per recording)."""
    out = []
    for cohort, folder in EXTERNAL.items():
        meta = json.load(open(os.path.join(EXT_ROOT, folder, "cohorts", f"{cohort}_v1", "age_metadata.json")))["recordings"]
        folds = _json_predictions(os.path.join(P4_EXT, f"{cohort}_all_p34.json"), "P4_L1_f")
        folds.update(_json_predictions(os.path.join(P4_EXT, f"{cohort}_all_p4f3.json"), "P4_L1_f"))
        f = pd.DataFrame(folds)[sorted(folds)]
        x = pd.DataFrame(index=f.index)
        x["age"] = pd.Series({r: float(v["age"]) for r, v in meta.items() if v.get("age") is not None}).reindex(f.index)
        x["subject"] = [f"{cohort}:{meta[r].get('subject') or r}" for r in f.index]
        x["cohort"] = cohort
        x["label"] = [meta[r].get("label") if meta[r].get("label") in ("normal", "abnormal") else "unlabelled"
                      for r in f.index]
        x[P4] = f["P4_L1_f0"]
        for m, (tag, key) in EXT_TAG.items():
            x[m] = _json_predictions(os.path.join(EVAL, f"{cohort}_all_{tag}.json"), key)[key].reindex(f.index)
        x[P4ENS] = f.mean(axis=1)
        x[P4ENS_RV] = (f.sum(axis=1) + x[RV]) / (f.shape[1] + 1)
        x[P4ENS_CB] = (f.sum(axis=1) + x[CB]) / (f.shape[1] + 1)
        if x[ALL + ["age"]].isna().any().any():
            raise ValueError(f"{cohort}: the models were not scored on the same recordings")
        out.append(x[x["age"] >= min_age])
    return pd.concat(out)


# ------------------------------------------------------------ statistics
def paired_delta(err_a, err_b, clusters, n_boot=2000, seed=0):
    """MAE(a) - MAE(b) on the same recordings and a 95 % interval from a bootstrap over clusters
    (subjects: a patient's recordings are resampled together)."""
    d = np.abs(np.asarray(err_a, float)) - np.abs(np.asarray(err_b, float))
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    s, n = np.bincount(inv, weights=d), np.bincount(inv).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    boot = s[idx].sum(1) / n[idx].sum(1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def cluster_mean(values, clusters):
    """Mean and its cluster-robust standard error (CR1; clusters = subjects)."""
    v = np.asarray(values, float)
    m = v.mean()
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    s = np.bincount(inv, weights=v - m)
    g = len(s)
    se = np.sqrt((s ** 2).sum() * g / max(g - 1, 1)) / len(v)
    return float(m), float(se)


def fmt_delta(d, lo, hi):
    sign = lambda v: f"{v:+.2f}".replace("-", "−")  # noqa: E731
    return f"{sign(d)} [{sign(lo)}, {sign(hi)}]"


def compare(sets, models, deltas=(), ref=P4, cluster="subject"):
    """One row per (name, frame) of ``sets``: n, subjects, the MAE of each model and, for each
    model in ``deltas``, its paired difference to ``ref`` (``Δ m``) with the 95 % subject-bootstrap
    interval (``Δ m lo`` / ``Δ m hi``). Rows below the small-cell limits are suppressed (NaN)."""
    rows = []
    for name, s in sets:
        row = {"": name, "n": len(s), "People": s[cluster].nunique()}
        if len(s) < MIN_N or row["People"] < MIN_SUBJECTS:
            rows.append({"": name})
            continue
        for m in models:
            row[m] = (s[m] - s["age"]).abs().mean()
        for m in deltas:
            row[f"Δ {m}"], row[f"Δ {m} lo"], row[f"Δ {m} hi"] = paired_delta(
                s[m] - s["age"], s[ref] - s["age"], s[cluster])
        rows.append(row)
    return pd.DataFrame(rows).set_index("")


def shown(df, deltas):
    """``compare`` output for display: each difference as 'Δ [lo, hi]', counts as integers."""
    out = df.copy()
    for m in deltas:
        out[f"Δ {m}"] = [fmt_delta(d, lo, hi) if pd.notna(d) else "–"
                         for d, lo, hi in zip(df[f"Δ {m}"], df[f"Δ {m} lo"], df[f"Δ {m} hi"])]
        out = out.drop(columns=[f"Δ {m} lo", f"Δ {m} hi"])
    for c in ("n", "People"):
        if c in out:
            out[c] = [f"{int(v):,}" if pd.notna(v) else "–" for v in out[c]]
    return out


def band_sets(x):
    """(label, subset) for all ages and each age band of ``BANDS``."""
    return [("all ages", x)] + [(name, x[(x["age"] >= lo) & (x["age"] < hi)]) for lo, hi, name in BANDS]


def bias_corrected(x, fit, models):
    """``x`` with each model's prediction corrected by the linear age bias fitted on ``fit``
    (gap = a + b * age, OLS; ``labram/eval/cv_compare.py::bias_correct``): prediction - (a + b * age)."""
    from labram.eval.cv_compare import bias_correct
    out = x.copy()
    for m in models:
        out[m] = x["age"] + bias_correct(x[m] - x["age"], x["age"], fit[m] - fit["age"], fit["age"])
    return out


def md_table(df, index=True, digits=2):
    """A frame as a Markdown pipe table (numbers with ``digits`` decimals, NaN as an en dash)."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "–" if np.isnan(v) else f"{v:.{digits}f}".replace("-", "−")
        return str(v)
    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join(([str(i)] if index else []) + [cell(v) for v in r.values]) + " |")
    return "\n".join(lines)


def band_table(x, models, deltas=(), ref=P4, cluster="subject"):
    """MAE per age band (and all ages): n, subjects, the MAE of each model and, for each model in
    ``deltas``, its paired difference to ``ref`` with a 95 % subject-bootstrap interval.
    Bands below the small-cell limits are suppressed."""
    rows = []
    for lo, hi, name in [(0, 200, "all ages")] + BANDS:
        s = x[(x["age"] >= lo) & (x["age"] < hi)]
        row = {"Age band": name, "n": len(s), "People": s[cluster].nunique()}
        if len(s) < MIN_N or row["People"] < MIN_SUBJECTS:
            rows.append({**row, "n": np.nan, "People": np.nan})
            continue
        for m in models:
            row[m] = (s[m] - s["age"]).abs().mean()
        for m in deltas:
            row[f"Δ {m}"] = fmt_delta(*paired_delta(s[m] - s["age"], s[ref] - s["age"], s[cluster]))
        rows.append(row)
    return pd.DataFrame(rows).set_index("Age band")


def band_gap(x, models, cluster="subject"):
    """Mean gap (prediction - age) per age band with its cluster-robust standard error."""
    rows = []
    for lo, hi, name in BANDS:
        s = x[(x["age"] >= lo) & (x["age"] < hi)]
        if len(s) < MIN_N or s[cluster].nunique() < MIN_SUBJECTS:
            continue
        for m in models:
            mean, se = cluster_mean(s[m] - s["age"], s[cluster])
            rows.append({"band": name, "model": m, "gap": mean, "se": se, "n": len(s)})
    return pd.DataFrame(rows)


def binned(x, model, width=5.0, lo=18.0, hi=95.0, cluster="subject"):
    """Per ``width``-year age bin: n, mean age, mean prediction, mean gap and its cluster-robust
    standard error. Bins below the small-cell limits are left out."""
    rows = []
    for a in np.arange(lo, hi, width):
        s = x[(x["age"] >= a) & (x["age"] < a + width)]
        if len(s) < MIN_N or s[cluster].nunique() < MIN_SUBJECTS:
            continue
        gap, se = cluster_mean(s[model] - s["age"], s[cluster])
        rows.append({"bin": a, "age": s["age"].mean(), "pred": s[model].mean(), "gap": gap, "se": se,
                     "n": len(s)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plots
def style(ax, grid_axis="both"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def curve(ax, b, color, label, y="gap", band=True, lw=2.0, ls="-"):
    """A binned mean (``binned``) as a line with markers and its 95 % band."""
    if b.empty:
        return
    ax.plot(b["age"], b[y], color=color, lw=lw, ls=ls, marker="o", ms=4.5, label=label, zorder=3)
    if band and y == "gap":
        ax.fill_between(b["age"], b["gap"] - 1.96 * b["se"], b["gap"] + 1.96 * b["se"], color=color,
                        alpha=0.15, lw=0, zorder=2)


def density(ax, x, model, gridsize=40, extent=(15, 100, 15, 100)):
    """Predicted against chronological age as a hexagon density; hexagons with fewer than
    ``MIN_SUBJECTS`` recordings are not drawn (small-cell suppression). Returns the collection."""
    hb = ax.hexbin(x["age"], x[model], gridsize=gridsize, extent=extent, mincnt=MIN_SUBJECTS, bins="log",
                   cmap="Greys", linewidths=0.2, zorder=1)
    ax.plot(extent[:2], extent[:2], color=INK, lw=1, ls=(0, (4, 3)), zorder=2)
    curve(ax, binned(x, model), COLOR[model], f"{model}: mean prediction", y="pred", band=False)
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_aspect("equal")
    return hb


def gap_boxes(ax, data, positions, color, label=None):
    """Horizontal box plots of gaps (years): box = quartiles, line = median, whiskers = 5th and
    95th percentiles, dot = mean. No single recordings are drawn (no outlier points). Returns a
    legend handle."""
    from matplotlib.patches import Patch
    bp = ax.boxplot(data, positions=positions, vert=False, widths=0.22, whis=(5, 95), showfliers=False,
                    patch_artist=True, manage_ticks=False, zorder=2)
    for box in bp["boxes"]:
        box.set(facecolor=color, alpha=0.35, edgecolor=color, linewidth=1.4)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set(color=color, linewidth=1.4)
    for line in bp["medians"]:
        line.set(color=INK, linewidth=1.6)
    ax.scatter([np.mean(d) for d in data], positions, s=22, color="white", edgecolor=color, linewidth=1.4, zorder=3)
    return Patch(facecolor=color, alpha=0.6, edgecolor=color, label=label)
