# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Analysis helpers for the per-recording meta-feature table written by
# dataset_maker/make_TUAB_meta.py: feature grouping, univariate separability,
# reference-range "abnormality signature" flags, and a metadata-only baseline
# classifier. Driven by notebooks/tuab_meta_features.ipynb.
# ---------------------------------------------------------

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# Which question each column answers. "acquisition" features describe *how* the
# EEG was recorded, not the patient: a model that leans on them is learning the
# hospital's workflow (a shortcut), not the brain.
FEATURE_GROUPS: Dict[str, List[str]] = {
    "demographics": ["age", "sex_male"],
    "history": ["session_number", "year"],
    "acquisition": [
        "duration_min", "sfreq", "physical_max_uv", "n_signals", "has_emg",
        "has_eog", "has_photic_channel", "has_sphenoidal", "has_numbered_extra",
        "photic_performed", "photic_stim_sec",
    ],
    "quality": [
        "q_clip_frac", "q_flat_epoch_frac", "q_high_amp_epoch_frac", "q_bad_epoch_frac",
        "q_n_flat_channels", "q_n_noisy_channels", "q_line60_log_ratio",
    ],
    "physiology": ["ekg_hr_bpm", "ekg_rr_std_ms", "trend_suppr_mean", "trend_suppr_gt50_frac",
                   "trend_bursts_mean"],
    "qeeg": [
        "qeeg_rel_delta", "qeeg_rel_theta", "qeeg_rel_alpha", "qeeg_rel_beta",
        "qeeg_log_total_power", "qeeg_theta_alpha_ratio", "qeeg_slowing_ratio",
        "qeeg_pdr_freq", "qeeg_pdr_prominence", "qeeg_aperiodic_slope",
        "qeeg_max_asymmetry", "qeeg_max_delta_asymmetry", "qeeg_median_kurtosis",
    ],
}

# The clinically motivated "why abnormal" signatures. Each is a qEEG feature and
# the direction in which it becomes abnormal; the threshold is set from the
# *normal* recordings so that ~5% of normals are flagged by construction.
SIGNATURES = {
    "diffuse_slowing": ("qeeg_slowing_ratio", "high"),
    "slow_pdr": ("qeeg_pdr_freq", "low"),
    "absent_pdr": ("qeeg_pdr_prominence", "low"),
    "asymmetry": ("qeeg_max_asymmetry", "high"),
    "focal_slowing": ("qeeg_max_delta_asymmetry", "high"),
    "sharp_transients": ("qeeg_median_kurtosis", "high"),
    "low_voltage": ("qeeg_log_total_power", "low"),
}


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric encodings the analyses share (``is_abnormal``, ``sex_male``, age bins)."""
    out = df.copy()
    out["is_abnormal"] = (out["label"] == "abnormal").astype(int)
    out["sex_male"] = out["sex"].map({"M": 1.0, "F": 0.0})
    out["age_group"] = pd.cut(out["age"], bins=[0, 20, 40, 60, 80, 120],
                              labels=["<20", "20-39", "40-59", "60-79", "80+"], right=False)
    for col in [c for cols in FEATURE_GROUPS.values() for c in cols]:
        if col in out and out[col].dtype == bool:
            out[col] = out[col].astype(float)
    return out


def available_features(df: pd.DataFrame, groups: Optional[Iterable[str]] = None) -> List[str]:
    groups = list(groups) if groups is not None else list(FEATURE_GROUPS)
    return [c for g in groups for c in FEATURE_GROUPS[g] if c in df.columns]


def feature_group_of(column: str) -> str:
    return next((g for g, cols in FEATURE_GROUPS.items() if column in cols), "other")


def univariate_auc(df: pd.DataFrame, features: Iterable[str], target: str = "is_abnormal") -> pd.DataFrame:
    """ROC-AUC of each feature on its own, oriented so 0.5 = useless, 1 = perfect.

    ``direction`` says whether higher values mean abnormal. Missing values are
    dropped per feature, and ``coverage`` reports how many rows that left.
    """
    from sklearn.metrics import roc_auc_score

    rows = []
    for col in features:
        sub = df[[col, target]].dropna()
        if sub[target].nunique() < 2 or sub[col].nunique() < 2:
            continue
        auc = roc_auc_score(sub[target], sub[col])
        rows.append({
            "feature": col,
            "group": feature_group_of(col),
            "auc": max(auc, 1 - auc),
            "direction": "higher=abnormal" if auc >= 0.5 else "lower=abnormal",
            "coverage": len(sub) / len(df),
            "median_normal": sub.loc[sub[target] == 0, col].median(),
            "median_abnormal": sub.loc[sub[target] == 1, col].median(),
        })
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def signature_flags(
    df: pd.DataFrame,
    reference: pd.Series,
    *,
    quantile: float = 0.95,
    signatures: Optional[Dict[str, tuple]] = None,
) -> pd.DataFrame:
    """Flag recordings outside the reference (normal) range on each signature.

    ``reference`` is a boolean mask selecting the reference recordings (e.g.
    normal recordings of the training split). Thresholds are per age group when
    ``age_group`` is present, because e.g. the PDR slows with age.
    """
    signatures = signatures or SIGNATURES
    flags = pd.DataFrame(index=df.index)
    has_groups = "age_group" in df.columns
    for name, (col, direction) in signatures.items():
        if col not in df:
            continue
        flag = pd.Series(False, index=df.index)
        strata = df["age_group"].cat.categories if has_groups else [None]
        for stratum in strata:
            in_stratum = (df["age_group"] == stratum) if has_groups else pd.Series(True, index=df.index)
            ref_values = df.loc[reference & in_stratum, col].dropna()
            if len(ref_values) < 20:                       # too few to set a range: pool
                ref_values = df.loc[reference, col].dropna()
            q = quantile if direction == "high" else 1 - quantile
            threshold = ref_values.quantile(q)
            values = df.loc[in_stratum, col]
            hit = values > threshold if direction == "high" else values < threshold
            flag[in_stratum] = hit.to_numpy(dtype=bool)          # NaN compares False
        flags[name] = flag
    return flags


def metadata_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: List[str],
    target: str = "is_abnormal",
    seed: int = 0,
):
    """Gradient-boosted trees on metadata only; returns (metrics, fitted model).

    Trees handle the NaNs (missing EKG, absent trend channels) natively, so no
    imputation decision leaks into the comparison.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           l2_regularization=1.0, random_state=seed)
    model.fit(train[features].astype(float), train[target])
    prob = model.predict_proba(test[features].astype(float))[:, 1]
    metrics = {
        "auroc": roc_auc_score(test[target], prob),
        "balanced_acc": balanced_accuracy_score(test[target], prob >= 0.5),
        "n_features": len(features),
    }
    return metrics, model
