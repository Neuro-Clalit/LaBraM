# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Analysis helpers for the per-recording meta-feature table written by
# dataset_maker/make_TUAB_meta.py (TUAB, from the EDFs) and
# dataset_maker/make_cohort_meta.py (a TUEG cohort, from the npy + clinical
# sheet + header dump): feature grouping, univariate separability,
# reference-range "abnormality signature" flags, signature prevalence per
# clinical diagnosis, and metadata-only baselines for normal/abnormal and age.
# Driven by notebooks/tuab_meta_features.ipynb.
# ---------------------------------------------------------

import re
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

# The TUEG cohort table (make_cohort_meta.py). No EKG/photic/trend *signals*
# (only the processed scalp npy is on disk), but the header dump gives the
# channel inventory and NEDC's sheet adds what TUAB never had: diagnoses,
# medication classes and the report's EEG findings. The findings (FIRDA, GPD,
# ...) are part of *why* the report says abnormal, so they are kept apart from
# the clinical context: a model that uses them reads the answer key.
EQUIPMENT_COLUMNS = ["equip_compact_32ch", "equip_compact_64ch", "equip_intution_ultrasom",
                     "equip_alliance", "equip_bravo", "equip_b545"]
COHORT_FEATURE_GROUPS: Dict[str, List[str]] = {
    "demographics": ["age", "sex_male"],
    "history": ["session_number", "year"],
    "acquisition": [
        "duration_min", "sfreq", "physical_max_uv", "n_signals_header", "has_ekg", "has_emg",
        "has_eog", "has_photic_channel", "has_sphenoidal", "has_numbered_extra",
        "has_trend_channels", "montage_le", "montage_reduced", "n_missing_channels",
        *EQUIPMENT_COLUMNS,
    ],
    "quality": ["q_flat_epoch_frac", "q_high_amp_epoch_frac", "q_bad_epoch_frac",
                "q_n_flat_channels", "q_n_noisy_channels", "q_median_amp_uv"],
    "clinical": [
        "dx_electrographic_seizure", "dx_clinical_seizure", "dx_stroke", "dx_epilepsy",
        "dx_parkinsons", "dx_alzheimers", "dx_tbi", "dx_schizophrenia", "dx_dementia",
        "dx_substance", "dx_anxiety", "dx_depression", "dx_bipolar", "dx_adhd", "dx_ms",
        "med_anesthetic", "med_benzodiazepine", "med_barbiturate", "med_opioid",
        "med_antiseizure", "med_antipsychotic",
    ],
    "report_findings": ["f_firda", "f_oirda", "f_tirda", "f_gpd", "f_lpd"],
    "qeeg": FEATURE_GROUPS["qeeg"],
}


# Reason categories of NEDC's free-text "abnormal due to" finding (TUEG clinical
# sheet), by keyword. Multi-label: "diffuse slowing and left temporal sharp
# waves" is both diffuse_slowing and epileptiform. The text is lower-cased
# first; the sheet spells "left" as "leftt" in places. Each maps onto the qEEG
# signature that should see it (or None when no whole-recording spectrum can).
_SIDE = r"\b(?:left|leftt|right|hemispher\w*|temporal|occipital|parietal|unilateral)\b"
_SLOW = r"(?:slow\w*|delta|theta)"
REPORT_REASONS: Dict[str, str] = {
    "diffuse_slowing": (r"(?:diffuse|generali[sz]ed|background|global|bilateral|marked\w*|severe|"
                        r"moderate|mild|shifting)[^.]{0,30}" + _SLOW + r"|slow pattern|"
                        r"excess\w* (?:background )?(?:theta|delta)|encephalopath|dysfunction|"
                        r"firda|oirda|triphasic|mixed frequency"),
    "focal_slowing": r"focal|lateraliz|tirda|lrda|" + _SIDE + r"[^.]{0,40}" + _SLOW + "|"
                     + _SLOW + r"[^.]{0,40}" + _SIDE,
    "pdr_abnormal": r"alpha|posterior (?:dominant )?rhythm|\bpdr\b",
    "epileptiform": r"spike|sharp|epileptiform|pled|\blpd|\bgpd|peds|periodic|polyspike|discharge|paroxysm",
    "seizure": r"seizure|ictal|status epilep",
    "excess_beta": r"beta|fast activity|excessive fast|drug effect|medication effect",
    "suppression": (r"suppress|burst|cerebral activity|electrocerebral|low voltage|attenuat|"
                    r"isoelectric|discontinu|paucity|no recognizable"),
    "asymmetry": r"asymmetr",
}
REASON_TO_SIGNATURE = {
    "diffuse_slowing": "diffuse_slowing", "focal_slowing": "focal_slowing",
    "pdr_abnormal": "slow_pdr", "asymmetry": "asymmetry", "suppression": "low_voltage",
    "epileptiform": "sharp_transients", "seizure": None, "excess_beta": None,
}


def report_reason_flags(text: pd.Series) -> pd.DataFrame:
    """Boolean reason categories (:data:`REPORT_REASONS`) of "abnormal due to" texts.

    A missing text gives False everywhere; ``reason_unmatched`` marks a text
    that no rule matched.
    """
    lower = text.fillna("").astype(str).str.lower()
    out = pd.DataFrame({name: lower.str.contains(pattern, regex=True)
                        for name, pattern in REPORT_REASONS.items()}, index=text.index)
    out["reason_unmatched"] = (lower.str.strip() != "") & ~out.any(axis=1)
    return out


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric encodings the analyses share (``is_abnormal``, ``sex_male``, age bins).

    ``is_abnormal`` is 1/0 for abnormal/normal and NaN for a recording without
    a label (TUEG sessions missing from the clinical sheet), never 0. Booleans
    of either feature-group table become floats (nullable ones keep NaN), and
    the cohort's montage / amplifier family become indicator columns.
    """
    out = df.copy()
    labelled = out["label"].isin(["abnormal", "normal"])
    out["is_abnormal"] = (out["label"] == "abnormal").astype(float).where(labelled)
    if labelled.all():
        out["is_abnormal"] = out["is_abnormal"].astype(int)
    out["sex_male"] = out["sex"].map({"M": 1.0, "F": 0.0})
    out["age_group"] = pd.cut(out["age"], bins=[0, 20, 40, 60, 80, 120],
                              labels=["<20", "20-39", "40-59", "60-79", "80+"], right=False)
    if "montage" in out:
        out["montage_le"] = out["montage"].str.contains("_le").astype(float).where(out["montage"].notna())
        out["montage_reduced"] = out["montage"].str.endswith("_a").astype(float).where(out["montage"].notna())
    if "equipment_family" in out:
        for col in EQUIPMENT_COLUMNS:
            family = col[len("equip_"):]
            out[col] = (out["equipment_family"].str.lower().str.replace("/", "_") == family).astype(float)
    all_cols = {c for groups in (FEATURE_GROUPS, COHORT_FEATURE_GROUPS) for cols in groups.values() for c in cols}
    for col in all_cols & set(out.columns):
        if out[col].dtype == bool or str(out[col].dtype) == "boolean":
            out[col] = out[col].astype("Float64").astype(float)
    return out


def available_features(df: pd.DataFrame, groups: Optional[Iterable[str]] = None,
                       feature_groups: Optional[Dict[str, List[str]]] = None) -> List[str]:
    feature_groups = feature_groups or FEATURE_GROUPS
    groups = list(groups) if groups is not None else list(feature_groups)
    return [c for g in groups for c in feature_groups[g] if c in df.columns]


def feature_group_of(column: str, feature_groups: Optional[Dict[str, List[str]]] = None) -> str:
    feature_groups = feature_groups or FEATURE_GROUPS
    return next((g for g, cols in feature_groups.items() if column in cols), "other")


def univariate_auc(df: pd.DataFrame, features: Iterable[str], target: str = "is_abnormal",
                   feature_groups: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
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
            "group": feature_group_of(col, feature_groups),
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


def signature_by_condition(flags: pd.DataFrame, conditions: pd.DataFrame,
                           min_positive: int = 30) -> pd.DataFrame:
    """Prevalence of each signature with vs. without each clinical condition.

    ``conditions`` holds boolean/0-1 columns (diagnoses, drug classes) aligned
    with ``flags``; rows where a condition is unknown (NaN) are left out of that
    condition. Returns one row per (condition, signature) with the prevalence in
    condition-positive and -negative recordings and their ratio (``lift``), so a
    qEEG phenotype can be checked against what the clinical sheet says.
    """
    rows = []
    for cond in conditions.columns:
        known = conditions[cond].notna()
        positive = conditions[cond].fillna(0).astype(bool) & known
        negative = ~conditions[cond].fillna(1).astype(bool) & known
        if positive.sum() < min_positive:
            continue
        for sig in flags.columns:
            p_pos, p_neg = flags.loc[positive, sig].mean(), flags.loc[negative, sig].mean()
            rows.append({"condition": cond, "signature": sig, "n_positive": int(positive.sum()),
                         "prev_with": p_pos, "prev_without": p_neg,
                         "lift": p_pos / max(p_neg, 1e-3)})
    return pd.DataFrame(rows)


def metadata_age_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: List[str],
    target: str = "age",
    seed: int = 0,
):
    """Gradient-boosted regression of age on metadata only; returns (metrics, model).

    The yardstick for brain-age models: how much of age a few non-EEG
    columns (or 13 qEEG summaries) already explain. ``mae_mean`` is the MAE of
    always predicting the training mean age, the trivial floor.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=seed)
    model.fit(train[features].astype(float), train[target])
    pred = model.predict(test[features].astype(float))
    y = test[target].to_numpy(dtype=float)
    err = pred - y
    metrics = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
        "pearson_r": float(np.corrcoef(pred, y)[0, 1]),
        "mae_mean": float(np.mean(np.abs(train[target].mean() - y))),
        "n_features": len(features),
    }
    return metrics, model
