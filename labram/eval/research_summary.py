"""Helpers for ``notebooks/brain_age_research_summary.ipynb``: one place that
turns the brain-age study's records into tables.

* :func:`parse_markdown_table` reads a numbered table (the experiment registry
  ``docs/age_overfitting_experiments.md``, Table 12.1) into a frame.
* :func:`parse_submit_script` reads the per-experiment arrays of
  ``scripts/submit_tuh_cohorts.sh`` (cohort, epochs, warmup, extra ``--set``
  overrides) with the script's ``${FAST}`` / ``${FEATS}`` expanded.
* :data:`PHASE2` is the catalog of the TUH cohort experiments (the names used
  in ``docs/data/tuh_cohort_eval_v1.csv``), with the overrides each one ran
  with; :func:`experiment_overrides` / :func:`resolve_config` rebuild a run's
  full :class:`FinetuneRunConfig` from them.
* :func:`config_diff` lists every scientific field two configs disagree on.
* :func:`seed_noise` and :func:`verdict` implement the noise floor and the
  significance rule of the notebook's Section 5.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Markdown tables


def _md_cell(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\[`?([^\]`]+)`?\]\([^)]*\)", r"\1", s)        # [`id`](url) -> id
    return s.replace("`", "").replace("$", "").replace("\\beta", "β").replace("e^{\\star}", "e*")


def _md_number(s: str) -> float:
    s = s.strip().replace("−", "-").rstrip("†*")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_markdown_table(text: str, heading: str) -> pd.DataFrame:
    """The first pipe table after the line that starts with ``heading`` (e.g.
    ``"### Table 12.1."``), as strings with links, backticks and math markup
    stripped."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(heading))
    rows: List[List[str]] = []
    for ln in lines[start + 1:]:
        if ln.startswith("|"):
            rows.append([_md_cell(c) for c in ln.strip().strip("|").split("|")])
        elif rows:
            break
    header, body = rows[0], [r for r in rows[1:] if not set("".join(r)) <= set("-: ")]
    return pd.DataFrame(body, columns=header)


def registry_master_table(path: str) -> pd.DataFrame:
    """Table 12.1 of the experiment registry with numeric columns: ``val``,
    ``test``, ``test_normal``, ``test_abnormal``, ``beta``, ``best_epoch``,
    ``epochs`` (years / epochs)."""
    with open(path) as f:
        t = parse_markdown_table(f.read(), "### Table 12.1.")
    change = next(c for c in t.columns if c.startswith("Change"))
    out = pd.DataFrame({
        "run": t["Run"], "family": t["Family"], "change": t[change], "clearml": t["ClearML"],
        "val": t["val MAE"].map(_md_number), "test": t["test MAE"].map(_md_number)})
    na = t[[c for c in t.columns if "normal" in c][0]].str.split("/")
    out["test_normal"] = na.map(lambda p: _md_number(p[0]) if len(p) == 2 else float("nan"))
    out["test_abnormal"] = na.map(lambda p: _md_number(p[1]) if len(p) == 2 else float("nan"))
    out["beta"] = t[[c for c in t.columns if "β" in c][0]].map(_md_number)
    ep = t[[c for c in t.columns if "epochs" in c][0]].str.split("/")
    out["best_epoch"] = ep.map(lambda p: _md_number(p[0]))
    out["epochs"] = ep.map(lambda p: _md_number(p[1]))
    return out


# ---------------------------------------------------------------------------
# scripts/submit_tuh_cohorts.sh


def parse_submit_script(path: str, fast_batch: int = 128) -> Dict[str, Dict[str, object]]:
    """Per experiment of ``submit_tuh_cohorts.sh``: ``cohort``, ``epochs``,
    ``warmup`` and ``extra`` (the list of ``key=value`` overrides of its
    ``EXTRA`` entry, with ``${FAST}`` / ``${FEATS}`` expanded the way the
    script does for ``FAST_BATCH=fast_batch``)."""
    with open(path) as f:
        text = f.read()
    fast_lr = 1e-4 * fast_batch / 64
    variables = {
        "FAST": (f"model.fused_attention=true data.num_workers=6 trainer.batch_size={fast_batch} "
                 f"optimizer.lr={fast_lr!r} data.train_minutes_per_epoch=5"),
        "FEATS": "data.session_features=session_features_v1.npz",
    }

    def array(name: str) -> Dict[str, str]:
        m = re.search(rf"declare -A {name}=\((.*?)\)\n", text, re.S)
        if m is None:
            raise ValueError(f"{path}: no 'declare -A {name}=(...)'")
        body = re.sub(r"#[^\n]*", "", m.group(1))
        out = {}
        for key, val in re.findall(r'\[(\w+)\]=("(?:[^"\\]|\\.)*"|\S*)', body):
            if val.startswith('"'):
                val = val[1:-1].replace('\\"', '"')
            out[key] = re.sub(r"\$\{(\w+)\}", lambda v: variables[v.group(1)], val)
        return out

    cohort, epochs, warmup, extra = (array(n) for n in ("COHORT", "EPOCHS", "WARMUP", "EXTRA"))
    return {e: {"cohort": cohort[e], "epochs": int(epochs[e]), "warmup": int(warmup[e]),
                "extra": extra.get(e, "").split()} for e in cohort}   # no override holds a space


# ---------------------------------------------------------------------------
# The phase-2 (TUH cohort) experiment catalog

#: Overrides every phase-2 run adds to ``finetune_tuab_age.json`` (scenario D):
#: the M1 recipe (mixup 0.4 on the batch, no C-Mixup) on the npy format.
M1 = ["data.data_format=npy", "mixup.enabled=true", "mixup.alpha=0.4", "mixup.sigma=0.0"]
#: The five-fold, patient-grouped cross-validation the cv* runs add.
CV5 = ["cross_validation.enabled=true", "cross_validation.n_folds=5",
       "cross_validation.split_by=subject", "cross_validation.pool=train_val"]
_T1T2 = ['channel_drop.enabled=true', 'channel_drop.channels=["T1","T2"]',
         "channel_drop.min_drop=1", "channel_drop.max_drop=2"]
_FAST = ["model.fused_attention=true", "data.num_workers=6", "trainer.batch_size=128",
         "optimizer.lr=0.0002", "data.train_minutes_per_epoch=5"]
_COND = ["data.session_features=conditions_v1.npz", "model.session_features=head"]


@dataclass(frozen=True)
class Experiment:
    """One phase-2 run: ``name`` as in ``tuh_cohort_eval_v1.csv``, the cohort
    it trained on, its schedule and the overrides beyond :data:`M1`;
    ``parent`` is the run it is compared with in Section 5 and ``change`` what
    differs from the parent; ``script`` is its key in ``submit_tuh_cohorts.sh``
    (empty for a run started locally)."""
    name: str
    cohort: str
    epochs: int
    warmup: int
    extra: Tuple[str, ...] = ()
    parent: str = ""
    change: str = ""
    script: str = ""
    conditions: bool = False          # status / pathology given to the head as input
    where: str = "SageMaker"


PHASE2: Tuple[Experiment, ...] = (
    Experiment("Control", "tuab_v3.0.1_interp50-60", 4, 1, (), "", "M1 on TUAB only (the TUAB baseline of phase 2)",
               where="local"),
    Experiment("Exp1", "normal_both_v1", 4, 1, (), "Control",
               "train on normal recordings only: TUAB normal + TUEG normal", where="local"),
    Experiment("Exp2", "normal_tueg_v1", 4, 1, (), "Exp1", "TUEG normal only (no TUAB in training)",
               script="exp2"),
    Experiment("Exp1b", "normal_both_v1", 10, 3, (), "Exp1", "10 epochs / 3 warmup instead of 4 / 1",
               where="local"),
    Experiment("Exp1t", "normal_both_v1", 4, 1,
               ("data.session_features=session_features_top10_v1.npz", "model.session_features=tokens",
                "model.session_feature_norm=post"), "Exp1",
               "classical EEG features as extra transformer tokens", script="exp1t"),
    Experiment("Exp1f", "normal_both_v1_meta", 4, 1,
               ("data.session_features=session_features_v1.npz", "model.session_features=only",
                "model.head_type=mlp", "model.mlp_head.hidden_dims=[256]", "model.mlp_head.drop=0.3",
                "model.session_feature_drop=0.2"), "Exp1",
               "classical EEG features only, MLP head, no EEG encoder", script="exp1f"),
    Experiment("Exp3", "all_both_v1", 4, 1, (), "Control",
               "add every TUEG recording (normal, abnormal, unlabelled): ~10x training data", script="exp3"),
    Experiment("Exp3b", "all_both_v1", 10, 3, (), "Exp3", "10 epochs / 3 warmup instead of 4 / 1", script="exp3b"),
    Experiment("Exp3q", "all_both_v1", 4, 1, ("data.window_max_abs_uv=800", "data.window_min_std_uv=0.5"),
               "Exp3", "drop windows beyond ±800 µV or with a flat channel", script="exp3q"),
    Experiment("Exp3c", "all_both_v1", 4, 1, ("model.input_clip_uv=800.0",), "Exp3",
               "clip the input to ±800 µV inside the model", where="local"),
    Experiment("Exp3v2", "all_both_v2", 4, 1, (), "Exp3",
               "cohort v2: one TUEG patient split shared by every cohort (data bookkeeping only)",
               where="local"),
    Experiment("Exp3cond", "all_both_v2", 4, 1, tuple(_COND), "Exp3v2",
               "status + pathology vectors concatenated to the pooled feature", script="exp3cond",
               conditions=True),
    Experiment("Exp3clscond", "all_both_v2", 4, 1, tuple(_COND) + ("model.use_mean_pooling=false",), "Exp3cond",
               "head on the class token instead of the mean patch token", script="exp3clscond",
               conditions=True),
    Experiment("Exp4drop", "all_both_v3", 4, 1, ('data.drop_channels=["T1","T2"]',) + tuple(_COND), "Exp3cond",
               "cohort v3 (+ recordings missing only T1/T2) on 21 channels (T1/T2 dropped everywhere)",
               conditions=True, where="local"),
    Experiment("Exp4base", "all_both_v3", 6, 2,
               tuple(_T1T2) + ("channel_drop.prob=0.0", "channel_drop.consistency_weight=0.0") + tuple(_FAST),
               "Exp3v2", "cohort v3, 23 channels with missing T1/T2 left out per sample; speed settings; 6 / 2 epochs",
               script="exp4base"),
    Experiment("Exp4basec", "all_both_v3_compact", 6, 2,
               tuple(_T1T2) + ("channel_drop.prob=0.0", "channel_drop.consistency_weight=0.0") + tuple(_FAST)
               + ("data.split_json=s3://eeg-data-public/TUH_EEG/cohorts/splits/exp4base_data_split.json",),
               "Exp4base", "compact copy of the cohort, same recorded split (reproduction check)",
               script="exp4basec"),
    Experiment("Exp4ctrl", "all_both_v3", 10, 3,
               tuple(_COND) + ("model.use_mean_pooling=false",) + tuple(_T1T2)
               + ("channel_drop.prob=0.0", "channel_drop.consistency_weight=0.0") + tuple(_FAST)
               + ("trainer.early_stopping=true",),
               "Exp4base", "class token + condition vectors, 10 / 3 epochs with early stopping",
               script="exp4ctrl", conditions=True),
    Experiment("Exp4aug", "all_both_v3", 6, 2,
               tuple(_COND) + ("model.use_mean_pooling=false",) + tuple(_T1T2)
               + ("channel_drop.prob=0.2", "channel_drop.consistency_weight=0.1") + tuple(_FAST),
               "Exp4ctrl", "T1/T2 batch dropout (p 0.2) + consistency loss (λ 0.1); 6 / 2 epochs",
               conditions=True, where="local"),
    Experiment("Exp4aug15", "all_both_v3", 15, 5,
               tuple(_COND) + ("model.use_mean_pooling=false",) + tuple(_T1T2)
               + ("channel_drop.prob=0.2", "channel_drop.consistency_weight=0.1") + tuple(_FAST)
               + ("trainer.early_stopping=true",),
               "Exp4aug", "15 / 5 epochs with early stopping (stopped at 11)", conditions=True, where="local"),
    Experiment("cvbase", "all_both_v3_cvcompact", 5, 1,
               tuple(_T1T2) + ("channel_drop.prob=0.0", "channel_drop.consistency_weight=0.0") + tuple(_FAST)
               + tuple(CV5), "Exp4base", "5-fold CV of Exp4base (5 / 1 epochs)", script="cvbase"),
    Experiment("cvctrl", "all_both_v3_cvcompact", 10, 3,
               tuple(_COND) + ("model.use_mean_pooling=false",) + tuple(_T1T2)
               + ("channel_drop.prob=0.0", "channel_drop.consistency_weight=0.0") + tuple(_FAST) + tuple(CV5),
               "cvbase", "class token + condition vectors, 10 / 3 epochs (CV of Exp4ctrl)",
               script="cvctrl", conditions=True),
    Experiment("cvaug", "all_both_v3_cvcompact", 10, 3,
               tuple(_COND) + ("model.use_mean_pooling=false",) + tuple(_T1T2)
               + ("channel_drop.prob=0.2", "channel_drop.consistency_weight=0.1") + tuple(_FAST) + tuple(CV5),
               "cvctrl", "T1/T2 batch dropout + consistency loss λ 0.1 (CV of Exp4aug)",
               script="cvaug", conditions=True),
)
EXPERIMENTS: Dict[str, Experiment] = {e.name: e for e in PHASE2}


def experiment_overrides(exp: Experiment, data_root: str = "") -> List[str]:
    """The ``--set`` overrides that turn ``finetune_tuab_age.json`` into the
    run: :data:`M1`, the schedule, the cohort path and the run's own extras.
    Logging, output and SageMaker settings are left out."""
    path = f"{data_root.rstrip('/')}/{exp.cohort}" if data_root else exp.cohort
    return (M1 + [f"data.data_path={path}", f"trainer.epochs={exp.epochs}",
                  f"optimizer.warmup_epochs={exp.warmup}"] + list(exp.extra))


def resolve_config(base_json: str, overrides: Sequence[str] = ()):
    """``FinetuneRunConfig`` of ``base_json`` with ``overrides`` applied, the
    way ``python -m labram.runs.run_finetune --config ... --set ...`` builds it."""
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.configs.utils_conf import parse_overrides
    return FinetuneRunConfig.load_config(base_json, **parse_overrides(list(overrides)))


# ---------------------------------------------------------------------------
# Config diffs

#: Field prefixes that do not change what is learned (bookkeeping, logging,
#: infrastructure, paths), left out of :func:`config_diff`.
NON_SCIENTIFIC = ("output.", "clearml.", "sagemaker.", "logging.", "shutdown.", "distributed.",
                  "data.num_workers", "data.pin_mem", "data.stage_", "data.data_path", "data.split_json",
                  "evaluation.log_", "evaluation.detailed_metrics", "trainer.debug", "finetune_checkpoint.model_",
                  "cross_validation.fold", "cross_validation.base_dir", "cross_validation.split_json")
#: Fields a regression fine-tune never reads (classification label smoothing;
#: the VQNSP loss weights, used only with ``model.codebook_reg``).
INERT_FOR_REGRESSION = ("loss.classification_label_smoothing", "loss.amplitude_weight", "loss.phase_weight",
                        "loss.embedding_weight", "loss.vq_commitment_beta", "loss.phase_loss",
                        "optimizer.smoothing")


def flatten(d: dict, prefix: str = "") -> Dict[str, object]:
    """Dotted path -> leaf value of a nested config dict (``py/object`` tags
    dropped; lists kept as values)."""
    out: Dict[str, object] = {}
    for k, v in d.items():
        if k == "py/object":
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _disabled_section(flat: Dict[str, object], key: str) -> bool:
    """True when ``key`` sits in a section switched off in ``flat`` (e.g.
    ``mixup.alpha`` with ``mixup.enabled=false``), whose other fields have no effect."""
    parts = key.split(".")
    for i in range(1, len(parts)):
        sw = ".".join(parts[:i]) + ".enabled"
        if sw != key and flat.get(sw) is False:
            return True
    return False


def config_diff(a, b, names: Tuple[str, str] = ("a", "b"),
                ignore: Iterable[str] = NON_SCIENTIFIC) -> pd.DataFrame:
    """Every field on which configs ``a`` and ``b`` (``ConfigBase`` or dicts)
    differ, except the ``ignore`` prefixes and fields of a section that is off
    in both. Columns ``field``, ``names[0]``, ``names[1]``."""
    fa = flatten(a.as_dict() if hasattr(a, "as_dict") else a)
    fb = flatten(b.as_dict() if hasattr(b, "as_dict") else b)
    ignore = tuple(ignore)
    rows = []
    for k in sorted(set(fa) | set(fb)):
        if k.startswith(ignore):
            continue
        va, vb = fa.get(k, "–"), fb.get(k, "–")
        if va == vb:
            continue
        if _disabled_section(fa, k) and _disabled_section(fb, k):
            continue
        rows.append({"field": k, names[0]: _show(va), names[1]: _show(vb)})
    return pd.DataFrame(rows, columns=["field", *names])


def _show(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return json.dumps(list(v))
    return str(v)


# ---------------------------------------------------------------------------
# Noise floor and significance


def seed_noise(ab_json: str, task: str = "age",
               metrics: Sequence[str] = ("val_mae", "test_mae", "test_normal_mae", "test_abnormal_mae")
               ) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Seed-to-seed spread of identical configs from the line-removal A/B
    (``docs/data/line_removal_ab.json``: 3 recipes x 3 seeds). Each run is read
    at its best-validation epoch; the pooled within-recipe sample standard
    deviation of each metric is returned with the per-run frame."""
    with open(ab_json) as f:
        runs = [r for r in json.load(f)["runs"] if r["task"] == task]
    rows = []
    for r in runs:
        e = min(r["epochs"], key=lambda x: x["val_mae"])
        rows.append({"variant": r["variant"], "seed": r["seed"], "epoch": e["epoch"],
                     **{m: e[m] for m in metrics}})
    d = pd.DataFrame(rows)
    g = d.groupby("variant")
    dof = len(d) - g.ngroups
    sd = {m: float(math.sqrt(((g[m].var(ddof=1)) * (g.size() - 1)).sum() / dof)) for m in metrics}
    return d, sd


def pair_threshold(sigma: float, z: float = 1.96) -> float:
    """Smallest difference between two single runs that is unlikely (two-sided
    ``z``) under seed noise alone: ``z * sqrt(2) * sigma``."""
    return z * math.sqrt(2.0) * sigma


def verdict(delta: float, threshold: float, lo: Optional[float] = None,
            hi: Optional[float] = None) -> str:
    """``improves`` / ``hurts`` / ``n.s.`` for a MAE difference ``delta``
    (variant - baseline, years; negative is better). With a confidence
    interval ``[lo, hi]`` the interval decides (it must exclude 0); otherwise
    ``|delta|`` must exceed ``threshold``."""
    if delta is None or (isinstance(delta, float) and math.isnan(delta)):
        return "–"
    if lo is not None and hi is not None:
        if hi < 0:
            return "improves"
        if lo > 0:
            return "hurts"
        return "n.s."
    if delta <= -threshold:
        return "improves"
    if delta >= threshold:
        return "hurts"
    return "n.s."


def cohort_summary(cohort: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    """Per split of a cohort (``cohort.parquet`` + ``manifest.json``):
    recordings, patients, recordings from TUAB, labelled normal / abnormal /
    without a label, and the age mean and sample sd (years)."""
    labels = manifest.get("cohort", {}).get("labels", {})
    rows = []
    for split in ("train", "val", "test"):
        s = cohort[cohort.split == split]
        if len(s) == 0:
            continue
        lab = labels.get(split, {})
        rows.append({"split": split, "recordings": len(s), "patients": s.subject.nunique(),
                     "from TUAB": int(s.corpora.str.contains("tuab").sum()),
                     "normal": lab.get("normal", 0), "abnormal": lab.get("abnormal", 0),
                     "no label": lab.get("null", 0),
                     "age mean": float(s.age.mean()), "age sd": float(s.age.std(ddof=1))})
    return pd.DataFrame(rows)


def tuab_split_summary(split_json: str, metadata_json: str) -> pd.DataFrame:
    """:func:`cohort_summary` for a TUAB-only dataset (``age_split.json`` +
    ``age_metadata.json``): window files are mapped back to their recording."""
    with open(split_json) as f:
        files = json.load(f)["files"]
    with open(metadata_json) as f:
        meta = json.load(f)["recordings"]
    rows = []
    for split, names in files.items():
        stems = sorted({re.sub(r"_\d+\.pkl$", "", n.split("/")[-1]) for n in names})
        m = pd.DataFrame([meta[s] for s in stems if s in meta])
        rows.append({"split": split, "recordings": len(m), "patients": m.subject.nunique(),
                     "from TUAB": len(m), "normal": int((m.label == "normal").sum()),
                     "abnormal": int((m.label == "abnormal").sum()), "no label": 0,
                     "age mean": float(m.age.mean()), "age sd": float(m.age.std(ddof=1))})
    return pd.DataFrame(rows)
