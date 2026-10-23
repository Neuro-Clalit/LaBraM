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
* :func:`load_clearml_runs` reads ``docs/data/brain_age_clearml_runs.csv``
  (run -> ClearML experiment id and name; regenerated from ClearML by
  :func:`fetch_clearml_runs`) and :func:`clearml_link` renders a run as a
  clickable link labelled with its ClearML experiment name.
* :func:`seed_noise` and :func:`verdict` implement the noise floor and the
  significance rule of the notebook's Section 5.
"""
from __future__ import annotations

import html
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


def parse_markdown_table(text: str, heading: str, link_columns: Sequence[str] = ()) -> pd.DataFrame:
    """The first pipe table after the line that starts with ``heading`` (e.g.
    ``"### Table 12.1."``), as strings with links, backticks and math markup
    stripped. Each column named in ``link_columns`` gets a companion column
    ``"<name> url"`` with the cell's link target ("" when it has none)."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(heading))
    raw: List[List[str]] = []
    for ln in lines[start + 1:]:
        if ln.startswith("|"):
            raw.append(ln.strip().strip("|").split("|"))
        elif raw:
            break
    rows = [[_md_cell(c) for c in r] for r in raw]
    header = rows[0]
    keep = [i for i, r in enumerate(rows) if i > 0 and not set("".join(r)) <= set("-: ")]
    out = pd.DataFrame([rows[i] for i in keep], columns=header)
    for name in link_columns:
        j = header.index(name)
        out[f"{name} url"] = [(re.search(r"\]\(([^)]*)\)", raw[i][j]) or [None, ""])[1] for i in keep]
    return out


def registry_master_table(path: str) -> pd.DataFrame:
    """Table 12.1 of the experiment registry (``clearml``: the short task id,
    ``clearml_url``: its link) with numeric columns: ``val``,
    ``test``, ``test_normal``, ``test_abnormal``, ``beta``, ``best_epoch``,
    ``epochs`` (years / epochs)."""
    with open(path) as f:
        t = parse_markdown_table(f.read(), "### Table 12.1.", link_columns=("ClearML",))
    change = next(c for c in t.columns if c.startswith("Change"))
    out = pd.DataFrame({
        "run": t["Run"], "family": t["Family"], "change": t[change], "clearml": t["ClearML"],
        "clearml_url": t["ClearML url"],
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

# ---------------------------------------------------------------------------
# Training loss of every run

#: Loss labels; the numbers are the notebook's Section-2 equations.
LOSSES = {
    "huber": "Huber δ 1 (2.3)",
    "huber_mixup": "Huber δ 1 + mixup (2.3, 2.5)",
    "huber_cmixup": "Huber δ 1 + C-Mixup σ 5 y (2.3, 2.5, 2.6)",
    "huber2": "Huber δ 2 (2.3)",
    "l1": "L1 (2.4)",
    "soft_label": "soft-label KL, σ 2.5 y (2.7)",
    "balance_sqrt": "Huber δ 1, age weights ρ^-1/2 (2.8)",
    "balance_inv": "Huber δ 1, age weights ρ^-1 (2.8)",
    "codebook_amp": "Huber δ 1 + amplitude reconstruction, w 1 (2.9)",
    "codebook_quant": "Huber δ 1 + quantization, w 1 (2.9)",
    "codebook_quant50": "Huber δ 1 + quantization, w 50 (2.9)",
    "multitask": "Huber δ 1 + mixup + 0.5 BCE abnormal (2.3, 2.5, 2.10)",
    "consistency": "Huber δ 1 + mixup + 0.1 consistency (2.3, 2.5, 2.11)",
}
#: Loss of each phase-1 run, read from its ClearML configuration
#: (``loss.regression_loss`` / ``huber_delta`` / ``balance``,
#: ``mixup.enabled`` / ``sigma``, the ``model.codebook_reg`` weights and
#: ``loss.abnormal_weight``); keys of :data:`LOSSES`.
PHASE1_LOSS: Dict[str, str] = {
    **dict.fromkeys(["A", "B", "D", "D-npy", "D-cohorts", "E2", "C2", "DX3", "F1", "F2", "F3", "F4", "G1",
                     "H1", "H2", "C2-head", "C2-last", "M3", "M2", "C", "D-clip1", "E1", "E1c", "E2c",
                     "D-cls", "D-clsmean", "S1"], "huber"),
    **dict.fromkeys(["L1", "M1", "M12", "N2L", "S2", "S2-seed1", "S3"], "huber_mixup"),
    "M1c": "huber_cmixup", "DX2": "huber2", "D2": "l1", "M4": "soft_label",
    "B1": "balance_sqrt", "B2": "balance_inv",
    "DX1": "codebook_amp", "DX4": "codebook_quant", "DX4b": "codebook_quant50",
    "MT1": "multitask", "MT1-seed1": "multitask",
}


def run_loss(run: str) -> str:
    """The training loss of a run as a :data:`LOSSES` label: phase 1 from
    :data:`PHASE1_LOSS`, phase 2 / CV from its overrides (every one is M1,
    Huber + mixup; a channel-dropout consistency weight adds Eq. (2.11))."""
    if run in PHASE1_LOSS:
        return LOSSES[PHASE1_LOSS[run]]
    extra = EXPERIMENTS[run].extra
    weight = [o.split("=", 1)[1] for o in extra if o.startswith("channel_drop.consistency_weight=")]
    return LOSSES["consistency" if weight and float(weight[0]) > 0 else "huber_mixup"]


# ---------------------------------------------------------------------------
# ClearML experiments

CLEARML_WEB = "https://app.clear.ml"
#: ClearML project ``LaBraM/brain_age``, which holds every single-split run.
CLEARML_PROJECT = "c44d37fe41b1483481d5495d42b4ffdc"
#: ClearML task id of each phase-2 run, and for a cross-validation run the id
#: of the sub-project ``LaBraM/brain_age/<study>`` holding its fold tasks.
PHASE2_CLEARML: Dict[str, str] = {
    "Control": "0e7e86c90b9741189fec7608a7cfd427",
    "Exp1": "9dc76b4d4b794e8980aa92668d993b77",
    "Exp2": "e41add850a7342d6a1d83c1e22db8343",
    "Exp1b": "908bad3a40aa4a84b02a69a6a70f40b8",
    "Exp1t": "be184e5a20364008a3e540f87d592207",
    "Exp1f": "2071d005ecdf4b53808dea946376aefb",
    "Exp3": "86f1c7488c3e4edd9b8891deb6b9a515",
    "Exp3b": "746e9d9c7fe548999e9b974a1654f861",
    "Exp3q": "ae431763b552447ba178c2d392246fea",
    "Exp3c": "bf41ca7ca0304c03844c55d73e063674",
    "Exp3v2": "ab96236c80234c6c9c92851df33ef3ee",
    "Exp3cond": "7642dde596ed42a6a32a664482260aa4",
    "Exp3clscond": "2fba9690f0dd46909b061ca2847eb422",
    "Exp4drop": "441f3b9920254cefa46075c7372b9033",
    "Exp4base": "33a926503fb347248dbc56252c53e609",
    "Exp4basec": "a28aa90b1eb8419d8db7b9b5eb3f2e75",
    "Exp4ctrl": "bb15c54792f24626b81ac3eac8b97661",
    "Exp4aug": "d15337ba1fe8412a83c386d7ddf69bb3",
    "Exp4aug15": "692b86d6f2ca4f4898488988722c7bc9",
    "cvbase": "d0379910d1e748d1ba2250b0e2157103",
    "cvctrl": "efe8b8525ed248ae968f24d6840e76bc",
    "cvaug": "d538fd4c7b1c484fa2f94438aecdc418",
}
_CV_RUNS = ("cvbase", "cvctrl", "cvaug")


def clearml_url(kind: str, ident: str, project_id: str = CLEARML_PROJECT) -> str:
    """Web-UI link of a ClearML task (``kind="task"``: its console log) or of a
    project's experiment list (``kind="project"``)."""
    if kind == "project":
        return f"{CLEARML_WEB}/projects/{ident}/experiments"
    return f"{CLEARML_WEB}/projects/{project_id}/experiments/{ident}/output/log"


def clearml_link(text: str, url: str) -> str:
    """An HTML anchor showing ``text`` (the ClearML experiment name) instead of
    the URL, opening in a new tab; ``"–"`` without a URL. The name may wrap
    after each ``_`` (``<wbr>``), so a long name does not widen its table
    past the page in the PDF export."""
    if not url:
        return "–"
    label = html.escape(text).replace("_", "_<wbr>")
    return f'<a href="{html.escape(url, quote=True)}" target="_blank">{label}</a>'


def clearml_targets(registry: pd.DataFrame) -> pd.DataFrame:
    """Every run of the study with its ClearML ``kind`` (task / project) and
    ``id``: the phase-1 runs from the registry links, the phase-2 and CV runs
    from :data:`PHASE2_CLEARML`."""
    rows = []
    for r in registry.itertuples():
        m = re.search(r"/experiments/([0-9a-f]{32})", r.clearml_url or "")
        if m:
            rows.append({"run": r.run, "phase": "1", "kind": "task", "id": m.group(1)})
    for name, ident in PHASE2_CLEARML.items():
        cv = name in _CV_RUNS
        rows.append({"run": name, "phase": "3" if cv else "2", "kind": "project" if cv else "task", "id": ident})
    return pd.DataFrame(rows)


def fetch_clearml_runs(registry: pd.DataFrame) -> pd.DataFrame:
    """:func:`clearml_targets` with each experiment's ClearML ``name`` (the
    task name, or the project's leaf name), queried from the ClearML server
    (needs the ``clearml`` package and credentials). Writes the rows that
    ``docs/data/brain_age_clearml_runs.csv`` holds."""
    from clearml.backend_api.session.client import APIClient
    client = APIClient()
    t = clearml_targets(registry)
    tasks = t[t.kind == "task"].id.tolist()
    names = {x.id: x.name for x in client.tasks.get_all(id=tasks, only_fields=["id", "name"])}
    projects = t[t.kind == "project"].id.tolist()
    names.update({x.id: x.name.rsplit("/", 1)[-1]
                  for x in client.projects.get_all(id=projects, only_fields=["id", "name"])})
    t["name"] = t.id.map(names)
    missing = t[t.name.isna()].run.tolist()
    if missing:
        raise KeyError(f"not found in ClearML: {missing}")
    return t


def load_clearml_runs(path: str) -> pd.DataFrame:
    """``docs/data/brain_age_clearml_runs.csv`` indexed by run, with a ``url``
    column (:func:`clearml_url`) and a ``link`` column (:func:`clearml_link`,
    the experiment name as a clickable link)."""
    t = pd.read_csv(path, dtype=str).set_index("run")
    t["url"] = [clearml_url(k, i) for k, i in zip(t.kind, t.id)]
    t["link"] = [clearml_link(n, u) for n, u in zip(t.name, t.url)]
    return t


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


# ---------------------------------------------------------------------------
# Data splits: which recordings each run trained, validated and tested on

SPLITS = ("train", "val", "test")


def split_fingerprint(data_split: dict) -> List[Dict[str, object]]:
    """One row per split of a run's recorded ``data_split.json`` (the
    ``data_split`` ClearML artifact): recordings, patients, windows and
    ``digest``, the first 12 hex digits of the SHA-1 of the sorted recording
    stems. Two runs used the same recordings in a split iff the digests match;
    the window count also reflects the window length and evaluation budget."""
    import hashlib
    rows = []
    for split in SPLITS:
        if split not in data_split:
            continue
        d = data_split[split]
        recs = sorted(d["recordings"])
        rows.append({"split": split, "recordings": len(recs), "patients": int(d["n_subjects"]),
                     "windows": int(d["n_windows"]),
                     "digest": hashlib.sha1("\n".join(recs).encode()).hexdigest()[:12]})
    return rows


def fetch_run_splits(runs: pd.DataFrame) -> pd.DataFrame:
    """:func:`split_fingerprint` of every ClearML task in ``runs`` (rows of
    :func:`load_clearml_runs` with ``kind == "task"``), downloading each task's
    ``data_split`` artifact. Writes the rows that
    ``docs/data/brain_age_run_splits.csv`` holds."""
    from clearml import Task
    rows = []
    for run, r in runs[runs.kind == "task"].iterrows():
        art = Task.get_task(task_id=r.id).artifacts.get("data_split")
        path = art.get_local_copy() if art is not None else None
        if path is None:
            raise KeyError(f"{run}: no data_split artifact")
        with open(path) as f:
            rows += [{"run": run, **row} for row in split_fingerprint(json.load(f))]
    return pd.DataFrame(rows)


def split_signature(splits: pd.DataFrame) -> pd.Series:
    """Per run, the concatenated train/val/test recording digests: equal
    signatures mean the same recordings in every split."""
    d = splits.pivot(index="run", columns="split", values="digest").reindex(columns=list(SPLITS))
    return d.fillna("-").agg("/".join, axis=1)


# ---------------------------------------------------------------------------
# TUEG, the source corpus

#: The other TUH corpora that are subsets of TUEG, with the label each adds.
TUEG_SUBSETS = {
    "tuab": "normal / abnormal (TUAB folder)",
    "tusz": "seizure events",
    "tuep": "epilepsy / no epilepsy (patient)",
    "tuar": "artifact events",
    "tusl": "slowing events",
}
STATUS = ("normal", "abnormal", "no label")


def tueg_recordings(index_parquet: str, sheet_xlsx: str) -> pd.DataFrame:
    """The TUEG recordings of the TUH recording index (one row per EDF, from
    ``dataset_maker/tuh_index.py``) joined with NEDC's clinical sheet
    (``dataset_maker/tuh_clinical.py``), plus ``status``: the TUAB folder label
    where there is one, else the report finding, else ``"no label"`` (the rule
    of ``dataset_maker/make_cohort_dataset.py::label_of``)."""
    from dataset_maker import tuh_clinical as tc
    j = tc.join_index(pd.read_parquet(index_parquet), tc.load_sheet(sheet_xlsx))
    t = j[j.corpora.str.split(",").map(lambda cs: "tueg" in cs)].reset_index(drop=True)
    report = t.abnormal_report.astype("boolean")
    status = pd.Series("no label", index=t.index, dtype=object)
    status[t.in_sheet & report.eq(False).fillna(False)] = "normal"
    status[t.in_sheet & report.eq(True).fillna(False)] = "abnormal"
    tuab = t.abnormal.notna()
    status[tuab] = np.where(t.abnormal[tuab].astype(int) == 1, "abnormal", "normal")
    t["status"] = status
    t["status_source"] = np.where(tuab, "TUAB folder", np.where(status != "no label", "NEDC report", "–"))
    return t


def _share(n: int, total: int) -> str:
    return f"{n:,} ({100.0 * n / total:.1f} %)"


def tueg_overview(t: pd.DataFrame) -> pd.DataFrame:
    """TUEG at a glance: size, hours, demographics and recording properties
    (one row per quantity, values as display strings)."""
    n = len(t)
    minutes = t.duration_sec / 60
    year = t.year[t.year.between(1990, 2025)]                 # a few headers carry 1899 / 2064
    missing = t.missing_channels.map(lambda m: {c.upper() for c in m})
    rows = [
        ("size", "recordings (EDF files)", f"{n:,}"),
        ("size", "patients", f"{t.subject.nunique():,}"),
        ("size", "sessions", f"{t[['subject', 'session']].drop_duplicates().shape[0]:,}"),
        ("size", "hours of EEG", f"{t.duration_sec.sum() / 3600:,.0f}"),
        ("size", "recordings per patient, median (max)",
         f"{t.groupby('subject').size().median():.0f} ({t.groupby('subject').size().max()})"),
        ("recording", "length (min), median [IQR]",
         f"{minutes.median():.1f} [{minutes.quantile(.25):.1f}, {minutes.quantile(.75):.1f}]"),
        ("recording", "year, median [5 %, 95 %]",
         f"{year.median():.0f} [{year.quantile(.05):.0f}, {year.quantile(.95):.0f}]"),
        ("recording", "sampling rate (Hz)",
         ", ".join(f"{int(k)}: {v / n:.0%}" for k, v in t.sfreq.value_counts().head(3).items())),
        ("recording", "montage", ", ".join(f"{k}: {v / n:.0%}" for k, v in t.montage.value_counts().head(3).items())),
        ("channels", "all 23 channels", _share(int(missing.map(len).eq(0).sum()), n)),
        ("channels", "missing only T1/T2", _share(int(missing.map(lambda m: bool(m) and m <= {"T1", "T2"}).sum()), n)),
        ("channels", "missing other channels",
         _share(int(missing.map(lambda m: bool(m - {"T1", "T2"})).sum()), n)),
        ("patient", "sex F / M / unknown (recordings)",
         " / ".join(f"{int((t.sex == s).sum()):,}" for s in ("F", "M")) + f" / {int(t.sex.isna().sum()):,}"),
        ("patient", "age known (1–89 y)", _share(int(t.age.notna().sum()), n)),
        ("patient", "age 90+ (redacted as 999)", _share(int((t.raw_age == 999).sum()), n)),
        ("patient", "age missing", _share(int((t.age.isna() & (t.raw_age != 999)).sum()), n)),
        ("patient", "age (y), mean ± sd; median [IQR]",
         f"{t.age.mean():.1f} ± {t.age.std(ddof=1):.1f}; {t.age.median():.0f} "
         f"[{t.age.quantile(.25):.0f}, {t.age.quantile(.75):.0f}]"),
    ]
    return pd.DataFrame(rows, columns=["group", "quantity", "value"])


def tueg_subsets(t: pd.DataFrame) -> pd.DataFrame:
    """Subsets of TUEG: the other TUH corpora inside it (a recording can be in
    several) and the subsets the brain-age cohorts select from."""
    from dataset_maker import tuh_clinical as tc
    n = len(t)
    corpora = t.corpora.str.split(",")
    masks = [(f"in {c.upper()}", lab, corpora.map(lambda cs, c=c: c in cs)) for c, lab in TUEG_SUBSETS.items()]
    masks += [
        ("TUEG only", "–", corpora.map(len).eq(1)),
        ("in the NEDC clinical sheet", "report finding, 15 diagnosis flags, medications", t.in_sheet),
        ("adult age (18–89 y)", "age", t.age.between(18, 89)),
        ("normative", "report-normal, no neurological diagnosis, not sedated, 18–89 y",
         tc.normative_mask(t).fillna(False).astype(bool)),
    ]
    rows = [{"subset": name, "label it carries": lab, "recordings": int(m.sum()),
             "% of TUEG": 100.0 * m.sum() / n, "patients": int(t.subject[m].nunique()),
             "hours": t.duration_sec[m].sum() / 3600} for name, lab, m in masks]
    return pd.DataFrame(rows)


def age_stats(age: pd.Series) -> Dict[str, float]:
    """n, mean, sample sd, median and quartiles of the known ages (years)."""
    a = age.dropna()
    return {"age n": len(a), "age mean": a.mean(), "age sd": a.std(ddof=1), "age median": a.median(),
            "age Q1": a.quantile(.25), "age Q3": a.quantile(.75)}


def tueg_status(t: pd.DataFrame) -> pd.DataFrame:
    """Recordings and patients per normal / abnormal / no-label status and per
    label source, with each group's age distribution (:func:`age_stats`); the
    last row is all of TUEG."""
    rows = []
    for status in STATUS:
        s = t[t.status == status]
        for source in sorted(s.status_source.unique()):
            g = s[s.status_source == source]
            rows.append({"status": status, "source": source, "recordings": len(g),
                         "patients": g.subject.nunique(), **age_stats(g.age)})
        if s.status_source.nunique() > 1:
            rows.append({"status": status, "source": "all", "recordings": len(s),
                         "patients": s.subject.nunique(), **age_stats(s.age)})
    rows.append({"status": "total", "source": "all", "recordings": len(t), "patients": t.subject.nunique(),
                 **age_stats(t.age)})
    out = pd.DataFrame(rows)
    out["% of TUEG"] = 100.0 * out.recordings / len(t)
    return out


def tueg_diagnoses(t: pd.DataFrame) -> pd.DataFrame:
    """The clinical sheet's diagnosis flags over the TUEG recordings in the
    sheet: recordings, patients, prevalence (% of sheet recordings), mean age
    and the share whose status is abnormal."""
    from dataset_maker import tuh_clinical as tc
    s = t[t.in_sheet]
    names = {v: k for k, v in tc.DIAGNOSES.items()}
    names.update({"any_neuro_dx": "any neurological diagnosis", "sedated": "sedating medication"})
    rows = []
    for col, name in names.items():
        m = s[col].fillna(False).astype(bool)
        rows.append({"flag": name, "recordings": int(m.sum()), "patients": int(s.subject[m].nunique()),
                     "% of sheet": 100.0 * m.mean(), "age mean": s.age[m].mean(),
                     "% abnormal": 100.0 * (s.status[m] == "abnormal").mean()})
    return pd.DataFrame(rows).sort_values("recordings", ascending=False)


# ---------------------------------------------------------------------------
# Per-recording predictions of the CV runs, by subcohort and status

PREDICTION_SETS = ("TUAB test, ensemble", "TUAB out-of-fold", "TUEG out-of-fold")


def prediction_sets(preds: pd.DataFrame, recordings: pd.DataFrame) -> pd.DataFrame:
    """Per CV experiment, one row per scored recording in each of
    :data:`PREDICTION_SETS`:

    * ``TUAB test, ensemble``: the TUAB test set, predicted by the median of the
      fold models (``labram/eval/cv_compare.py::ensemble``);
    * ``TUAB out-of-fold`` / ``TUEG out-of-fold``: every recording of the CV
      pool, predicted by the one fold model that held it out, split by whether
      the recording is in TUAB.

    ``preds`` is ``cv_compare.long_frame`` of the CV predictions; ``recordings``
    is :func:`tueg_recordings` (``stem``, ``status``, ``corpora``). Columns
    ``exp``, ``set``, ``stem``, ``pred``, ``true``, ``err``, ``status``."""
    from labram.eval import cv_compare as cc
    info = recordings.drop_duplicates("stem").set_index("stem")
    ens = cc.ensemble(preds, "heldout", "median")[["exp", "stem", "pred", "true"]].assign(set=PREDICTION_SETS[0])
    oof = preds[preds.split == "test"][["exp", "stem", "pred", "true"]].copy()
    in_tuab = oof.stem.map(info.corpora).fillna("").str.split(",").map(lambda cs: "tuab" in cs)
    oof["set"] = np.where(in_tuab, PREDICTION_SETS[1], PREDICTION_SETS[2])
    out = pd.concat([ens, oof], ignore_index=True)
    out["err"] = out.pred - out.true
    out["status"] = out.stem.map(info.status).fillna("no label")
    return out


def age_fit_metrics(pred: pd.Series, true: pd.Series) -> Dict[str, float]:
    """n, MAE (Eq. (0.1)), mean error (Eq. (0.3)), R² (Eq. (0.2)) and the bias
    slope β (Eq. (0.5), population moments) of one set of recordings."""
    p, t = np.asarray(pred, float), np.asarray(true, float)
    e = p - t
    var = float(np.var(t))
    return {"n": len(t), "MAE": float(np.mean(np.abs(e))), "bias": float(np.mean(e)),
            "R²": 1.0 - float(np.sum(e ** 2)) / float(np.sum((t - t.mean()) ** 2)) if len(t) > 1 else float("nan"),
            "β": float(np.mean((t - t.mean()) * (e - e.mean()))) / var if var > 0 else float("nan")}


def prediction_metrics(sets: pd.DataFrame) -> pd.DataFrame:
    """:func:`age_fit_metrics` per experiment, prediction set and status
    (``all`` plus each status present)."""
    rows = []
    for (exp, name), g in sets.groupby(["exp", "set"], sort=False):
        for status in ("all", *STATUS):
            s = g if status == "all" else g[g.status == status]
            if len(s):
                rows.append({"exp": exp, "set": name, "status": status, **age_fit_metrics(s.pred, s.true)})
    return pd.DataFrame(rows)
