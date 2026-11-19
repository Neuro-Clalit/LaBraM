# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# MIL Stage 1: bag pooling trained on cached (frozen) window features.
# ---------------------------------------------------------
"""Stage 1 of ``docs/b1_attention_mil_plan.md``: train a bag pooling layer
(``labram.layers.bag_pool``) and the age head on cached window features of a
trained run (``labram.mil.cache``), the encoder frozen.

A recording is a bag of its (up to 30) eval-budget windows; the pool maps the
window features ``h_i`` (and optionally the descriptors ``q_i``) to ``z_c`` and
the head -- initialised from the run's own head -- predicts the z-scored age
with a Huber loss, plus ``entropy_weight * (log n - H(alpha))`` for the
competing (softmax) pools.

Protocols (:func:`run_variant`):

* ``val_cf`` -- fit on val, cross-fitted over 5 folds grouped by subject
  (out-of-fold val predictions), then refit on all of val for test (as A3);
* ``train`` -- fit on the training bags (the first 30 windows of each training
  recording), epoch selected on val, test reported.

Benchmarks: ``M0-run`` (mean of the run's own window predictions, no fitting
-- the reported metric) and ``M0`` (mean pooling of ``h`` with the head refit,
the like-for-like control of every attention variant).
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from labram.layers.bag_pool import DescriptorNorm, attention_entropy_gap, build_bag_pool
from labram.mil.cache import Cache

COMPETING = ("gated", "query")


@dataclass
class Variant:
    name: str
    kind: str = "gated"            # mean | gated | sigmoid | query
    score_input: str = "h"         # h | q | hq
    attn_dim: int = 128
    branches: int = 1
    heads: int = 4
    self_attention: bool = False
    temperature: float = 1.0
    topk_mask_prob: float = 0.0
    dropout: float = 0.1
    time_encoding: bool = False
    entropy_weight: float = 0.01
    lr: float = 1e-3
    head_lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 64
    fit_head: bool = True

    def pool_kwargs(self):
        kw = {"score_input": self.score_input, "dropout": self.dropout,
              "time_encoding": self.time_encoding}
        if self.kind == "gated":
            kw.update(attn_dim=self.attn_dim, branches=self.branches, temperature=self.temperature,
                      topk_mask_prob=self.topk_mask_prob)
        elif self.kind == "sigmoid":
            kw.update(attn_dim=self.attn_dim)
        elif self.kind == "query":
            kw.update(heads=self.heads, self_attention=self.self_attention)
        return kw


# ------------------------------------------------------------------- bags
@dataclass
class Bags:
    """All bags of one split as padded tensors (on ``device``)."""
    h: torch.Tensor          # [R, K, D]
    q: torch.Tensor          # [R, K, Q] raw descriptors (+ minute)
    t: torch.Tensor          # [R, K] minute
    mask: torch.Tensor       # [R, K]
    y: torch.Tensor          # [R] z-scored age
    win_pred: np.ndarray     # [R, K] the run's window predictions (years), nan = pad
    rec: pd.DataFrame        # recording, age, cohort, dataset (one row per bag)
    desc_columns: List[str] = field(default_factory=list)

    def __len__(self):
        return len(self.rec)


def load_bags(cache: Cache, split: str, target_stats, device, max_k: int = 30) -> Bags:
    d = cache.split(split)
    meta = d["meta"]
    codes, stems = pd.factorize(meta["recording"], sort=False)
    slot = meta.groupby(codes).cumcount().to_numpy()
    keep = slot < max_k
    codes, slot = codes[keep], slot[keep]
    r, k = len(stems), int(slot.max()) + 1
    h = np.zeros((r, k, d["h"].shape[1]), np.float32)
    h[codes, slot] = np.asarray(d["h"])[keep]
    desc = np.asarray(d["desc"])[keep]
    nq = desc.shape[1] + 1
    q = np.zeros((r, k, nq), np.float32)
    q[codes, slot, :-1] = desc
    tt = np.zeros((r, k), np.float32)
    tt[codes, slot] = meta["minute"].to_numpy()[keep]
    q[codes, slot, -1] = tt[codes, slot]
    mask = np.zeros((r, k), bool)
    mask[codes, slot] = True
    wp = np.full((r, k), np.nan, np.float32)
    wp[codes, slot] = d["pred"][keep]
    rec = meta[keep].groupby(codes, sort=True)[["recording", "age", "cohort", "dataset"]].first()
    rec = rec.reset_index(drop=True)
    mean, std = target_stats
    y = (rec["age"].to_numpy() - mean) / std
    as_t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
    return Bags(as_t(h), as_t(q), as_t(tt), as_t(mask, torch.bool), as_t(y), wp, rec,
                d["desc_columns"] + ["minute"])


# ------------------------------------------------------------------ model
class BagModel(nn.Module):
    def __init__(self, variant: Variant, dim: int, n_desc: int, head_state: dict, desc_norm: DescriptorNorm):
        super().__init__()
        self.variant = variant
        self.desc_norm = desc_norm
        self.pool = build_bag_pool(variant.kind, dim, n_desc, **variant.pool_kwargs())
        self.head = nn.Linear(dim, 1)
        self.head.load_state_dict(head_state)

    def forward(self, h, mask, q, t):
        z, alpha, extra = self.pool(h, mask, self.desc_norm(q), t)
        return self.head(z).squeeze(-1), alpha, extra


def _batches(idx, batch_size, shuffle, gen):
    idx = np.asarray(idx)
    if shuffle:
        idx = idx[torch.randperm(len(idx), generator=gen).numpy()]
    for s in range(0, len(idx), batch_size):
        yield torch.as_tensor(idx[s:s + batch_size])


@torch.no_grad()
def predict(model: BagModel, bags: Bags, idx=None, batch_size: int = 512):
    model.eval()
    idx = np.arange(len(bags)) if idx is None else np.asarray(idx)
    preds, alphas, gates = [], [], []
    for b in _batches(idx, batch_size, False, None):
        b = b.to(bags.h.device)
        p, a, extra = model(bags.h[b], bags.mask[b], bags.q[b], bags.t[b])
        preds.append(p.cpu())
        alphas.append(a.cpu())
        if "gate" in extra:
            gates.append(extra["gate"].cpu())
    out = {"pred_z": torch.cat(preds).numpy(), "alpha": torch.cat(alphas).numpy()}
    if gates:
        out["gate"] = torch.cat(gates).numpy()
    return out


def train_model(variant: Variant, bags: Bags, idx, head_state, desc_norm, seed=0,
                val: Optional[Bags] = None, target_stats=None, log=None):
    """Fit ``variant`` on bags ``idx``. With ``val``, returns the state of the
    epoch with the lowest val MAE (and the per-epoch val MAEs)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = BagModel(variant, bags.h.shape[-1], bags.q.shape[-1], head_state, desc_norm).to(bags.h.device)
    groups = [{"params": list(model.pool.parameters()), "lr": variant.lr}]
    groups.append({"params": list(model.head.parameters()),
                   "lr": variant.head_lr if variant.fit_head else 0.0})
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=variant.weight_decay)
    steps = max(1, variant.epochs * math.ceil(len(idx) / variant.batch_size))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    best, curve = (math.inf, None, -1), []
    for epoch in range(variant.epochs):
        model.train()
        for b in _batches(idx, variant.batch_size, True, gen):
            b = b.to(bags.h.device)
            p, alpha, _ = model(bags.h[b], bags.mask[b], bags.q[b], bags.t[b])
            loss = F.huber_loss(p, bags.y[b], delta=1.0)
            if variant.kind in COMPETING and variant.entropy_weight > 0:
                loss = loss + variant.entropy_weight * attention_entropy_gap(alpha, bags.mask[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if val is not None:
            mae = float(np.mean(np.abs(predict(model, val)["pred_z"] - val.y.cpu().numpy())) * target_stats[1])
            curve.append(mae)
            if mae < best[0]:
                best = (mae, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
            if log:
                log(f"    epoch {epoch}: val MAE {mae:.3f}")
    if val is not None:
        model.load_state_dict(best[1])
    return model, {"val_curve": curve, "best_epoch": best[2]}


# --------------------------------------------------------------- protocols
def subject_folds(stems, n_folds=5, seed=0):
    from labram.eval.window_confidence import subject_folds as sf
    return sf(list(stems), n_folds, seed)


def run_variant(variant: Variant, data: Dict[str, Bags], head_state, desc_norm, target_stats,
                protocol: str = "val_cf", seed: int = 0, log=print) -> Dict[str, dict]:
    """Predictions (years) and pooling weights per split for one variant."""
    mean, std = target_stats
    to_years = lambda z: z * std + mean   # noqa: E731
    out = {}
    if variant.kind == "mean" and not variant.fit_head:          # M0-run: nothing to fit
        model = BagModel(variant, data["val"].h.shape[-1], data["val"].q.shape[-1], head_state,
                         desc_norm).to(data["val"].h.device)
        for split in ("val", "test"):
            p = predict(model, data[split])
            out[split] = {"pred": to_years(p["pred_z"]), "alpha": p["alpha"], **_extras(p)}
        return out
    if protocol == "val_cf":
        val = data["val"]
        folds = subject_folds(val.rec["recording"], 5, seed)
        pred = np.zeros(len(val))
        alpha = np.zeros(val.mask.shape, np.float32)
        gate = None
        for k in range(5):
            model, _ = train_model(variant, val, np.flatnonzero(folds != k), head_state, desc_norm, seed + k)
            te = np.flatnonzero(folds == k)
            p = predict(model, val, te)
            pred[te], alpha[te] = p["pred_z"], p["alpha"]
            if "gate" in p:
                gate = np.zeros(val.mask.shape, np.float32) if gate is None else gate
                gate[te] = p["gate"]
        out["val"] = {"pred": to_years(pred), "alpha": alpha, **({"gate": gate} if gate is not None else {})}
        model, _ = train_model(variant, val, np.arange(len(val)), head_state, desc_norm, seed + 99)
        p = predict(model, data["test"])
        out["test"] = {"pred": to_years(p["pred_z"]), "alpha": p["alpha"], **_extras(p)}
    elif protocol == "train":
        tr = data["train"]
        model, info = train_model(variant, tr, np.arange(len(tr)), head_state, desc_norm, seed,
                                  val=data["val"], target_stats=target_stats)
        for split in ("val", "test"):
            p = predict(model, data[split])
            out[split] = {"pred": to_years(p["pred_z"]), "alpha": p["alpha"], **_extras(p)}
        out["info"] = info
    else:
        raise ValueError(protocol)
    out["head_after"] = {k: v.detach().cpu() for k, v in model.head.state_dict().items()}
    return out


def _extras(p):
    return {"gate": p["gate"]} if "gate" in p else {}


# ---------------------------------------------------------------- metrics
GROUPS = ["all", "temple", "harvard"] + [f"{d}_{c}" for d in ("temple", "harvard")
                                          for c in ("normal", "abnormal", "unlabelled")]


def group_mask(rec: pd.DataFrame, g: str) -> np.ndarray:
    if g == "all":
        return np.ones(len(rec), bool)
    if "_" not in g:
        return (rec["dataset"] == g).to_numpy()
    d, c = g.split("_")
    return ((rec["dataset"] == d) & (rec["cohort"] == c)).to_numpy()


def metrics_rows(name, split, rec, pred, ref_pred=None, n_boot=1000, seed=0):
    err = np.abs(pred - rec["age"].to_numpy())
    ref = None if ref_pred is None else np.abs(ref_pred - rec["age"].to_numpy())
    rng = np.random.default_rng(seed)
    rows = []
    for g in GROUPS:
        m = group_mask(rec, g)
        if m.sum() < 20:
            continue
        row = {"variant": name, "split": split, "group": g, "n": int(m.sum()), "mae": float(err[m].mean()),
               "rmse": float(np.sqrt(((pred - rec["age"].to_numpy())[m] ** 2).mean())),
               "r": float(np.corrcoef(pred[m], rec["age"].to_numpy()[m])[0, 1])}
        if ref is not None:
            diff = err[m] - ref[m]
            boots = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(1)
            row.update(delta_vs_ref=float(diff.mean()), ci_lo=float(np.percentile(boots, 2.5)),
                       ci_hi=float(np.percentile(boots, 97.5)))
        rows.append(row)
    return rows


def window_frame(bags: Bags, alpha, head_state, target_stats, gate=None) -> pd.DataFrame:
    """One row per real window: the pooling weight, the window's own
    prediction through the (refit) head (years), its |error|, descriptors."""
    w = head_state["weight"].float().numpy()[0]
    b = float(head_state["bias"].float().numpy()[0])
    hz = bags.h.cpu().numpy() @ w + b
    mean, std = target_stats
    mask = bags.mask.cpu().numpy()
    r, k = np.nonzero(mask)
    df = pd.DataFrame({"split": "x", "recording": bags.rec["recording"].to_numpy()[r], "slot": k,
                       "alpha": alpha[r, k], "win_pred": hz[r, k] * std + mean,
                       "age": bags.rec["age"].to_numpy()[r], "cohort": bags.rec["cohort"].to_numpy()[r],
                       "dataset": bags.rec["dataset"].to_numpy()[r]})
    if gate is not None:
        df["gate"] = gate[r, k]
    q = bags.q.cpu().numpy()
    for j, c in enumerate(bags.desc_columns):
        if c in ("emg", "max_abs", "min_std", "stft_ch_sd", "stft_logpow", "cwt_peak_theta", "minute"):
            df[c] = q[r, k, j]
    df["abs_err"] = (df["win_pred"] - df["age"]).abs()
    return df


def attention_analytics(win: pd.DataFrame, rec: pd.DataFrame, pred: np.ndarray) -> dict:
    """Does the pooling weight track the window's own error, and what does it
    follow? Plus concentration and recording-level confidence from the weights."""
    from labram.eval import window_confidence as wc
    out = {}
    w = win.rename(columns={"alpha": "w"}).assign(split="x")
    corr = wc.weight_error_correlation(w, "w", "abs_err")
    out.update({f"win_{k}": v for k, v in corr.items()})
    out["excess_by_decile"] = wc.excess_error_by_weight_rank(w, "w", "abs_err").round(3).to_dict()
    for c in ("emg", "max_abs", "min_std", "stft_ch_sd", "minute", "cwt_peak_theta"):
        if c in w:
            vals = [wc.spearman(g["w"], g[c]) for _, g in w.groupby("recording", sort=False) if len(g) >= 5]
            out[f"rho_within_w_{c}"] = float(np.nanmean(vals))
    g = w.groupby("recording", sort=False)
    n = g["w"].size()
    ess = 1.0 / g["w"].apply(lambda a: float((a ** 2).sum()))
    conc = (1 - ess / n)
    ent = g["w"].apply(lambda a: float(-(a * np.log(np.clip(a, 1e-12, None))).sum()))
    out["mean_entropy_gap"] = float((np.log(n) - ent).mean())
    out["mean_max_w"] = float(g["w"].max().mean())
    out["mean_ess_frac"] = float((ess / n).mean())
    # weighted sum: sum_i w_i * window prediction vs the pooled prediction (equal for a linear head)
    ws = (w["w"] * w["win_pred"]).groupby(w["recording"], sort=False).sum()
    pr = pd.Series(pred, index=rec["recording"].to_numpy())
    out["max_abs_weighted_sum_minus_pred"] = float((ws - pr.reindex(ws.index)).abs().max())
    # recording confidence from the weights
    err = pd.Series(np.abs(pred - rec["age"].to_numpy()), index=rec["recording"].to_numpy())
    wmean = ws
    wsd = np.sqrt(((w["win_pred"] - w["recording"].map(wmean)) ** 2 * w["w"]).groupby(w["recording"], sort=False).sum())
    scores = {"conc": conc, "wsd": wsd}
    if "gate" in w:
        scores["gate_deficit"] = 1 - g["gate"].mean()
    for name, u in scores.items():
        e = err.reindex(u.index)
        out[f"rec_rho_{name}"] = wc.spearman(u.to_numpy(), e.to_numpy())
        rc = wc.risk_coverage(u.to_numpy(), e.to_numpy())
        out[f"rec_{name}_mae@50"] = rc["mae@50"]
        out[f"rec_{name}_aurc"] = rc["aurc"]
    return out


def variant_record(v: Variant) -> dict:
    return asdict(v)
