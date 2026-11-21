#!/usr/bin/env python
"""Pictures of the handcrafted window features (``docs/window_features.md``, Figures 2.1-7.3).

Example windows come from the Dortmund Vital Study (OpenNeuro ds005385, CC0), the only cohort we may
show recording by recording in a public repository: two people (a young and an old adult), each with
one 10 s window of the eyes-closed and one of the eyes-open run before the task block of session 1.
The features are computed by the model's own module (``labram/layers/window_features.py``) with the
reference matrices, standardisation and projection of the trained checkpoint. The age figures use one
value per person: the median over the clean windows of the person's eyes-closed run.

    python scripts/window_feature_figures.py --run /data/runs/heedb/mil_features/P4F_L1/cv/fold_0 \\
        --cohort /data/datasets/EEG-public/DORTMUND/cohorts/dortmund_v1 --out docs/figures/window_features
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from labram.eval.age_plots import assert_max_columns, panel_labels  # noqa: E402
from labram.features.tuab import clean_mask, window_quality  # noqa: E402
from labram.layers.window_features import (BANDS, FILTERBANK, N_CHANNELS, WindowFeatureBranch,  # noqa: E402
                                           _sym_fn)

DROP = ("A1", "A2", "T1", "T2")
POSTERIOR = ("O1", "O2", "P3", "P4", "PZ", "T5", "T6")
CORE_NAMES = ([f"log abs {b}" for b in BANDS] + [f"rel {b}" for b in BANDS]
              + ["log theta/alpha", "log slowing ratio", "SEF95", "spectral entropy", "peak alpha freq",
                 "1/f exponent", "1/f offset"] + [f"bin {e} Hz" for e in range(1, 45)]
              + ["log activity", "log line length", "skewness", "kurtosis", "Hjorth mobility",
                 "Hjorth complexity", "zero-crossing rate"])
FAMILIES = [("spectral", "#2e86c1"), ("1 Hz bins", "#85c1e9"), ("time domain", "#27ae60"),
            ("covariance (tangent)", "#e67e22"), ("connectivity", "#8e44ad")]
WINDOW_COLORS = {"A, eyes closed": "#1f618d", "A, eyes open": "#85c1e9",
                 "B, eyes closed": "#a04000", "B, eyes open": "#f0b27a"}


def finish(fig, path, axes=None):
    fig.tight_layout()
    if axes is not None and len(axes) > 1:
        panel_labels(axes)
    assert_max_columns(fig)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def load_branch(run, device):
    state = torch.load(os.path.join(run, "checkpoint-best.pth"), map_location="cpu", weights_only=False)["model"]
    sub = {k[len("window_feat."):]: v for k, v in state.items() if k.startswith("window_feat.")}
    branch = WindowFeatureBranch(sub["proj.weight"].shape[0], 0.1)
    branch.load_state_dict(sub)
    return branch.to(device).eval()


def feature_layout(names19):
    """(family, name) of each of the 4028 columns."""
    fam, name = [], []
    for ch in names19:
        for k, n in enumerate(CORE_NAMES):
            fam.append("spectral" if k < 17 else "1 Hz bins" if k < 61 else "time domain")
            name.append(f"{ch} {n}")
    for b in FILTERBANK:
        for i in range(171):
            fam.append("covariance (tangent)")
            name.append(f"cov {b} t{i}")
    pairs = [(names19[i], names19[j]) for i in range(N_CHANNELS) for j in range(i + 1, N_CHANNELS)]
    for b in BANDS:
        for kind in ("iCOH", "MSC"):
            for a, c in pairs:
                fam.append("connectivity")
                name.append(f"{kind} {b} {a}-{c}")
    return np.array(fam), np.array(name)


class Cohort:
    def __init__(self, path):
        self.path = path
        man = json.load(open(os.path.join(path, "manifest.json")))
        self.keep = [i for i, c in enumerate(man["channels"]) if c not in DROP]
        self.names = [man["channels"][i] for i in self.keep]
        recs = man["recordings"]
        self.recs = recs if isinstance(recs, list) else list(recs.values())
        self.win = int(man["window_samples"])

    def windows(self, stem, with_quality=False):
        """``(windows [W, 19, L] µV float32, clean [W])`` of a recording (+ ``window_quality``)."""
        arr = np.load(os.path.join(self.path, "recordings", f"{stem}.npy"), mmap_mode="r")
        n = arr.shape[0] // self.win
        x = np.asarray(arr[: n * self.win], np.float32).T[self.keep]
        w = x.reshape(len(self.keep), n, self.win).transpose(1, 0, 2)
        q = window_quality(w)
        return (w, clean_mask(q), q) if with_quality else (w, clean_mask(q))


@torch.no_grad()
def per_person(cohort, branch, device):
    """Median raw and standardised features of every person's session-1 eyes-closed run (before the task)."""
    ex = branch.extract
    rows, raw, std, tnorm = [], [], [], []
    for r in cohort.recs:
        if r["session"] != "1" or r["task"] != "EC" or r["acq"] != "pre":
            continue
        if not all(r["channel_mask"][i] for i in cohort.keep):
            continue
        w, clean = cohort.windows(r["stem"])
        if clean.sum() < 3:
            continue
        x = torch.from_numpy(w[clean]).to(device)
        core, cov, conn = ex.raw(x)
        u = torch.cat([core, ex._tangent(cov), conn], 1)
        z = ((torch.nan_to_num(u) - ex.mean) / ex.std).clamp(-6, 6)
        t = _sym_fn(ex.whiten @ cov @ ex.whiten, lambda v: v.clamp_min(1e-12).log())
        rows.append({"subject": r["subject"], "age": float(r["age"]), "stem": r["stem"]})
        raw.append(u.median(0).values.cpu().numpy())
        std.append(z.median(0).values.cpu().numpy())
        tnorm.append(torch.linalg.matrix_norm(t).median(0).values.cpu().numpy())
    return pd.DataFrame(rows), np.stack(raw), np.stack(std), np.stack(tnorm)


def pick_people(people, raw, names19):
    """A: age <= 25, B: age >= 70, each the person whose posterior relative alpha is the median of the group."""
    post = [names19.index(c) for c in POSTERIOR]
    rel_alpha = raw[:, [c * 68 + 7 for c in post]].mean(1)
    out = {}
    for tag, m in (("A", people.age <= 25), ("B", people.age >= 70)):
        idx = np.flatnonzero(m.to_numpy())
        out[tag] = int(idx[np.argsort(rel_alpha[idx])[len(idx) // 2]])
    return out


def calmest_clean(clean, quality):
    """The clean window with the smallest largest peak-to-peak amplitude (fewest eye / muscle artifacts)."""
    idx = np.flatnonzero(clean)
    return int(idx[np.argmin(quality[idx, 1])])


@torch.no_grad()
def example_windows(cohort, people, picks, branch, device):
    """The four example windows and everything drawn from them."""
    ex = branch.extract
    out = {}
    for tag, i in picks.items():
        stem_ec = people.stem[i]
        for state, stem in (("eyes closed", stem_ec), ("eyes open", stem_ec.replace("_EC-", "_EO-"))):
            w, clean, q = cohort.windows(stem, with_quality=True)
            k = calmest_clean(clean, q)
            x = torch.from_numpy(w[k:k + 1]).to(device)
            xc, psd, csd = ex._spectra(x)
            core, cov, conn = ex.raw(x)
            df = float(ex.freqs[1] - ex.freqs[0])
            cov19 = torch.einsum("nijf,bf->nbij", csd.real.double(), ex.fbank.double())[0] * df
            t = _sym_fn(ex.whiten @ cov @ ex.whiten, lambda v: v.clamp_min(1e-12).log())[0]
            u = torch.cat([core, ex._tangent(cov), conn], 1)
            z = ((torch.nan_to_num(u) - ex.mean) / ex.std).clamp(-6, 6)
            e = ex_proj = branch.proj(z)[0]
            out[f"{tag}, {state}"] = {
                "age": float(people.age[i]), "x": xc[0].cpu().numpy(), "psd": psd[0].cpu().numpy(),
                "core": core[0].reshape(N_CHANNELS, 68).cpu().numpy(), "cov19": cov19.cpu().numpy(),
                "tangent": t.cpu().numpy(), "conn": conn[0].reshape(5, 2, 171).cpu().numpy(),
                "z": z[0].cpu().numpy(), "e": ex_proj.cpu().numpy() if e is not None else None}
    return out


def pair_matrix(values171):
    m = np.zeros((N_CHANNELS, N_CHANNELS))
    iu = np.triu_indices(N_CHANNELS, 1)
    m[iu] = values171
    return m + m.T


def draw(out_dir, cohort, ex_freqs, W, people, raw, std, tnorm, fam, fname):
    names = cohort.names
    keys = list(W)
    freqs = ex_freqs
    o1 = names.index("O1")
    lab = {k: f"{k.split(',')[0]} ({W[k]['age']:.0f} y),{k.split(',')[1]}" for k in keys}

    # Figure 2.1: the four windows, 19 channels after the common average reference
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), sharex=True)
    t = np.arange(W[keys[0]]["x"].shape[-1]) / 200.0
    for ax, k in zip(axes.ravel(), keys):
        x = W[k]["x"]
        for c in range(N_CHANNELS):
            ax.plot(t, x[c] - 60 * c, lw=0.5, color="#17202a")
        ax.set_yticks(-60 * np.arange(N_CHANNELS), names)
        ax.set_xlabel("time (s)")
        ax.set_title(f"Person {lab[k]}")
        ax.plot([9.6, 9.6], [-60 * 18.5, -60 * 18.5 + 50], color="#c0392b", lw=2)
        ax.text(9.55, -60 * 18.5 + 25, "50 µV", ha="right", va="center", fontsize=8, color="#c0392b")
    finish(fig, os.path.join(out_dir, "eeg_examples.png"), axes)

    # Figure 3.1: power spectra
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for k in keys:
        axes[0].semilogy(freqs, W[k]["psd"][o1], color=WINDOW_COLORS[k], lw=1.6, label=lab[k])
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("PSD at O1 (µV²/Hz)")
    axes[0].set_title("O1 spectrum, Eq. (3.4)")
    axes[0].legend(fontsize=8)
    im = axes[1].imshow(np.log10(W[keys[0]]["psd"] + 1e-12), aspect="auto", cmap="viridis",
                        extent=[freqs[0], freqs[-1] + 0.5, N_CHANNELS - 0.5, -0.5])
    axes[1].set_yticks(range(N_CHANNELS), names, fontsize=7)
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_title(f"All channels, {lab[keys[0]]}")
    fig.colorbar(im, ax=axes[1], label="log10 PSD (µV²/Hz)")
    finish(fig, os.path.join(out_dir, "spectra.png"), axes)

    # Figure 4.1: the spectral features on one channel
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    k0 = keys[0]
    core = W[k0]["core"][o1]
    ax = axes[0]
    ax.loglog(freqs, W[k0]["psd"][o1], color="#17202a", lw=1.4, label="PSD (3.4)")
    fit = 10 ** (core[16] - core[15] * np.log10(freqs))
    ax.loglog(freqs, fit, color="#c0392b", ls="--", lw=1.4, label=f"1/f fit (4.11), exponent {core[15]:.2f}")
    for (b, (lo, hi)), col in zip(BANDS.items(), ["#f9e79f", "#d5f5e3", "#d6eaf8", "#fadbd8", "#e8daef"]):
        ax.axvspan(lo, hi, color=col, alpha=0.6, lw=0)
        ax.text(np.sqrt(lo * hi), 0.97, b, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    ax.axvline(core[14], color="#1f618d", lw=1.2, label=f"peak alpha freq. (4.10) {core[14]:.1f} Hz")
    ax.axvline(core[12], color="#7d3c98", lw=1.2, ls=":", label=f"SEF95 (4.8) {core[12]:.1f} Hz")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (µV²/Hz)")
    ax.set_title(f"O1, {lab[k0]}")
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    width = 0.2
    for j, k in enumerate(keys):
        ax.bar(np.arange(5) + (j - 1.5) * width, W[k]["core"][o1][5:10], width, color=WINDOW_COLORS[k], label=lab[k])
    ax.set_xticks(range(5), list(BANDS))
    ax.set_ylabel("relative power at O1 (4.4)")
    ax.set_title("Relative band power")
    ax.legend(fontsize=8)
    finish(fig, os.path.join(out_dir, "spectral_features.png"), axes)

    # Figure 4.2: 1 Hz bins
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    a = W[keys[0]]["core"][:, 17:61]
    d = W[keys[2]]["core"][:, 17:61] - a
    for ax, m, cmap, title, cl in ((axes[0], a, "viridis", f"log10 power per 1 Hz bin (4.13), {lab[keys[0]]}",
                                    "log10 µV²"),
                                   (axes[1], d, "RdBu_r", "Person B minus person A, eyes closed", "difference (log10)")):
        lim = np.abs(m).max() if cmap == "RdBu_r" else None
        im = ax.imshow(m, aspect="auto", cmap=cmap, extent=[1, 45, N_CHANNELS - 0.5, -0.5],
                       vmin=-lim if lim else None, vmax=lim)
        ax.set_yticks(range(N_CHANNELS), names, fontsize=7)
        ax.set_xlabel("bin start (Hz)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=cl)
    finish(fig, os.path.join(out_dir, "bins.png"), axes)

    # Figure 4.3: time domain
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = W[k0]["x"][o1]
    full = np.fft.rfft(x)
    ff = np.fft.rfftfreq(len(x), 1 / 200.0)
    xb = np.fft.irfft(full * ((ff >= 1) & (ff <= 45)), n=len(x))
    tt = np.arange(400) / 200.0
    axes[0].plot(tt, x[:400], color="#aab7b8", lw=1, label="after reference (2.3)")
    axes[0].plot(tt, xb[:400], color="#17202a", lw=1, label="1-45 Hz (4.14)")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("O1 (µV)")
    axes[0].set_title(f"First 2 s at O1, {lab[k0]}")
    axes[0].legend(fontsize=8)
    names7 = CORE_NAMES[61:]
    for j, k in enumerate(keys):
        zz = W[k]["z"][o1 * 68 + 61: o1 * 68 + 68]
        axes[1].bar(np.arange(7) + (j - 1.5) * 0.2, zz, 0.2, color=WINDOW_COLORS[k], label=lab[k])
    axes[1].set_xticks(range(7), names7, rotation=25, ha="right", fontsize=8)
    axes[1].axhline(0, color="#34495e", lw=0.8)
    axes[1].set_ylabel("standardised value z (7.2)")
    axes[1].set_title("Time-domain features at O1, Eqs. (4.15)-(4.18)")
    axes[1].legend(fontsize=8)
    finish(fig, os.path.join(out_dir, "time_domain.png"), axes)

    # Figure 4.4: spectral features against age, one point per person
    post = [names.index(c) for c in POSTERIOR]

    def colmean(k, chans):
        return raw[:, [c * 68 + k for c in chans]].mean(1)
    panels = [("peak alpha frequency, posterior (Hz)", colmean(14, post)),
              ("1/f exponent, mean of channels", colmean(15, range(N_CHANNELS))),
              ("relative alpha power, posterior", colmean(7, post)),
              ("log10 theta / alpha, mean of channels", colmean(10, range(N_CHANNELS)))]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    rho_rows = []
    for ax, (title, v) in zip(axes.ravel(), panels):
        rho = pd.Series(v).corr(people.age, method="spearman")
        rho_rows.append({"feature": title, "spearman_rho": rho})
        ax.scatter(people.age, v, s=6, alpha=0.5, color="#2e86c1")
        ax.set_xlabel("age (years)")
        ax.set_title(title)
        ax.text(0.98, 0.95, f"Spearman rho {rho:+.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    finish(fig, os.path.join(out_dir, "features_vs_age.png"), axes)

    # Figure 5.1: a band covariance and its tangent matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    b = list(FILTERBANK).index("alpha")
    c19 = W[k0]["cov19"][b]
    im = axes[0].imshow(c19, cmap="RdBu_r", vmin=-np.abs(c19).max(), vmax=np.abs(c19).max())
    axes[0].set_xticks(range(N_CHANNELS), names, rotation=90, fontsize=7)
    axes[0].set_yticks(range(N_CHANNELS), names, fontsize=7)
    axes[0].set_title(f"Alpha-band covariance (5.1), {lab[k0]}")
    fig.colorbar(im, ax=axes[0], label="µV²")
    tm = W[k0]["tangent"][b]
    im = axes[1].imshow(tm, cmap="RdBu_r", vmin=-np.abs(tm).max(), vmax=np.abs(tm).max())
    axes[1].set_xlabel("basis direction (5.2)")
    axes[1].set_ylabel("basis direction (5.2)")
    axes[1].set_title("Tangent matrix T, alpha band (5.5)")
    fig.colorbar(im, ax=axes[1], label="log ratio to the training reference")
    finish(fig, os.path.join(out_dir, "covariance.png"), axes)

    # Figure 5.2: distance from the typical training window per band
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for j, k in enumerate(keys):
        norms = np.linalg.norm(W[k]["tangent"], axis=(1, 2))
        axes[0].bar(np.arange(6) + (j - 1.5) * 0.2, norms, 0.2, color=WINDOW_COLORS[k], label=lab[k])
    axes[0].set_xticks(range(6), list(FILTERBANK))
    axes[0].set_ylabel("Frobenius norm of T (5.5)")
    axes[0].set_title("Distance from the training reference per band")
    axes[0].legend(fontsize=8)
    rho = pd.Series(tnorm[:, b]).corr(people.age, method="spearman")
    axes[1].scatter(people.age, tnorm[:, b], s=6, alpha=0.5, color="#e67e22")
    axes[1].set_xlabel("age (years)")
    axes[1].set_ylabel("Frobenius norm of T, alpha band")
    axes[1].set_title("Alpha-band distance, one point per person")
    axes[1].text(0.98, 0.95, f"Spearman rho {rho:+.2f}", transform=axes[1].transAxes, ha="right", va="top", fontsize=9)
    rho_rows.append({"feature": "alpha tangent norm", "spearman_rho": rho})
    finish(fig, os.path.join(out_dir, "tangent_norms.png"), axes)

    # Figure 6.1 / 6.2: connectivity matrices
    ab = list(BANDS).index("alpha")

    def mat(ax, m, title, cmap, label, sym=False):
        lim = np.abs(m).max()
        im = ax.imshow(m, cmap=cmap, vmin=-lim if sym else 0, vmax=lim)
        ax.set_xticks(range(N_CHANNELS), names, rotation=90, fontsize=7)
        ax.set_yticks(range(N_CHANNELS), names, fontsize=7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=label)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    mat(axes[0], pair_matrix(W[k0]["conn"][ab, 0]), f"Imaginary coherence (6.2), alpha, {lab[k0]}", "magma", "iCOH")
    mat(axes[1], pair_matrix(W[k0]["conn"][ab, 1]), "Coherence (6.3), alpha, same window", "magma", "MSC")
    finish(fig, os.path.join(out_dir, "connectivity.png"), axes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    mat(axes[0], pair_matrix(W[keys[0]]["conn"][ab, 1] - W[keys[1]]["conn"][ab, 1]),
        "Coherence, alpha: person A eyes closed minus open", "RdBu_r", "MSC difference", sym=True)
    mat(axes[1], pair_matrix(W[keys[2]]["conn"][ab, 1] - W[keys[0]]["conn"][ab, 1]),
        "Coherence, alpha: person B minus A, eyes closed", "RdBu_r", "MSC difference", sym=True)
    finish(fig, os.path.join(out_dir, "connectivity_contrast.png"), axes)

    # Figure 7.1: the standardised vector of two windows
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    starts = {f: np.flatnonzero(fam == f)[0] for f, _ in FAMILIES}
    for ax, k in zip(axes, (keys[0], keys[2])):
        for f, col in FAMILIES:
            m = np.flatnonzero(fam == f)
            ax.scatter(m, W[k]["z"][m], s=1.5, color=col, label=f)
        ax.set_ylabel("z (7.2)")
        ax.set_ylim(-6.5, 6.5)
        ax.set_title(f"Standardised feature vector, {lab[k]}")
    axes[1].set_xlabel("feature index (Table 1.1 order)")
    axes[0].legend(markerscale=6, fontsize=8, ncol=5, loc="lower center")
    for ax in axes:                          # the per-channel families interleave; mark the two block starts
        for f in ("covariance (tangent)", "connectivity"):
            ax.axvline(starts[f], color="#bdc3c7", lw=0.8)
    finish(fig, os.path.join(out_dir, "standardized_vector.png"), axes)

    # Figure 7.2: the 64 projected numbers e
    fig, ax = plt.subplots(figsize=(14, 3.6))
    E = np.stack([W[k]["e"] for k in keys])
    lim = np.abs(E).max()
    im = ax.imshow(E, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_yticks(range(4), [lab[k] for k in keys])
    ax.set_xlabel("output of the projection (7.4), 64 numbers")
    fig.colorbar(im, ax=ax, label="e")
    finish(fig, os.path.join(out_dir, "projection_output.png"))

    # Figure 7.3: which features follow age across people
    rho_all = pd.DataFrame(std).corrwith(people.age.reset_index(drop=True), method="spearman").to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    data = [np.abs(rho_all[fam == f]) for f, _ in FAMILIES]
    bp = axes[0].boxplot(data, vert=False, tick_labels=[f for f, _ in FAMILIES], patch_artist=True, showfliers=False)
    for patch, (_, col) in zip(bp["boxes"], FAMILIES):
        patch.set_facecolor(col)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("|Spearman rho| with age")
    axes[0].set_title("Every feature, by family")
    top = np.argsort(-np.abs(rho_all))[:15]
    colors = dict(FAMILIES)
    axes[1].barh(range(15), rho_all[top], color=[colors[fam[i]] for i in top])
    axes[1].set_yticks(range(15), fname[top], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#34495e", lw=0.8)
    axes[1].set_xlabel("Spearman rho with age")
    axes[1].set_title("The 15 features most related to age")
    finish(fig, os.path.join(out_dir, "age_correlation.png"), axes)

    summary = pd.DataFrame(rho_rows)
    fam_rows = [{"feature": f"family {f}: median |rho|", "spearman_rho": float(np.median(np.abs(rho_all[fam == f])))}
                for f, _ in FAMILIES]
    top_rows = [{"feature": f"top: {fname[i]}", "spearman_rho": float(rho_all[i])} for i in top]
    ages = [{"feature": f"age of person {k.split(',')[0]}", "spearman_rho": W[k]["age"]} for k in (keys[0], keys[2])]
    people_row = [{"feature": "people", "spearman_rho": float(len(people))}]
    pd.concat([summary, pd.DataFrame(fam_rows + top_rows + ages + people_row)]).to_csv(
        os.path.join(out_dir, "figure_numbers.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    branch = load_branch(args.run, args.device)
    cohort = Cohort(args.cohort)
    people, raw, std, tnorm = per_person(cohort, branch, args.device)
    picks = pick_people(people, raw, cohort.names)
    W = example_windows(cohort, people, picks, branch, args.device)
    fam, fname = feature_layout(cohort.names)
    draw(args.out, cohort, branch.extract.freqs.cpu().numpy(), W, people, raw, std, tnorm, fam, fname)
    print(pd.read_csv(os.path.join(args.out, "figure_numbers.csv")).to_string(index=False))


if __name__ == "__main__":
    main()
