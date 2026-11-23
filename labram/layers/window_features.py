# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Handcrafted features of each 10 s window, computed inside the model (mil.window_features).
# ---------------------------------------------------------
"""Handcrafted window features as a second input next to the class token.

``mil.window_features=handcrafted`` gives every 10 s window a fixed feature
vector, computed from the model's own input (so training, evaluation and any
external cohort see the same thing), projected to ``mil.feature_dim`` and
concatenated to the window's class-token embedding. The joined vector is what
the attention scores read, what the pool averages and what the age head
predicts from (``NeuralTransformer.forward`` / ``forward_bag``).

The features are the three blocks that did best in the recording-level study
(``scripts/feature_stack.py``), here per window on the 19 scalp channels after a
common average reference:

* spectral / spectrum / time (68 per channel): band powers, ratios, edge
  frequency, entropy, alpha peak, 1/f fit; log power in 1 Hz bins; Hjorth and
  moments of the 1-45 Hz signal (``labram/features/extractors.py``);
* band covariances between the channels (6 bands) in the Riemannian tangent
  space at reference matrices fitted on training windows;
* connectivity per channel pair and band: imaginary coherence and coherence.

Everything up to the standardized feature vector is fixed (no gradient); the
standardization and the tangent-space references are buffers fitted once on
training windows (:meth:`HandcraftedWindowFeatures.fit`) and travel with the
checkpoint. Only the projection is trained.

Optional blocks (``mil.feature_extra``), each tested as an addition to the vector above:

* ``microstates``: back-fitting of every sample of the 2-20 Hz signal to 5 polarity-free
  scalp maps (modified k-means on GFP peaks of the fit windows): coverage, mean duration,
  occurrence and transition probabilities of each map, global explained variance (36);
* ``aec``: orthogonalised amplitude-envelope correlation (Hipp et al. 2012) of every
  channel pair in five bands (855);
* ``complexity``: Higuchi, Katz and Petrosian fractal dimension, DFA exponent of the
  signal and permutation entropy (order 4, delays 1 and 4) per channel (114);
* ``green``: learned filter-bank covariances (after GREEN, Paillard et al. 2025):
  Gaussian band filters with trainable centre and width, the band covariances reduced by a
  trainable bilinear map and taken to the log domain, batch-normalised and projected
  (:class:`LearnedFilterbankCov`, trained with the model).

``mil.recording_features`` adds one vector per *recording* (slow amplitude dynamics over
minutes, ``labram/features/slow_dynamics.py``), looked up by recording, standardised with
training statistics and projected (:class:`RecordingFeatures`).
"""
import contextlib
import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

EPS = 1e-12
# Same bands as labram/features/extractors.py (kept in sync by tests/test_window_features.py;
# not imported from there: that module needs PyWavelets, which a training image may lack).
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}
FILTERBANK = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta_low": (13.0, 20.0),
              "beta_high": (20.0, 30.0), "gamma": (30.0, 45.0)}
N_CHANNELS = 19
FIXED_EXTRA = ("microstates", "aec", "complexity")
LEARNED_EXTRA = ("green",)
N_MICROSTATES = 5
# The fixed families of the base vector (mil.feature_families; empty = all): per channel
# spectral (17), 1 Hz bins (44) and time domain (7), then band covariances and connectivity.
FAMILIES = ("spectral", "bins", "time", "covariance", "connectivity")
N_EXTRA = {"microstates": 3 * N_MICROSTATES + N_MICROSTATES * (N_MICROSTATES - 1) + 1,
           "aec": 5 * 171, "complexity": N_CHANNELS * 6}


def _sym_fn(c: torch.Tensor, fn) -> torch.Tensor:
    """A scalar function of the eigenvalues of symmetric matrices ``[..., n, n]``."""
    w, v = torch.linalg.eigh(c)
    return (v * fn(w)[..., None, :]) @ v.transpose(-1, -2)


def riemann_mean(covs: torch.Tensor, tol: float = 1e-8, max_iter: int = 50) -> torch.Tensor:
    """Affine-invariant (Karcher) mean of SPD matrices ``[N, n, n]`` (``labram/features/riemann.py``)."""
    ref = covs.mean(0)
    for _ in range(max_iter):
        s = _sym_fn(ref, torch.sqrt)
        si = _sym_fn(ref, lambda w: w.clamp_min(EPS).rsqrt())
        t = _sym_fn(si @ covs @ si, lambda w: w.clamp_min(EPS).log()).mean(0)
        ref = s @ _sym_fn(t, torch.exp) @ s
        if torch.linalg.norm(t) < tol:
            break
    return ref


class HandcraftedWindowFeatures(nn.Module):
    """Fixed features of windows ``[N, 19, L]`` (µV at ``fs`` Hz) -> standardized ``[N, n_features]``."""

    CHUNK = 256

    def __init__(self, fs: float = 200.0, seg_sec: float = 2.0, lo: float = 1.0, hi: float = 45.0,
                 shrink: float = 1e-3, extra: Sequence[str] = (), families: Sequence[str] = ()):
        super().__init__()
        self.extra = tuple(b for b in FIXED_EXTRA if b in extra)
        unknown = set(families) - set(FAMILIES)
        if unknown:
            raise ValueError(f"unknown feature families {sorted(unknown)}; expected some of {FAMILIES}")
        self.families = tuple(f for f in FAMILIES if f in families) or FAMILIES
        self.fs, self.nper, self.lo, self.hi, self.shrink = fs, int(seg_sec * fs), lo, hi, shrink
        self.register_buffer("hann", torch.hann_window(self.nper, periodic=True), persistent=False)
        f = torch.fft.rfftfreq(self.nper, 1.0 / fs)
        sel = (f >= lo) & (f < hi)
        f = f[sel]
        self.register_buffer("sel", sel, persistent=False)
        self.register_buffer("freqs", f, persistent=False)

        def masks(bands):
            return torch.stack([((f >= a) & (f < b)).float() for a, b in bands])
        self.register_buffer("band", masks(BANDS.values()), persistent=False)             # [5, F]
        self.register_buffer("fbank", masks(FILTERBANK.values()), persistent=False)       # [6, F]
        self.register_buffer("bins", masks([(e, e + 1.0) for e in torch.arange(lo, hi - 1e-9).tolist()]),
                             persistent=False)                                            # [44, F]
        fit = (f >= 2.0) & (f <= 40.0) & ~((f >= 7.0) & (f <= 14.0))                      # 1/f fit, alpha left out
        lf = torch.log10(f[fit])
        self.register_buffer("fit_bins", fit, persistent=False)
        self.register_buffer("lf_c", lf - lf.mean(), persistent=False)
        self.lf_mean = float(lf.mean())
        self.register_buffer("alpha_bins", (f >= 6.0) & (f <= 14.0), persistent=False)
        # orthonormal basis of the complement of the all-ones vector: the covariance of an
        # average-referenced montage has rank 18
        q, _ = torch.linalg.qr(torch.eye(N_CHANNELS, dtype=torch.float64) - 1.0 / N_CHANNELS)
        self.register_buffer("basis", q[:, : N_CHANNELS - 1].contiguous(), persistent=False)   # [19, 18]
        m = N_CHANNELS - 1
        iu = torch.triu_indices(m, m)
        self.register_buffer("iu", iu, persistent=False)
        self.register_buffer("iu_coef", torch.where(iu[0] == iu[1], 1.0, math.sqrt(2.0)).double(), persistent=False)
        self.register_buffer("pairs", torch.triu_indices(N_CHANNELS, N_CHANNELS, offset=1), persistent=False)
        self.n_core = N_CHANNELS * (17 + self.bins.shape[0] + 7)
        self.n_cov = len(FILTERBANK) * iu.shape[1]
        self.n_conn = len(BANDS) * 2 * self.pairs.shape[1]
        self.n_extra = {b: N_EXTRA[b] for b in self.extra}
        # columns kept from the full vector (all of the optional blocks, the chosen base families)
        per_channel = {"spectral": range(0, 17), "bins": range(17, 61), "time": range(61, 68)}
        keep = torch.zeros(self.n_core + self.n_cov + self.n_conn + sum(self.n_extra.values()), dtype=torch.bool)
        for f, ks in per_channel.items():
            if f in self.families:
                for c in range(N_CHANNELS):
                    keep[[c * 68 + k for k in ks]] = True
        if "covariance" in self.families:
            keep[self.n_core:self.n_core + self.n_cov] = True
        if "connectivity" in self.families:
            keep[self.n_core + self.n_cov:self.n_core + self.n_cov + self.n_conn] = True
        keep[self.n_core + self.n_cov + self.n_conn:] = True
        self.register_buffer("keep", keep, persistent=False)
        self.n_features = int(keep.sum())
        if "microstates" in self.extra:     # fitted scalp maps, unit norm, zero mean over channels
            self.register_buffer("ms_maps", torch.zeros(N_MICROSTATES, N_CHANNELS))
        # fitted on training windows, saved with the checkpoint
        self.register_buffer("whiten", torch.eye(m, dtype=torch.float64).repeat(len(FILTERBANK), 1, 1))
        self.register_buffer("mean", torch.zeros(self.n_features))
        self.register_buffer("std", torch.ones(self.n_features))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    # ------------------------------------------------------------ pieces
    @staticmethod
    def car(x):
        """Common average reference over the present channels (an absent channel is stored as zeros)."""
        present = x.abs().amax(-1, keepdim=True) > 0
        n = present.sum(1, keepdim=True).clamp_min(1)
        return (x - (x * present).sum(1, keepdim=True) / n) * present

    def _spectra(self, x):
        """Common average reference, then Welch segments: ``(x_car, psd [N,C,F], csd [N,C,C,F])``."""
        x = self.car(x)
        seg = x.unfold(-1, self.nper, self.nper // 2)                   # [N, C, S, nper]
        spec = torch.fft.rfft((seg - seg.mean(-1, keepdim=True)) * self.hann, dim=-1)[..., self.sel]
        scale = 2.0 / (self.fs * float((self.hann ** 2).sum()) * seg.shape[-2])
        csd = torch.einsum("nisf,njsf->nijf", spec, spec.conj()) * scale
        return x, torch.diagonal(csd, dim1=1, dim2=2).real.permute(0, 2, 1), csd

    def _core(self, x, psd):
        """``[N, C, 68]``: spectral (17), 1 Hz log-power bins (44), time domain (7)."""
        df = float(self.freqs[1] - self.freqs[0])
        bp = psd @ self.band.T * df                                     # [N, C, 5]
        total = psd.sum(-1, keepdim=True) * df + EPS
        d, t, a, b = (bp[..., i] + EPS for i in range(4))
        p = psd / (psd.sum(-1, keepdim=True) + EPS)
        entropy = -(p * torch.log(p + EPS)).sum(-1) / math.log(psd.shape[-1])
        sef95 = self.freqs[((p.cumsum(-1) >= 0.95).float().argmax(-1))]
        paf = self.freqs[self.alpha_bins][psd[..., self.alpha_bins].argmax(-1)]
        lp = torch.log10(psd[..., self.fit_bins] + EPS)
        slope = (lp - lp.mean(-1, keepdim=True)) @ self.lf_c / (self.lf_c @ self.lf_c)
        offset = lp.mean(-1) - slope * self.lf_mean
        spectral = torch.cat([torch.log10(bp + EPS), bp / total,
                              torch.stack([torch.log10(t / a), torch.log10((d + t) / (a + b)), sef95, entropy,
                                           paf, -slope, offset], -1)], -1)
        bins = torch.log10(psd @ self.bins.T * df + EPS)
        # time domain on the 1-45 Hz part of the window
        full = torch.fft.rfft(x, dim=-1)
        ff = torch.fft.rfftfreq(x.shape[-1], 1.0 / self.fs, device=x.device)
        xb = torch.fft.irfft(full * ((ff >= self.lo) & (ff <= self.hi)), n=x.shape[-1], dim=-1)
        dx = xb.diff(dim=-1)
        ddx = dx.diff(dim=-1)
        v0, v1, v2 = (v.var(-1, unbiased=False) + EPS for v in (xb, dx, ddx))
        mobility = torch.sqrt(v1 / v0)
        c = xb - xb.mean(-1, keepdim=True)
        sd = c.pow(2).mean(-1).sqrt() + EPS
        time = torch.stack([torch.log10(v0), torch.log10(dx.abs().mean(-1) + EPS), (c / sd[..., None]).pow(3).mean(-1),
                            (c / sd[..., None]).pow(4).mean(-1) - 3.0, mobility, torch.sqrt(v2 / v1) / mobility,
                            (torch.signbit(c)[..., 1:] != torch.signbit(c)[..., :-1]).float().mean(-1)], -1)
        return torch.cat([spectral, bins, time], -1)

    def _cov(self, csd):
        """Band covariances from the cross-spectra, on the 18-dimensional basis, lightly shrunk: ``[N, 6, 18, 18]``."""
        df = float(self.freqs[1] - self.freqs[0])
        cov = torch.einsum("nijf,bf->nbij", csd.real.double(), self.fbank.double()) * df
        cov = self.basis.T @ cov @ self.basis
        tr = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1)[..., None, None] / cov.shape[-1]
        return (1 - self.shrink) * cov + self.shrink * tr * torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)

    def _tangent(self, cov):
        """Tangent vectors at the fitted references: ``[N, 6 * 171]``."""
        log = _sym_fn(self.whiten @ cov @ self.whiten, lambda w: w.clamp_min(EPS).log())
        return (log[..., self.iu[0], self.iu[1]] * self.iu_coef).flatten(1).float()

    def _conn(self, psd, csd):
        """Per band: |imaginary part of the mean coherency| and mean squared coherence of the 171 pairs."""
        coh = csd / torch.sqrt(psd[:, :, None, :] * psd[:, None, :, :] + EPS)
        coh = coh[:, self.pairs[0], self.pairs[1]]                      # [N, P, F]
        n = self.band.sum(-1)                                           # bins per band
        icoh = (coh.imag @ self.band.T / n).abs()                       # [N, P, 5]
        msc = (coh.real ** 2 + coh.imag ** 2) @ self.band.T / n
        return torch.stack([icoh, msc], 2).permute(0, 3, 2, 1).flatten(1)   # band, kind, pair

    # ------------------------------------------------------------ optional fixed blocks
    def _fft_band(self, xc, lo, hi):
        """``xc`` limited to ``lo``-``hi`` Hz by zeroing its FFT (real signal)."""
        full = torch.fft.rfft(xc, dim=-1)
        ff = torch.fft.rfftfreq(xc.shape[-1], 1.0 / self.fs, device=xc.device)
        return torch.fft.irfft(full * ((ff >= lo) & (ff <= hi)), n=xc.shape[-1], dim=-1)

    def _ms_corr(self, xs):
        """Spatial correlation ``[N, K, T]`` of every sample of ``xs [N, C, T]`` with the maps, and GFP ``[N, T]``."""
        gfp = xs.pow(2).mean(1).sqrt()
        xn = xs / (xs.norm(dim=1, keepdim=True) + EPS)
        return torch.einsum("kc,nct->nkt", self.ms_maps, xn), gfp

    def _microstates(self, xc):
        """Coverage, mean duration (s), occurrence (per s) of each map, transition probabilities, GEV: ``[N, 36]``."""
        k = N_MICROSTATES
        corr, gfp = self._ms_corr(self._fft_band(xc, 2.0, 20.0))
        lab = corr.abs().argmax(1)                                       # [N, T]
        n, t = lab.shape
        # the usual back-fitting: labels at the GFP peaks, every sample takes the label of its nearest peak
        pos = torch.arange(t, device=lab.device).expand(n, t)
        peak = torch.zeros_like(lab, dtype=torch.bool)
        peak[:, 1:-1] = (gfp[:, 1:-1] > gfp[:, :-2]) & (gfp[:, 1:-1] > gfp[:, 2:])
        peak[:, 0] |= ~peak.any(1)                                       # a window without a peak keeps sample 0
        prev = torch.where(peak, pos, torch.full_like(pos, -10 * t)).cummax(1).values
        nxt = torch.where(peak, pos, torch.full_like(pos, 10 * t)).flip(1).cummin(1).values.flip(1)
        near = torch.where((pos - prev) <= (nxt - pos), prev, nxt).clamp(0, t - 1)
        lab = lab.gather(1, near)
        onehot = torch.nn.functional.one_hot(lab, k).float()             # [N, T, K]
        coverage = onehot.mean(1)
        change = lab[:, 1:] != lab[:, :-1]
        starts = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=lab.device), change], 1)
        n_seg = (starts.unsqueeze(-1) * onehot).sum(1)                   # segments per map
        duration = coverage * t / n_seg.clamp_min(1) / self.fs           # mean segment length (s)
        occurrence = n_seg / (t / self.fs)
        pair = lab[:, :-1] * k + lab[:, 1:]
        trans = torch.zeros(n, k * k, device=lab.device).scatter_add_(1, pair, change.float())
        off = ~torch.eye(k, dtype=torch.bool, device=lab.device).flatten()
        trans = trans[:, off] / trans.sum(1, keepdim=True).clamp_min(1)
        best = corr.abs().amax(1)
        gev = (gfp.pow(2) * best.pow(2)).sum(1) / (gfp.pow(2).sum(1) + EPS)
        return torch.cat([coverage, duration, occurrence, trans, gev[:, None]], 1)

    def _aec(self, xc):
        """Orthogonalised amplitude-envelope correlation, symmetrised, of the 171 pairs in 5 bands: ``[N, 855]``."""
        n, c, t = xc.shape
        spec = torch.fft.fft(xc, dim=-1)
        f = torch.fft.fftfreq(t, 1.0 / self.fs, device=xc.device)
        out = []
        for lo, hi in BANDS.values():
            z = torch.fft.ifft(spec * (((f >= lo) & (f < hi)).float() * 2.0), dim=-1)   # analytic signal
            env = z.abs()
            envc = env - env.mean(-1, keepdim=True)
            envn = envc.norm(dim=-1) + EPS
            unit = z / (env + EPS)
            a = xc.new_zeros(n, c, c)
            for i0 in range(0, c, 5):
                i1 = min(c, i0 + 5)
                # the part of every channel orthogonal to channels i0..i1 (no zero-lag leakage), its envelope
                yp = (z[:, None] * unit[:, i0:i1, None].conj()).imag.abs()      # [N, I, C, T]
                yc = yp - yp.mean(-1, keepdim=True)
                a[:, i0:i1] = ((yc * envc[:, i0:i1, None]).sum(-1) / (yc.norm(dim=-1) + EPS)
                               / envn[:, i0:i1, None])
            a = (a + a.transpose(1, 2)) / 2
            out.append(a[:, self.pairs[0], self.pairs[1]])
        return torch.stack(out, 1).flatten(1)

    def _complexity(self, xc):
        """Per channel: Higuchi, Katz and Petrosian fractal dimension, DFA exponent, permutation
        entropy at delays 1 and 4, of the 1-45 Hz signal: ``[N, 19 * 6]``."""
        xb = self._fft_band(xc, self.lo, self.hi)
        t = xb.shape[-1]
        lk = []                                                          # Higuchi, kmax = 10
        for k in range(1, 11):
            lm = [xb[..., m::k].diff(dim=-1).abs().sum(-1) * (t - 1) / ((xb[..., m::k].shape[-1] - 1) * k) / k
                  for m in range(k)]
            lk.append(torch.stack(lm, -1).mean(-1))
        lx = torch.log(1.0 / torch.arange(1, 11, device=xb.device, dtype=xb.dtype))
        ly = torch.log(torch.stack(lk, -1) + EPS)
        lxc = lx - lx.mean()
        higuchi = ((ly - ly.mean(-1, keepdim=True)) * lxc).sum(-1) / (lxc ** 2).sum()
        dx = xb.diff(dim=-1)
        length = dx.abs().sum(-1) + EPS
        d = (xb - xb[..., :1]).abs().amax(-1) + EPS
        katz = math.log10(t - 1) / (math.log10(t - 1) + torch.log10(d / length))
        n_delta = (torch.signbit(dx[..., 1:]) != torch.signbit(dx[..., :-1])).float().sum(-1)
        petrosian = math.log10(t) / (math.log10(t) + torch.log10(t / (t + 0.4 * n_delta)))
        y = torch.cumsum(xb - xb.mean(-1, keepdim=True), -1)             # DFA over 0.05-1 s
        logf, logs = [], []
        for s in (10, 14, 20, 28, 40, 57, 80, 113, 160, 200):
            seg = y[..., : t // s * s].reshape(*y.shape[:-1], t // s, s)
            u = torch.arange(s, device=xb.device, dtype=xb.dtype) - (s - 1) / 2.0
            slope = (seg * u).sum(-1, keepdim=True) / (u ** 2).sum()
            res = seg - seg.mean(-1, keepdim=True) - slope * u
            logf.append(0.5 * torch.log10(res.pow(2).mean((-1, -2)) + EPS))
            logs.append(math.log10(s))
        logs = torch.tensor(logs, device=xb.device, dtype=xb.dtype)
        lsc = logs - logs.mean()
        logf = torch.stack(logf, -1)
        dfa = ((logf - logf.mean(-1, keepdim=True)) * lsc).sum(-1) / (lsc ** 2).sum()
        pe = []
        for delay in (1, 4):
            emb = xb.unfold(-1, 3 * delay + 1, 1)[..., ::delay]          # [N, C, T', 4]
            code = (emb.argsort(-1) * torch.tensor([1, 4, 16, 64], device=xb.device)).sum(-1)
            counts = torch.zeros(*code.shape[:-1], 256, device=xb.device).scatter_add_(
                -1, code, torch.ones(code.shape, device=xb.device))
            p = counts / counts.sum(-1, keepdim=True)
            pe.append(-(p * torch.log(p + EPS)).sum(-1) / math.log(24))
        return torch.stack([higuchi, katz, petrosian, dfa, pe[0], pe[1]], -1).flatten(1)

    def _extra(self, xc):
        fns = {"microstates": self._microstates, "aec": self._aec, "complexity": self._complexity}
        return [fns[b](xc) for b in self.extra]

    @torch.no_grad()
    def vector(self, x: torch.Tensor) -> torch.Tensor:
        """The raw (unstandardised) feature vector ``[N, n_features]`` of ``x [N, 19, L]`` µV."""
        if x.shape[1] != N_CHANNELS:
            raise ValueError(f"window features need the {N_CHANNELS} scalp channels, got {x.shape[1]}")
        rows = []
        for i in range(0, max(x.shape[0], 1), self.CHUNK):
            xc, psd, csd = self._spectra(x[i:i + self.CHUNK].float())
            parts = [self._core(xc, psd).flatten(1), self._tangent(self._cov(csd)), self._conn(psd, csd)]
            rows.append(torch.cat(parts + self._extra(xc), 1)[:, self.keep])
        return torch.nan_to_num(torch.cat(rows), nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def fit_microstates(self, x: torch.Tensor, n_peaks: int = 50000, iters: int = 40, restarts: int = 5):
        """The 5 microstate maps: modified (polarity-free) k-means on the normalised scalp maps at GFP
        peaks of the 2-20 Hz signal of windows ``x [M, 19, L]``; the restart with the largest explained
        variance is kept. Returns that explained variance."""
        maps = []
        for i in range(0, x.shape[0], self.CHUNK):
            xs = self._fft_band(self.car(x[i:i + self.CHUNK].float()), 2.0, 20.0)
            gfp = xs.pow(2).mean(1).sqrt()
            peak = (gfp[:, 1:-1] > gfp[:, :-2]) & (gfp[:, 1:-1] > gfp[:, 2:])
            nidx, tidx = peak.nonzero(as_tuple=True)
            v = xs[nidx, :, tidx + 1]
            maps.append(v / (v.norm(dim=1, keepdim=True) + EPS))
        v = torch.cat(maps)
        g = torch.Generator(device="cpu").manual_seed(0)
        v = v[torch.randperm(v.shape[0], generator=g)[:n_peaks].to(v.device)].double()
        best, best_gev = None, -1.0
        for _ in range(restarts):
            m = v[torch.randperm(v.shape[0], generator=g)[:N_MICROSTATES].to(v.device)]
            for _ in range(iters):
                lab = (v @ m.T).abs().argmax(1)
                m = torch.stack([m[k] if not (lab == k).any() else torch.linalg.eigh(v[lab == k].T @ v[lab == k])[1][:, -1]
                                 for k in range(N_MICROSTATES)])
            gev = float((v @ m.T).abs().amax(1).pow(2).mean())
            if gev > best_gev:
                best, best_gev = m, gev
        self.ms_maps.copy_(best.float())
        return best_gev

    @torch.no_grad()
    def raw(self, x: torch.Tensor):
        """``(core [N, 1292], band covariances [N, 6, 18, 18], connectivity [N, 1710])`` of ``x [N, 19, L]`` µV."""
        if x.shape[1] != N_CHANNELS:
            raise ValueError(f"window features need the {N_CHANNELS} scalp channels, got {x.shape[1]}")
        core, cov, conn = [], [], []
        for i in range(0, max(x.shape[0], 1), self.CHUNK):
            xc, psd, csd = self._spectra(x[i:i + self.CHUNK].float())
            core.append(self._core(xc, psd).flatten(1))
            cov.append(self._cov(csd))
            conn.append(self._conn(psd, csd))
        return torch.cat(core), torch.cat(cov), torch.cat(conn)

    @torch.no_grad()
    def fit(self, x: torch.Tensor, max_reference: int = 4000):
        """Fit the tangent-space references, the microstate maps (``microstates``) and the
        standardization on windows ``x [M, 19, L]`` (µV)."""
        core, cov, conn = self.raw(x)
        pick = torch.randperm(cov.shape[0], generator=torch.Generator().manual_seed(0))[:max_reference].to(cov.device)
        for b in range(cov.shape[1]):
            self.whiten[b] = _sym_fn(riemann_mean(cov[pick, b]), lambda w: w.clamp_min(EPS).rsqrt())
        if "microstates" in self.extra:
            self.fit_microstates(x)
        feats = self.vector(x)
        self.mean.copy_(feats.mean(0))
        self.std.copy_(feats.std(0).clamp_min(1e-6))
        self.fitted.fill_(True)
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((self.vector(x) - self.mean) / self.std).clamp(-6.0, 6.0)


class LearnedFilterbankCov(nn.Module):
    """Learned filter-bank covariances of a window (after GREEN, Paillard et al. 2025).

    ``K`` Gaussian band filters with trainable centre and width (initialised log-spaced over
    2-40 Hz) are applied in the frequency domain; each band's channel covariance (Parseval, on the
    18-dimensional average-reference basis, lightly shrunk) is reduced by a trainable bilinear map
    ``W_k^T C W_k`` (18 -> ``rank``), taken to the log domain (eigendecomposition, float64),
    vectorised, standardised and projected to ``out_dim``. Trained with the model.

    The standardisation uses running statistics (initialised on the fit windows,
    :meth:`init_stats`, then updated with momentum from training batches), never the batch's
    own, so a group of one window (the channel-drop path splits batches by channel set) works.
    """

    def __init__(self, n_filters: int = 10, rank: int = 8, out_dim: int = 32, fs: float = 200.0,
                 shrink: float = 1e-2):
        super().__init__()
        f0 = torch.logspace(math.log10(2.0), math.log10(40.0), n_filters)
        self.log_f = nn.Parameter(f0.log())
        self.log_bw = nn.Parameter((f0 / 4).clamp_min(1.0).log())
        q, _ = torch.linalg.qr(torch.eye(N_CHANNELS, dtype=torch.float64) - 1.0 / N_CHANNELS)
        self.register_buffer("basis", q[:, : N_CHANNELS - 1].float().contiguous(), persistent=False)
        g = torch.Generator().manual_seed(0)
        self.bimap = nn.Parameter(torch.linalg.qr(torch.randn(n_filters, N_CHANNELS - 1, rank, generator=g))[0])
        iu = torch.triu_indices(rank, rank)
        self.register_buffer("iu", iu, persistent=False)
        self.register_buffer("iu_coef", torch.where(iu[0] == iu[1], 1.0, math.sqrt(2.0)).double(), persistent=False)
        n_vec = n_filters * iu.shape[1]
        self.register_buffer("run_mean", torch.zeros(n_vec))
        self.register_buffer("run_var", torch.ones(n_vec))
        self.gain = nn.Parameter(torch.ones(n_vec))
        self.bias = nn.Parameter(torch.zeros(n_vec))
        self.momentum = 0.01
        self.out = nn.Linear(n_vec, out_dim)
        self.fs, self.shrink, self.out_dim = fs, shrink, out_dim

    @torch.no_grad()
    def init_stats(self, xc: torch.Tensor, chunk: int = 256):
        """Running statistics from windows ``xc [M, 19, L]`` (µV, average-referenced)."""
        v = torch.cat([self.vectors(xc[i:i + chunk]) for i in range(0, xc.shape[0], chunk)])
        self.run_mean.copy_(v.mean(0))
        self.run_var.copy_(v.var(0, unbiased=False).clamp_min(1e-6))

    def forward(self, xc: torch.Tensor) -> torch.Tensor:
        """``xc [N, 19, L]`` µV, average-referenced -> ``[N, out_dim]``."""
        v = self.vectors(xc)
        if self.training and v.shape[0] > 1:
            with torch.no_grad():
                self.run_mean.lerp_(v.mean(0), self.momentum)
                self.run_var.lerp_(v.var(0, unbiased=False), self.momentum)
        z = (v - self.run_mean) / torch.sqrt(self.run_var + 1e-5) * self.gain + self.bias
        return self.out(z)

    def vectors(self, xc: torch.Tensor) -> torch.Tensor:
        """The log-domain band covariances, vectorised: ``[N, n_filters * rank * (rank + 1) / 2]``."""
        n, c, t = xc.shape
        spec = torch.fft.rfft(xc.float(), dim=-1)
        f = torch.fft.rfftfreq(t, 1.0 / self.fs, device=xc.device)
        gain = torch.exp(-0.5 * ((f - self.log_f.exp()[:, None]) / self.log_bw.exp()[:, None]) ** 2)   # [K, F]
        xk = spec[:, None] * gain[None, :, None, :]                               # [N, K, C, F]
        cov = torch.einsum("nkcf,nkdf->nkcd", xk, xk.conj()).real * (2.0 / t ** 2)
        cov = (self.basis.T @ cov @ self.basis).double()                          # [N, K, 18, 18]
        eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
        tr = torch.diagonal(cov, dim1=-2, dim2=-1).mean(-1)[..., None, None]
        cov = (1 - self.shrink) * cov + (self.shrink * tr + 1e-9) * eye
        w = self.bimap.double()
        red = w.transpose(-1, -2) @ cov @ w                                       # [N, K, r, r]
        eye_r = torch.eye(red.shape[-1], dtype=red.dtype, device=red.device)
        red = red + 1e-6 * torch.diagonal(red, dim1=-2, dim2=-1).mean(-1)[..., None, None] * eye_r
        ev, vec = torch.linalg.eigh(red)
        log = (vec * ev.clamp_min(1e-12).log()[..., None, :]) @ vec.transpose(-1, -2)
        return (log[..., self.iu[0], self.iu[1]] * self.iu_coef).flatten(1).float()


class RecordingFeatures(nn.Module):
    """One vector per recording (``mil.recording_features``): looked up by recording stem from the
    tables loaded with :meth:`load_table`, standardised with training statistics (missing values ->
    the training mean), clipped and projected to ``out_dim``."""

    def __init__(self, n_in: int, out_dim: int = 16):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))
        self.proj = nn.Linear(n_in, out_dim)
        nn.init.normal_(self.proj.weight, std=n_in ** -0.5)
        nn.init.zeros_(self.proj.bias)
        self.n_in, self.out_dim, self.table = n_in, out_dim, {}

    def load_table(self, path: str) -> int:
        import numpy as np
        z = np.load(path, allow_pickle=False)
        feats = z["features"].astype("float32")
        if feats.shape[1] != self.n_in:
            raise ValueError(f"{path}: {feats.shape[1]} recording features, the model expects {self.n_in}")
        self.table.update(zip(z["stems"].tolist(), feats))
        return len(feats)

    def lookup(self, stems, device) -> torch.Tensor:
        """Raw vectors ``[len(stems), n_in]`` (NaN where a recording or a value is missing)."""
        import numpy as np
        nan = np.full(self.n_in, np.nan, dtype="float32")
        return torch.from_numpy(np.stack([self.table.get(s, nan) for s in stems])).to(device)

    @torch.no_grad()
    def fit(self, raw: torch.Tensor):
        mean = torch.nanmean(raw, 0)
        std = torch.sqrt(torch.nanmean((raw - mean) ** 2, 0))
        self.mean.copy_(torch.nan_to_num(mean))
        self.std.copy_(torch.nan_to_num(std, nan=1.0).clamp_min(1e-6))
        self.fitted.fill_(True)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        z = torch.nan_to_num((raw - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0).clamp(-6.0, 6.0)
        return self.proj(z)


class WindowFeatureBranch(nn.Module):
    """The model's second window input: handcrafted features -> ``dim`` (trained projection, dropout).

    ``forward`` takes the model input ``[N, C, A, T]`` (µV / ``input_scale``); with fewer than 19
    channels (a window encoded without its absent channels) ``channel_indices`` places them.
    """

    def __init__(self, dim: int = 64, dropout: float = 0.1, input_scale: float = 100.0,
                 extra: Sequence[str] = (), recording_features: int = 0, norm: str = "none",
                 families: Sequence[str] = ()):
        super().__init__()
        unknown = set(extra) - set(FIXED_EXTRA) - set(LEARNED_EXTRA)
        if unknown:
            raise ValueError(f"unknown window feature blocks {sorted(unknown)}; "
                             f"expected some of {FIXED_EXTRA + LEARNED_EXTRA}")
        self.extract = HandcraftedWindowFeatures(extra=extra, families=families)
        self.proj = nn.Linear(self.extract.n_features, dim)
        nn.init.normal_(self.proj.weight, std=self.extract.n_features ** -0.5)    # unit-scale output at the start
        nn.init.zeros_(self.proj.bias)
        self.green = LearnedFilterbankCov() if "green" in extra else None
        self.rec = RecordingFeatures(recording_features) if recording_features else None
        self.drop = nn.Dropout(dropout)
        self.out_dim = (dim + (self.green.out_dim if self.green is not None else 0)
                        + (self.rec.out_dim if self.rec is not None else 0))
        self.input_scale = input_scale
        # "layernorm": the joined output normalised per window (learned gain and bias), so its scale
        # matches the class token's instead of varying ~10x more per dimension across windows
        self.out_norm = nn.LayerNorm(self.out_dim) if norm == "layernorm" else None
        # positions of the 19 channels in the model's channel table (set when the features are fitted)
        self.register_buffer("channel_index", torch.full((N_CHANNELS,), -1, dtype=torch.long))

    def set_channels(self, channel_indices: Sequence[int]) -> None:
        """``channel_indices``: the model's indices of the 19 channels (class-token entry first)."""
        idx = list(channel_indices)[1:]
        if len(idx) != N_CHANNELS:
            raise ValueError(f"window features need the {N_CHANNELS} scalp channels, got {len(idx)}")
        self.channel_index.copy_(torch.as_tensor(idx, dtype=torch.long))

    def _signal(self, x4, channel_indices):
        x = x4.detach().flatten(2).float() * self.input_scale
        if x.shape[1] == N_CHANNELS:
            return x
        if channel_indices is None or int(self.channel_index.min()) < 0:
            raise ValueError(f"window features: {x.shape[1]} channels given and no channel table to place them")
        table = {int(c): i for i, c in enumerate(self.channel_index.tolist())}
        pos = [table[int(c)] for c in list(channel_indices)[1:]]
        full = x.new_zeros(x.shape[0], N_CHANNELS, x.shape[-1])
        full[:, pos] = x
        return full

    def forward(self, x4: torch.Tensor, channel_indices: Optional[Sequence[int]] = None,
                frozen: bool = False, stems: Optional[Sequence[str]] = None) -> torch.Tensor:
        """``stems``: the recording of every window (``mil.recording_features``); without them the
        recording block reads the training mean."""
        with torch.autocast(device_type=x4.device.type, enabled=False):
            x = self._signal(x4, channel_indices)
            feats = self.extract(x)
            with (torch.no_grad() if frozen else contextlib.nullcontext()):
                parts = [self.proj(feats)]
                if self.green is not None:
                    parts.append(self.green(self.extract.car(x)))
                if self.rec is not None:
                    raw = (self.rec.lookup(stems, x.device) if stems is not None
                           else torch.full((x.shape[0], self.rec.n_in), float("nan"), device=x.device))
                    parts.append(self.rec(raw))
                out = torch.cat(parts, -1)
                if self.out_norm is not None:
                    out = self.out_norm(out)
        return self.drop(out)
