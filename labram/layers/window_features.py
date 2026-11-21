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
                 shrink: float = 1e-3):
        super().__init__()
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
        self.n_features = self.n_core + self.n_cov + self.n_conn
        # fitted on training windows, saved with the checkpoint
        self.register_buffer("whiten", torch.eye(m, dtype=torch.float64).repeat(len(FILTERBANK), 1, 1))
        self.register_buffer("mean", torch.zeros(self.n_features))
        self.register_buffer("std", torch.ones(self.n_features))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    # ------------------------------------------------------------ pieces
    def _spectra(self, x):
        """Common average reference, then Welch segments: ``(x_car, psd [N,C,F], csd [N,C,C,F])``."""
        present = x.abs().amax(-1, keepdim=True) > 0                    # an absent channel is stored as zeros
        n = present.sum(1, keepdim=True).clamp_min(1)
        x = (x - (x * present).sum(1, keepdim=True) / n) * present
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
        """Fit the tangent-space references and the standardization on windows ``x [M, 19, L]`` (µV)."""
        core, cov, conn = self.raw(x)
        pick = torch.randperm(cov.shape[0], generator=torch.Generator().manual_seed(0))[:max_reference].to(cov.device)
        for b in range(cov.shape[1]):
            self.whiten[b] = _sym_fn(riemann_mean(cov[pick, b]), lambda w: w.clamp_min(EPS).rsqrt())
        feats = torch.nan_to_num(torch.cat([core, self._tangent(cov), conn], 1), nan=0.0, posinf=0.0, neginf=0.0)
        self.mean.copy_(feats.mean(0))
        self.std.copy_(feats.std(0).clamp_min(1e-6))
        self.fitted.fill_(True)
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        core, cov, conn = self.raw(x)
        feats = torch.cat([core, self._tangent(cov), conn], 1)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return ((feats - self.mean) / self.std).clamp(-6.0, 6.0)


class WindowFeatureBranch(nn.Module):
    """The model's second window input: handcrafted features -> ``dim`` (trained projection, dropout).

    ``forward`` takes the model input ``[N, C, A, T]`` (µV / ``input_scale``); with fewer than 19
    channels (a window encoded without its absent channels) ``channel_indices`` places them.
    """

    def __init__(self, dim: int = 64, dropout: float = 0.1, input_scale: float = 100.0):
        super().__init__()
        self.extract = HandcraftedWindowFeatures()
        self.proj = nn.Linear(self.extract.n_features, dim)
        nn.init.normal_(self.proj.weight, std=self.extract.n_features ** -0.5)    # unit-scale output at the start
        nn.init.zeros_(self.proj.bias)
        self.drop = nn.Dropout(dropout)
        self.out_dim, self.input_scale = dim, input_scale
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
                frozen: bool = False) -> torch.Tensor:
        with torch.autocast(device_type=x4.device.type, enabled=False):
            feats = self.extract(self._signal(x4, channel_indices))
            with (torch.no_grad() if frozen else contextlib.nullcontext()):
                out = self.proj(feats)
        return self.drop(out)
