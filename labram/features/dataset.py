# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Loaders for the training-ready handcrafted-feature export.
# ---------------------------------------------------------
"""Read a ``flat_v1`` export (:mod:`labram.features.export`) for training.

Two entry points:

* :func:`load_flat` -> :class:`FlatFeatures` with ``X`` (memory-mapped
  ``[n, F]`` float32), ``meta`` and ``columns`` DataFrames, and the optional
  filter-bank covariances. Use it for scikit-learn style training::

      ff = load_flat(root, groups=["riemann", "psd_bins", "time", "cepstral"])
      dev = ff.meta.fold_age.ge(0).to_numpy()
      X, y = ff.design(), ff.meta.age.to_numpy()

* :class:`HandcraftedFeatureDataset`, a ``torch.utils.data.Dataset`` yielding
  ``{"x", "y", "index"}`` (plus ``"fbcov"``) per recording, with an optional
  standardiser fitted on a chosen subset of rows::

      ds = HandcraftedFeatureDataset(root, target="age", rows=ff.meta.fold_age.between(1, 4))
      mean, std = ds.fit_standardizer()
      val = HandcraftedFeatureDataset(root, target="age", rows=ff.meta.fold_age.eq(0), mean=mean, std=std)

The Riemann group is stored as raw covariances; :meth:`FlatFeatures.design`
returns them flattened after the per-channel columns, ready for
:class:`labram.features.modeling.FBTangentSpace` (fit it on training rows only).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FlatFeatures:
    X: np.ndarray            # [n, F] float32 (selected columns; memory-mapped when all are kept)
    meta: pd.DataFrame       # one row per recording
    columns: pd.DataFrame    # one row per column of X
    fbcov: np.ndarray | None  # [n, bands, C, C] float32, if requested
    info: dict

    def design(self) -> tuple[np.ndarray, int]:
        """``(X_design, n_flat)``: per-channel columns, then the flattened covariances.

        ``n_flat`` is the ``n_passthrough`` argument of ``FBTangentSpace``.
        Without covariances, ``X_design`` is just ``X``.
        """
        if self.fbcov is None:
            return np.asarray(self.X), self.X.shape[1]
        return np.concatenate([np.asarray(self.X), self.fbcov.reshape(len(self.fbcov), -1)], axis=1), self.X.shape[1]


def load_flat(root: str, groups=None, families=None, with_fbcov: bool = None, mmap: bool = True) -> FlatFeatures:
    """Load a ``flat_v1`` directory.

    ``groups`` (e.g. ``["riemann", "psd_bins", "time", "cepstral"]``) or
    ``families`` select columns; ``"riemann"`` in ``groups`` loads the
    covariances (``with_fbcov`` defaults to that).
    """
    with open(os.path.join(root, "info.json")) as fh:
        info = json.load(fh)
    meta = pd.read_parquet(os.path.join(root, "meta.parquet"))
    cols = pd.read_parquet(os.path.join(root, "columns.parquet"))
    X = np.load(os.path.join(root, "features.npy"), mmap_mode="r" if mmap else None)
    if with_fbcov is None:
        with_fbcov = groups is not None and "riemann" in groups
    flat_groups = None if groups is None else [g for g in groups if g != "riemann"]
    keep = np.ones(len(cols), dtype=bool)
    if flat_groups is not None:
        unknown = set(flat_groups) - set(cols.group)
        if unknown:
            raise KeyError(f"groups not in this export: {sorted(unknown)}")
        keep &= cols.group.isin(flat_groups).to_numpy()
    if families is not None:
        keep &= cols.family.isin(families).to_numpy()
    if not keep.all():
        X = np.ascontiguousarray(X[:, keep])
        cols = cols[keep].reset_index(drop=True)
    fb = np.load(os.path.join(root, "fbcov.npy"), mmap_mode="r" if mmap else None) if with_fbcov else None
    return FlatFeatures(X, meta, cols, fb, info)


try:  # torch is optional for the scikit-learn path
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover
    torch, _TorchDataset = None, object


class HandcraftedFeatureDataset(_TorchDataset):
    """One item per recording: ``{"x": [F], "y": scalar, "index": row}`` (+ ``"fbcov"``).

    ``target`` is a ``meta`` column (``"age"``, ``"abnormal"``, ...); rows with a
    missing target are dropped. ``groups`` / ``families`` / ``with_fbcov`` are
    passed to :func:`load_flat` (``"riemann"`` in ``groups`` adds ``"fbcov"``). ``rows`` is a boolean mask / index array over
    the export's rows. ``mean``/``std`` standardise ``x`` (see
    :meth:`fit_standardizer`).
    """

    def __init__(self, root: str, target: str = "age", rows=None, groups=None, families=None,
                 with_fbcov: bool = None, mean=None, std=None):
        if torch is None:  # pragma: no cover
            raise ImportError("HandcraftedFeatureDataset needs torch")
        self.ff = load_flat(root, groups=groups, families=families, with_fbcov=with_fbcov)
        y = pd.to_numeric(self.ff.meta[target], errors="coerce").to_numpy(dtype=np.float64)
        sel = np.ones(len(y), dtype=bool) if rows is None else np.zeros(len(y), dtype=bool)
        if rows is not None:
            rows = np.asarray(rows)
            sel[rows if rows.dtype != bool else np.where(rows)[0]] = True
        self.index = np.where(sel & np.isfinite(y))[0]
        self.y = y.astype(np.float32)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        for name, v in (("mean", self.mean), ("std", self.std)):
            if v is not None and v.shape != (self.ff.X.shape[1],):
                raise ValueError(f"{name} has shape {v.shape}, expected ({self.ff.X.shape[1]},) for the selected columns")

    def __len__(self) -> int:
        return len(self.index)

    def fit_standardizer(self, eps: float = 1e-6):
        """Mean/std over this dataset's rows; applied to its items from now on."""
        X = np.asarray(self.ff.X[self.index], dtype=np.float64)
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std = np.maximum(X.std(axis=0), eps).astype(np.float32)
        return self.mean, self.std

    def __getitem__(self, i: int) -> dict:
        r = int(self.index[i])
        x = np.asarray(self.ff.X[r], dtype=np.float32)
        if self.mean is not None:
            x = (x - self.mean) / self.std
        item = {"x": torch.from_numpy(x.copy()), "y": torch.tensor(self.y[r]), "index": r}
        if self.ff.fbcov is not None:
            item["fbcov"] = torch.from_numpy(np.asarray(self.ff.fbcov[r], dtype=np.float32).copy())
        return item
