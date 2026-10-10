# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Riemannian (affine-invariant) mean and tangent-space mapping of SPD covariances.
# ---------------------------------------------------------
"""Minimal Riemannian-geometry helpers for covariance features.

The tangent-space projection of Barachant et al. (2012), as used for
filter-bank brain-age models by Sabbagh et al. (2020) / Engemann et al. (2022):
covariances are whitened by the reference (Riemannian mean) ``C_ref``,
mapped through the matrix logarithm, and vectorised (upper triangle,
off-diagonal scaled by sqrt(2) so the Euclidean norm of the vector equals the
Riemannian distance to ``C_ref``). Kept dependency-free (NumPy only).
"""

from __future__ import annotations

import numpy as np


def _eig_fn(c: np.ndarray, fn) -> np.ndarray:
    """Apply a scalar function to the eigenvalues of symmetric matrices ``[..., n, n]``."""
    w, v = np.linalg.eigh(c)
    return (v * fn(w)[..., None, :]) @ np.swapaxes(v, -1, -2)


def mean_riemann(covs: np.ndarray, tol: float = 1e-8, max_iter: int = 50) -> np.ndarray:
    """Affine-invariant Riemannian (Karcher) mean of SPD matrices ``[N, n, n]``.

    Fixed-point iteration from the Euclidean mean: average the matrices in the
    tangent space at the current estimate and map back, until the step is
    below ``tol``.
    """
    ref = covs.mean(axis=0)
    for _ in range(max_iter):
        s = _eig_fn(ref, np.sqrt)
        si = _eig_fn(ref, lambda w: 1.0 / np.sqrt(w))
        t = _eig_fn(si @ covs @ si, np.log).mean(axis=0)
        ref = s @ _eig_fn(t, np.exp) @ s
        if np.linalg.norm(t) < tol:
            break
    return ref


def tangent_space(covs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Map SPD matrices ``[N, n, n]`` to tangent vectors ``[N, n(n+1)/2]`` at ``ref``."""
    si = _eig_fn(ref, lambda w: 1.0 / np.sqrt(w))
    logm = _eig_fn(si @ covs @ si, np.log)
    n = ref.shape[-1]
    iu = np.triu_indices(n)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return logm[:, iu[0], iu[1]] * coef
