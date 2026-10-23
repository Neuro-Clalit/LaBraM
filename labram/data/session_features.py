# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Session-level classical EEG features, one row per recording, for
# model.session_features (extra transformer tokens, or the only model input).
# ---------------------------------------------------------
"""Read a session-feature file written by ``dataset_maker/make_session_features.py``.

The ``.npz`` holds ``stems`` [N] and, per feature group ``g`` listed in
``groups``, ``tokens/<g>`` [N, n_g, dim_g] float32, already standardized with
training-split statistics. A model sees one recording's groups flattened and
concatenated in ``groups`` order (``matrix`` row), and splits them back into
``[n_g, dim_g]`` tokens using :attr:`SessionFeatures.groups`.
"""
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

_CACHE: Dict[str, "SessionFeatures"] = {}


@dataclass
class SessionFeatures:
    rows: Dict[str, int]                # recording stem -> row of ``matrix``
    matrix: np.ndarray                  # [N, sum n_g * dim_g] float32
    groups: List[List[int]]             # [[n_g, dim_g], ...]
    group_names: List[str]


def load_session_features(path: str) -> SessionFeatures:
    key = os.path.abspath(path)
    if key not in _CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"session features {path} not found; build them with "
                f"`python -m dataset_maker.make_session_features`")
        with np.load(path) as z:
            names = [str(g) for g in z["groups"]]
            blocks = [z[f"tokens/{g}"].astype(np.float32) for g in names]
            stems = [str(s) for s in z["stems"]]
        n = len(stems)
        _CACHE[key] = SessionFeatures(
            rows={s: i for i, s in enumerate(stems)},
            matrix=np.ascontiguousarray(np.concatenate([b.reshape(n, -1) for b in blocks], axis=1)),
            groups=[[int(b.shape[1]), int(b.shape[2])] for b in blocks],
            group_names=names)
    return _CACHE[key]
