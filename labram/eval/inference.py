# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Inference helpers: run a fine-tuned model over a split and collect the raw
# per-window predictions (probabilities, logits, targets, case ids, file names)
# needed for offline evaluation, aggregation and visualization.
# ---------------------------------------------------------

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.utils.data
from einops import rearrange

import labram.utils as utils
from labram.models.outputs import PredictorOutput

logger = utils.get_logger(__name__)


@dataclass
class PredictionResult:
    """Raw per-window model output over a split.

    Attributes:
        probs: Positive-class probability ``(N,)`` for binary, or per-class
            softmax probabilities ``(N, C)`` for multiclass.
        logits: Raw model output ``(N, 1)`` / ``(N, C)`` (pre-sigmoid/softmax).
        targets: Integer labels ``(N,)``.
        groups: Per-window case id ``(N,)`` (recording / subject), or ``None``
            when the loader did not surface ids.
        files: Per-window source file name ``(N,)`` when known, else ``None``.
        is_binary: Whether the task is binary.
        nb_classes: Number of classes (1 for binary).
    """

    probs: np.ndarray
    logits: np.ndarray
    targets: np.ndarray
    groups: Optional[np.ndarray]
    files: Optional[np.ndarray]
    is_binary: bool
    nb_classes: int

    def __len__(self) -> int:
        return len(self.targets)

    @property
    def entropy(self) -> np.ndarray:
        """Normalized (``[0, 1]``) per-window prediction entropy."""
        return utils.prediction_entropy(self.probs, self.is_binary)

    @property
    def scores(self) -> np.ndarray:
        """Score array for metrics: positive-class prob (binary) / logits
        (multiclass, so the downstream softmax recovers the probabilities)."""
        return self.probs if self.is_binary else self.logits


def ordered_files(dataset) -> Optional[List[str]]:
    """Window file names in iteration order, unwrapping a ``Subset``.

    Returns ``None`` for datasets that do not expose a ``files`` list. Only
    valid when the loader iterates the dataset sequentially (no shuffling), as
    the evaluation loaders do.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        base = dataset.dataset
        files = getattr(base, 'files', None)
        if files is None:
            return None
        return [files[i] for i in dataset.indices]
    files = getattr(dataset, 'files', None)
    return list(files) if files is not None else None


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    is_binary: bool,
    nb_classes: int,
    ch_names: Optional[Sequence[str]] = None,
    files: Optional[Sequence[str]] = None,
) -> PredictionResult:
    """Run ``model`` over ``data_loader`` and gather raw per-window predictions.

    Mirrors the preprocessing of :func:`labram.train.train_finetune.evaluate`
    (scale by ``1/100``, reshape to ``[B, N, A, T=200]``) but returns the
    predictions rather than reducing them to metrics, so aggregation, threshold
    sweeps and per-window inspection can be done offline. The loader may yield
    2-tuples ``(X, y)`` or 3-tuples ``(X, y, case_id)`` (see
    :func:`labram.runs.finetune_setup.enable_window_ids`).

    ``files`` (window file names in loader order) is attached verbatim when
    given; use :func:`ordered_files` on the underlying dataset to obtain it.
    """
    channel_indices = None
    if ch_names is not None:
        channel_indices = utils.get_channel_indices(list(ch_names))

    model.eval()
    logits_all: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []
    true_all: List[np.ndarray] = []
    groups: List = []

    for batch in data_loader:
        eeg = batch[0]
        target = batch[1] if len(batch) >= 3 else batch[-1]
        group_batch = batch[2] if len(batch) >= 3 else None

        eeg = eeg.float().to(device, non_blocking=True) / 100
        eeg = rearrange(eeg, 'B N (A T) -> B N A T', T=200)

        with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
            output = model(eeg, channel_indices=channel_indices, classify_only=True)
            if isinstance(output, PredictorOutput):
                output = output.logits

        logits = output.float().cpu()
        if is_binary:
            probs = torch.sigmoid(logits).numpy().reshape(-1)
            logits = logits.numpy().reshape(-1, 1)
        else:
            probs = torch.softmax(logits, dim=-1).numpy()
            logits = logits.numpy()

        logits_all.append(logits)
        probs_all.append(probs)
        true_all.append(np.asarray(target).reshape(-1))
        if group_batch is not None:
            groups.extend(list(group_batch))

    logits_arr = np.concatenate(logits_all, axis=0) if logits_all else np.empty((0, nb_classes))
    probs_arr = np.concatenate(probs_all, axis=0) if probs_all else np.empty((0,))
    true_arr = np.concatenate(true_all, axis=0).astype(int) if true_all else np.empty((0,), dtype=int)
    groups_arr = np.asarray(groups, dtype=object) if groups else None

    files_arr = None
    if files is not None:
        files_arr = np.asarray(list(files), dtype=object)
        if len(files_arr) != len(true_arr):
            logger.warning(
                "files length (%d) != number of predictions (%d); dropping file names",
                len(files_arr), len(true_arr))
            files_arr = None

    logger.info("Collected %d window predictions (%d cases)", len(true_arr),
                len(set(groups)) if groups else len(true_arr))
    return PredictionResult(
        probs=probs_arr, logits=logits_arr, targets=true_arr,
        groups=groups_arr, files=files_arr,
        is_binary=is_binary, nb_classes=nb_classes)
