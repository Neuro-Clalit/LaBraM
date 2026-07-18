"""Cross-validation configuration for fine-tuning.

``CrossValidationConfig`` turns a normal fine-tune run into a K-fold
cross-validation study. The data pool (train+val, or all splits) is partitioned
into ``n_folds`` group-disjoint folds so the same subject/recording never
straddles train/val/test. Each fold is trained as its own sub-experiment whose
name/output folder embeds the fold number, and the fold partition is recorded to
a ``cv_split.json`` artifact for reproducibility.

The single switch is ``config.cross_validation`` on :class:`FinetuneRunConfig`;
disabled by default so existing fine-tune runs are unaffected.
"""

from dataclasses import dataclass

from labram.configs.base_configs import ConfigBase
from labram.configs.defaults import (
    DEFAULT_CV_BASE_DIR,
    DEFAULT_CV_ENABLED,
    DEFAULT_CV_FOLD,
    DEFAULT_CV_N_FOLDS,
    DEFAULT_CV_POOL,
    DEFAULT_CV_SEED,
    DEFAULT_CV_SHUFFLE,
    DEFAULT_CV_SPLIT_BY,
    DEFAULT_CV_SPLIT_JSON,
)


@dataclass
class CrossValidationConfig(ConfigBase):
    """K-fold cross-validation settings for fine-tuning.

    Attributes:
        enabled: Master switch. When False the fine-tune runner behaves exactly
            as before (single train/val/test run).
        n_folds: Number of folds K.
        fold: Which fold to train. ``-1`` (default) makes the CV runner iterate
            over every fold in-process; ``>=0`` trains only that single fold —
            used when each fold is dispatched as its own process / SageMaker job.
        split_by: How a "group" is defined so it never straddles splits —
            ``subject`` (default) / ``recording`` / ``window``.
        shuffle: Shuffle groups before partitioning into folds.
        seed: RNG seed for the (deterministic) fold assignment.
        pool: Which data is re-partitioned — ``train_val`` (keep the original
            test set untouched, cross-validate over train+val) or ``all``
            (pool train+val+test).
        split_json: Path (local or s3://) to a precomputed ``cv_split.json`` to
            reuse. Empty -> compute the folds from (split_by, seed, shuffle).
        base_dir: Base experiment/output folder for the per-fold sub-runs.
            Empty -> derived from the fine-tune ``output.output_dir``. Each fold
            is written to ``<base_dir>/fold_<k>``.
    """

    enabled: bool = DEFAULT_CV_ENABLED
    n_folds: int = DEFAULT_CV_N_FOLDS
    fold: int = DEFAULT_CV_FOLD
    split_by: str = DEFAULT_CV_SPLIT_BY
    shuffle: bool = DEFAULT_CV_SHUFFLE
    seed: int = DEFAULT_CV_SEED
    pool: str = DEFAULT_CV_POOL
    split_json: str = DEFAULT_CV_SPLIT_JSON
    base_dir: str = DEFAULT_CV_BASE_DIR

    def validate(self) -> None:
        """Guard against misconfiguration that would silently mis-split data."""
        if not self.enabled:
            return
        if self.n_folds < 2:
            raise ValueError(f"cross_validation.n_folds must be >= 2, got {self.n_folds}")
        if self.fold >= self.n_folds:
            raise ValueError(
                f"cross_validation.fold={self.fold} is out of range for "
                f"n_folds={self.n_folds} (valid: 0..{self.n_folds - 1}, or -1 for all)")
        if self.split_by not in ('subject', 'recording', 'window'):
            raise ValueError(
                f"cross_validation.split_by must be subject|recording|window, got {self.split_by!r}")
        if self.pool not in ('train_val', 'all'):
            raise ValueError(
                f"cross_validation.pool must be train_val|all, got {self.pool!r}")
