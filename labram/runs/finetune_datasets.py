# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Per-task dataset registry for run_class_finetuning.
# ---------------------------------------------------------

from dataclasses import dataclass
from typing import List

import torch.utils.data

import labram.utils as utils


# Channel montage shared by TUAB and TUEV TUH-EEG corpora.
_TUH_EEG_CH_NAMES = [
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF',
    'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
    'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF',
    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF',
]


def _normalize_ch_names(raw_names: List[str]) -> List[str]:
    """Strip the 'EEG '/'-REF' decorations to match the standard 10-20 names."""
    return [name.split(' ')[-1].split('-')[0] for name in raw_names]


@dataclass
class DatasetBundle:
    train: torch.utils.data.Dataset
    val: torch.utils.data.Dataset
    test: torch.utils.data.Dataset
    ch_names: List[str]
    nb_classes: int
    metrics: List[str]


def get_dataset_bundle(dataset_name: str, data_path: str) -> DatasetBundle:
    if dataset_name == 'TUAB':
        root = data_path or "path/to/TUAB"
        train, test, val = utils.prepare_TUAB_dataset(root)
        return DatasetBundle(
            train=train, val=val, test=test,
            ch_names=_normalize_ch_names(_TUH_EEG_CH_NAMES),
            nb_classes=1,
            metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
        )
    if dataset_name == 'TUEV':
        root = data_path or "path/to/TUEV"
        train, test, val = utils.prepare_TUEV_dataset(root)
        return DatasetBundle(
            train=train, val=val, test=test,
            ch_names=_normalize_ch_names(_TUH_EEG_CH_NAMES),
            nb_classes=6,
            metrics=["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"],
        )
    raise ValueError(f"Unknown dataset: {dataset_name!r} (expected 'TUAB' or 'TUEV')")
