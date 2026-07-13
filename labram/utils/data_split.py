# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Record which EEG cases/windows were assigned to train/val/test, so an
# experiment's split is reproducible and auditable (saved to disk + ClearML).
# ---------------------------------------------------------

import json
import os
from typing import Any, Dict, List, Optional

import torch.utils.data

from labram.utils.logging import get_logger

logger = get_logger(__name__)


def _unwrap(dataset):
    """Return (base_dataset, files) resolving a Subset to its selected files."""
    if dataset is None:
        return None, None
    if isinstance(dataset, torch.utils.data.Subset):
        base = dataset.dataset
        files = getattr(base, 'files', None)
        if files is not None:
            files = [files[i] for i in dataset.indices]
        return base, files
    return dataset, getattr(dataset, 'files', None)


def _recording_id(base, filename: str) -> str:
    sep = getattr(base, '_recording_sep', '_')
    b = filename[:-4] if filename.endswith('.pkl') else filename
    return b.rsplit(sep, 1)[0]


def _subject_id(filename: str) -> str:
    b = filename[:-4] if filename.endswith('.pkl') else filename
    return b.split('_')[0]


def _split_entry(dataset) -> Optional[Dict[str, Any]]:
    base, files = _unwrap(dataset)
    if files is None:
        return None
    files = sorted(files)
    recordings = sorted({_recording_id(base, f) for f in files})
    subjects = sorted({_subject_id(f) for f in files})
    return {
        'n_windows': len(files),
        'n_recordings': len(recordings),
        'n_subjects': len(subjects),
        'recordings': recordings,
        'subjects': subjects,
        'files': files,
    }


def build_data_split(dataset_train, dataset_val, dataset_test,
                     dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """Summarize the train/val/test assignment: per-split window/recording/
    subject counts and identifiers, plus a subject-overlap (leakage) check."""
    split: Dict[str, Any] = {'dataset': dataset_name}
    entries = {}
    for name, ds in (('train', dataset_train), ('val', dataset_val), ('test', dataset_test)):
        entry = _split_entry(ds)
        if entry is not None:
            entries[name] = entry
            split[name] = entry
    # Data-leakage guard: a subject appearing in more than one split is usually a bug.
    subj = {k: set(v['subjects']) for k, v in entries.items()}
    overlap = {}
    names = list(subj)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            common = sorted(subj[names[i]] & subj[names[j]])
            if common:
                overlap[f'{names[i]}∩{names[j]}'] = common
    split['subject_overlap'] = overlap
    return split


def save_data_split(split: Dict[str, Any], output_dir: str,
                    filename: str = 'data_split.json') -> Optional[str]:
    """Write the split record to ``output_dir/filename``; return the path."""
    if not output_dir:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    counts = {k: v.get('n_windows') for k, v in split.items() if isinstance(v, dict) and 'n_windows' in v}
    logger.info("Saved data split (%s) -> %s", counts, path)
    if split.get('subject_overlap'):
        logger.warning("Subject overlap between splits detected: %s",
                       {k: len(v) for k, v in split['subject_overlap'].items()})
    return path
