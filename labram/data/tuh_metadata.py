# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Patient metadata (age / sex) for the TUH EEG corpora, read straight from the
# EDF headers. TUH no longer distributes the clinical reports, so the header is
# the only source of demographics. See docs/age_regression.md.
# ---------------------------------------------------------

import json
import logging
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from glob import iglob
from typing import Dict, Iterable, Optional, Tuple

# Plain stdlib logging, not labram.utils.get_logger: labram.utils imports
# labram.data, so a labram.utils import here would be circular.
logger = logging.getLogger(__name__)

# An EDF header starts with a fixed 256-byte block. Two of its ASCII fields
# carry the demographics TUH fills in:
#
#   bytes 8:88   "local patient identification"
#                'aaaaantl F 01-JAN-0000 aaaaantl Age:42'
#                 subject   sex dob      subject   age
#   bytes 88:168 "local recording identification"
#                'Startdate 01-JAN-2012 aaaaantl_s001 XXX X'
#
# The date of birth is anonymised to 01-JAN-0000, which is why MNE reports no
# age for these files (``raw.info['subject_info']`` yields only his_id/sex/
# last_name): there is nothing to subtract the birth year from. The literal
# ``Age:`` token is the only usable age, and it must be read from the raw bytes.
EDF_HEADER_BYTES = 256
_PATIENT_FIELD = slice(8, 88)
_RECORDING_FIELD = slice(88, 168)

_AGE_RE = re.compile(r"Age:(\d+)")
_SEX_RE = re.compile(r"^\s*\S+\s+([MF])\b")
_YEAR_RE = re.compile(r"Startdate\s+\d{2}-[A-Z]{3}-(\d{4})")
# Filenames are '<subject>_s<NNN>_t<NNN>.edf' across every TUH corpus.
_STEM_RE = re.compile(r"^(?P<subject>[^_]+)_(?P<session>s\d+)_(?P<token>t\d+)$")

# ``Age:999`` is TUH's redaction for patients aged 90+ (HIPAA requires ages over
# 89 to be aggregated). Verified on TUAB v3.0.0: the 12 files carrying 999 match
# the AAREADME demographics table's "90-100" row exactly, per split and per
# normal/abnormal label. ``Age:0`` is ambiguous -- it may be a genuine neonate or
# a missing value -- so it is excluded by the default 1..89 range rather than
# being named a sentinel.
REDACTED_AGE = 999

DEFAULT_MIN_AGE = 1
DEFAULT_MAX_AGE = 89

SIDECAR_FILENAME = "age_metadata.json"


@dataclass(frozen=True)
class RecordingMetadata:
    """Demographics for one TUH recording (one EDF file).

    ``age`` is ``None`` when ``raw_age`` falls outside the accepted range, so
    callers can never silently train on a sentinel. ``raw_age`` keeps the
    as-parsed value for auditing.
    """

    stem: str
    subject: str
    session: Optional[str]
    token: Optional[str]
    age: Optional[int]
    sex: Optional[str]
    year: Optional[int]
    raw_age: Optional[int]
    # TUAB's ``normal`` / ``abnormal`` folder the EDF sits in (None when the
    # path names neither, e.g. a non-TUAB corpus or a pre-label sidecar).
    label: Optional[str] = None

    @property
    def is_redacted(self) -> bool:
        return self.raw_age == REDACTED_AGE


CASE_LABELS = ("normal", "abnormal")


def label_from_path(path: str) -> Optional[str]:
    """TUAB's normal/abnormal label from an EDF path's folder names
    (``.../train/abnormal/01_tcp_ar/x.edf`` -> ``abnormal``)."""
    parts = os.path.normpath(path).split(os.sep)
    for label in CASE_LABELS:
        if label in parts:
            return label
    return None


def _split_stem(stem: str) -> Tuple[str, Optional[str], Optional[str]]:
    match = _STEM_RE.match(stem)
    if match:
        return match["subject"], match["session"], match["token"]
    # TUEG is heterogeneous; fall back to the leading token, which is what every
    # other grouping helper in the codebase uses as the subject id.
    return stem.split("_")[0], None, None


def parse_edf_header_metadata(
    path: str,
    *,
    min_age: int = DEFAULT_MIN_AGE,
    max_age: int = DEFAULT_MAX_AGE,
) -> RecordingMetadata:
    """Read demographics from an EDF file's 256-byte header block.

    Only the header is read, so this is cheap enough to sweep a corpus the size
    of TUEG. Malformed fields yield ``None`` attributes rather than raising --
    TUH's older recordings are not uniformly populated.
    """
    with open(path, "rb") as fh:
        header = fh.read(EDF_HEADER_BYTES)
    if len(header) < EDF_HEADER_BYTES:
        raise ValueError(f"{path}: truncated EDF header ({len(header)} bytes)")

    patient = header[_PATIENT_FIELD].decode("latin-1").strip()
    recording = header[_RECORDING_FIELD].decode("latin-1").strip()

    stem = os.path.splitext(os.path.basename(path))[0]
    subject, session, token = _split_stem(stem)

    age_match = _AGE_RE.search(patient)
    raw_age = int(age_match.group(1)) if age_match else None
    age = raw_age if raw_age is not None and min_age <= raw_age <= max_age else None

    sex_match = _SEX_RE.match(patient)
    year_match = _YEAR_RE.search(recording)

    return RecordingMetadata(
        stem=stem,
        subject=subject,
        session=session,
        token=token,
        age=age,
        sex=sex_match.group(1) if sex_match else None,
        year=int(year_match.group(1)) if year_match else None,
        raw_age=raw_age,
        label=label_from_path(path),
    )


def scan_corpus_metadata(
    root: str,
    *,
    pattern: str = "**/*.edf",
    min_age: int = DEFAULT_MIN_AGE,
    max_age: int = DEFAULT_MAX_AGE,
) -> Dict[str, RecordingMetadata]:
    """Parse every EDF header under *root*, keyed by recording stem.

    The stem (``<subject>_s<NNN>_t<NNN>``) is the join key against the
    preprocessed window pickles, whose names are ``<stem>_<window index>.pkl``.
    """
    found: Dict[str, RecordingMetadata] = {}
    failures = 0
    for path in iglob(os.path.join(root, pattern), recursive=True):
        try:
            meta = parse_edf_header_metadata(path, min_age=min_age, max_age=max_age)
        except (OSError, ValueError) as exc:
            failures += 1
            logger.warning("Could not read EDF header %s: %s", path, exc)
            continue
        if meta.stem in found and found[meta.stem] != meta:
            logger.warning("Duplicate recording stem %s with differing metadata", meta.stem)
        found[meta.stem] = meta
    if failures:
        logger.warning("Failed to parse %d EDF header(s) under %s", failures, root)
    if not found:
        raise FileNotFoundError(f"No EDF files matched {pattern!r} under {root}")
    return found


def summarize_metadata(meta: Dict[str, RecordingMetadata]) -> Dict[str, object]:
    """Counts used to sanity-check a corpus scan against its AAREADME tables."""
    usable = [m for m in meta.values() if m.age is not None]
    ages = [m.age for m in usable]
    with_raw_age = sum(1 for m in meta.values() if m.raw_age is not None)
    decades = Counter(min(age // 10 * 10, 90) for age in ages)
    mean = sum(ages) / len(ages) if ages else 0.0
    var = sum((a - mean) ** 2 for a in ages) / (len(ages) - 1) if len(ages) > 1 else 0.0
    return {
        "recordings": len(meta),
        "with_age_field": with_raw_age,
        "usable_ages": len(usable),
        "redacted_90_plus": sum(1 for m in meta.values() if m.is_redacted),
        "excluded_out_of_range": len(meta) - len(usable),
        "subjects": len({m.subject for m in meta.values()}),
        "age_mean": round(mean, 1),
        "age_std": round(var ** 0.5, 1),
        "age_min": min(ages) if ages else None,
        "age_max": max(ages) if ages else None,
        "sex_counts": dict(Counter(m.sex for m in meta.values())),
        "label_counts": dict(Counter(m.label for m in meta.values())),
        "age_decades": {f"{d}-{d + 10}": decades[d] for d in sorted(decades)},
    }


def save_metadata_sidecar(meta: Dict[str, RecordingMetadata], path: str) -> None:
    """Write the scan to JSON so training never needs the raw EDFs again."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "version": 1,
        "recordings": {stem: asdict(m) for stem, m in sorted(meta.items())},
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    logger.info("Wrote metadata for %d recording(s) to %s", len(meta), path)


def load_metadata_sidecar(path: str) -> Dict[str, RecordingMetadata]:
    with open(path) as fh:
        payload = json.load(fh)
    records = payload["recordings"] if "recordings" in payload else payload
    return {stem: RecordingMetadata(**fields) for stem, fields in records.items()}


def age_lookup(meta: Dict[str, RecordingMetadata]) -> Dict[str, float]:
    """``stem -> age in years`` for recordings with a usable age."""
    return {stem: float(m.age) for stem, m in meta.items() if m.age is not None}


def label_lookup(meta: Dict[str, RecordingMetadata]) -> Dict[str, str]:
    """``stem -> 'normal' | 'abnormal'`` for recordings with a known label."""
    return {stem: m.label for stem, m in meta.items() if m.label is not None}


def find_metadata_sidecar(start: str, *, filename: str = SIDECAR_FILENAME) -> Optional[str]:
    """Search *start* and its parents for a metadata sidecar.

    Lets a dataset loader recover its age labels from nothing but its data
    directory, which is what keeps cross-validation's positional loader
    reconstruction working.
    """
    current = os.path.abspath(start)
    while True:
        candidate = os.path.join(current, filename)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def load_age_lookup_for(root: str, *, filename: str = SIDECAR_FILENAME) -> Dict[str, float]:
    """Resolve and load the age lookup covering the window directory *root*."""
    # A non-existent root walks straight up to '/' and reports a missing
    # sidecar, which reads as "run the scan step" when the real fault is a
    # mistyped data_path. Name that case separately.
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Data directory does not exist: {root}\n"
            f"  Check data.data_path -- it should point at the corpus directory "
            f"holding the processed window pickles."
        )
    path = find_metadata_sidecar(root, filename=filename)
    if path is None:
        raise FileNotFoundError(
            f"No {filename} found in {root} or any parent directory. Generate it with:\n"
            f"  python -m dataset_maker.make_TUAB_age scan --root <corpus edf dir>"
        )
    return age_lookup(load_metadata_sidecar(path))


def load_label_lookup_for(root: str, *, filename: str = SIDECAR_FILENAME) -> Dict[str, str]:
    """Resolve and load the normal/abnormal labels covering *root*.

    Raises when the sidecar predates the ``label`` field, since an empty lookup
    would otherwise silently filter every recording out."""
    path = find_metadata_sidecar(root, filename=filename)
    if path is None:
        raise FileNotFoundError(f"No {filename} found in {root} or any parent directory.")
    labels = label_lookup(load_metadata_sidecar(path))
    if not labels:
        raise ValueError(
            f"{path} has no normal/abnormal labels (written before the 'label' field "
            f"existed). Regenerate it with:\n"
            f"  python dataset_maker/make_TUAB_age.py scan --root <corpus edf dir>")
    return labels


def recording_stem(filename: str, *, sep: str = "_") -> str:
    """Recording stem of a window file (``<stem><sep><window index>.pkl``).

    Accepts a bare filename or a path relative to the window root -- the age
    split stores files as ``<subdir>/<name>.pkl`` so that a split pooled from two
    directories is still a single loader.
    """
    base = os.path.basename(filename)
    if base.endswith(".pkl"):
        base = base[:-4]
    return base.rsplit(sep, 1)[0]


def filter_files_with_age(
    files: Iterable[str],
    ages: Dict[str, float],
    *,
    sep: str = "_",
) -> list:
    """Keep only window files whose recording has a usable age.

    Filtering up front matters because ``TUHLoader.__getitem__`` swallows
    ``KeyError`` and silently substitutes a different window -- a missing age
    would otherwise corrupt the targets invisibly.
    """
    return [f for f in files if recording_stem(f, sep=sep) in ages]


# ------------------------------------------------------------------------------
# Whole-corpus scan: signal header, montage and per-corpus labels.
#
# TUEG mixes "over 40 channel configurations" and 250-1024 Hz sampling, so the
# demographics alone do not say whether a recording can feed the 23-channel
# model. The rest of the EDF header (one 256-byte block per signal) does: labels,
# samples per data record, record count and length -- still header-only, so a
# 70k-file corpus scans in minutes.

# The 23 channels the TUAB pipeline keeps, in model order (FP1 ... T2).
TUH_CHANNELS = ("FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
                "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ", "T1", "T2")
_TUH_CHANNEL_SET = frozenset(TUH_CHANNELS)
# 'EEG FP1-REF' (average reference) / 'EEG FP1-LE' (linked ears).
_CHANNEL_RE = re.compile(r"^EEG\s+(?P<name>[A-Z0-9]+)-(?:REF|LE)$")
# '01_tcp_ar', '02_tcp_le', '03_tcp_ar_a' (no A1/A2), '04_tcp_le_a'.
_MONTAGE_RE = re.compile(r"^\d{2}_tcp_(?:ar|le)(?:_a)?$")
# TUEG session dirs carry the date: 's001_2015_12_30'.
_SESSION_DIR_RE = re.compile(r"^s\d+_(\d{4})_(\d{2})_(\d{2})$")
# TUEP's subject-level label: '00_epilepsy' / '01_no_epilepsy'.
_EPILEPSY_DIRS = {"00_epilepsy": 1, "01_no_epilepsy": 0}


@dataclass(frozen=True)
class EdfSignalHeader:
    labels: Tuple[str, ...]
    sfreqs: Tuple[float, ...]
    n_records: int
    record_duration: float

    @property
    def duration_sec(self) -> float:
        return self.n_records * self.record_duration


def read_edf_signal_header(path: str) -> EdfSignalHeader:
    """Signal labels, per-signal sampling rate and duration from an EDF header."""
    with open(path, "rb") as fh:
        head = fh.read(EDF_HEADER_BYTES)
        if len(head) < EDF_HEADER_BYTES:
            raise ValueError(f"{path}: truncated EDF header ({len(head)} bytes)")
        n_signals = int(head[252:256])
        block = fh.read(n_signals * 256)
    if len(block) < n_signals * 256:
        raise ValueError(f"{path}: truncated signal header")
    n_records = int(head[236:244])
    record_duration = float(head[244:252])

    def field(offset, width):
        start = offset * n_signals
        return [block[start + i * width:start + (i + 1) * width].decode("latin-1").strip()
                for i in range(n_signals)]

    # Per-signal fields, each stored for all signals before the next one:
    # label 16, transducer 80, dimension 8, phys min/max 8+8, dig min/max 8+8,
    # prefiltering 80, samples per record 8, reserved 32.
    labels = field(0, 16)
    samples = [int(s) for s in field(16 + 80 + 8 * 5 + 80, 8)]
    sfreqs = tuple(s / record_duration if record_duration > 0 else 0.0 for s in samples)
    return EdfSignalHeader(tuple(labels), sfreqs, n_records, record_duration)


def canonical_channel(label: str) -> Optional[str]:
    """'EEG FP1-REF' / 'EEG FP1-LE' -> 'FP1' if it is one of the 23 TUH channels."""
    match = _CHANNEL_RE.match(label.strip().upper())
    if match and match["name"] in _TUH_CHANNEL_SET:
        return match["name"]
    return None


def channel_map(labels: Iterable[str]) -> Dict[str, str]:
    """``{canonical name: raw EDF label}`` for the TUH channels present (first wins)."""
    found: Dict[str, str] = {}
    for label in labels:
        name = canonical_channel(label)
        if name is not None and name not in found:
            found[name] = label
    return found


def path_component(path: str, pattern: "re.Pattern") -> Optional[str]:
    for part in os.path.normpath(path).split(os.sep):
        if pattern.match(part):
            return part
    return None


def corpus_labels(path: str) -> Dict[str, Optional[int]]:
    """Pathology labels encoded in a TUH path: TUAB's normal/abnormal folder and
    TUEP's epilepsy/no-epilepsy folder (None where the path carries neither)."""
    case = label_from_path(path)
    parts = os.path.normpath(path).split(os.sep)
    epilepsy = next((_EPILEPSY_DIRS[p] for p in parts if p in _EPILEPSY_DIRS), None)
    return {"abnormal": None if case is None else int(case == "abnormal"),
            "epilepsy": epilepsy}


def recording_row(path: str, root: str, corpus: str, *,
                  min_age: int = DEFAULT_MIN_AGE, max_age: int = DEFAULT_MAX_AGE) -> Dict[str, object]:
    """One flat, JSON-able index row for an EDF file (header-only read)."""
    meta = parse_edf_header_metadata(path, min_age=min_age, max_age=max_age)
    sig = read_edf_signal_header(path)
    present = channel_map(sig.labels)
    eeg_sfreqs = [sig.sfreqs[sig.labels.index(raw)] for raw in present.values()]
    session_dir = path_component(path, _SESSION_DIR_RE)
    date = None
    if session_dir:
        y, m, d = _SESSION_DIR_RE.match(session_dir).groups()
        date = f"{y}-{m}-{d}"
    row = asdict(meta)
    row.update(
        corpus=corpus,
        path=os.path.relpath(path, root),
        montage=path_component(path, _MONTAGE_RE),
        session_date=date,
        sfreq=max(eeg_sfreqs) if eeg_sfreqs else None,
        mixed_sfreq=len(set(eeg_sfreqs)) > 1,
        n_signals=len(sig.labels),
        duration_sec=round(sig.duration_sec, 3),
        missing_channels=[c for c in TUH_CHANNELS if c not in present],
        **corpus_labels(path),
    )
    return row


def scan_recordings(root: str, *, corpus: str, pattern: str = "**/*.edf",
                    min_age: int = DEFAULT_MIN_AGE, max_age: int = DEFAULT_MAX_AGE) -> list:
    """``recording_row`` for every EDF under *root*; unreadable files are logged
    and returned as rows with an ``error`` field so they stay countable."""
    rows = []
    for path in sorted(iglob(os.path.join(root, pattern), recursive=True)):
        try:
            rows.append(recording_row(path, root, corpus, min_age=min_age, max_age=max_age))
        except (OSError, ValueError) as exc:
            logger.warning("Could not read EDF header %s: %s", path, exc)
            rows.append({"corpus": corpus, "path": os.path.relpath(path, root),
                         "stem": os.path.splitext(os.path.basename(path))[0], "error": str(exc)})
    return rows
