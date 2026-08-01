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

    @property
    def is_redacted(self) -> bool:
        return self.raw_age == REDACTED_AGE


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
    path = find_metadata_sidecar(root, filename=filename)
    if path is None:
        raise FileNotFoundError(
            f"No {filename} found in {root} or any parent directory. Generate it with:\n"
            f"  python -m dataset_maker.make_TUAB_age scan --root <corpus edf dir>"
        )
    return age_lookup(load_metadata_sidecar(path))


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
