# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Build the age-regression labels and splits for a TUH corpus.
#
# Unlike make_TUAB.py this does NOT touch the EEG signals: the age lives in the
# EDF headers, and the window pickles make_TUAB.py already produced are joined to
# it by filename. So both subcommands are read-only with respect to the pickles
# and take seconds, not hours.
#
#   scan  -- parse every EDF header into processed/age_metadata.json
#   split -- write a subject-disjoint processed/age_split.json
#
# See docs/age_regression.md.
# ---------------------------------------------------------

import argparse
import json
import logging
import os
import sys

from labram.data.age_splits import (
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    SPLIT_FILENAME,
    build_age_split,
    save_age_split,
)
from labram.data.tuh_metadata import (
    DEFAULT_MAX_AGE,
    DEFAULT_MIN_AGE,
    SIDECAR_FILENAME,
    age_lookup,
    load_metadata_sidecar,
    save_metadata_sidecar,
    scan_corpus_metadata,
    summarize_metadata,
)

logger = logging.getLogger("make_TUAB_age")


def _processed_dir(root: str) -> str:
    """The window directory: ``<root>/processed`` when it exists, else ``root``."""
    candidate = os.path.join(root, "processed")
    return candidate if os.path.isdir(candidate) else root


def cmd_scan(args: argparse.Namespace) -> int:
    meta = scan_corpus_metadata(
        args.root, pattern=args.pattern, min_age=args.min_age, max_age=args.max_age)
    out = args.out or os.path.join(_processed_dir(args.root), SIDECAR_FILENAME)
    save_metadata_sidecar(meta, out)

    summary = summarize_metadata(meta)
    print(f"\nScanned {args.root}")
    print(json.dumps(summary, indent=2))
    coverage = 100.0 * summary["with_age_field"] / max(1, summary["recordings"])
    print(f"\nAge field present in {coverage:.1f}% of recordings; "
          f"{summary['usable_ages']} usable in [{args.min_age}, {args.max_age}].")
    if summary["redacted_90_plus"]:
        print(f"{summary['redacted_90_plus']} recording(s) carry Age:999 "
              f"(TUH's redaction for 90+), excluded.")
    print(f"\nCross-check the age_decades and sex_counts above against the "
          f"DEMOGRAPHICS tables in the corpus AAREADME.txt.")
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    processed = _processed_dir(args.root)
    sidecar = args.metadata or os.path.join(processed, SIDECAR_FILENAME)
    if not os.path.isfile(sidecar):
        logger.error("No metadata sidecar at %s -- run the 'scan' subcommand first.", sidecar)
        return 1

    ages = age_lookup(load_metadata_sidecar(sidecar))
    split = build_age_split(
        processed, ages, val_fraction=args.val_fraction, seed=args.seed)
    out = args.out or os.path.join(processed, SPLIT_FILENAME)
    save_age_split(split, out)

    print(f"\nSubject-disjoint age split (seed={args.seed}):")
    print(f"{'split':<6} {'windows':>9} {'recordings':>11} {'subjects':>9}")
    for name in ("train", "val", "test"):
        print(f"{name:<6} {len(split.files[name]):>9} "
              f"{len(split.recordings(name)):>11} {len(split.subjects(name)):>9}")
    # build_age_split raises on any overlap, so reaching here proves disjointness.
    print("\nSubject overlap between all split pairs: 0 (asserted).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser(
        "scan", help="parse EDF headers into an age/sex metadata sidecar")
    scan.add_argument("--root", required=True,
                      help="corpus edf/ directory (searched recursively)")
    scan.add_argument("--out", default=None,
                      help=f"output JSON (default: <root>/processed/{SIDECAR_FILENAME})")
    scan.add_argument("--pattern", default="**/*.edf",
                      help="glob for EDF files under --root")
    scan.add_argument("--min_age", type=int, default=DEFAULT_MIN_AGE)
    scan.add_argument("--max_age", type=int, default=DEFAULT_MAX_AGE,
                      help="upper bound; excludes TUH's Age:999 redaction for 90+")
    scan.set_defaults(func=cmd_scan)

    split = sub.add_parser(
        "split", help="write a subject-disjoint train/val/test window split")
    split.add_argument("--root", required=True,
                       help="corpus edf/ directory containing processed/")
    split.add_argument("--out", default=None,
                       help=f"output JSON (default: <root>/processed/{SPLIT_FILENAME})")
    split.add_argument("--metadata", default=None,
                       help=f"metadata sidecar (default: <root>/processed/{SIDECAR_FILENAME})")
    split.add_argument("--val_fraction", type=float, default=DEFAULT_VAL_FRACTION,
                       help="fraction of pooled subjects held out for validation")
    split.add_argument("--seed", type=int, default=DEFAULT_SEED,
                       help="shuffle seed; fixed so the split is reproducible")
    split.set_defaults(func=cmd_split)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
