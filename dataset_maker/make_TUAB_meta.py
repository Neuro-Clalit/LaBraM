# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Extract per-recording meta features (demographics, acquisition setup, signal
# quality, EKG / photic / burst-suppression channels, qEEG summaries) for a TUH
# corpus into one CSV. Read-only with respect to the EDFs and the window pickles.
#
#   python dataset_maker/make_TUAB_meta.py --root /data/TUAB/edf --n_jobs 8
#
# Writes <root>/processed/meta_features.csv by default. The table drives
# notebooks/tuab_meta_features.ipynb. See docs/meta_features.md.
# ---------------------------------------------------------

import argparse
import logging
import os
import sys
import time

from labram.data.tuh_meta_features import extract_corpus_features

META_FILENAME = "meta_features.csv"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="corpus edf/ directory")
    parser.add_argument("--out", default=None,
                        help=f"output CSV (default <root>/processed/{META_FILENAME})")
    parser.add_argument("--pattern", default="**/*.edf")
    parser.add_argument("--n_jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--limit", type=int, default=None, help="only the first N files (smoke test)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start = time.time()
    table = extract_corpus_features(args.root, pattern=args.pattern,
                                    n_jobs=args.n_jobs, limit=args.limit)
    processed = os.path.join(args.root, "processed")
    out = args.out or os.path.join(processed if os.path.isdir(processed) else args.root,
                                   META_FILENAME)
    table.to_csv(out, index=False)

    failed = (table["error"] != "").sum()
    print(f"Wrote {len(table)} recordings x {table.shape[1]} columns to {out} "
          f"in {time.time() - start:.0f}s ({failed} failed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
