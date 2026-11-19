#!/usr/bin/env python
"""Spectrogram descriptors of every val/test window of a run's split.

Model-free companion of ``scripts/window_confidence_predict.py``: reads each
window once (the run's own window selection and ``data.drop_channels``) and
stores the classical-spectrogram (STFT) and Morlet-wavelet descriptors of
``labram/eval/window_descriptors.py``, keyed by ``split, recording, window``
so they merge onto the ``<NAME>_windows.parquet`` caches. Runs that share a
split share this file.

    python scripts/window_spectral_descriptors.py \\
        --data-path /data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1 \\
        --run-dir /data/runs/heedb/e2l/finetune_... \\
        --out /data/runs/heedb/window_conf/spectral_descriptors.parquet
"""
import argparse
import time

import pandas as pd
import torch

from labram.eval.age_analysis import load_age_bundle
from labram.eval.window_descriptors import TTAWindows, cwt_descriptors, stft_descriptors


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--run-dir", required=True, help="any run with the split to describe")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--max-windows", type=int, default=0, help="first N windows per split (smoke test)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, bundle = load_age_bundle(args.run_dir, args.data_path, stage_dir="none")

    frames = []
    for split in args.splits:
        data = TTAWindows(getattr(bundle, split), n_crops=0, pad=0)
        if args.max_windows:
            data = torch.utils.data.Subset(data, range(min(args.max_windows, len(data))))
        loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
        feats, recs, wins = {}, [], []
        t0, done = time.time(), 0
        with torch.no_grad():
            for copies, _, stem, k, _ in loader:
                x = copies[:, 0].to(device, non_blocking=True)
                for fn in (stft_descriptors, cwt_descriptors):
                    for key, v in fn(x).items():
                        feats.setdefault(key, []).append(v.float().cpu())
                recs += list(stem)
                wins += [int(v) for v in k]
                done += len(stem)
                if done // args.batch_size % 200 == 0:
                    rate = done / (time.time() - t0)
                    print(f"  {split}: {done}/{len(data)} windows, {rate:.0f}/s, "
                          f"eta {(len(data) - done) / rate / 60:.1f} min", flush=True)
        df = pd.DataFrame({"split": split, "recording": recs, "window": wins})
        for key, v in feats.items():
            df[key] = torch.cat(v).numpy()
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(args.out)
    print(f"wrote {args.out}: {len(out)} windows x {out.shape[1] - 3} descriptors")


if __name__ == "__main__":
    main()
