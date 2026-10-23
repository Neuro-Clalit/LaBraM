#!/usr/bin/env python
"""Training throughput of the age fine-tune under different speed settings.

For each combination of batch size, data-loader workers and fused attention,
build the run's model and training loader (the given config + data path, its
window selection), run ``--warmup`` untimed then ``--steps`` timed training
steps (forward, backward, AdamW step under AMP, as in train_finetune), and
report samples/s, ms/step, peak GPU memory and the projected wall time of one
epoch of ``--epoch-samples`` samples.

    python scripts/benchmark_finetune_speed.py \\
        --data-path /data/.../cohorts/all_both_v2 --batch 64 128 256 \\
        --workers 10 6 --fused 0 1 --out benchmark.json
"""
import argparse
import itertools
import json
import time

import torch
from einops import rearrange

import labram.models.registry  # noqa: F401
import labram.utils as utils
from labram.configs.run_configs import FinetuneRunConfig
from labram.data import get_dataset_bundle
from labram.data.window_selection import WindowSelection, apply_window_selection
from labram.runs.run_finetune import get_model


def bench(cfg, train, ch_names, batch, workers, fused, steps, warmup, device):
    cfg.model.fused_attention = bool(fused)
    model = get_model(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    loader = torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, drop_last=True,
                                         num_workers=workers, pin_memory=True,
                                         persistent_workers=workers > 0)
    ci = utils.get_channel_indices(ch_names)
    torch.cuda.reset_peak_memory_stats()
    it = iter(loader)
    t0 = None
    for step in range(warmup + steps):
        if step == warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        x, y = next(it)
        x = rearrange(x.float().to(device, non_blocking=True) / 100, "B N (A T) -> B N A T", T=200)
        y = y.float().to(device).reshape(len(y), -1)[:, :1]
        with torch.amp.autocast("cuda"):
            loss = torch.nn.functional.huber_loss(model(x, ci), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    del loader, it
    return {"batch": batch, "workers": workers, "fused": bool(fused),
            "samples_per_s": steps * batch / dt, "ms_per_step": 1000 * dt / steps,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="labram/configs/defaults/finetune_tuab_age.json")
    p.add_argument("--data-path", required=True)
    p.add_argument("--batch", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--workers", type=int, nargs="+", default=[10, 6])
    p.add_argument("--fused", type=int, nargs="+", default=[0, 1])
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--epoch-samples", type=int, default=1733329, help="samples per epoch to project")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cfg = FinetuneRunConfig.load_config(args.config, **{"data.data_path": args.data_path,
                                                         "finetune_checkpoint.finetune": ""})
    bundle = get_dataset_bundle(cfg.data.dataset, args.data_path, data_format=cfg.data.data_format)
    bundle = apply_window_selection(bundle, WindowSelection.from_data_config(cfg.data))
    cfg.model.nb_classes, cfg.model.task = bundle.nb_classes, bundle.task
    device = torch.device("cuda")
    rows = []
    for batch, workers, fused in itertools.product(args.batch, args.workers, args.fused):
        r = bench(cfg, bundle.train, bundle.ch_names, batch, workers, fused, args.steps,
                  args.warmup, device)
        r["epoch_min"] = args.epoch_samples / r["samples_per_s"] / 60
        rows.append(r)
        print(f"batch {batch:4d}  workers {workers:2d}  fused {int(fused)}  "
              f"{r['samples_per_s']:7.0f} samples/s  {r['ms_per_step']:6.1f} ms/step  "
              f"{r['peak_mem_gb']:5.1f} GB  epoch {r['epoch_min']:5.1f} min", flush=True)
        torch.cuda.empty_cache()
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
