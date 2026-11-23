#!/usr/bin/env python
"""Fit the tokenizer's projection layer to a fine-tune encoder (``codebook_aux``).

The released VQNSP tokenizer maps *its own* encoder's patch tokens to the
codebook through ``encode_task_layer`` (Linear 200 -> 200, tanh, Linear 200 ->
64). The patch tokens of ``labram-base`` live in another space: through that
layer the frozen decoder reconstructs nothing (amplitude loss about 0.9, with
1.0 = no information). This script trains only that layer -- encoder, codebook
and decoder frozen -- on unlabelled training windows, with the tokenizer's own
loss (amplitude reconstruction + quantization), and writes a tokenizer
checkpoint that differs from the input one in ``encode_task_layer`` only. That
file is ``codebook_aux.tokenizer_weight`` of a fine-tune.

The encoder is built from the fine-tune config and overrides given here, so it
sees the windows exactly as the run will (channels, average reference,
class-token or mean-pooled tokens, missing channels left out per window).

    python scripts/fit_codebook_projection.py \
        --cohort /data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1 \
        --cv-split <run>/cv_split.json --fold 0 \
        --out checkpoints/vqnsp_proj_labram_base_cls19.pth \
        --set 'data.drop_channels=["A1","A2","T1","T2"]' model.use_mean_pooling=false
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import labram.utils as utils  # noqa: E402
from labram.configs.run_configs import FinetuneRunConfig  # noqa: E402
from labram.configs.utils_conf import parse_overrides  # noqa: E402
from labram.models.codebook_aux import CodebookAuxiliary  # noqa: E402
from labram.models.vqnsp import load_vqnsp_weights, remap_legacy_keys, vqnsp_codebook_shape  # noqa: E402
from labram.runs.finetune_setup import load_finetune_checkpoint  # noqa: E402
from labram.train.channel_drop import ChannelDropper  # noqa: E402
from labram.utils.checkpoint import load_pretrained_weights  # noqa: E402


def split_recordings(manifest, cv_split, fold):
    """(train, val) manifest rows: by the CV split (``fold`` = the val fold,
    held-out test excluded) or, without one, by ``source_split``."""
    recs = manifest["recordings"]
    if not cv_split:
        return ([r for r in recs if r.get("source_split") == "train"],
                [r for r in recs if r.get("source_split") == "val"])
    cv = json.load(open(cv_split))
    val_groups = set(cv["folds"][fold]["groups"])
    train_groups = {g for f in cv["folds"] if f["fold"] != fold for g in f["groups"]}

    def member(r, groups):
        return bool({str(r.get("subject")), "sub-" + str(r.get("subject")), r["stem"].split("_")[0]} & groups)
    return [r for r in recs if member(r, train_groups)], [r for r in recs if member(r, val_groups)]


def read_windows(root, rows, per_recording, keep, trim_windows, rng):
    """[n, C, 2000] float32 in 0.1 mV (the model's input unit), random windows."""
    out = []
    for r in rows:
        lo, hi = trim_windows, r["n_windows"] - trim_windows
        if hi - lo < per_recording:
            continue
        a = np.load(os.path.join(root, "recordings", r["stem"] + ".npy"), mmap_mode="r")
        for w in rng.choice(np.arange(lo, hi), per_recording, replace=False):
            out.append(np.asarray(a[w * 2000:(w + 1) * 2000], dtype=np.float32).T[keep])
    return torch.from_numpy(np.stack(out)) / 100


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True, help="npy cohort root (manifest.json + recordings/)")
    ap.add_argument("--config", default="labram/configs/defaults/finetune_tuab_age.json")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides", help="fine-tune config overrides")
    ap.add_argument("--encoder", default="./checkpoints/labram-base.pth")
    ap.add_argument("--tokenizer", default="./checkpoints/vqnsp.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cv-split", default="", help="cv_split.json of the run's study (train folds only)")
    ap.add_argument("--fold", type=int, default=0, help="the val fold of --cv-split")
    ap.add_argument("--recordings", type=int, default=2000)
    ap.add_argument("--val-recordings", type=int, default=200)
    ap.add_argument("--windows", type=int, default=4, help="windows per recording")
    ap.add_argument("--trim-windows", type=int, default=6, help="windows skipped at each end (60 s)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--freq-fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = FinetuneRunConfig.load_config(args.config, **parse_overrides(args.overrides))
    cfg.finetune_checkpoint.finetune = args.encoder
    cfg.mil.enabled = False
    cfg.codebook_aux.enabled = False
    from labram.runs.run_finetune import get_model
    model = get_model(cfg)
    load_finetune_checkpoint(model, cfg.finetune_checkpoint)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    manifest = json.load(open(os.path.join(args.cohort, "manifest.json")))
    names = [c.upper() for c in manifest["channels"]]
    dropped = {c.upper() for c in cfg.data.drop_channels}
    keep = [i for i, n in enumerate(names) if n not in dropped]
    ch_names = [names[i] for i in keep]
    ci_all = utils.get_channel_indices(ch_names)
    train_rows, val_rows = split_recordings(manifest, args.cv_split, args.fold)
    train_rows = [train_rows[i] for i in rng.permutation(len(train_rows))[:args.recordings]]
    val_rows = [val_rows[i] for i in rng.permutation(len(val_rows))[:args.val_recordings]]
    x_train = read_windows(args.cohort, train_rows, args.windows, keep, args.trim_windows, rng)
    x_val = read_windows(args.cohort, val_rows, args.windows, keep, args.trim_windows, rng)
    print(f"{len(ch_names)} channels; train {tuple(x_train.shape)} from {len(train_rows)} recordings, "
          f"val {tuple(x_val.shape)} from {len(val_rows)}")

    n_codes, code_dim = vqnsp_codebook_shape(args.tokenizer)
    from timm.models import create_model
    tokenizer = create_model(cfg.codebook_aux.tokenizer_model, pretrained=False,
                             num_codebook_tokens=n_codes, quantizer_dim=code_dim)
    load_vqnsp_weights(tokenizer, args.tokenizer)
    aux = CodebookAuxiliary(tokenizer, amplitude_weight=1.0, embedding_weight=1.0,
                            freq_fraction=args.freq_fraction).to(device)

    def losses(x):
        """Token-weighted (reconstruction, quantization, code indices) of a
        batch, every window on its own present channels."""
        x = x.to(device).reshape(x.shape[0], x.shape[1], -1, 200)
        present = ChannelDropper.present(x)
        patterns, inverse = torch.unique(present, dim=0, return_inverse=True)
        parts = []
        for p in range(patterns.shape[0]):
            chans = patterns[p].nonzero().squeeze(1)
            if chans.numel() == 0:
                continue
            xp = x[inverse == p][:, chans]
            ci = [ci_all[0]] + [ci_all[int(c) + 1] for c in chans]
            with torch.no_grad():
                tokens = model.forward_features(xp, channel_indices=ci, return_patch_tokens=True)
                if model.fc_norm is not None:
                    pass                      # return_patch_tokens already applied fc_norm
                x_pre = model.maybe_preprocess_input(xp)
            quantized, quant, idx = aux.quantize(tokens)
            recon = aux.reconstruction_loss(quantized, x_pre, ci)
            parts.append((idx.numel(), recon, quant, idx.reshape(-1)))
        total = sum(n for n, *_ in parts)
        return (sum(r * (n / total) for n, r, _, _ in parts), sum(q * (n / total) for n, _, q, _ in parts),
                torch.cat([i for *_, i in parts]))

    @torch.no_grad()
    def evaluate():
        rec, qua, codes, n = 0.0, 0.0, [], 0
        for b in range(0, len(x_val), args.batch):
            r, q, idx = losses(x_val[b:b + args.batch])
            rec += float(r) * idx.numel(); qua += float(q) * idx.numel(); n += idx.numel(); codes.append(idx)
        return rec / n, qua / n, int(torch.unique(torch.cat(codes)).numel())

    before = evaluate()
    print("released projection layer : recon %.4f quant %.5f codes %d" % before)
    for m in aux.encode_task_layer.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            torch.nn.init.zeros_(m.bias)
    for p in aux.encode_task_layer.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(aux.encode_task_layer.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.1)
    for step in range(args.steps):
        r, q, _ = losses(x_train[torch.randint(0, len(x_train), (args.batch,))])
        opt.zero_grad()
        (r + q).backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            print("step %5d: train recon %.4f quant %.5f | val recon %.4f quant %.5f codes %d"
                  % ((step, float(r), float(q)) + evaluate()))
    after = evaluate()

    weights = remap_legacy_keys(load_pretrained_weights(args.tokenizer))
    for k in [k for k in weights if k.startswith(("loss", "teacher", "scaling"))]:
        del weights[k]
    fitted = {f"encode_task_layer.{k}": v.detach().cpu() for k, v in aux.encode_task_layer.state_dict().items()}
    assert set(fitted) <= set(weights), sorted(set(fitted) - set(weights))
    weights.update(fitted)
    meta = {"encoder": args.encoder, "tokenizer": args.tokenizer, "config": args.config,
            "overrides": args.overrides, "cohort": args.cohort, "cv_split": args.cv_split, "fold": args.fold,
            "channels": ch_names, "freq_fraction": args.freq_fraction, "steps": args.steps,
            "train_windows": len(x_train), "val_windows": len(x_val),
            "val_before": dict(zip(("recon", "quant", "codes"), before)),
            "val_after": dict(zip(("recon", "quant", "codes"), after))}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model": weights, "codebook_projection_fit": meta}, args.out)
    print("fitted projection layer   : recon %.4f quant %.5f codes %d" % after)
    print("wrote", args.out)
    print(json.dumps(meta["val_after"]))


if __name__ == "__main__":
    main()
