#!/usr/bin/env python
"""How much of the patch spectrum can be rebuilt from an encoder's patch tokens?

Freezes the encoder (any fine-tune or pre-training checkpoint) and the released
VQNSP codebook + decoder, trains only a fresh projection layer between them
(Linear 200->200, tanh, Linear 200->64; 1,500 AdamW steps, one-cycle LR 1e-3,
batches of 64 windows) on amplitude reconstruction + quantization, and reports
the amplitude reconstruction loss (Eq. 1.3 of docs/codebook_aux_p4_results.md,
0-99 Hz, 1 = no information) on held-out windows. Windows: 300 training and 48
held-out recordings of an npy cohort with every channel present, 8 / 4 random
windows each (60 s trimmed at both ends), average reference applied.

    python scripts/codebook_decodability_probe.py \\
        --npy-root /data/datasets/EEG-public/TUH_EEG/tuab/v3.0.1/edf/processed_npy \\
        --encoder <run>/checkpoint-best.pth --pooling cls --channels 19
"""
import argparse
import copy
import json
import types

import numpy as np
import torch
from timm.models import create_model

import labram.models.registry  # noqa: F401
from labram import utils
from labram.configs.loss_config import LossConfig
from labram.data.preprocess import common_average_reference
from labram.losses.spectral import SpectralReconstructionLoss
from labram.models.vqnsp import load_vqnsp_weights, vqnsp_codebook_shape
from labram.runs.finetune_setup import load_finetune_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npy-root", required=True)
    ap.add_argument("--encoder", required=True, help="checkpoint whose encoder is probed")
    ap.add_argument("--tokenizer", default="./checkpoints/vqnsp.pth")
    ap.add_argument("--pooling", choices=("cls", "mean"), default="cls",
                    help="the encoder's readout (decides whether its fc_norm is built)")
    ap.add_argument("--channels", type=int, choices=(19, 23), default=19, help="19 = without A1, A2, T1, T2")
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    rng = np.random.default_rng(1)

    man = json.load(open(f"{args.npy_root}/manifest.json"))
    names = [c.upper() for c in man["channels"]]
    sel = [i for i, n in enumerate(names) if args.channels == 23 or n not in ("A1", "A2", "T1", "T2")]
    ci = utils.get_channel_indices([names[i] for i in sel])
    recs = [r for r in man["recordings"] if all(r["channel_mask"]) and r["n_windows"] >= 40]
    order = rng.permutation(len(recs))

    def load(idx, per):
        xs = []
        for i in idx:
            r = recs[i]
            a = np.load(f"{args.npy_root}/recordings/{r['stem']}.npy", mmap_mode="r")
            for w in rng.choice(np.arange(6, r["n_windows"] - 6), per, replace=False):
                xs.append(np.asarray(a[w * 2000:(w + 1) * 2000], dtype=np.float32).T[sel])
        x = torch.from_numpy(np.stack(xs)) / 100
        return common_average_reference(x.reshape(len(x), len(sel), 10, 200))

    x_train, x_test = load(order[:300], 8), load(order[300:348], 4)
    n, d = vqnsp_codebook_shape(args.tokenizer)
    tok = create_model("vqnsp_encoder_base_decoder_3x200x12", pretrained=False, num_codebook_tokens=n, quantizer_dim=d)
    load_vqnsp_weights(tok, args.tokenizer)
    tok = tok.to(dev).eval()
    tok.quantizer.embedding.update = False
    enc = create_model("labram_base_patch200_200", pretrained=False, num_classes=0, drop_rate=0.0, drop_path_rate=0.0,
                       attn_drop_rate=0.0, use_mean_pooling=args.pooling == "mean", init_scale=0.001,
                       use_rel_pos_bias=False, use_abs_pos_emb=True, init_values=0.1, qkv_bias=False)
    load_finetune_checkpoint(enc, types.SimpleNamespace(finetune=args.encoder, model_key="model|module",
                                                        model_filter_name="gzp", model_prefix=""))
    enc = enc.to(dev).eval()
    for p in list(tok.parameters()) + list(enc.parameters()):
        p.requires_grad_(False)
    spectral = SpectralReconstructionLoss(LossConfig(freq_fraction=0.5))

    def losses(x, adapter):
        with torch.no_grad():
            tokens = enc.forward_features(x, channel_indices=ci, return_patch_tokens=True).float()
        f = adapter(tokens)
        f = f.view(x.shape[0], x.shape[1], -1, f.shape[-1]).permute(0, 3, 1, 2)
        q, quant, idx = tok.quantizer(f)
        magnitude, _ = tok.decode_spectrum(q, ci)
        target, _ = spectral.spectrum_targets(x)
        return spectral.reconstruction_loss(magnitude.float(), target), quant, idx

    @torch.no_grad()
    def evaluate(adapter):
        rec, qs, codes = [], [], set()
        for b in range(0, len(x_test), 48):
            r, q, idx = losses(x_test[b:b + 48].to(dev), adapter)
            rec.append(float(r)); qs.append(float(q)); codes |= set(idx.flatten().tolist())
        return np.mean(rec), np.mean(qs), len(codes)

    print("released projection layer: recon %.3f quantize %.4f codes %d" % evaluate(tok.encode_task_layer))
    adapter = copy.deepcopy(tok.encode_task_layer)
    for m in adapter.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            torch.nn.init.zeros_(m.bias)
    for p in adapter.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=args.steps, pct_start=0.1)
    for _ in range(args.steps):
        r, q, _ = losses(x_train[torch.randint(0, len(x_train), (64,))].to(dev), adapter)
        opt.zero_grad()
        (r + q).backward()
        opt.step()
        sched.step()
    print("fitted projection layer: recon %.3f quantize %.4f codes %d" % evaluate(adapter))


if __name__ == "__main__":
    main()
