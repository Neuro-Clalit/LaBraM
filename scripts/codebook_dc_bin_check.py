#!/usr/bin/env python
"""What does each 1 Hz bin of a 1 s patch hold, the 0 Hz bin in particular?

Pretrained encoder + the fitted projection layer + the frozen VQNSP decoder
(``codebook_aux``), P4's input (19 channels, average reference), 200 Temple and
200 Harvard training recordings of the CV fold (4 random windows each, 60 s
trimmed). Per bin: mean and SD across recordings of the standardised target
(Eq. 1.2 of docs/codebook_aux_p4_results.md), the decoder's squared error, the
raw amplitude per source, and the Spearman correlation of the per-recording
target with age (Table 5.3 of that doc).

    python scripts/codebook_dc_bin_check.py --cohort /data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1 \
        --cv-split <P4 run>/cv_split.json --tokenizer-weight checkpoints/vqnsp_proj_labram_base_cls19_fold0.pth
"""
import argparse
import json
import os
import sys
import numpy as np, torch
from scipy.stats import spearmanr
from einops import rearrange
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_codebook_projection import split_recordings  # noqa: E402
from labram import utils
from labram.configs.run_configs import FinetuneRunConfig
from labram.runs.run_finetune import get_model
from labram.runs.finetune_setup import load_finetune_checkpoint
from labram.models.codebook_aux import get_codebook_aux
from labram.train.channel_drop import ChannelDropper

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--cohort", required=True)
ap.add_argument("--cv-split", required=True)
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--tokenizer-weight", required=True, help="tokenizer with the fitted projection layer")
ap.add_argument("--encoder", default="./checkpoints/labram-base.pth")
args = ap.parse_args()
COHORT, CV = args.cohort, args.cv_split
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rng = np.random.default_rng(7); torch.manual_seed(0)
cfg = FinetuneRunConfig.load_config("labram/configs/defaults/finetune_tuab_age.json", **{
    "data.drop_channels": ["A1", "A2", "T1", "T2"], "model.use_mean_pooling": False, "model.nb_classes": 1,
    "codebook_aux.enabled": True, "codebook_aux.amplitude_weight": 1.0,
    "codebook_aux.tokenizer_weight": args.tokenizer_weight, "finetune_checkpoint.finetune": args.encoder})
m = get_model(cfg); load_finetune_checkpoint(m, cfg.finetune_checkpoint); m.to(dev).eval()
aux = get_codebook_aux(m).to(dev)
man = json.load(open(f"{COHORT}/manifest.json"))
names = [c.upper() for c in man["channels"]]
keep = [i for i, n in enumerate(names) if n not in {"A1", "A2", "T1", "T2"}]
ci_all = utils.get_channel_indices([names[i] for i in keep])
train, _ = split_recordings(man, CV, args.fold)
train = [r for r in train if r["n_windows"] >= 30 and r.get("age") is not None]
pick = {"tuh": [], "heedb": []}
for i in rng.permutation(len(train)):
    c = train[i]["corpus"]
    if len(pick[c]) < 200: pick[c].append(train[i])
rows = pick["tuh"] + pick["heedb"]
sq_sum = np.zeros(100); n_el = 0
per_rec = []          # (corpus, age, mean standardized target per bin, raw |FFT| bin0 / bin1-4 mean in uV)
with torch.no_grad():
    for r in rows:
        a = np.load(f"{COHORT}/recordings/{r['stem']}.npy", mmap_mode="r")
        ws = rng.choice(np.arange(6, r["n_windows"] - 6), 4, replace=False)
        x = torch.from_numpy(np.stack([np.asarray(a[w * 2000:(w + 1) * 2000], dtype=np.float32).T[keep] for w in ws])).to(dev) / 100
        x = x.reshape(4, len(keep), 10, 200)
        present = ChannelDropper.present(x)[0]
        chans = present.nonzero().squeeze(1)
        x = x[:, chans]; ci = [ci_all[0]] + [ci_all[int(c) + 1] for c in chans]
        tokens = m.forward_features(x, channel_indices=ci, return_patch_tokens=True).float()
        xp = m.maybe_preprocess_input(x)
        q, _, _ = aux.quantize(tokens)
        grid = q.view(4, x.shape[1], -1, q.shape[-1]).permute(0, 3, 1, 2)
        mag = aux.decode_task_layer_magnitude(aux.decoder(grid, ci, return_patch_tokens=True)).float()[..., :100]
        tgt, _ = aux.spectral.spectrum_targets(xp.float())
        tgt = rearrange(tgt, "b n a c -> b (n a) c")
        sq = (mag - tgt).pow(2)
        sq_sum += sq.sum((0, 1)).cpu().numpy(); n_el += sq.shape[0] * sq.shape[1]
        raw = torch.fft.fft(xp.float(), dim=-1).abs()[..., :100] * 100 / 200   # uV amplitude per bin (x100 undoes /100; /T)
        per_rec.append((r["corpus"], r["age"], tgt.mean((0, 1)).cpu().numpy(), sq.mean((0, 1)).cpu().numpy(),
                        raw.mean((0, 1, 2)).cpu().numpy()))
err = sq_sum / n_el
T = np.stack([p[2] for p in per_rec]); E = np.stack([p[3] for p in per_rec]); RAW = np.stack([p[4] for p in per_rec])
age = np.array([p[1] for p in per_rec]); src = np.array([p[0] for p in per_rec])
print(f"{len(per_rec)} recordings (Temple {np.sum(src == 'tuh')}, Harvard {np.sum(src == 'heedb')}), 4 windows each")
print("mean squared error over all 100 bins %.3f | bins 1-45 %.3f | bins 0-45 %.3f" % (err.mean(), err[1:46].mean(), err[0:46].mean()))
print("bin | target mean (std units) | target SD across recordings | sq. error | error share of 0-45 | raw amplitude uV Temple / Harvard | Spearman with age")
for b in (0, 1, 2, 3, 4, 6, 8, 10, 12, 20, 30, 45, 60, 80):
    rho = spearmanr(T[:, b], age)[0]
    print(f"{b:3d} | {T[:, b].mean():+.2f} | {T[:, b].std():.2f} | {err[b]:.3f} | {100 * err[b] / err[0:46].sum():.1f}% | "
          f"{RAW[src == 'tuh', b].mean():.2f} / {RAW[src == 'heedb', b].mean():.2f} | {rho:+.2f}")
print("bin 0 error / mean error of bins 1-45: %.1fx" % (err[0] / err[1:46].mean()))
print("bin 0 target, Temple vs Harvard (std units): %.2f vs %.2f; bin 1: %.2f vs %.2f" % (
    T[src == 'tuh', 0].mean(), T[src == 'heedb', 0].mean(), T[src == 'tuh', 1].mean(), T[src == 'heedb', 1].mean()))
