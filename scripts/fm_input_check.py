#!/usr/bin/env python
"""Input checks for the foreign pretrained encoders, run before any training.

``parity``: the vendored network against the upstream implementation on the
same random input, with the released weights loaded strictly into both.
``--upstream`` is a folder with the upstream files: ``cbramod.py`` and
``criss_cross_transformer.py`` (github.com/wjq-learning/CBraMod, ``models/``),
or ``modeling_reve.py`` and ``configuration_reve.py``
(huggingface.co/brain-bzh/reve-base). REVE's file imports ``transformers`` only
for its base classes; a stub stands in, so the package is not needed.

    python scripts/fm_input_check.py parity --encoder cbramod \\
        --weights ./checkpoints/cbramod-base.pth --upstream /path/to/CBraMod/models

``recon`` (CBraMod, label-free): half of the channel x patch tokens are masked
and reconstructed with the pretrained output layer, as in pretraining. The
relative error on the masked patches of a window, Eq. (1), is reported (median
over windows, and pooled over windows) for our input convention and for
controls that must be clearly worse (no division by 100, shuffled channel order):

    e = sum_masked (x_hat - x)^2 / sum_masked x^2                        (1)

``audit``: amplitude statistics of each cohort in model units: median channel
standard deviation, share of windows above CBraMod's 100 µV pretraining limit,
share of samples a 15 SD clip touches.

    python scripts/fm_input_check.py recon --weights ./checkpoints/cbramod-base.pth \\
        --config /data/runs/heedb/mil_phased_cv/P4_L1/fold_0/cv/fold_0/run_config.yaml \\
        --cohort harvard_temple=/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1 \\
        --cohort lemon=/data/datasets/EEG-public/LEMON/cohorts/lemon_v1 --out recon.csv
"""
import argparse
import importlib.util
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import torch

from labram.data.preprocess import band_limit, common_average_reference
from labram.models.backbones import CBRAMOD_CHANNELS, CBraModBackbone, REVEBackbone
from labram.models.backbones.base import read_weight_file


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------- parity
def parity_cbramod(weights: str, upstream: str) -> dict:
    pkg = types.ModuleType("models")
    pkg.__path__ = [upstream]
    sys.modules["models"] = pkg
    _load_module("models.criss_cross_transformer", os.path.join(upstream, "criss_cross_transformer.py"))
    up = _load_module("models.cbramod", os.path.join(upstream, "cbramod.py"))
    state = read_weight_file(weights)
    ref = up.CBraMod()
    ref.load_state_dict(state)
    ref.eval()
    ours = CBraModBackbone(keep_recon_head=True)
    ours.load_pretrained(state)
    ours.eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 19, 10, 200, generator=g) * 0.3
    mask = (torch.rand(2, 19, 10, generator=g) < 0.5).long()
    with torch.no_grad():
        d_plain = (ref(x) - ours.net(x)).abs().max().item()
        d_mask = (ref(x, mask) - ours.encode(x, CBRAMOD_CHANNELS, mask=mask)).abs().max().item()
        feats = CBraModBackbone()
        feats.load_pretrained(state)
        feats.eval()
        ref.proj_out = torch.nn.Identity()
        d_feat = (ref(x) - feats.encode(x, CBRAMOD_CHANNELS)).abs().max().item()
    return {"encoder": "cbramod", "tensors": len(state), "max_diff_output": d_plain,
            "max_diff_masked": d_mask, "max_diff_features": d_feat}


def parity_reve(weights: str, upstream: str) -> dict:
    stub = types.ModuleType("transformers")

    class PretrainedConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class PreTrainedModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def post_init(self):
            pass

    stub.PretrainedConfig, stub.PreTrainedModel = PretrainedConfig, PreTrainedModel
    sys.modules.setdefault("transformers", stub)
    pkg = types.ModuleType("reve_upstream")
    pkg.__path__ = [upstream]
    sys.modules["reve_upstream"] = pkg
    conf = _load_module("reve_upstream.configuration_reve", os.path.join(upstream, "configuration_reve.py"))
    up = _load_module("reve_upstream.modeling_reve", os.path.join(upstream, "modeling_reve.py"))
    with open(os.path.join(upstream, "config.json")) as fh:
        cfg = {k: v for k, v in json.load(fh).items()
               if k in ("embed_dim", "depth", "heads", "head_dim", "mlp_dim_ratio", "use_geglu",
                        "freqs", "noise_ratio", "patch_size", "patch_overlap")}
    state = read_weight_file(weights)
    ref = up.Reve(conf.ReveConfig(**cfg))
    ref.load_state_dict(state)
    ref.eval()
    ours = REVEBackbone(embed_dim=cfg["embed_dim"], depth=cfg["depth"], heads=cfg["heads"],
                        head_dim=cfg["head_dim"], mlp_dim_ratio=cfg["mlp_dim_ratio"], freqs=cfg["freqs"])
    ours.load_pretrained(state)
    ours.eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 19, 10, 200, generator=g) * 0.3
    pos = ours.positions(CBRAMOD_CHANNELS).unsqueeze(0).expand(2, -1, -1)
    with torch.no_grad():
        a = ref(x.reshape(2, 19, 2000), pos.clone())
        b = ours.encode(x, CBRAMOD_CHANNELS)
        d_tok = (a - b).abs().max().item()
        d_pool = (ref.attention_pooling(a) - ours.pool(b)).abs().max().item()
    return {"encoder": "reve", "tensors": len(state), "tokens_shape": list(b.shape),
            "max_diff_tokens": d_tok, "max_diff_pooled": d_pool}


# ------------------------------------------------------------------ windows
def sample_windows(config: str, data_path: str, n: int, overrides=(), split: str = "test", seed: int = 0,
                   num_workers: int = 8):
    """``(x [n, C, 2000] µV, channel names)``: a seeded sample of the windows a
    run config selects on ``data_path``."""
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.configs.utils_conf import parse_overrides
    from labram.eval.age_analysis import age_bundle_for_config
    from labram.eval.window_descriptors import TTAWindows
    cfg = FinetuneRunConfig.load_config(config, raise_exception=False, **parse_overrides(list(overrides)))
    bundle = age_bundle_for_config(cfg, data_path, "none")
    data = TTAWindows(getattr(bundle, split), n_crops=0, pad=0)
    index = np.sort(np.random.default_rng(seed).choice(len(data), size=min(n, len(data)), replace=False))
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, index.tolist()), batch_size=256,
                                         num_workers=num_workers)
    x = torch.cat([copies[:, 0] for copies, *_ in loader])
    return x, list(bundle.ch_names)


def _patches(x_uv: torch.Tensor) -> torch.Tensor:
    return (x_uv / 100).reshape(x_uv.shape[0], x_uv.shape[1], -1, 200)


# -------------------------------------------------------------------- recon
RECON_VARIANTS = ("stored", "car", "car_highpass_0.3", "car_clip_100uv",
                  "control_no_div100", "control_shuffled_channels")


@torch.no_grad()
def recon_errors(backbone: CBraModBackbone, x_uv: torch.Tensor, ch_names, device, batch: int = 256,
                 seed: int = 0) -> dict:
    """Eq. (1) per input variant, with one mask shared by all variants: the
    median over windows of the per-window error (``<variant>``), and the error
    pooled over all windows (``<variant>_pooled``), which a few large-amplitude
    windows dominate."""
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(x_uv.shape[1], generator=g)
    num = {v: [] for v in RECON_VARIANTS}
    den = {v: [] for v in RECON_VARIANTS}
    for i in range(0, len(x_uv), batch):
        x = _patches(x_uv[i:i + batch].to(device).float())
        mask = (torch.rand(x.shape[:3], generator=g) < 0.5).long().to(device)
        car = common_average_reference(x)
        variants = {"stored": x, "car": car, "car_highpass_0.3": band_limit(car, 0.3, 99.0),
                    "car_clip_100uv": car.clamp(-1.0, 1.0), "control_no_div100": car * 100,
                    "control_shuffled_channels": car[:, order]}
        for name, v in variants.items():
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                out = backbone.encode(v, ch_names, mask=mask).float()
            sel = (mask == 1)[..., None].float()
            num[name].append((((out - v) ** 2) * sel).sum(dim=(1, 2, 3)).double().cpu())
            den[name].append(((v ** 2) * sel).sum(dim=(1, 2, 3)).double().cpu())
    res = {}
    for name in RECON_VARIANTS:
        n, d = torch.cat(num[name]), torch.cat(den[name])
        res[name] = float((n / d.clamp_min(1e-12)).median())
    for name in RECON_VARIANTS:
        res[f"{name}_pooled"] = float(torch.cat(num[name]).sum() / torch.cat(den[name]).sum())
    return res


# -------------------------------------------------------------------- audit
def audit(x_uv: torch.Tensor) -> dict:
    """Amplitude statistics of windows ``[N, C, L]`` (µV) after CAR."""
    x = common_average_reference(_patches(x_uv.float())).reshape(x_uv.shape) * 100
    std = x.std(-1)                                             # [N, C]
    present = std > 1e-6
    z = (x - x.mean(-1, keepdim=True)) / std.clamp_min(1e-6)[..., None]
    peak = x.abs().amax(dim=(1, 2))
    return {"windows": int(len(x)), "channels": int(x.shape[1]),
            "missing_channel_share": float(1 - present.float().mean()),
            "median_channel_std_uv": float(std[present].median()),
            "p95_channel_std_uv": float(std[present].quantile(0.95)),
            "windows_peak_above_100uv": float((peak > 100).float().mean()),
            "windows_peak_above_800uv": float((peak > 800).float().mean()),
            "samples_above_100uv": float((x.abs() > 100).float().mean()),
            "samples_clipped_at_15sd": float((z.abs() > 15).float().mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parity")
    p.add_argument("--encoder", choices=["cbramod", "reve"], required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--upstream", required=True)
    for name in ("recon", "audit"):
        q = sub.add_parser(name)
        q.add_argument("--config", required=True, help="run config giving the window selection and channels")
        q.add_argument("--cohort", action="append", required=True, metavar="NAME=PATH[::key=value...]",
                       help="cohort folder; config overrides for it after '::' (e.g. data.trim_start_sec=0)")
        q.add_argument("--n-windows", type=int, default=2000)
        q.add_argument("--split", default="test")
        q.add_argument("--out", default=None)
        if name == "recon":
            q.add_argument("--weights", required=True)
    args = ap.parse_args()
    if args.cmd == "parity":
        res = (parity_cbramod if args.encoder == "cbramod" else parity_reve)(args.weights, args.upstream)
        print(json.dumps(res, indent=2))
        worst = max(v for k, v in res.items() if k.startswith("max_diff"))
        if worst > 1e-4:
            raise SystemExit(f"parity FAILED: max difference {worst:g} > 1e-4")
        print("parity OK")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = None
    if args.cmd == "recon":
        backbone = CBraModBackbone(keep_recon_head=True)
        backbone.load_pretrained(args.weights)
        backbone.to(device).eval()
    rows = []
    for spec in args.cohort:
        name, rest = spec.split("=", 1)
        path, *overrides = rest.split("::")
        x, ch_names = sample_windows(args.config, path, args.n_windows, overrides, args.split)
        row = recon_errors(backbone, x, ch_names, device) if args.cmd == "recon" else audit(x)
        rows.append({"cohort": name, **row})
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    table = pd.DataFrame(rows).set_index("cohort")
    print(table.round(4).T.to_string())
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        table.to_csv(args.out)


if __name__ == "__main__":
    main()
