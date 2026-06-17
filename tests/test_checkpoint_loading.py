"""End-to-end checkpoint / pretrained-weight loading tests (config-driven trunk).

Round-trips actual model weights through the three loaders that matter:

- ``load_pretrained_weights``        -- checkpoint layout handling
- ``load_vqnsp_weights``             -- tokenizer load for the pre-training phase
- ``load_finetune_checkpoint``       -- pretrained -> fine-tune transfer
                                        (strip ``student.`` prefix, drop a
                                        mismatched head, skip relpos buffers)

Also pins the Phase-B backward-compat wiring (channel helpers resolve to the
same objects via ``labram.data`` and the ``labram.utils`` re-export).
"""
from types import SimpleNamespace

import torch
import torch.nn as nn

from labram.configs.model_config import QuantizerConfig, TransformerArchConfig, VQNSPArchConfig
from labram.models.neural_transformer import NeuralTransformer
from labram.models.vqnsp import VQNSP, load_vqnsp_weights
from labram.runs.finetune_setup import load_finetune_checkpoint
from labram.utils.checkpoint import load_pretrained_weights


# ---------------------------------------------------------------------------
# model builders (config-driven, mirror the tiny configs used elsewhere)
# ---------------------------------------------------------------------------

def _ft_arch_cfg(num_classes: int) -> TransformerArchConfig:
    return TransformerArchConfig(
        eeg_window_size=200, patch_size=200, in_chans=1, out_chans=8,
        num_classes=num_classes, embed_dim=200, depth=2, num_heads=10,
        init_values=0.1, qkv_bias=True, use_abs_pos_emb=False, use_rel_pos_bias=True,
    )


def _make_ft_model(num_classes: int) -> NeuralTransformer:
    return NeuralTransformer(_ft_arch_cfg(num_classes))


def _vqnsp_arch_cfg(**overrides) -> TransformerArchConfig:
    defaults = dict(
        eeg_window_size=400, patch_size=200, in_chans=1, out_chans=8, num_classes=0,
        embed_dim=200, depth=2, num_heads=10, mlp_ratio=4.0, qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, init_values=0.1,
        use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
        use_mean_pooling=True, init_scale=0.001,
    )
    return TransformerArchConfig(**{**defaults, **overrides})


def _make_tiny_vqnsp() -> VQNSP:
    enc = _vqnsp_arch_cfg()
    dec = _vqnsp_arch_cfg(eeg_window_size=400 // 200, patch_size=1, in_chans=8, depth=1)
    cfg = VQNSPArchConfig(
        encoder=enc, decoder=dec,
        quantizer=QuantizerConfig(num_codebook_tokens=32, quantizer_dim=8, kmeans_init=False),
        decoder_out_dim=200,
    )
    return VQNSP(cfg)


def _perturb(model: nn.Module) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)


# ---------------------------------------------------------------------------
# load_pretrained_weights: checkpoint layout handling
# ---------------------------------------------------------------------------

class TestLoadPretrainedWeights:
    def test_model_layout(self, tmp_path):
        path = tmp_path / "m.pth"
        torch.save({"model": {"w": torch.ones(3)}, "epoch": 7}, path)
        assert set(load_pretrained_weights(str(path)).keys()) == {"w"}

    def test_state_dict_layout(self, tmp_path):
        path = tmp_path / "s.pth"
        torch.save({"state_dict": {"w": torch.ones(3)}}, path)
        assert set(load_pretrained_weights(str(path)).keys()) == {"w"}

    def test_bare_dict_layout(self, tmp_path):
        path = tmp_path / "b.pth"
        torch.save({"w": torch.ones(3)}, path)
        assert set(load_pretrained_weights(str(path)).keys()) == {"w"}


# ---------------------------------------------------------------------------
# load_vqnsp_weights: tokenizer load for the pre-training phase
# ---------------------------------------------------------------------------

class TestLoadVQNSPWeights:
    def test_roundtrip_matches_and_filters_aux_keys(self, tmp_path):
        source = _make_tiny_vqnsp()
        _perturb(source)
        src_sd = source.state_dict()

        payload = {k: v.clone() for k, v in src_sd.items()}
        payload["loss.fake"] = torch.zeros(1)
        payload["teacher.fake"] = torch.zeros(1)
        payload["scaling.fake"] = torch.zeros(1)
        path = tmp_path / "vqnsp.pth"
        torch.save({"model": payload}, path)

        target = _make_tiny_vqnsp()
        probe = "encoder.blocks.0.mlp.fc1.weight"
        assert not torch.allclose(target.state_dict()[probe], src_sd[probe])

        load_vqnsp_weights(target, str(path))  # strict load -> aux keys must be gone

        tgt_sd = target.state_dict()
        for k, v in src_sd.items():
            assert torch.allclose(tgt_sd[k], v), f"mismatch after load: {k}"


# ---------------------------------------------------------------------------
# load_finetune_checkpoint: pretrained -> fine-tune transfer
# ---------------------------------------------------------------------------

class TestLoadFinetuneCheckpoint:
    def _save_pretrained(self, tmp_path, source: NeuralTransformer):
        sd = source.state_dict()
        ckpt = {"model": {f"student.{k}": v.clone() for k, v in sd.items()}}
        path = tmp_path / "pretrain.pth"
        torch.save(ckpt, path)
        return path

    def _cfg(self, path):
        return SimpleNamespace(
            finetune=str(path), model_key="model",
            model_filter_name="student.", model_prefix="",
        )

    def test_backbone_loads_and_mismatched_head_dropped(self, tmp_path):
        source = _make_ft_model(num_classes=5)
        _perturb(source)
        src_sd = source.state_dict()
        path = self._save_pretrained(tmp_path, source)

        target = _make_ft_model(num_classes=1)
        probe = "blocks.0.mlp.fc1.weight"
        assert not torch.allclose(target.state_dict()[probe], src_sd[probe])

        load_finetune_checkpoint(target, self._cfg(path))

        tgt_sd = target.state_dict()
        for k, v in src_sd.items():
            if k.startswith("head.") or "relative_position_index" in k:
                continue
            assert torch.allclose(tgt_sd[k], v), f"backbone not transferred: {k}"
        assert tgt_sd["head.weight"].shape[0] == 1

    def test_matching_head_is_transferred(self, tmp_path):
        source = _make_ft_model(num_classes=3)
        _perturb(source)
        src_sd = source.state_dict()
        path = self._save_pretrained(tmp_path, source)

        target = _make_ft_model(num_classes=3)
        load_finetune_checkpoint(target, self._cfg(path))

        assert torch.allclose(target.state_dict()["head.weight"], src_sd["head.weight"])

    def test_loaded_model_runs_forward(self, tmp_path):
        source = _make_ft_model(num_classes=2)
        _perturb(source)
        path = self._save_pretrained(tmp_path, source)
        target = _make_ft_model(num_classes=2)
        load_finetune_checkpoint(target, self._cfg(path))

        target.eval()
        with torch.no_grad():
            out = target(torch.randn(2, 4, 1, 200), channel_indices=None)
        assert out.shape == (2, 2)


# ---------------------------------------------------------------------------
# Phase-B wiring: channel helpers are the same object via data / utils re-export
# ---------------------------------------------------------------------------

def test_channel_helpers_reexport_identity():
    import labram.data as data
    import labram.utils as utils

    assert utils.get_channel_indices is data.get_channel_indices
    assert utils.standard_1020 is data.standard_1020
    assert utils.build_pretraining_dataset is data.build_pretraining_dataset
