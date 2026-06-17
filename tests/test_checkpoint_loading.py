"""End-to-end checkpoint / pretrained-weight loading tests.

The existing suite only covered the *no-op* (empty `--finetune`) path, so the
real weight-loading code was untested. These tests round-trip actual model
weights through the three loaders that matter:

- ``load_pretrained_weights``        -- checkpoint layout handling
- ``load_vqnsp_weights``             -- tokenizer load for the pre-training phase
- ``load_finetune_checkpoint``       -- pretrained -> fine-tune transfer
                                        (strip ``student.`` prefix, drop a
                                        mismatched head, skip relpos buffers)

They also pin the Phase-B backward-compat wiring: the channel helpers used by
the model forward path resolve to the same objects via ``labram.data`` and the
``labram.utils`` re-export.
"""
from functools import partial
from types import SimpleNamespace

import torch
import torch.nn as nn

from labram.models.neural_transformer import NeuralTransformer
from labram.models.vqnsp import VQNSP, load_vqnsp_weights
from labram.runners.finetune_setup import load_finetune_checkpoint
from labram.utils.checkpoint import load_pretrained_weights


PATCH_SIZE = 200
EMBED_DIM = 200  # forced by TemporalConv output dim (25 patches * 8 out_chans)


# ---------------------------------------------------------------------------
# model builders (mirror the tiny configs used elsewhere in the suite)
# ---------------------------------------------------------------------------

def _make_ft_model(num_classes: int) -> NeuralTransformer:
    return NeuralTransformer(
        eeg_window_size=PATCH_SIZE, patch_size=PATCH_SIZE, in_chans=1, out_chans=8,
        num_classes=num_classes, embed_dim=EMBED_DIM, depth=2, num_heads=10,
        init_values=0.1, qkv_bias=True, use_abs_pos_emb=False, use_rel_pos_bias=True,
    )


def _base_transformer_config():
    return dict(
        eeg_window_size=400, patch_size=PATCH_SIZE, in_chans=1, out_chans=8,
        num_classes=0, embed_dim=EMBED_DIM, depth=2, num_heads=10, mlp_ratio=4.0,
        qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
        drop_path_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.1, use_abs_pos_emb=True, use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001,
    )


def _make_tiny_vqnsp() -> VQNSP:
    encoder_config = _base_transformer_config()
    decoder_config = _base_transformer_config()
    decoder_config['eeg_window_size'] = 400 // PATCH_SIZE  # = 2
    decoder_config['patch_size'] = 1
    decoder_config['in_chans'] = 8
    decoder_config['depth'] = 1
    return VQNSP(
        encoder_config, decoder_config,
        num_codebook_tokens=32, quantizer_dim=8, decoder_out_dim=PATCH_SIZE,
        quantize_kmeans_init=False,
    )


def _perturb(model: nn.Module) -> None:
    """Move params away from fresh init so a successful load is observable."""
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
        sd = load_pretrained_weights(str(path))
        assert set(sd.keys()) == {"w"}

    def test_state_dict_layout(self, tmp_path):
        path = tmp_path / "s.pth"
        torch.save({"state_dict": {"w": torch.ones(3)}}, path)
        sd = load_pretrained_weights(str(path))
        assert set(sd.keys()) == {"w"}

    def test_bare_dict_layout(self, tmp_path):
        path = tmp_path / "b.pth"
        torch.save({"w": torch.ones(3)}, path)
        sd = load_pretrained_weights(str(path))
        assert set(sd.keys()) == {"w"}


# ---------------------------------------------------------------------------
# load_vqnsp_weights: tokenizer load for the pre-training phase
# ---------------------------------------------------------------------------

class TestLoadVQNSPWeights:
    def test_roundtrip_matches_and_filters_aux_keys(self, tmp_path):
        source = _make_tiny_vqnsp()
        _perturb(source)
        src_sd = source.state_dict()

        # Save with extra loss/teacher/scaling keys that the loader must drop.
        payload = {k: v.clone() for k, v in src_sd.items()}
        payload["loss.fake"] = torch.zeros(1)
        payload["teacher.fake"] = torch.zeros(1)
        payload["scaling.fake"] = torch.zeros(1)
        path = tmp_path / "vqnsp.pth"
        torch.save({"model": payload}, path)

        target = _make_tiny_vqnsp()
        # A representative param differs before the load.
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
        """Simulate a masked-EEG pretrain checkpoint: 'student.'-prefixed keys."""
        sd = source.state_dict()
        ckpt = {"model": {f"student.{k}": v.clone() for k, v in sd.items()}}
        path = tmp_path / "pretrain.pth"
        torch.save(ckpt, path)
        return path

    def _args(self, path):
        return SimpleNamespace(
            finetune=str(path), model_key="model",
            model_filter_name="student.", model_prefix="",
        )

    def test_backbone_loads_and_mismatched_head_dropped(self, tmp_path):
        # Pretrained backbone with a 5-class head; fine-tune target wants 1 class.
        source = _make_ft_model(num_classes=5)
        _perturb(source)
        src_sd = source.state_dict()
        path = self._save_pretrained(tmp_path, source)

        target = _make_ft_model(num_classes=1)
        probe = "blocks.0.mlp.fc1.weight"
        assert not torch.allclose(target.state_dict()[probe], src_sd[probe])

        load_finetune_checkpoint(target, self._args(path))

        tgt_sd = target.state_dict()
        # Every shared backbone weight now equals the pretrained source.
        for k, v in src_sd.items():
            if k.startswith("head.") or "relative_position_index" in k:
                continue
            assert torch.allclose(tgt_sd[k], v), f"backbone not transferred: {k}"
        # The mismatched head was dropped, not overwritten -> keeps target shape.
        assert tgt_sd["head.weight"].shape[0] == 1

    def test_matching_head_is_transferred(self, tmp_path):
        # Same num_classes on both sides -> head should load too.
        source = _make_ft_model(num_classes=3)
        _perturb(source)
        src_sd = source.state_dict()
        path = self._save_pretrained(tmp_path, source)

        target = _make_ft_model(num_classes=3)
        load_finetune_checkpoint(target, self._args(path))

        assert torch.allclose(target.state_dict()["head.weight"], src_sd["head.weight"])

    def test_loaded_model_runs_forward(self, tmp_path):
        source = _make_ft_model(num_classes=2)
        _perturb(source)
        path = self._save_pretrained(tmp_path, source)
        target = _make_ft_model(num_classes=2)
        load_finetune_checkpoint(target, self._args(path))

        target.eval()
        with torch.no_grad():
            out = target(torch.randn(2, 4, 1, 200), channel_indices=None)
        assert out.shape == (2, 2)


# ---------------------------------------------------------------------------
# Phase-B wiring: channel helpers used by the forward path are the same object
# whether reached via labram.data or the labram.utils re-export.
# ---------------------------------------------------------------------------

def test_channel_helpers_reexport_identity():
    import labram.data as data
    import labram.utils as utils

    assert utils.get_channel_indices is data.get_channel_indices
    assert utils.standard_1020 is data.standard_1020
    assert utils.build_pretraining_dataset is data.build_pretraining_dataset
