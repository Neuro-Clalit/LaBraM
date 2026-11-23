"""codebook_aux: quantization / amplitude-reconstruction terms from a frozen
VQNSP projection layer + codebook + decoder, added to a plain fine-tune (window
level and MIL), with the model's state_dict left untouched."""
import glob
import json
import os

import pytest
import torch
from einops import rearrange


def _tokenizer(n_codes=32, dim=64):
    from timm.models import create_model
    import labram.models.registry  # noqa: F401
    return create_model("vqnsp_encoder_base_decoder_3x200x12", num_codebook_tokens=n_codes,
                        quantizer_dim=dim, quantize_kmeans_init=False)


def _tokenizer_file(tmp_path):
    path = tmp_path / "vqnsp_proj.pth"
    torch.save({"model": _tokenizer().state_dict()}, path)
    return str(path)


def _aux(**kw):
    from labram.models.codebook_aux import CodebookAuxiliary
    tok = _tokenizer()
    return CodebookAuxiliary(tok, **kw), tok


def test_quantize_equals_the_tokenizer_quantizer():
    torch.manual_seed(0)
    aux, tok = _aux(embedding_weight=1.0)
    tok.eval()
    tokens = torch.randn(3, 5 * 4, 200)                       # 5 channels x 4 patches
    quantized, loss, idx = aux.quantize(tokens)
    ref_q, ref_loss, ref_idx = tok.quantize_patch_tokens(tokens, num_channels=5)
    assert torch.equal(idx.reshape(-1), ref_idx.reshape(-1))
    assert torch.allclose(loss, ref_loss, atol=1e-7)
    assert torch.allclose(quantized, rearrange(ref_q, "b c h w -> b (h w) c"), atol=1e-6)


def test_reconstruction_equals_the_tokenizer_decoder_and_is_frozen():
    from labram import utils
    from labram.configs.loss_config import LossConfig
    from labram.losses.spectral import SpectralReconstructionLoss
    torch.manual_seed(0)
    aux, tok = _aux(amplitude_weight=1.0, freq_fraction=0.5)
    tok.eval()
    ci = utils.get_channel_indices(["FP1", "FP2", "C3", "C4", "O1"])
    x = torch.randn(2, 5, 4, 200)
    tokens = torch.randn(2, 5 * 4, 200, requires_grad=True)
    quantized, _, _ = aux.quantize(tokens)
    loss = aux.reconstruction_loss(quantized, x, ci)
    ref_q, _, _ = tok.quantize_patch_tokens(tokens, num_channels=5)
    magnitude, _ = tok.decode_spectrum(ref_q, ci)
    spectral = SpectralReconstructionLoss(LossConfig(freq_fraction=0.5))
    target, _ = spectral.spectrum_targets(x)
    assert target.shape[-1] == 100
    assert torch.allclose(loss, spectral.reconstruction_loss(magnitude, target), atol=1e-6)
    loss.backward()
    assert tokens.grad.abs().sum() > 0                         # the gradient passes through ...
    assert all(not p.requires_grad and p.grad is None for p in aux.parameters())   # ... nothing in it trains
    aux.train(True)
    assert not aux.training and not aux.decoder.training      # always eval


def _model(mean_pool, tmp_path, **aux_kw):
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.runs.run_finetune import get_model
    over = {"model.use_mean_pooling": mean_pool, "codebook_aux.enabled": True,
            "codebook_aux.tokenizer_weight": _tokenizer_file(tmp_path), **aux_kw}
    cfg = FinetuneRunConfig.load_config("labram/configs/defaults/finetune_tuab_age.json", **over)
    return get_model(cfg), cfg


@pytest.mark.parametrize("mean_pool", [False, True])
def test_attached_auxiliary_stays_out_of_the_model_and_collects_per_step(tmp_path, mean_pool):
    from labram import utils
    from labram.models.codebook_aux import get_codebook_aux
    from labram.runs.run_finetune import get_model
    model, cfg = _model(mean_pool, tmp_path, **{"codebook_aux.amplitude_weight": 1.0,
                                                "codebook_aux.embedding_weight": 2.0})
    aux = get_codebook_aux(model)
    assert aux is not None
    cfg.codebook_aux.enabled = False
    plain = get_model(cfg)
    assert list(model.state_dict()) == list(plain.state_dict())            # checkpoints are unchanged
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in plain.parameters())
    assert not any(k.startswith("codebook_aux") for k in model.state_dict())

    ci = utils.get_channel_indices(["FP1", "FP2", "C3", "C4", "O1"])
    x = torch.randn(2, 5, 10, 200)
    model.train()
    out_plain = model(x, ci)                                               # not started: nothing collected
    assert aux.finish() == (None, {})
    aux.start()
    out = model(x, ci)
    model(x[:1, :3], ci[:4])                                               # a second channel pattern
    loss, stats = aux.finish()
    assert out.shape == out_plain.shape
    assert set(stats) == {"aux_quant_loss", "aux_recon_loss", "aux_codes", "aux_loss"}
    assert loss.item() == pytest.approx(stats["aux_recon_loss"] + 2.0 * stats["aux_quant_loss"], rel=1e-5)
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if n.startswith("blocks.0.")]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)         # the encoder gets the gradient
    aux.start()
    with torch.no_grad():                                                  # eval-style forward: ignored
        model(x, ci)
    assert aux.finish() == (None, {})


def test_zero_weight_reconstruction_is_monitored_without_gradient(tmp_path):
    from labram import utils
    from labram.models.codebook_aux import get_codebook_aux
    model, _ = _model(False, tmp_path, **{"codebook_aux.embedding_weight": 10.0,
                                          "codebook_aux.monitor_every": 2})
    aux = get_codebook_aux(model)
    ci = utils.get_channel_indices(["FP1", "FP2", "C3"])
    x = torch.randn(2, 3, 10, 200)
    model.train()
    seen = []
    for _ in range(4):
        aux.start()
        model(x, ci)
        loss, stats = aux.finish()
        seen.append("aux_recon_loss" in stats)
        assert loss.item() == pytest.approx(10.0 * stats["aux_quant_loss"], rel=1e-5)
    assert seen == [True, False, True, False]


def test_frequency_band_scores_only_its_bins():
    """freq_band_hz=[1, 45]: the scored loss is the mean squared error over bins
    1..45 of the same decoder output and target as the full loss; the target is
    standardised over all kept bins either way."""
    from labram import utils
    torch.manual_seed(0)
    full_aux, tok = _aux(amplitude_weight=1.0, freq_fraction=0.5)
    band_aux, _ = _aux(amplitude_weight=1.0, freq_fraction=0.5, freq_band_hz=[1, 45])
    band_aux.load_state_dict(full_aux.state_dict())
    ci = utils.get_channel_indices(["FP1", "FP2", "C3", "C4", "O1"])
    x = torch.randn(2, 5, 4, 200)
    q, _, _ = full_aux.quantize(torch.randn(2, 5 * 4, 200))
    full, full_again = full_aux.reconstruction_losses(q, x, ci)
    assert full is full_again
    band, band_full = band_aux.reconstruction_losses(q, x, ci)
    assert torch.allclose(band_full, full, atol=1e-6)
    grid = q.view(2, 5, -1, q.shape[-1]).permute(0, 3, 1, 2)
    magnitude = band_aux.decode_task_layer_magnitude(band_aux.decoder(grid, ci, return_patch_tokens=True))
    target, _ = band_aux.spectral.spectrum_targets(x)
    target = rearrange(target, "b n a c -> b (n a) c")
    assert band_aux.band_bins(200, target.shape[-1]) == (1, 45)
    expected = (magnitude[..., 1:46] - target[..., 1:46]).pow(2).mean()
    assert torch.allclose(band, expected, atol=1e-6) and not torch.allclose(band, full)


def test_config_validation(tmp_path):
    from labram.configs.train_config import CodebookAuxConfig
    CodebookAuxConfig().validate()                                         # off: nothing to check
    with pytest.raises(ValueError, match="tokenizer_weight"):
        CodebookAuxConfig(enabled=True, amplitude_weight=1.0).validate()
    with pytest.raises(ValueError, match="amplitude_weight > 0 or embedding_weight > 0"):
        CodebookAuxConfig(enabled=True, tokenizer_weight="x.pth").validate()
    with pytest.raises(ValueError, match="freq_fraction"):
        CodebookAuxConfig(enabled=True, tokenizer_weight="x.pth", amplitude_weight=1.0,
                          freq_fraction=0.0).validate()
    ok = dict(enabled=True, tokenizer_weight="x.pth", amplitude_weight=1.0)
    CodebookAuxConfig(**ok, freq_band_hz=[1, 45]).validate()
    for bad in ([45, 1], [1], [1, 100], [-1, 45]):           # reversed, one value, above the kept 0-99 Hz, negative
        with pytest.raises(ValueError, match="freq_band_hz"):
            CodebookAuxConfig(**ok, freq_band_hz=bad).validate()


def test_missing_weight_warns_at_build_and_fails_a_training_run(tmp_path):
    from labram.models.codebook_aux import get_codebook_aux
    from labram.runs import run_finetune
    from test_mil_e2e import _cfg
    cfg, _ = _cfg(tmp_path, **{"mil.enabled": False, "codebook_aux.enabled": True,
                               "codebook_aux.amplitude_weight": 1.0,
                               "codebook_aux.tokenizer_weight": "/opt/ml/input/data/codebook_aux/gone.pth"})
    assert get_codebook_aux(run_finetune.get_model(cfg)) is None           # evaluation can rebuild the model
    with pytest.raises(FileNotFoundError, match="codebook_aux.tokenizer_weight"):
        run_finetune.main(cfg)


def _epochs(out):
    (log,) = glob.glob(os.path.join(str(out) + "*", "log.txt"))
    return [json.loads(line) for line in open(log) if line.startswith("{")]


@pytest.mark.parametrize("weights", [{"codebook_aux.amplitude_weight": 1.0},
                                     {"codebook_aux.amplitude_weight": 1.0, "codebook_aux.freq_band_hz": [1.0, 45.0]},
                                     {"codebook_aux.embedding_weight": 10.0, "codebook_aux.monitor_every": 1}])
def test_p4_recipe_with_codebook_aux(tmp_path, weights):
    """Window phase, attention-only epoch (no auxiliary), joint bag phase; the
    checkpoint is that of a run without the auxiliary."""
    from labram.runs import run_finetune
    from test_mil_e2e import _cfg
    tok = _tokenizer_file(tmp_path)
    cfg, out = _cfg(tmp_path, **{
        "mil.pool": "gated", "mil.gated": False, "mil.score_input": "h", "trainer.epochs": 3,
        "mil.window_epochs": 1, "mil.window_minutes_per_epoch": 0.3334, "mil.window_bag_repeats": 2,
        "mil.window_batch_size": 4, "mil.freeze_encoder_epochs": 1, "mil.freeze_head": True,
        "mil.head_lr_scale": 5.0, "mil.window_loss_weight": 0.5, "mil.bag_repeats": 2,
        "mixup.enabled": True, "model.use_mean_pooling": False,
        "data.drop_channels": ["A1", "A2", "T1", "T2"],
        "channel_drop.enabled": True, "channel_drop.prob": 0.0, "channel_drop.consistency_weight": 0.0,
        "channel_drop.eval_consistency": False,
        "codebook_aux.enabled": True, "codebook_aux.tokenizer_weight": tok, **weights})
    summary = run_finetune.main(cfg)
    assert summary["best_val_stats"]["mae"] > 0
    rows = _epochs(out)
    assert len(rows) == 3
    for row in (rows[0], rows[2]):                             # window phase, joint bag phase
        assert row["train_aux_quant_loss"] > 0 and row["train_aux_recon_loss"] > 0
        assert row["train_aux_codes"] >= 1 and row["train_aux_task_loss"] > 0
        assert row["train_loss"] == pytest.approx(row["train_aux_task_loss"] + row["train_aux_loss"], rel=1e-3)
        assert ("train_aux_recon_full_loss" in row) == ("codebook_aux.freq_band_hz" in weights)
    assert not any(k.startswith("train_aux") for k in rows[1])     # attention only: encoder frozen
    (ckpt,) = glob.glob(os.path.join(str(out) + "*", "checkpoint-best.pth"))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    assert not any("codebook_aux" in k for k in state)
    os.remove(tok)                                             # as in evaluation: no tokenizer file
    model = run_finetune.get_model(cfg)
    model.load_state_dict(state)                               # strict


def test_window_level_run_with_mean_pooling(tmp_path):
    from labram.runs import run_finetune
    from test_mil_e2e import _cfg
    cfg, out = _cfg(tmp_path, **{"mil.enabled": False, "trainer.epochs": 1, "codebook_aux.enabled": True,
                                 "codebook_aux.tokenizer_weight": _tokenizer_file(tmp_path),
                                 "codebook_aux.amplitude_weight": 0.5, "codebook_aux.embedding_weight": 0.5})
    assert run_finetune.main(cfg)["best_val_stats"]["mae"] > 0
    (row,) = _epochs(out)
    assert row["train_aux_recon_loss"] > 0 and row["train_aux_quant_loss"] > 0


def test_sagemaker_stages_the_tokenizer_weight_only_when_enabled():
    import labram.runs.submit_sagemaker as sub
    from labram.configs.run_configs import FinetuneRunConfig

    def config(enabled):
        c = FinetuneRunConfig()
        c.data.data_path = "s3://bucket/data/cohort"
        c.finetune_checkpoint.finetune = "s3://bucket/ckpts/labram-base.pth"
        c.codebook_aux.tokenizer_weight = "s3://bucket/ckpts/vqnsp_proj.pth"
        c.codebook_aux.enabled = enabled
        return c
    c = config(True)
    staged = sub.stage_s3_inputs(c, "finetune")
    assert staged.channels["codebook_aux"] == "s3://bucket/ckpts/vqnsp_proj.pth"
    assert "codebook_aux" in staged.object_channels
    assert c.codebook_aux.tokenizer_weight == "/opt/ml/input/data/codebook_aux/vqnsp_proj.pth"
    assert "codebook_aux" not in sub.stage_s3_inputs(config(False), "finetune").channels
