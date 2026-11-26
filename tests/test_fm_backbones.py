"""Foreign pretrained encoders (CBraMod, REVE) behind the fine-tune model interface."""
import pytest
import torch
from timm.models import create_model

import labram.models.registry  # noqa: F401
from labram.data.eeg_constants import get_channel_indices
from labram.models.backbones import CBRAMOD_CHANNELS, CBraModBackbone, REVEBackbone

NAMES = list(CBRAMOD_CHANNELS)                       # the 19 channels of the brain-age cohorts
TINY = ["cbramod_tiny_patch200", "reve_tiny_patch200"]


def _model(name, **kw):
    torch.manual_seed(0)
    return create_model(name, num_classes=1, **kw).eval()


def _x(batch=3, channels=19, patches=2):
    return torch.randn(batch, channels, patches, 200, generator=torch.Generator().manual_seed(1)) * 0.3


@pytest.mark.parametrize("name", TINY)
def test_call_contract_matches_the_labram_model(name):
    m, x, ci = _model(name), _x(), get_channel_indices(NAMES)
    out = m(x, ci)
    out2, feat = m(x, channel_indices=ci, return_features=True, classify_only=True, condition=None)
    assert out.shape == (3, 1) and torch.allclose(out, out2)
    assert feat.shape == (3, m.num_features) and m.num_features == m.embed_dim == m.head.in_features
    tokens = m.forward_tokens(x, ci)
    patches = 2 if name.startswith("cbramod") else (2 * 200 - 200) // 180 + 1     # REVE: 20-sample overlap
    assert tokens.shape == (3, 19, patches, m.embed_dim)
    assert m.forward_features(x, ci, return_patch_tokens=True).shape == (3, 19 * patches, m.embed_dim)
    assert m.patch_size == 200 and m.get_num_layers() == 2
    assert m.head_dtype() == torch.float32


@pytest.mark.parametrize("name", TINY)
def test_channel_names_come_from_the_indices(name):
    m, x = _model(name), _x()
    with pytest.raises(ValueError, match="channel_indices"):
        m(x)                                                   # None: no names
    with pytest.raises(ValueError, match="channel_indices"):
        m(x, list(range(1, 20)))                               # no class-token slot
    with pytest.raises(ValueError, match="channels in the input"):
        m(x[:, :5], get_channel_indices(NAMES))


def test_cbramod_places_channels_in_its_fixed_layout():
    m, x, ci = _model("cbramod_tiny_patch200"), _x(), get_channel_indices(NAMES)
    full = m.forward_tokens(x, ci)
    # A permuted channel order gives the same tokens, permuted back to the caller's order.
    perm = torch.randperm(19, generator=torch.Generator().manual_seed(2)).tolist()
    ci_perm = [ci[0]] + [ci[i + 1] for i in perm]
    assert torch.allclose(m.forward_tokens(x[:, perm], ci_perm), full[:, perm], atol=1e-5)
    # A subset runs as the full layout with zeros in the empty slots; its tokens are returned.
    keep = [0, 3, 18]
    zeroed = torch.zeros_like(x)
    zeroed[:, keep] = x[:, keep]
    ci_keep = [ci[0]] + [ci[i + 1] for i in keep]
    assert torch.allclose(m.forward_tokens(x[:, keep], ci_keep), m.forward_tokens(zeroed, ci)[:, keep], atol=1e-5)
    with pytest.raises(ValueError, match="no slot"):           # A1 is not one of the 19
        m(x[:, :1], get_channel_indices(["A1"]))


def test_reve_looks_up_positions_and_is_a_set_model():
    m, x, ci = _model("reve_tiny_patch200"), _x(), get_channel_indices(NAMES)
    full = m.forward_tokens(x, ci)
    perm = torch.randperm(19, generator=torch.Generator().manual_seed(2)).tolist()
    ci_perm = [ci[0]] + [ci[i + 1] for i in perm]
    assert torch.allclose(m.forward_tokens(x[:, perm], ci_perm), full[:, perm], atol=1e-4)
    bank = m.backbone._bank
    assert bank["T3"] == bank["T7"] and bank["T5"] == bank["P7"]          # old and new 10-20 names
    assert all(n in bank for n in NAMES + ["A1", "A2"])
    with pytest.raises(ValueError, match="no electrode position"):        # T1 has no position
        m(x[:, :1], get_channel_indices(["T1"]))
    assert m.backbone.positions(NAMES).shape == (19, 3)
    assert not any("pos" in k for k in m.state_dict())                    # positions are not weights


def test_reve_attention_readout_uses_the_pretrained_query():
    mean = _model("reve_tiny_patch200")
    attn = _model("reve_tiny_patch200", encoder={"readout": "attn"})
    assert not mean.backbone.net.cls_query_token.requires_grad            # unused by a mean readout
    assert attn.backbone.net.cls_query_token.requires_grad
    x, ci = _x(), get_channel_indices(NAMES)
    assert attn(x, ci).shape == (3, 1)
    with pytest.raises(NotImplementedError):                              # CBraMod has no pooling of its own
        _model("cbramod_tiny_patch200", encoder={"readout": "attn"})(x, ci)
    with pytest.raises(ValueError, match="readout"):
        _model("cbramod_tiny_patch200", encoder={"readout": "cls"})


@pytest.mark.parametrize("name", TINY)
def test_layer_ids_cover_stem_layers_and_head(name):
    from labram.optim_factory import LayerDecayValueAssigner, get_parameter_groups
    m = _model(name)
    layers = m.get_num_layers()
    ids = {n: m.layer_id(n) for n, _ in m.named_parameters()}
    assert min(ids.values()) == 0 and max(ids.values()) == layers + 1
    assert set(ids.values()) == set(range(layers + 2))
    assert all(v == layers + 1 for n, v in ids.items() if n.startswith(("head.", "fc_norm.")))
    stem = [n for n, v in ids.items() if v == 0]
    assert stem and all(n.startswith("backbone.net.") for n in stem)
    decay = 0.5
    assigner = LayerDecayValueAssigner([decay ** (layers + 1 - i) for i in range(layers + 2)],
                                       layer_id_fn=m.layer_id)
    groups = get_parameter_groups(m, 0.05, m.no_weight_decay(), assigner.get_layer_id, assigner.get_scale)
    scales = sorted({g["lr_scale"] for g in groups})
    assert scales[0] == pytest.approx(decay ** (layers + 1)) and scales[-1] == 1.0
    grouped = sum(len(g["params"]) for g in groups)
    assert grouped == sum(p.requires_grad for p in m.parameters())
    bad = LayerDecayValueAssigner([1.0] * (layers + 2), layer_id_fn=lambda n: layers + 2)
    with pytest.raises(ValueError, match="outside"):
        bad.get_layer_id("head.weight")


@pytest.mark.parametrize("name", TINY)
def test_state_dict_is_stable_and_loads_strictly(name):
    a, b = _model(name), create_model(name, num_classes=1)
    b.load_state_dict(a.state_dict())                          # strict, as the evaluation scripts load
    x, ci = _x(), get_channel_indices(NAMES)
    assert torch.allclose(a(x, ci), b.eval()(x, ci))
    a(x[:, :4], [ci[0]] + ci[1:5])                             # a forward creates no new state
    assert set(a.state_dict()) == set(b.state_dict())


def test_released_weight_files_load_strictly(tmp_path):
    from safetensors.torch import save_file
    # CBraMod: a torch state dict that also holds the pretraining output layer.
    src = CBraModBackbone(n_layer=2, dim_feedforward=64, keep_recon_head=True)
    path = tmp_path / "cbramod.pth"
    torch.save(src.net.state_dict(), path)
    assert any(k.startswith("proj_out.") for k in src.net.state_dict())
    dst = CBraModBackbone(n_layer=2, dim_feedforward=64)
    dst.load_pretrained(str(path))
    assert all(torch.equal(v, src.net.state_dict()[k]) for k, v in dst.net.state_dict().items())
    with pytest.raises(ValueError, match="does not match"):    # another depth
        CBraModBackbone(n_layer=3, dim_feedforward=64).load_pretrained(str(path))
    # REVE: safetensors.
    src = REVEBackbone(embed_dim=64, depth=2, heads=2, head_dim=16)
    path = tmp_path / "reve.safetensors"
    save_file(src.net.state_dict(), str(path))
    dst = REVEBackbone(embed_dim=64, depth=2, heads=2, head_dim=16)
    dst.load_pretrained(str(path))
    assert all(torch.equal(v, src.net.state_dict()[k]) for k, v in dst.net.state_dict().items())
    state = dict(src.net.state_dict())
    state["extra.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected"):
        dst.load_pretrained(state)


def test_finetune_checkpoint_loader_tells_released_weights_from_run_checkpoints(tmp_path):
    from labram.configs.model_config import FinetuneCheckpointConfig
    from labram.runs.finetune_setup import load_finetune_checkpoint
    src = _model("cbramod_tiny_patch200")
    with torch.no_grad():
        for p in src.parameters():
            p.add_(0.01)
    released, run = tmp_path / "released.pth", tmp_path / "checkpoint-best.pth"
    torch.save(src.backbone.net.state_dict(), released)
    torch.save({"model": src.state_dict(), "epoch": 3, "args": object}, run)   # not weights-only loadable

    dst = create_model("cbramod_tiny_patch200", num_classes=1)
    load_finetune_checkpoint(dst, FinetuneCheckpointConfig(finetune=str(released)))
    assert all(torch.equal(v, src.backbone.state_dict()[k]) for k, v in dst.backbone.state_dict().items())
    assert not torch.equal(dst.head.weight, src.head.weight)   # the released file holds no head

    dst = create_model("cbramod_tiny_patch200", num_classes=1)
    load_finetune_checkpoint(dst, FinetuneCheckpointConfig(finetune=str(run)))
    assert all(torch.equal(v, src.state_dict()[k]) for k, v in dst.state_dict().items())
    with pytest.raises(ValueError, match="local or s3"):
        load_finetune_checkpoint(dst, FinetuneCheckpointConfig(finetune="https://example.org/w.pth"))


def test_input_preprocessing_is_model_owned():
    from labram.configs.labram_plus_config import LaBraMPlusConfig
    from labram.models.fm_regressor import zscore_window
    x = _x()
    x[:, 4] = 0.0                                              # a missing channel
    z = zscore_window(x * 50, clip_sd=2.0)
    assert z.shape == x.shape and (z[:, 4] == 0).all()
    flat = z[:, 0].reshape(3, -1)
    assert flat.abs().max() <= 2.0 and abs(float(flat.mean())) < 0.05
    # z-scoring makes the model blind to a global gain; the raw unit does not.
    ci = get_channel_indices(NAMES)
    zs = _model("cbramod_tiny_patch200", encoder={"input_norm": "zscore_window"})
    raw = _model("cbramod_tiny_patch200")
    assert torch.allclose(zs(x, ci), zs(3 * x, ci), atol=1e-4)
    assert not torch.allclose(raw(x, ci), raw(3 * x, ci), atol=1e-4)
    # CAR (labram_plus) and the clip are applied inside the model.
    car = _model("cbramod_tiny_patch200", labram_plus=LaBraMPlusConfig(enabled=True, common_average_reference=True,
                                                                         z_score_patches=False))
    shifted = x + 0.7                                          # a common offset on every channel
    assert torch.allclose(car(x, ci), car(shifted, ci), atol=1e-4)
    clip = _model("cbramod_tiny_patch200", input_clip=0.5)
    assert float(clip.maybe_preprocess_input(x * 10).abs().max()) == pytest.approx(0.5)


def test_factory_builds_foreign_models_and_rejects_labram_only_options():
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.eval.loading import build_finetune_model
    from labram.models.fm_regressor import FoundationRegressor
    from labram.runs import run_finetune
    base = {"finetune_checkpoint.finetune": "/nonexistent/weights.pth",      # never opened by the factory
            "model.nb_classes": 1, "model.task": "regression", "model.model": "cbramod_tiny_patch200"}

    def cfg(**over):
        return FinetuneRunConfig.load_config("labram/configs/defaults/finetune_tuab_age.json", **{**base, **over})
    model = run_finetune.get_model(cfg(**{"mil.enabled": True, "model.head_type": "mlp"}))
    assert isinstance(model, FoundationRegressor) and hasattr(model, "mil")
    assert set(build_finetune_model(cfg(**{"mil.enabled": True, "model.head_type": "mlp"})).state_dict()) \
        == set(model.state_dict())                              # evaluation rebuilds the same model
    assert model.labram_plus.use_car                             # the age config's CAR applies to it too
    for over in ({"model.lora.enabled": True}, {"model.head_type": "attention"},
                 {"model.condition_on_abnormal": True}, {"model.depth": 6},
                 {"codebook_aux.enabled": True}):
        with pytest.raises(ValueError, match="foreign encoder|not supported"):
            run_finetune.get_model(cfg(**over))
    native = run_finetune.get_model(cfg(**{"model.model": "labram_base_patch200_200"}))
    assert not isinstance(native, FoundationRegressor) and not hasattr(native, "layer_id")


def test_reve_stochastic_depth_is_optional_and_adds_no_weights():
    plain = _model("reve_tiny_patch200")
    reg = _model("reve_tiny_patch200", encoder={"drop_path": 0.5})
    assert set(reg.state_dict()) == set(plain.state_dict())    # the released weights still load strictly
    reg.load_state_dict(plain.state_dict())
    assert reg.backbone.net.transformer.drop_path_rates == [0.0, 0.5]     # linear over the 2 layers
    assert plain.backbone.net.transformer.drop_path_rates == [0.0, 0.0]
    x, ci = _x(batch=16), get_channel_indices(NAMES)
    assert torch.allclose(reg(x, ci), plain(x, ci), atol=1e-6)            # evaluation: no layer is dropped
    reg.train()
    torch.manual_seed(0)
    a = reg.forward_tokens(x, ci)
    b = reg.forward_tokens(x, ci)
    assert not torch.allclose(a, b)                                       # training: branches drop per sample
    plain.train()
    assert torch.allclose(plain.forward_tokens(x, ci), plain.forward_tokens(x, ci))
    with pytest.raises(ValueError, match="drop_path"):                    # CBraMod has its own dropout
        _model("cbramod_tiny_patch200", encoder={"drop_path": 0.1})


def test_cbramod_flatten_readout_projects_every_token_of_the_layout():
    m = _model("cbramod_tiny_patch200", encoder={"readout": "flatten", "readout_dropout": 0.5}, window_patches=2)
    x, ci = _x(), get_channel_indices(NAMES)
    assert m.token_grid == (19, 2) and m.readout_proj.in_features == 19 * 2 * m.embed_dim
    out, feat = m(x, ci, return_features=True)
    assert out.shape == (3, 1) and feat.shape == (3, m.embed_dim)
    assert m.layer_id("readout_proj.weight") == m.get_num_layers() + 1          # trains at the head's rate
    # The feature is tied to the layout, not to the caller's channel order ...
    perm = torch.randperm(19, generator=torch.Generator().manual_seed(2)).tolist()
    assert torch.allclose(m(x[:, perm], [ci[0]] + [ci[i + 1] for i in perm]), out, atol=1e-5)
    # ... and a missing channel is a flat slot.
    keep = [0, 3, 18]
    zeroed = torch.zeros_like(x)
    zeroed[:, keep] = x[:, keep]
    assert torch.allclose(m(x[:, keep], [ci[0]] + [ci[i + 1] for i in keep]), m(zeroed, ci), atol=1e-5)
    assert m.forward_features(x, ci, return_patch_tokens=True).shape == (3, 19 * 2, m.embed_dim)
    with pytest.raises(ValueError, match="window_sec"):                         # built for 2 patches
        m(_x(patches=3), ci)
    m.train()
    assert not torch.allclose(m(x, ci), m(x, ci))                               # readout dropout
    plain = _model("cbramod_tiny_patch200")
    assert plain.readout_proj is None and "readout_proj.weight" not in plain.state_dict()
    with pytest.raises(NotImplementedError, match="fixed token grid"):          # REVE has no fixed layout
        _model("reve_tiny_patch200", encoder={"readout": "flatten"})


def test_encoder_lr_scale_puts_every_layer_below_the_head():
    from labram.optim_factory import LayerDecayValueAssigner, get_parameter_groups, layer_scale_values
    assert layer_scale_values(2, 0.5) == [0.5 ** (3 - i) for i in range(4)]     # the usual layer decay
    assert layer_scale_values(2, 1.0, 0.2) == [0.2, 0.2, 0.2, 1.0]
    assert layer_scale_values(2, 0.5, 0.2) == pytest.approx([0.025, 0.05, 0.1, 1.0])
    m = _model("reve_tiny_patch200", encoder={"readout": "attn"})
    assigner = LayerDecayValueAssigner(layer_scale_values(m.get_num_layers(), 1.0, 0.2), layer_id_fn=m.layer_id)
    groups = get_parameter_groups(m, 0.05, m.no_weight_decay(), assigner.get_layer_id, assigner.get_scale)
    by_param = {id(p): g["lr_scale"] for g in groups for p in g["params"]}
    for n, p in m.named_parameters():
        encoder = n.startswith("backbone.") and n != "backbone.net.cls_query_token"
        assert by_param[id(p)] == (0.2 if encoder else 1.0), n


@pytest.mark.parametrize("readout", ["attn", "mean"])
def test_probe_stage_freezes_the_encoder_and_releases_it(readout):
    m = _model("reve_tiny_patch200", encoder={"readout": readout})
    before = {n: p.requires_grad for n, p in m.named_parameters()}
    assert before["backbone.net.cls_query_token"] == (readout == "attn")
    m.set_encoder_trainable(False)
    frozen = {n for n, p in m.named_parameters() if not p.requires_grad}
    assert {n for n in before if n.startswith(("head.", "fc_norm."))}.isdisjoint(frozen)
    assert all(n in frozen for n in before if n.startswith("backbone.net.transformer."))
    # The pooling query is part of the probe when it is the readout.
    assert ("backbone.net.cls_query_token" in frozen) == (readout == "mean")
    m.set_encoder_trainable(True)
    assert {n: p.requires_grad for n, p in m.named_parameters()} == before


def test_reve_position_noise_only_in_training():
    plain = _model("reve_tiny_patch200")
    noisy = _model("reve_tiny_patch200", encoder={"position_noise": 0.0025})
    assert set(noisy.state_dict()) == set(plain.state_dict())
    noisy.load_state_dict(plain.state_dict())
    x, ci = _x(), get_channel_indices(NAMES)
    assert torch.allclose(noisy(x, ci), plain(x, ci), atol=1e-6)                # evaluation: exact positions
    noisy.train()
    a, b = noisy.forward_tokens(x, ci), noisy.forward_tokens(x, ci)
    assert not torch.allclose(a, b)                                             # a new draw per forward
    assert torch.equal(noisy.backbone.positions(NAMES), plain.backbone.positions(NAMES))   # the table is untouched
    with pytest.raises(ValueError, match="position_noise"):
        _model("cbramod_tiny_patch200", encoder={"position_noise": 0.0025})


def test_encoder_options_are_validated():
    from labram.configs.run_configs import FinetuneRunConfig
    base = {"model.nb_classes": 1, "model.task": "regression", "model.model": "cbramod_tiny_patch200"}
    for over in ({"model.encoder.readout": "max"}, {"model.encoder.readout_dropout": 1.0},
                 {"model.encoder.lr_scale": 0.0}, {"model.encoder.probe_epochs": -1},
                 {"model.encoder.position_noise": -0.1}):
        cfg = FinetuneRunConfig.load_config("labram/configs/defaults/finetune_tuab_age.json", **{**base, **over})
        with pytest.raises(ValueError, match="model.encoder"):
            cfg.model.validate()
