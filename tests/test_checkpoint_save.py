"""Tests for checkpoint naming, the 'final'/'best' string-epoch tags used by
save_only_final_model, and final-checkpoint resolution."""

import torch

from labram.configs.train_config import OutputConfig, TrainerConfig
from labram.utils.checkpoint import save_model
from labram.utils.clearml_artifacts import resolve_final_checkpoint


def _save(tmp_path, epoch, save_ckpt_freq=5):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    save_model(
        output_cfg=OutputConfig(output_dir=str(tmp_path), save_ckpt_freq=save_ckpt_freq),
        trainer_cfg=TrainerConfig(), epoch=epoch, model=model, model_without_ddp=model,
        optimizer=opt, loss_scaler=None)


def test_final_tag_writes_named_file(tmp_path):
    _save(tmp_path, 'final')
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {'checkpoint-final.pth'}  # single dedicated file, no rolling


def test_best_tag_writes_best_only(tmp_path):
    _save(tmp_path, 'best')
    assert {p.name for p in tmp_path.iterdir()} == {'checkpoint-best.pth'}


def test_periodic_epoch_writes_rolling_only_off_cycle(tmp_path):
    _save(tmp_path, epoch=3, save_ckpt_freq=5)  # (3+1)%5 != 0
    assert {p.name for p in tmp_path.iterdir()} == {'checkpoint.pth'}


def test_periodic_epoch_writes_named_on_cycle(tmp_path):
    _save(tmp_path, epoch=4, save_ckpt_freq=5)  # (4+1)%5 == 0
    assert {p.name for p in tmp_path.iterdir()} == {'checkpoint.pth', 'checkpoint-4.pth'}


def test_resolve_prefers_best_then_final_then_rolling(tmp_path):
    (tmp_path / 'checkpoint.pth').write_text('')
    assert resolve_final_checkpoint(str(tmp_path)).endswith('checkpoint.pth')
    (tmp_path / 'checkpoint-final.pth').write_text('')
    assert resolve_final_checkpoint(str(tmp_path)).endswith('checkpoint-final.pth')
    (tmp_path / 'checkpoint-best.pth').write_text('')
    assert resolve_final_checkpoint(str(tmp_path)).endswith('checkpoint-best.pth')
