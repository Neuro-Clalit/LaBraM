# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Checkpoint I/O: state-dict loading, model save/auto-resume, DeepSpeed config writer.
# ---------------------------------------------------------

import glob
import io
import json
import os
from pathlib import Path

import torch
from timm.utils import get_state_dict

from labram.utils.distributed import get_world_size, save_on_master


def load_pretrained_weights(path: str) -> dict:
    """Load a checkpoint from a URL or local path and return the model state dict.

    Handles both ``{'model': ...}`` and ``{'state_dict': ...}`` checkpoint layouts.
    """
    if path.startswith('https'):
        weights = torch.hub.load_state_dict_from_url(path, map_location='cpu', check_hash=True)
    else:
        weights = torch.load(path, map_location='cpu', weights_only=False)
    if 'model' in weights:
        return weights['model']
    if 'state_dict' in weights:
        return weights['state_dict']
    return weights


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """Workaround for ModelEma._load_checkpoint to accept an already-loaded object."""
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def save_nan_model(output_dir: str, model):
    """Snapshot the model's state_dict when training NaNs out, for offline debug.

    Called by train_vqnsp.train_one_epoch right before sys.exit(1) when the
    loss becomes non-finite. Best-effort: if anything in here itself fails
    (no output_dir, disk full, no module to unwrap) we swallow the error so
    the original NaN failure is what surfaces, not a secondary exception.
    """
    if not output_dir:
        return
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        inner = model.module if hasattr(model, 'module') else model
        save_on_master(
            {'model': inner.state_dict()},
            Path(output_dir) / 'nan_checkpoint.pth',
        )
        print(f"Saved NaN-loss state to {output_dir}/nan_checkpoint.pth")
    except Exception as e:
        print(f"save_nan_model: failed to dump checkpoint ({e}); continuing to exit")


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def save_model(output_cfg, trainer_cfg, epoch, model, model_without_ddp, optimizer,
               loss_scaler, model_ema=None, optimizer_disc=None, enable_deepspeed: bool = False):
    output_dir = Path(output_cfg.output_dir)
    save_ckpt_freq = output_cfg.save_ckpt_freq
    epoch_name = str(epoch)

    if not enable_deepspeed:
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        # String epoch tags ('best', 'final') name a dedicated single file.
        if isinstance(epoch, str):
            checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        elif (epoch + 1) % save_ckpt_freq == 0:
            checkpoint_paths.append(output_dir / ('checkpoint-%s.pth' % epoch_name))

        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if loss_scaler is not None:
                to_save['scaler'] = loss_scaler.state_dict()

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            if optimizer_disc is not None:
                to_save['optimizer_disc'] = optimizer_disc.state_dict()

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=output_cfg.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(output_cfg, trainer_cfg, model, model_without_ddp, optimizer,
                    loss_scaler, model_ema=None, optimizer_disc=None,
                    enable_deepspeed: bool = False, model_ema_enabled: bool = False):
    output_dir = Path(output_cfg.output_dir)

    if not enable_deepspeed:
        if output_cfg.auto_resume and not output_cfg.resume:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint.pth'))
            if len(all_checkpoints) > 0:
                output_cfg.resume = os.path.join(output_dir, 'checkpoint.pth')
            else:
                all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('-')[-1].split('.')[0]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    output_cfg.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % output_cfg.resume)

        if output_cfg.resume:
            if output_cfg.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    output_cfg.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(output_cfg.resume, map_location='cpu', weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % output_cfg.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"Resume checkpoint at epoch {checkpoint['epoch']}")
                trainer_cfg.start_epoch = 1
                if model_ema_enabled and model_ema is not None:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
            if optimizer_disc is not None and 'optimizer_disc' in checkpoint:
                optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
    else:
        # deepspeed, only support '--auto_resume'.
        if output_cfg.auto_resume:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                output_cfg.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(output_cfg.output_dir, tag='checkpoint-%d' % latest_ckpt)
                trainer_cfg.start_epoch = client_states['epoch'] + 1
                if model_ema_enabled and model_ema is not None:
                    _load_checkpoint_for_ema(model_ema, client_states['model_ema'])


def create_ds_config(output_cfg, trainer_cfg, optim_cfg):
    Path(output_cfg.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_cfg.output_dir, "latest"), mode="w"):
        pass

    deepspeed_config_path = os.path.join(output_cfg.output_dir, "deepspeed_config.json")
    with open(deepspeed_config_path, mode="w") as writer:
        ds_config = {
            "train_batch_size": trainer_cfg.batch_size * trainer_cfg.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": trainer_cfg.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": optim_cfg.lr,
                    "weight_decay": optim_cfg.weight_decay,
                    "bias_correction": True,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                },
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128,
            },
        }
        writer.write(json.dumps(ds_config, indent=2))
    return deepspeed_config_path
