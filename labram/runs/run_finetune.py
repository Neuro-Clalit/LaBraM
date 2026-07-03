# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# ---------------------------------------------------------

import argparse
import numpy as np
import torch
from pathlib import Path

from timm.models import create_model
from timm.utils import ModelEma

import labram.models.registry  # noqa: F401
import labram.runs.common as runner_common
import labram.utils as utils
from labram.data import get_dataset_bundle
from labram.losses import CodebookRegularizedCriterion, LossConfig, build_classification_criterion
from labram.configs.run_configs import FinetuneRunConfig
from labram.configs.utils_conf import parse_overrides
from labram.train.train_finetune import evaluate, train_loop
from labram.runs.codebook_setup import (
    CodebookRegLayerAssigner, build_codebook_classifier, loss_config_from_codebook_reg,
)
from labram.runs.finetune_setup import (
    build_dataloaders, build_samplers, load_finetune_checkpoint,
    subset_for_debug,
)
from labram.optim_factory import LayerDecayValueAssigner, create_optimizer, get_parameter_groups
from labram.utils import NativeScalerWithGradNormCount as NativeScaler

logger = utils.get_logger(__name__)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser('LaBraM fine-tuning (config-driven)', add_help=True)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a JSON or YAML FinetuneRunConfig file.')
    parser.add_argument('--set', dest='overrides', nargs='*', default=[],
                        metavar='KEY=VALUE')
    return parser.parse_args()


def get_model(config: FinetuneRunConfig):
    if config.codebook_reg.enabled:
        return build_codebook_classifier(config)
    m = config.model
    return create_model(
        m.model, pretrained=False,
        num_classes=m.nb_classes, drop_rate=m.drop,
        drop_path_rate=m.drop_path, attn_drop_rate=m.attn_drop_rate,
        use_mean_pooling=m.use_mean_pooling, init_scale=m.init_scale,
        use_rel_pos_bias=m.rel_pos_bias, use_abs_pos_emb=m.abs_pos_emb,
        init_values=m.layer_scale_init_value, qkv_bias=m.qkv_bias,
    )


def main(config: FinetuneRunConfig):
    # setup_environment handles distributed init, device resolution, and seeding.
    # For finetune, config.distributed.device defaults to 'auto', which
    # setup_environment resolves to cuda/mps/cpu.
    device, num_tasks, global_rank = runner_common.setup_environment(config)

    if config.enable_deepspeed:
        utils.create_ds_config(config.output, config.trainer, config.optimizer)

    if config.debug:
        logger.info("[DEBUG MODE] Overriding training schedule for fast iteration")
        config.trainer.epochs = max(1, min(config.trainer.epochs, 2))
        config.trainer.batch_size = min(config.trainer.batch_size, 4)
        config.data.num_workers = 0
        config.optimizer.warmup_epochs = 0
        config.output.save_ckpt = False
        config.distributed.dist_eval = False
        if config.output.output_dir:
            config.output.log_dir = config.output.log_dir or config.output.output_dir

    logger.info("%s", config)

    bundle = get_dataset_bundle(config.dataset, config.data_path)
    config.model.nb_classes = bundle.nb_classes
    dataset_train, dataset_val, dataset_test = bundle.train, bundle.val, bundle.test
    ch_names, metrics = bundle.ch_names, bundle.metrics

    if config.debug:
        n = config.debug_samples
        dataset_train = subset_for_debug(dataset_train, n)
        dataset_val   = subset_for_debug(dataset_val, n)
        dataset_test  = subset_for_debug(dataset_test, n)

    if config.disable_eval_during_finetuning:
        dataset_val = dataset_test = None

    num_tasks, global_rank = utils.get_world_size(), utils.get_rank()
    sampler_train, sampler_val, sampler_test = build_samplers(
        dataset_train, dataset_val, dataset_test,
        num_tasks, global_rank, config.distributed.dist_eval,
    )

    log_writer = runner_common.create_log_writer(
        config.output, global_rank, config.clearml, config)
    pin_memory = config.data.pin_mem and device.type == 'cuda'
    loaders = build_dataloaders(
        dataset_train, dataset_val, dataset_test,
        sampler_train, sampler_val, sampler_test,
        batch_size=config.trainer.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )

    model = get_model(config)
    patch_size = model.patch_size
    window_size = (1, config.model.input_size // patch_size)

    # Codebook-regularized: the pre-trained backbone loads into model.encoder
    # (decoder + quantizer come from the VQNSP checkpoint at build time).
    if config.codebook_reg.enabled:
        load_finetune_checkpoint(model.encoder, config.finetune_checkpoint)
    else:
        load_finetune_checkpoint(model, config.finetune_checkpoint)
    model.to(device)

    model_ema = None
    if config.model_ema:
        model_ema = ModelEma(model, decay=config.model_ema_decay,
                             device='cpu' if config.model_ema_force_cpu else '', resume='')

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_parameters}")

    total_batch_size = config.trainer.batch_size * config.trainer.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    logger.info(f"LR={config.optimizer.lr:.8f}  batch={total_batch_size}  update_freq={config.trainer.update_freq}")

    num_layers = model_without_ddp.get_num_layers()
    assigner = None
    if config.codebook_reg.enabled:
        # Per-component LR scales (encoder/decoder slower than the head),
        # with optional layer-wise decay folded into the encoder group.
        assigner = CodebookRegLayerAssigner(config.codebook_reg, num_layers, config.layer_decay)
    elif config.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            [config.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)])

    skip_weight_decay_list = model.no_weight_decay()
    if config.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add(f"blocks.{i}.attn.relative_position_bias_table")

    if config.enable_deepspeed:
        ds_init = utils.get_ds_init()
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, config.optimizer.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner else None,
            assigner.get_scale if assigner else None)
        from argparse import Namespace as _NS
        _ds_args = _NS(distributed=config.distributed.distributed, gpu=config.distributed.gpu)
        model, optimizer, _, _ = ds_init(
            args=_ds_args, model=model, model_parameters=optimizer_params,
            dist_init_required=not config.distributed.distributed)
    else:
        if config.distributed.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[config.distributed.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
        optimizer = create_optimizer(
            config.optimizer, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner else None,
            get_layer_scale=assigner.get_scale if assigner else None)
        loss_scaler = NativeScaler()

    nb_classes = config.model.nb_classes
    if config.codebook_reg.enabled:
        loss_cfg = loss_config_from_codebook_reg(config.codebook_reg, config.smoothing)
        criterion = CodebookRegularizedCriterion(
            build_classification_criterion(nb_classes, loss_cfg), loss_cfg)
    else:
        criterion = build_classification_criterion(
            nb_classes, LossConfig(classification_label_smoothing=config.smoothing))

    utils.auto_load_model(
        output_cfg=config.output, trainer_cfg=config.trainer,
        model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema,
        enable_deepspeed=config.enable_deepspeed,
        model_ema_enabled=config.model_ema,
    )

    if config.eval:
        accuracy, balanced_accuracy = [], []
        for data_loader in loaders.test:
            test_stats = evaluate(data_loader, model, device, header='Test:',
                                  ch_names=ch_names, metrics=metrics, is_binary=(nb_classes == 1))
            accuracy.append(test_stats['accuracy'])
            balanced_accuracy.append(test_stats['balanced_accuracy'])
        logger.info(f"Accuracy: {np.mean(accuracy):.3f} ± {np.std(accuracy):.3f}, "
                    f"balanced: {np.mean(balanced_accuracy):.3f} ± {np.std(balanced_accuracy):.3f}")
        return

    train_loop(
        config=config,
        model=model, model_without_ddp=model_without_ddp,
        criterion=criterion, loaders=loaders,
        optimizer=optimizer, device=device, loss_scaler=loss_scaler,
        log_writer=log_writer,
        ch_names=ch_names, metrics=metrics,
        n_parameters=n_parameters,
        num_training_steps_per_epoch=num_training_steps_per_epoch,
        model_ema=model_ema,
        enable_deepspeed=config.enable_deepspeed,
    )


def build_config(cli: argparse.Namespace) -> FinetuneRunConfig:
    overrides = parse_overrides(cli.overrides)
    return FinetuneRunConfig.load_config(cli.config, **overrides)


if __name__ == '__main__':
    cli = parse_cli()
    config = build_config(cli)
    if config.output.output_dir:
        out_dir = Path(config.output.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        config.save_to(str(out_dir / 'run_config.yaml'))
    main(config)
