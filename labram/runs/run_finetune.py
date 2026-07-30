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
    build_dataloaders, build_samplers, enable_window_ids, load_finetune_checkpoint,
    subset_for_debug,
)
from labram.optim_factory import (
    LayerDecayValueAssigner, create_optimizer, get_parameter_groups,
    log_trainable_parameters,
)
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
    if config.model.codebook_reg.enabled:
        return build_codebook_classifier(config)
    m = config.model
    return create_model(
        m.model, pretrained=False,
        num_classes=m.nb_classes, drop_rate=m.drop,
        drop_path_rate=m.drop_path, attn_drop_rate=m.attn_drop_rate,
        use_mean_pooling=m.use_mean_pooling, init_scale=m.init_scale,
        use_rel_pos_bias=m.rel_pos_bias, use_abs_pos_emb=m.abs_pos_emb,
        init_values=m.layer_scale_init_value, qkv_bias=m.qkv_bias,
        labram_plus=config.labram_plus,
    )


def main(config: FinetuneRunConfig, bundle=None):
    """Run a single fine-tune. When ``bundle`` is provided it overrides the
    dataset built from ``config.data`` — the cross-validation runner passes a
    fold's train/val/test :class:`DatasetBundle` this way. Returns the
    best-epoch metric summary from the training loop (or the eval metrics for an
    eval-only run)."""
    # setup_environment handles distributed init, device resolution, and seeding.
    # For finetune, config.distributed.device defaults to 'auto', which
    # setup_environment resolves to cuda/mps/cpu.
    device, num_tasks, global_rank = runner_common.setup_environment(config)

    if config.trainer.enable_deepspeed:
        utils.create_ds_config(config.output, config.trainer, config.optimizer)

    if config.trainer.debug:
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

    if bundle is None:
        bundle = get_dataset_bundle(config.data.dataset, config.data.data_path)
        # Optionally pin the train/val/test split to a recorded data_split.json
        # (local or s3://) so several models train on an identical split.
        if config.data.split_json:
            from labram.data.data_split_reuse import apply_data_split, load_data_split_json
            logger.info("Reusing recorded data split from %s", config.data.split_json)
            bundle = apply_data_split(bundle, load_data_split_json(config.data.split_json))
    config.model.nb_classes = bundle.nb_classes
    dataset_train, dataset_val, dataset_test = bundle.train, bundle.val, bundle.test
    ch_names, metrics = bundle.ch_names, bundle.metrics

    if config.trainer.debug:
        n = config.trainer.debug_samples
        dataset_train = subset_for_debug(dataset_train, n)
        dataset_val   = subset_for_debug(dataset_val, n)
        dataset_test  = subset_for_debug(dataset_test, n)

    if config.trainer.disable_eval_during_finetuning:
        dataset_val = dataset_test = None

    # Per-case window aggregation needs each eval dataset to surface a per-window
    # case id (recording/subject). Enable it for val *and* test whenever an
    # aggregation mode is configured — both during training (so per-epoch eval
    # reports case-level metrics alongside per-window ones) and in eval-only mode.
    if config.evaluation.agg_windows != 'none':
        for eval_ds in (dataset_val, dataset_test):
            if eval_ds is not None:
                enable_window_ids(eval_ds, config.evaluation.agg_case_by)

    num_tasks, global_rank = utils.get_world_size(), utils.get_rank()
    sampler_train, sampler_val, sampler_test = build_samplers(
        dataset_train, dataset_val, dataset_test,
        num_tasks, global_rank, config.distributed.dist_eval,
    )

    log_writer = runner_common.create_log_writer(
        config.output, global_rank, config.clearml, config)

    # Record which cases went to train/val/test (saved to disk + ClearML artifact).
    runner_common.log_data_split_artifact(
        config.logging, config.output, dataset_train, dataset_val, dataset_test,
        log_writer=log_writer, dataset_name=config.data.dataset)

    # For a cross-validation fold, also attach the fold partition (cv_split.json,
    # written to the fold's output dir by the CV runner) to this experiment.
    if getattr(config, 'cross_validation', None) is not None and config.cross_validation.enabled:
        runner_common.log_cv_split_artifact(config.output, log_writer=log_writer)

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
    if config.model.codebook_reg.enabled:
        load_finetune_checkpoint(model.encoder, config.finetune_checkpoint)
    else:
        load_finetune_checkpoint(model, config.finetune_checkpoint)
    model.to(device)

    model_ema = None
    if config.optimizer.model_ema:
        model_ema = ModelEma(model, decay=config.optimizer.model_ema_decay,
                             device='cpu' if config.optimizer.model_ema_force_cpu else '', resume='')

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_parameters}")

    # Log the model architecture and which layers are frozen (and thus excluded
    # from the optimizer) — e.g. when codebook_reg.encoder.n_last_trainable_layers
    # keeps only the last N encoder blocks trainable.
    logger.info("Model structure:\n%s", str(model_without_ddp))
    log_trainable_parameters(model_without_ddp, logger)

    # Visualize the architecture, colouring frozen vs trainable layers, and log
    # it to TensorBoard (figure) and ClearML (vector SVG media + artifacts).
    runner_common.log_model_visualization(
        config.logging, config.output, model_without_ddp, log_writer)

    total_batch_size = config.trainer.batch_size * config.trainer.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    logger.info(f"LR={config.optimizer.lr:.8f}  batch={total_batch_size}  update_freq={config.trainer.update_freq}")

    num_layers = model_without_ddp.get_num_layers()
    assigner = None
    if config.model.codebook_reg.enabled:
        # Per-component LR scales (encoder/decoder slower than the head),
        # with optional layer-wise decay folded into the encoder group.
        assigner = CodebookRegLayerAssigner(config.model.codebook_reg, num_layers, config.optimizer.layer_decay)
    elif config.optimizer.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            [config.optimizer.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)])

    skip_weight_decay_list = model.no_weight_decay()
    if config.trainer.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add(f"blocks.{i}.attn.relative_position_bias_table")

    if config.trainer.enable_deepspeed:
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
    if config.model.codebook_reg.enabled:
        loss_cfg = loss_config_from_codebook_reg(
            config.model.codebook_reg, config.optimizer.smoothing,
            phase_loss=config.labram_plus.resolved_phase_loss)
        criterion = CodebookRegularizedCriterion(
            build_classification_criterion(nb_classes, loss_cfg), loss_cfg)
    else:
        criterion = build_classification_criterion(
            nb_classes, LossConfig(classification_label_smoothing=config.optimizer.smoothing))

    utils.auto_load_model(
        output_cfg=config.output, trainer_cfg=config.trainer,
        model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema,
        enable_deepspeed=config.trainer.enable_deepspeed,
        model_ema_enabled=config.optimizer.model_ema,
    )

    if config.trainer.eval:
        # loaders.test may be a single loader or a list; normalize to a list.
        test_loaders = loaders.test if isinstance(loaders.test, list) else [loaders.test]
        accuracy, balanced_accuracy = [], []
        for data_loader in test_loaders:
            test_stats = evaluate(data_loader, model, device, header='Test:',
                                  ch_names=ch_names, metrics=metrics, is_binary=(nb_classes == 1),
                                  nb_classes=nb_classes, eval_cfg=config.evaluation,
                                  log_writer=log_writer, head='test', epoch=0)
            accuracy.append(test_stats['accuracy'])
            balanced_accuracy.append(test_stats['balanced_accuracy'])
        logger.info(f"Accuracy: {np.mean(accuracy):.3f} ± {np.std(accuracy):.3f}, "
                    f"balanced: {np.mean(balanced_accuracy):.3f} ± {np.std(balanced_accuracy):.3f}")
        eval_summary = {"accuracy": float(np.mean(accuracy)),
                        "balanced_accuracy": float(np.mean(balanced_accuracy))}
        runner_common.log_summary_metrics(log_writer, eval_summary, config)
        return eval_summary

    summary = train_loop(
        config=config,
        model=model, model_without_ddp=model_without_ddp,
        criterion=criterion, loaders=loaders,
        optimizer=optimizer, device=device, loss_scaler=loss_scaler,
        log_writer=log_writer,
        ch_names=ch_names, metrics=metrics,
        n_parameters=n_parameters,
        num_training_steps_per_epoch=num_training_steps_per_epoch,
        model_ema=model_ema,
        enable_deepspeed=config.trainer.enable_deepspeed,
    )

    # Persist the best-epoch metrics as ClearML single values (comparable in
    # compare mode) + a final_metrics config section before finishing.
    runner_common.log_summary_metrics(log_writer, summary, config)

    runner_common.finalize_run(config, log_writer)
    return summary


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
