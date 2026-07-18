# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# ---------------------------------------------------------

import argparse
from pathlib import Path

from timm.models import create_model

import labram.models.registry  # noqa: F401
import labram.runs.common as runner_common
import labram.utils as utils
from labram.configs.run_configs import VQNSPRunConfig
from labram.configs.utils_conf import parse_overrides
from labram.configs.defaults import DEFAULT_EVAL_BATCH_SCALE
from labram.train.train_vqnsp import calculate_codebook_usage, evaluate, train_loop
from labram.optim_factory import create_optimizer, log_trainable_parameters
from labram.utils import NativeScalerWithGradNormCount as NativeScaler

logger = utils.get_logger(__name__)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser('LaBraM VQNSP training (config-driven)', add_help=True)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a JSON or YAML VQNSPRunConfig file.')
    parser.add_argument('--set', dest='overrides', nargs='*', default=[],
                        metavar='KEY=VALUE')
    return parser.parse_args()


def get_model(config: VQNSPRunConfig):
    m = config.model
    return create_model(
        m.model, pretrained=False, as_tokenzer=False,
        num_codebook_tokens=m.codebook_size,
        quantizer_dim=m.quantizer_dim,
        eeg_window_size=m.input_size,
        decay=m.ema_decay,
        quantize_kmeans_init=m.quantize_kmeans_init,
        labram_plus=config.labram_plus,
    )


def _log_model_param_counts(model):
    for part in ['encoder', 'decoder']:
        model_part = getattr(model, part)
        n_learnable = sum(p.numel() for p in model_part.parameters() if p.requires_grad)
        n_fixed = sum(p.numel() for p in model_part.parameters() if not p.requires_grad)
        logger.info(f'learnable params in model.{part}: {n_learnable / 1e6:.2f} M')
        logger.info(f'fixed params in model.{part}: {n_fixed / 1e6:.2f} M')
    n_learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'total learnable params: {n_learnable / 1e6:.2f} M')
    return n_learnable


def main(config: VQNSPRunConfig):
    device, num_tasks, global_rank = runner_common.setup_environment(config)
    logger.info("%s", config)

    model = get_model(config)

    dataset_train_list, train_ch_names_list = utils.build_pretraining_dataset(
        config.data.datasets_train, config.data.time_window,
        stride=config.data.stride,
        start_percentage=config.data.start_percentage,
        end_percentage=config.data.end_percentage,
    )

    if config.disable_eval:
        dataset_val_list = val_ch_names_list = None
    else:
        dataset_val_list, val_ch_names_list = utils.build_pretraining_dataset(
            config.data.datasets_val or [[]], config.data.val_time_window or [4],
        )

    num_training_steps_per_epoch = (
        sum(len(d) for d in dataset_train_list) // config.trainer.batch_size // num_tasks
    )

    sampler_train_list = runner_common.build_distributed_train_sampler_list(
        dataset_train_list, num_tasks, global_rank,
    )
    sampler_eval_list = None
    if dataset_val_list is not None:
        sampler_eval_list = runner_common.build_distributed_eval_sampler_list(
            dataset_val_list, num_tasks, global_rank, config.distributed.dist_eval,
        )

    log_writer = runner_common.create_log_writer(
        config.output, global_rank, config.clearml, config)

    data_loader_train_list = runner_common.build_dataloader_list(
        dataset_train_list, sampler_train_list,
        batch_size=config.trainer.batch_size, num_workers=config.data.num_workers,
        pin_memory=config.data.pin_mem, drop_last=True,
    )
    data_loader_val_list = None
    if dataset_val_list is not None:
        data_loader_val_list = runner_common.build_dataloader_list(
            dataset_val_list, sampler_eval_list,
            batch_size=int(DEFAULT_EVAL_BATCH_SCALE * config.trainer.batch_size),
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_mem, drop_last=False,
        )

    model.to(device)
    model_without_ddp = model
    n_learnable_parameters = _log_model_param_counts(model)
    logger.info("Model structure:\n%s", str(model_without_ddp))
    log_trainable_parameters(model_without_ddp, logger)
    runner_common.log_model_visualization(config.logging, config.output, model_without_ddp, log_writer)

    total_batch_size = config.trainer.batch_size * num_tasks
    scaled_lr = total_batch_size / 128 * config.optimizer.lr
    config.optimizer.lr = scaled_lr
    logger.info(f"LR = {scaled_lr:.8f}  batch = {total_batch_size}  steps/epoch = {num_training_steps_per_epoch}")

    optimizer = create_optimizer(config.optimizer, model_without_ddp)
    loss_scaler = NativeScaler()
    model, model_without_ddp = runner_common.wrap_distributed(config.distributed, model)

    utils.auto_load_model(
        output_cfg=config.output, trainer_cfg=config.trainer,
        model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler,
    )

    if config.eval:
        evaluate(data_loader_val_list, model, device, log_writer, 0,
                 ch_names_list=val_ch_names_list)
        return

    if config.calculate_codebook_usage:
        calculate_codebook_usage(data_loader_val_list, model, device,
                                 codebook_size=config.model.codebook_size,
                                 log_writer=log_writer, epoch=0)
        return

    train_loop(
        config=config,
        model=model, model_without_ddp=model_without_ddp,
        data_loader_train_list=data_loader_train_list,
        data_loader_val_list=data_loader_val_list,
        train_ch_names_list=train_ch_names_list,
        val_ch_names_list=val_ch_names_list,
        optimizer=optimizer, device=device, loss_scaler=loss_scaler,
        log_writer=log_writer,
        n_learnable_parameters=n_learnable_parameters,
        num_training_steps_per_epoch=num_training_steps_per_epoch,
    )

    runner_common.finalize_run(config, log_writer)


def build_config(cli: argparse.Namespace) -> VQNSPRunConfig:
    overrides = parse_overrides(cli.overrides)
    return VQNSPRunConfig.load_config(cli.config, **overrides)


if __name__ == '__main__':
    cli = parse_cli()
    config = build_config(cli)
    if config.output.output_dir:
        out_dir = Path(config.output.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        config.save_to(str(out_dir / 'run_config.yaml'))
    main(config)
