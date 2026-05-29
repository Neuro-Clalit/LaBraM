# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import argparse
import time
from pathlib import Path

from timm.models import create_model

import labram.models.registry  # noqa: F401  -- registers timm models
import labram.runs.common as runner_common
import labram.utils as utils
from labram.configs.runner_configs import VQNSPRunConfig, parse_overrides
from labram.configs.defaults import DEFAULT_EVAL_BATCH_SCALE
from labram.trainers.train_vqnsp import calculate_codebook_usage, evaluate, train_one_epoch
from labram.optim_factory import create_optimizer
from labram.utils import NativeScalerWithGradNormCount as NativeScaler


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser('LaBraM VQNSP training (config-driven)', add_help=True)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a JSON or YAML VQNSPRunConfig file.')
    parser.add_argument('--set', dest='overrides', nargs='*', default=[],
                        metavar='KEY=VALUE',
                        help='Dotted-path overrides, e.g. --set trainer.epochs=50')
    return parser.parse_args()


def get_model(config: VQNSPRunConfig):
    m = config.model
    return create_model(
        m.model,
        pretrained=False,
        as_tokenzer=False,
        num_codebook_tokens=m.codebook_size,
        quantizer_dim=m.quantizer_dim,
        eeg_window_size=m.input_size,
        decay=m.ema_decay,
        quantize_kmeans_init=m.quantize_kmeans_init,
    )


def _log_model_param_counts(model):
    for part in ['encoder', 'decoder']:
        model_part = getattr(model, part)
        n_learnable = sum(p.numel() for p in model_part.parameters() if p.requires_grad)
        n_fixed = sum(p.numel() for p in model_part.parameters() if not p.requires_grad)
        print(f'number of learnable params in model.{part}: {n_learnable / 1e6} M')
        print(f'number of fixed params in model.{part}: {n_fixed / 1e6} M')
    n_learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_fixed = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'total number of learnable params: {n_learnable / 1e6} M')
    print(f'total number of fixed params in : {n_fixed / 1e6} M')
    return n_learnable


def main(config: VQNSPRunConfig):
    args = config.to_namespace()
    device, num_tasks, global_rank = runner_common.setup_environment(args)
    print(config)

    model = get_model(config)

    dataset_train_list, train_ch_names_list = utils.build_pretraining_dataset(
        config.data.datasets_train,
        config.data.time_window,
        stride=config.data.stride,
        start_percentage=config.data.start_percentage,
        end_percentage=config.data.end_percentage,
    )

    if config.disable_eval:
        dataset_val_list = val_ch_names_list = None
    else:
        dataset_val_list, val_ch_names_list = utils.build_pretraining_dataset(
            config.data.datasets_val or [[]],
            config.data.val_time_window or [4],
        )

    num_training_steps_per_epoch = (
        sum(len(d) for d in dataset_train_list) // config.trainer.batch_size // num_tasks
    )

    sampler_train_list = runner_common.build_distributed_train_sampler_list(
        dataset_train_list, num_tasks, global_rank,
    )
    print("Sampler_train = %s" % str(sampler_train_list[-1]))

    sampler_eval_list = None
    if dataset_val_list is not None:
        sampler_eval_list = runner_common.build_distributed_eval_sampler_list(
            dataset_val_list, num_tasks, global_rank, config.distributed.dist_eval,
        )

    log_writer = runner_common.create_log_writer(args, global_rank)

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
    if not config.eval:
        print("Model = %s" % str(model_without_ddp))
    n_learnable_parameters = _log_model_param_counts(model)

    total_batch_size = config.trainer.batch_size * num_tasks
    # VQNSP scales LR linearly with batch size relative to reference of 128
    scaled_lr = total_batch_size / 128 * config.optimizer.lr
    args.lr = scaled_lr
    print("LR = %.8f" % scaled_lr)
    print("Min LR = %.8f" % config.optimizer.min_lr)
    print("Weight Decay = %.8f" % config.optimizer.weight_decay)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    model, model_without_ddp = runner_common.wrap_distributed(args, model)

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = runner_common.make_lr_schedule(args, num_training_steps_per_epoch)

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
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

    print(f"Start training for {config.trainer.epochs} epochs")
    start_time = time.time()

    for epoch in range(config.trainer.start_epoch, config.trainer.epochs):
        if args.distributed:
            for dl in data_loader_train_list:
                dl.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model, data_loader_train_list, optimizer, device, epoch, loss_scaler,
            clip_grad=config.optimizer.clip_grad or 0,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            ch_names_list=train_ch_names_list,
            output_dir=config.output.output_dir,
        )
        if config.output.output_dir:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                save_ckpt_freq=config.output.save_ckpt_freq,
            )

        if data_loader_val_list is not None:
            test_stats = evaluate(
                data_loader_val_list, model, device, log_writer, epoch,
                ch_names_list=val_ch_names_list,
            )
            print(
                f"Validation loss of the network on the "
                f"{sum(len(d) for d in dataset_val_list)} test EEG: "
                f"{test_stats['loss']:.4f}"
            )
            if log_writer is not None:
                log_writer.update(**test_stats, head="val/loss")

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch, 'n_parameters': n_learnable_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'n_parameters': n_learnable_parameters}

        if log_writer is not None and config.output.output_dir and utils.is_main_process():
            log_writer.flush()
        runner_common.append_log_line(args, log_stats)

    runner_common.print_training_time(start_time)


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
