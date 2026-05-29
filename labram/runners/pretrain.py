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
import labram.runners.common as runner_common
import labram.utils as utils
from labram.configs.runner_configs import PretrainRunConfig, parse_overrides
from labram.trainers.train_pretrain import train_one_epoch
from labram.optim_factory import create_optimizer
from labram.utils import NativeScalerWithGradNormCount as NativeScaler


def parse_cli() -> argparse.Namespace:
    """Thin CLI: a config file path plus optional dotted-key overrides.

    Example:
        python -m labram.runners.pretrain \
            --config conf/pretrain.yaml \
            --set trainer.epochs=5 optimizer.lr=1e-4
    """
    parser = argparse.ArgumentParser('LaBraM pre-training (config-driven)', add_help=True)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a JSON or YAML PretrainRunConfig file. '
                             'Omit to use built-in defaults.')
    parser.add_argument('--set', dest='overrides', nargs='*', default=[],
                        metavar='KEY=VALUE',
                        help='Dotted-path overrides applied after loading, '
                             'e.g. --set trainer.epochs=5 optimizer.lr=1e-4')
    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    return create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        vocab_size=args.codebook_size,
    )


def get_visual_tokenizer(args):
    print(f"Creating visual tokenizer: {args.tokenizer_model}")
    return create_model(
        args.tokenizer_model,
        pretrained=True,
        pretrained_weight=args.tokenizer_weight,
        as_tokenzer=True,
        num_codebook_tokens=args.codebook_size,
        quantizer_dim=args.quantizer_dim,
    ).eval()


def main(config: PretrainRunConfig):
    # Flat namespace adapter for legacy consumers (init_distributed_mode,
    # create_optimizer, auto_load_model, save_model, cosine_scheduler).
    # Mutations to `args` (e.g. rank, gpu, distributed, resume) do NOT
    # propagate back to `config`.
    args = config.to_namespace()

    device, num_tasks, global_rank = runner_common.setup_environment(args)
    print(config)

    model = get_model(args)
    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    # Dataset spec lives on the typed config. ``datasets_train`` is a nested
    # list of HDF5 paths grouped by channel montage (see DataConfig docstring).
    dataset_train_list, train_ch_names_list = utils.build_pretraining_dataset(
        config.data.datasets_train,
        config.data.time_window,
        stride=config.data.stride,
        start_percentage=config.data.start_percentage,
        end_percentage=config.data.end_percentage,
    )
    vqnsp = get_visual_tokenizer(args).to(device)

    num_training_steps_per_epoch = (
        sum(len(d) for d in dataset_train_list) // args.batch_size // num_tasks
    )

    sampler_train_list = runner_common.build_distributed_train_sampler_list(
        dataset_train_list, num_tasks, global_rank,
    )
    print("Sampler_train = %s" % str(sampler_train_list[-1]))

    log_writer = runner_common.create_log_writer(args, global_rank)

    data_loader_train_list = runner_common.build_dataloader_list(
        dataset_train_list, sampler_train_list,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True,
    )

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)
    print("Tokenizer = %s" % str(vqnsp))

    total_batch_size = args.batch_size * num_tasks * args.gradient_accumulation_steps
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    model, model_without_ddp = runner_common.wrap_distributed(args, model)

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = runner_common.make_lr_schedule(args, num_training_steps_per_epoch)
    wd_schedule_values = runner_common.make_wd_schedule(args, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler,
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for data_loader_train in data_loader_train_list:
                data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model, vqnsp, data_loader_train_list,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            ch_names_list=train_ch_names_list,
            args=args,
        )
        if args.output_dir:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                save_ckpt_freq=args.save_ckpt_freq,
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if log_writer is not None and args.output_dir and utils.is_main_process():
            log_writer.flush()
        runner_common.append_log_line(args, log_stats)

    runner_common.print_training_time(start_time)


def build_config(cli: argparse.Namespace) -> PretrainRunConfig:
    """Load config from file (if given) and apply CLI overrides."""
    overrides = parse_overrides(cli.overrides)
    return PretrainRunConfig.load_config(cli.config, **overrides)


if __name__ == '__main__':
    cli = parse_cli()
    config = build_config(cli)

    if config.output.output_dir:
        out_dir = Path(config.output.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Snapshot the resolved config alongside the run for reproducibility.
        config.save_to(str(out_dir / 'run_config.yaml'))

    main(config)
