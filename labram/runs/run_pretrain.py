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
from labram.configs.runner_configs import PretrainRunConfig, parse_overrides
from labram.train.train_pretrain import train_loop
from labram.optim_factory import create_optimizer
from labram.utils import NativeScalerWithGradNormCount as NativeScaler


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser('LaBraM pre-training (config-driven)', add_help=True)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a JSON or YAML PretrainRunConfig file.')
    parser.add_argument('--set', dest='overrides', nargs='*', default=[],
                        metavar='KEY=VALUE',
                        help='Dotted-path overrides, e.g. --set trainer.epochs=5')
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
    args = config.to_namespace()
    device, num_tasks, global_rank = runner_common.setup_environment(args)
    print(config)

    model = get_model(args)
    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

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
    log_writer = runner_common.create_log_writer(args, global_rank)

    data_loader_list = runner_common.build_dataloader_list(
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
    print("Training steps/epoch = %d" % num_training_steps_per_epoch)

    model, model_without_ddp = runner_common.wrap_distributed(args, model)
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_schedule_values = runner_common.make_lr_schedule(args, num_training_steps_per_epoch)
    wd_schedule_values = runner_common.make_wd_schedule(args, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler,
    )

    train_loop(
        config=config,
        model=model, model_without_ddp=model_without_ddp,
        vqnsp=vqnsp, data_loader_list=data_loader_list,
        optimizer=optimizer, device=device, loss_scaler=loss_scaler,
        lr_schedule_values=lr_schedule_values,
        wd_schedule_values=wd_schedule_values,
        train_ch_names_list=train_ch_names_list,
        log_writer=log_writer, args=args,
        n_parameters=n_parameters,
        num_training_steps_per_epoch=num_training_steps_per_epoch,
    )


def build_config(cli: argparse.Namespace) -> PretrainRunConfig:
    overrides = parse_overrides(cli.overrides)
    return PretrainRunConfig.load_config(cli.config, **overrides)


if __name__ == '__main__':
    cli = parse_cli()
    config = build_config(cli)
    if config.output.output_dir:
        out_dir = Path(config.output.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        config.save_to(str(out_dir / 'run_config.yaml'))
    main(config)
