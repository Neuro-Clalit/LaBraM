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

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import ModelEma

import labram.models.registry  # noqa: F401  -- registers timm models
import labram.runs.common as runner_common
import labram.utils as utils
from labram.configs.runner_configs import FinetuneRunConfig, parse_overrides
from labram.configs.defaults import DEFAULT_EVAL_BATCH_SCALE
from labram.trainers.train_finetune import evaluate, train_one_epoch
from labram.runs.finetune_datasets import get_dataset_bundle
from labram.runs.finetune_setup import (
    build_dataloaders,
    build_samplers,
    load_finetune_checkpoint,
    resolve_device,
    subset_for_debug,
)
from labram.optim_factory import LayerDecayValueAssigner, create_optimizer, get_parameter_groups
from labram.utils import NativeScalerWithGradNormCount as NativeScaler


_LOGGED_EVAL_KEYS = (
    'accuracy', 'balanced_accuracy', 'f1_weighted', 'pr_auc', 'roc_auc', 'cohen_kappa', 'loss',
)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser('LaBraM fine-tuning (config-driven)', add_help=True)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a JSON or YAML FinetuneRunConfig file.')
    parser.add_argument('--set', dest='overrides', nargs='*', default=[],
                        metavar='KEY=VALUE',
                        help='Dotted-path overrides, e.g. --set trainer.epochs=30')
    return parser.parse_args()


def get_model(config: FinetuneRunConfig):
    m = config.model
    return create_model(
        m.model,
        pretrained=False,
        num_classes=m.nb_classes,
        drop_rate=m.drop,
        drop_path_rate=m.drop_path,
        attn_drop_rate=m.attn_drop_rate,
        use_mean_pooling=m.use_mean_pooling,
        init_scale=m.init_scale,
        use_rel_pos_bias=m.rel_pos_bias,
        use_abs_pos_emb=m.abs_pos_emb,
        init_values=m.layer_scale_init_value,
        qkv_bias=m.qkv_bias,
    )


def _log_eval_stats(log_writer, stats, head, epoch):
    if log_writer is None:
        return
    for key, value in stats.items():
        if key in _LOGGED_EVAL_KEYS:
            log_writer.update(**{key: value}, head=head, step=epoch)


def main(config: FinetuneRunConfig):
    args = config.to_namespace()

    utils.init_distributed_mode(args)

    if config.enable_deepspeed:
        utils.create_ds_config(args)

    # Apply debug overrides directly on the config so downstream code sees them.
    if config.debug:
        n = config.debug_samples
        print(f"[DEBUG MODE] Overriding training schedule for fast iteration")
        config.trainer.epochs = max(1, min(config.trainer.epochs, 2))
        config.trainer.batch_size = min(config.trainer.batch_size, 4)
        config.data.num_workers = 0
        config.optimizer.warmup_epochs = 0
        config.output.save_ckpt = False
        config.distributed.dist_eval = False
        if config.output.output_dir:
            config.output.log_dir = config.output.log_dir or config.output.output_dir
        # Rebuild namespace after debug mutations
        args = config.to_namespace()

    print(config)

    device = resolve_device(config.distributed.device)
    args.device = str(device)

    seed = config.distributed.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        cudnn.benchmark = True

    bundle = get_dataset_bundle(config.dataset, config.data_path)
    config.model.nb_classes = bundle.nb_classes
    dataset_train, dataset_val, dataset_test = bundle.train, bundle.val, bundle.test
    ch_names, metrics = bundle.ch_names, bundle.metrics

    if config.debug:
        n = config.debug_samples
        print(f"[DEBUG MODE] Subsetting datasets to first {n} samples per split")
        dataset_train = subset_for_debug(dataset_train, n)
        dataset_val = subset_for_debug(dataset_val, n)
        dataset_test = subset_for_debug(dataset_test, n)

    if config.disable_eval_during_finetuning:
        dataset_val = dataset_test = None

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train, sampler_val, sampler_test = build_samplers(
        dataset_train, dataset_val, dataset_test,
        num_tasks, global_rank, config.distributed.dist_eval,
    )

    log_writer = runner_common.create_log_writer(args, global_rank)

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
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, config.model.input_size // patch_size)
    args.patch_size = patch_size

    load_finetune_checkpoint(model, args)
    model.to(device)

    model_ema = None
    if config.model_ema:
        model_ema = ModelEma(
            model,
            decay=config.model_ema_decay,
            device='cpu' if config.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % config.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = (config.trainer.batch_size * config.trainer.update_freq
                        * utils.get_world_size())
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % config.optimizer.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update freq = %d" % config.trainer.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Training steps per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if config.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(config.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        print("Assigned values = %s" % str(assigner.values))
    else:
        assigner = None

    skip_weight_decay_list = model.no_weight_decay()
    if config.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if config.enable_deepspeed:
        ds_init = utils.get_ds_init()
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, config.optimizer.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )
        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == config.trainer.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = runner_common.make_lr_schedule(args, num_training_steps_per_epoch)
    wd_schedule_values = runner_common.make_wd_schedule(args, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    nb_classes = config.model.nb_classes
    if nb_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif config.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    is_binary = nb_classes == 1
    if config.eval:
        balanced_accuracy, accuracy = [], []
        for data_loader in loaders.test:
            test_stats = evaluate(data_loader, model, device, header='Test:',
                                  ch_names=ch_names, metrics=metrics, is_binary=is_binary)
            accuracy.append(test_stats['accuracy'])
            balanced_accuracy.append(test_stats['balanced_accuracy'])
        print(f"======Accuracy: {np.mean(accuracy)} {np.std(accuracy)}, "
              f"balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}")
        return

    print(f"Start training for {config.trainer.epochs} epochs")
    start_time = time.time()
    max_accuracy = max_accuracy_test = 0.0

    for epoch in range(config.trainer.start_epoch, config.trainer.epochs):
        if args.distributed:
            loaders.train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * config.trainer.update_freq)

        train_stats = train_one_epoch(
            model, criterion, loaders.train, optimizer,
            device, epoch, loss_scaler,
            max_norm=config.optimizer.clip_grad or 0,
            model_ema=model_ema,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=config.trainer.update_freq,
            ch_names=ch_names,
            is_binary=is_binary,
        )

        if config.output.output_dir and config.output.save_ckpt:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler,
                epoch=epoch, model_ema=model_ema,
                save_ckpt_freq=config.output.save_ckpt_freq)

        if loaders.val is not None:
            val_stats = evaluate(loaders.val, model, device, header='Val:',
                                 ch_names=ch_names, metrics=metrics, is_binary=is_binary)
            print(f"Accuracy of the network on the {len(dataset_val)} val EEG: "
                  f"{val_stats['accuracy']:.2f}%")
            test_stats = evaluate(loaders.test, model, device, header='Test:',
                                  ch_names=ch_names, metrics=metrics, is_binary=is_binary)
            print(f"Accuracy of the network on the {len(dataset_test)} test EEG: "
                  f"{test_stats['accuracy']:.2f}%")

            if max_accuracy < val_stats["accuracy"]:
                max_accuracy = val_stats["accuracy"]
                if config.output.output_dir and config.output.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler,
                        epoch="best", model_ema=model_ema)
                max_accuracy_test = test_stats["accuracy"]
            print(f'Max accuracy val: {max_accuracy:.2f}%, '
                  f'max accuracy test: {max_accuracy_test:.2f}%')

            _log_eval_stats(log_writer, val_stats, head="val", epoch=epoch)
            _log_eval_stats(log_writer, test_stats, head="test", epoch=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch, 'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'n_parameters': n_parameters}

        if log_writer is not None and config.output.output_dir and utils.is_main_process():
            log_writer.flush()
        runner_common.append_log_line(args, log_stats)

    runner_common.print_training_time(start_time)


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
