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

import labram.models.pretrain  # noqa: F401  -- registers timm models
import labram.models.vqnsp  # noqa: F401  -- registers timm models
import labram.runners.common as runner_common
import labram.utils as utils
from labram.engines.pretrain import train_one_epoch
from labram.optim_factory import create_optimizer
from labram.utils import NativeScalerWithGradNormCount as NativeScaler


def get_args():
    parser = argparse.ArgumentParser('LaBraM pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

    # tokenizer settings
    parser.add_argument("--tokenizer_weight", type=str)
    parser.add_argument("--tokenizer_model", type=str, default="vqnsp_encoder_base_decoder_3x200x12")

    # Model parameters
    parser.add_argument('--model', default='labram_base_patch200_1600_8k_vocab', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_true', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=False)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=True)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=1600, type=int,
                        help='EEG input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Tokenizer parameters
    parser.add_argument('--codebook_size', default=8192, type=int, help='number of entries in the codebook')
    parser.add_argument('--quantizer_dim', default=32, type=int, help='dimension of each codebook entry')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD.
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)

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


def main(args):
    device, num_tasks, global_rank = runner_common.setup_environment(args)
    print(args)

    model = get_model(args)
    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    # get dataset
    # datasets with the same montage can be packed within a sublist
    datasets_train = [
        ["path/to/dataset1", "path/to/dataset2"],  # e.g., 64 channels for dataset1 and dataset2
        ["path/to/dataset3", "path/to/dataset4"],  # e.g., 32 channels for dataset3 and dataset4
    ]
    # time window for each sublist; sequence length is window * channel_count
    time_window = [4, 8]
    dataset_train_list, train_ch_names_list = utils.build_pretraining_dataset(
        datasets_train, time_window, stride=800, start_percentage=0, end_percentage=1,
    )
    # prepare visual tokenizer
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


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
