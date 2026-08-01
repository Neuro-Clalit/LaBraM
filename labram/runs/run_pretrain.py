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
from labram.configs.run_configs import PretrainRunConfig
from labram.configs.utils_conf import add_override_arg, parse_overrides
from labram.train.train_pretrain import train_loop
from labram.optim_factory import create_optimizer, log_trainable_parameters
from labram.utils import NativeScalerWithGradNormCount as NativeScaler

logger = utils.get_logger(__name__)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser('LaBraM pre-training (config-driven)', add_help=True)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a JSON or YAML PretrainRunConfig file.')
    add_override_arg(parser)
    return parser.parse_args()


def get_model(config: PretrainRunConfig):
    m = config.model
    t = m.tokenizer
    logger.info(f"Creating model: {m.model}")
    return create_model(
        m.model,
        pretrained=False,
        drop_path_rate=m.drop_path,
        use_shared_rel_pos_bias=m.rel_pos_bias,
        use_abs_pos_emb=m.abs_pos_emb,
        init_values=m.layer_scale_init_value,
        vocab_size=t.codebook_size,
        labram_plus=config.labram_plus,
    )


def get_visual_tokenizer(config: PretrainRunConfig):
    t = config.model.tokenizer
    logger.info(f"Creating visual tokenizer: {t.tokenizer_model}")
    return create_model(
        t.tokenizer_model,
        pretrained=True,
        pretrained_weight=t.tokenizer_weight,
        as_tokenzer=True,
        num_codebook_tokens=t.codebook_size,
        quantizer_dim=t.quantizer_dim,
        labram_plus=config.labram_plus,
    ).eval()


def main(config: PretrainRunConfig):
    device, num_tasks, global_rank = runner_common.setup_environment(config)
    logger.info("%s", config)

    model = get_model(config)
    patch_size = model.patch_size
    logger.info("Patch size = %s", str(patch_size))

    dataset_train_list, train_ch_names_list = utils.build_pretraining_dataset(
        config.data.datasets_train,
        config.data.time_window,
        stride=config.data.stride,
        start_percentage=config.data.start_percentage,
        end_percentage=config.data.end_percentage,
    )
    vqnsp = get_visual_tokenizer(config).to(device)

    num_training_steps_per_epoch = (
        sum(len(d) for d in dataset_train_list) // config.trainer.batch_size // num_tasks
    )

    sampler_train_list = runner_common.build_distributed_train_sampler_list(
        dataset_train_list, num_tasks, global_rank,
    )
    log_writer = runner_common.create_log_writer(
        config.output, global_rank, config.clearml, config)

    data_loader_list = runner_common.build_dataloader_list(
        dataset_train_list, sampler_train_list,
        batch_size=config.trainer.batch_size, num_workers=config.data.num_workers,
        pin_memory=config.data.pin_mem, drop_last=True,
    )

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model = %s", str(model_without_ddp))
    logger.info('number of params: %s', n_parameters)
    log_trainable_parameters(model_without_ddp, logger)
    runner_common.log_model_visualization(config.logging, config.output, model_without_ddp, log_writer)
    logger.info("Tokenizer = %s", str(vqnsp))

    total_batch_size = config.trainer.batch_size * num_tasks * config.trainer.gradient_accumulation_steps
    logger.info("LR = %.8f", config.optimizer.lr)
    logger.info("Batch size = %d", total_batch_size)
    logger.info("Training steps/epoch = %d", num_training_steps_per_epoch)

    model, model_without_ddp = runner_common.wrap_distributed(config.distributed, model)
    optimizer = create_optimizer(config.optimizer, model_without_ddp)
    loss_scaler = NativeScaler()

    utils.auto_load_model(
        output_cfg=config.output, trainer_cfg=config.trainer,
        model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler,
    )

    train_loop(
        config=config,
        model=model, model_without_ddp=model_without_ddp,
        vqnsp=vqnsp, data_loader_list=data_loader_list,
        train_ch_names_list=train_ch_names_list,
        optimizer=optimizer, device=device, loss_scaler=loss_scaler,
        log_writer=log_writer,
        n_parameters=n_parameters,
        num_training_steps_per_epoch=num_training_steps_per_epoch,
    )

    runner_common.finalize_run(config, log_writer)


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
