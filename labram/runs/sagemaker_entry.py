# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# In-container entry point for SageMaker training jobs. SageMaker invokes this as
#   python labram/runs/sagemaker_entry.py --config <path> --fold <k> [--set ...]
# passing the run config via the ``config`` input channel (mounted at
# /opt/ml/input/data/config) and the CV fold as a hyperparameter. It resolves the
# config, points outputs at the SageMaker model dir (auto-uploaded to S3 at job
# end), and dispatches to the cross-validation runner (or a plain fine-tune).
# ---------------------------------------------------------

import argparse
import os
from typing import List, Optional

import labram.utils as utils
from labram.configs.run_configs import (
    FinetuneRunConfig,
    PretrainRunConfig,
    VQNSPRunConfig,
)
from labram.configs.utils_conf import parse_overrides

logger = utils.get_logger(__name__)

_CONFIG_NAMES = ('run_config.yaml', 'run_config.json', 'config.yaml', 'config.json')

# Trainer phase -> RunConfig class. Mirrors labram.runs.submit_sagemaker.
PHASE_CONFIGS = {
    'vqnsp': VQNSPRunConfig,
    'pretrain': PretrainRunConfig,
    'finetune': FinetuneRunConfig,
}


def find_config_path(explicit: Optional[str]) -> Optional[str]:
    """Locate the run config: an explicit path, else a well-known name (or the
    first YAML/JSON) in the SageMaker ``config`` input channel."""
    if explicit and os.path.exists(explicit):
        return explicit
    channel = os.environ.get('SM_CHANNEL_CONFIG')
    if channel and os.path.isdir(channel):
        for name in _CONFIG_NAMES:
            candidate = os.path.join(channel, name)
            if os.path.exists(candidate):
                return candidate
        cfgs = sorted(f for f in os.listdir(channel)
                      if f.endswith(('.yaml', '.yml', '.json')))
        if cfgs:
            return os.path.join(channel, cfgs[0])
    return explicit


def parse_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser('LaBraM SageMaker training entry point')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--phase', choices=sorted(PHASE_CONFIGS), default='finetune',
                        help='Trainer to run: vqnsp | pretrain | finetune.')
    parser.add_argument('--fold', type=int, default=None,
                        help='CV fold to train (finetune only; -1 or omitted -> all / non-CV).')
    parser.add_argument('--set', dest='overrides', nargs='*', default=[],
                        metavar='KEY=VALUE')
    return parser.parse_args(argv)


def build_config(cli: argparse.Namespace):
    cfg_path = find_config_path(cli.config)
    overrides = parse_overrides(cli.overrides)
    config = PHASE_CONFIGS[cli.phase].load_config(cfg_path, **overrides)

    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    if not config.output.output_dir:
        config.output.output_dir = os.path.join(model_dir, cli.phase)

    # Cross-validation applies to fine-tuning only.
    if cli.phase == 'finetune':
        if cli.fold is not None and cli.fold >= 0:
            config.cross_validation.enabled = True
            config.cross_validation.fold = cli.fold
        if config.cross_validation.enabled and not config.cross_validation.base_dir:
            # Keep every fold's outputs under the SageMaker model dir so they are
            # uploaded to S3 when the job completes.
            config.cross_validation.base_dir = os.path.join(model_dir, 'cv')
    return config


def main(argv: Optional[List[str]] = None) -> None:
    cli = parse_cli(argv)
    config = build_config(cli)
    if cli.phase == 'vqnsp':
        from labram.runs.run_vqnsp import main as vqnsp_main
        vqnsp_main(config)
    elif cli.phase == 'pretrain':
        from labram.runs.run_pretrain import main as pretrain_main
        pretrain_main(config)
    elif config.cross_validation.enabled:
        from labram.runs.finetune_cv import run_cross_validation
        run_cross_validation(config)
    else:
        from labram.runs.run_finetune import main as finetune_main
        finetune_main(config)


if __name__ == '__main__':
    main()
