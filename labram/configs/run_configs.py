"""Top-level run configurations for LaBraM training scripts.

Each `*RunConfig` composes the shared sub-configs with phase-specific
model/tokenizer settings. Sub-config types live in their own modules:

    labram.configs.data_config    — DataConfig
    labram.configs.model_config   — PretrainModelConfig, VQNSPModelConfig,
                                    FinetuneModelConfig, TokenizerConfig,
                                    FinetuneCheckpointConfig
    labram.configs.optim_config   — OptimizerConfig
    labram.configs.train_config   — TrainerConfig, OutputConfig, DistributedConfig

``RunConfig.to_namespace()`` flattens every leaf field into an
``argparse.Namespace`` for legacy consumers (``init_distributed_mode``,
``create_optimizer``, ``auto_load_model``, ``save_model``) that read a
flat ``args.X`` surface. Mutations to the namespace are isolated from the
typed config.
"""

from argparse import Namespace
from dataclasses import dataclass, field
from typing import List

from labram.configs.base_configs import ConfigBase
from labram.configs.data_config import DataConfig
from labram.configs import defaults as conf_consts
from labram.configs.loss_config import LossConfig
from labram.configs.model_config import (
    CodebookRegConfig,
    FinetuneCheckpointConfig,
    FinetuneModelConfig,
    PretrainModelConfig,
    TokenizerConfig,
    VQNSPModelConfig,
)
from labram.configs.optim_config import OptimizerConfig
from labram.configs.train_config import ClearMLConfig, DistributedConfig, OutputConfig, TrainerConfig


# ============================================================
# Base run config
# ============================================================


@dataclass
class RunConfig(ConfigBase):
    """Base for any single-phase training run. Subclasses inject the
    phase-specific ``model`` (and any extras like ``tokenizer``)."""

    model: ConfigBase = None
    output: OutputConfig = field(default_factory=OutputConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    clearml: ClearMLConfig = field(default_factory=ClearMLConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def to_namespace(self) -> Namespace:
        """Flatten every leaf field into a single ``argparse.Namespace``.

        Legacy consumers (``init_distributed_mode``, ``create_optimizer``,
        ``auto_load_model``, ``save_model``) read a flat ``args.X`` surface
        and sometimes mutate it. We give them a separate Namespace so those
        mutations (rank, gpu, distributed, resume) don't leak back into the
        typed config.
        """
        ns = Namespace()

        def _flatten(obj):
            for k, v in obj.__dict__.items():
                if isinstance(v, ConfigBase):
                    _flatten(v)
                else:
                    setattr(ns, k, v)

        _flatten(self)
        return ns


# ============================================================
# Phase-specific run configs
# ============================================================


@dataclass
class PretrainRunConfig(RunConfig):
    model: PretrainModelConfig = field(default_factory=PretrainModelConfig)

    @property
    def tokenizer(self) -> TokenizerConfig:
        return self.model.tokenizer

    @tokenizer.setter
    def tokenizer(self, value: TokenizerConfig):
        self.model.tokenizer = value

    @property
    def dataset(self) -> str:
        return self.data.dataset

    @dataset.setter
    def dataset(self, value: str):
        self.data.dataset = value

    @property
    def batch_size(self) -> int:
        return self.trainer.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self.trainer.batch_size = value

    @property
    def lr(self) -> float:
        return self.optimizer.lr

    @lr.setter
    def lr(self, value: float):
        self.optimizer.lr = value

    @property
    def epochs(self) -> int:
        return self.trainer.epochs

    @epochs.setter
    def epochs(self, value: int):
        self.trainer.epochs = value

    @property
    def update_freq(self) -> int:
        return self.trainer.update_freq

    @update_freq.setter
    def update_freq(self, value: int):
        self.trainer.update_freq = value

    @property
    def data_path(self) -> str:
        return self.data.data_path

    @data_path.setter
    def data_path(self, value: str):
        self.data.data_path = value


@dataclass
class VQNSPRunConfig(RunConfig):
    model: VQNSPModelConfig = field(default_factory=VQNSPModelConfig)
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            lr=conf_consts.DEFAULT_VQNSP_LR,
            weight_decay=conf_consts.DEFAULT_VQNSP_WEIGHT_DECAY,
        ),
    )
    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(epochs=conf_consts.DEFAULT_VQNSP_EPOCHS),
    )
    distributed: DistributedConfig = field(
        default_factory=lambda: DistributedConfig(dist_eval=conf_consts.DEFAULT_VQNSP_DIST_EVAL),
    )
    data: DataConfig = field(
        default_factory=lambda: DataConfig(stride=conf_consts.DEFAULT_VQNSP_STRIDE),
    )
    output: OutputConfig = field(
        default_factory=lambda: OutputConfig(save_ckpt_freq=conf_consts.DEFAULT_VQNSP_SAVE_CKPT_FREQ),
    )
    disable_eval: bool = False
    eval: bool = False
    calculate_codebook_usage: bool = False


@dataclass
class FinetuneRunConfig(RunConfig):
    model: FinetuneModelConfig = field(default_factory=FinetuneModelConfig)
    finetune_checkpoint: FinetuneCheckpointConfig = field(default_factory=FinetuneCheckpointConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            lr=conf_consts.DEFAULT_FINETUNE_LR,
            min_lr=conf_consts.DEFAULT_FINETUNE_MIN_LR,
        ),
    )
    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(epochs=conf_consts.DEFAULT_FINETUNE_EPOCHS),
    )
    distributed: DistributedConfig = field(
        default_factory=lambda: DistributedConfig(device=conf_consts.DEFAULT_FINETUNE_DEVICE),
    )
    output: OutputConfig = field(
        default_factory=lambda: OutputConfig(save_ckpt_freq=conf_consts.DEFAULT_FINETUNE_SAVE_CKPT_FREQ),
    )

    @property
    def layer_decay(self) -> float:
        return self.optimizer.layer_decay

    @layer_decay.setter
    def layer_decay(self, value: float):
        self.optimizer.layer_decay = value

    @property
    def smoothing(self) -> float:
        return self.optimizer.smoothing

    @smoothing.setter
    def smoothing(self, value: float):
        self.optimizer.smoothing = value

    @property
    def model_ema(self) -> bool:
        return self.optimizer.model_ema

    @model_ema.setter
    def model_ema(self, value: bool):
        self.optimizer.model_ema = value

    @property
    def model_ema_decay(self) -> float:
        return self.optimizer.model_ema_decay

    @model_ema_decay.setter
    def model_ema_decay(self, value: float):
        self.optimizer.model_ema_decay = value

    @property
    def model_ema_force_cpu(self) -> bool:
        return self.optimizer.model_ema_force_cpu

    @model_ema_force_cpu.setter
    def model_ema_force_cpu(self, value: bool):
        self.optimizer.model_ema_force_cpu = value

    @property
    def disable_eval_during_finetuning(self) -> bool:
        return self.trainer.disable_eval_during_finetuning

    @disable_eval_during_finetuning.setter
    def disable_eval_during_finetuning(self, value: bool):
        self.trainer.disable_eval_during_finetuning = value

    @property
    def disable_weight_decay_on_rel_pos_bias(self) -> bool:
        return self.trainer.disable_weight_decay_on_rel_pos_bias

    @disable_weight_decay_on_rel_pos_bias.setter
    def disable_weight_decay_on_rel_pos_bias(self, value: bool):
        self.trainer.disable_weight_decay_on_rel_pos_bias = value

    @property
    def eval(self) -> bool:
        return self.trainer.eval

    @eval.setter
    def eval(self, value: bool):
        self.trainer.eval = value

    @property
    def enable_deepspeed(self) -> bool:
        return self.trainer.enable_deepspeed

    @enable_deepspeed.setter
    def enable_deepspeed(self, value: bool):
        self.trainer.enable_deepspeed = value

    @property
    def debug(self) -> bool:
        return self.trainer.debug

    @debug.setter
    def debug(self, value: bool):
        self.trainer.debug = value

    @property
    def debug_samples(self) -> int:
        return self.trainer.debug_samples

    @debug_samples.setter
    def debug_samples(self, value: int):
        self.trainer.debug_samples = value

    @property
    def datasets_train(self) -> List[str]:
        return self.data.datasets_train

    @datasets_train.setter
    def datasets_train(self, value: List[str]):
        self.data.datasets_train = value

    @property
    def data_path(self) -> str:
        return self.data.data_path

    @data_path.setter
    def data_path(self, value: str):
        self.data.data_path = value

    @property
    def robust_test(self) -> str:
        return self.data.robust_test

    @robust_test.setter
    def robust_test(self, value: str):
        self.data.robust_test = value

    @property
    def batch_size(self) -> int:
        return self.trainer.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self.trainer.batch_size = value

    @property
    def lr(self) -> float:
        return self.optimizer.lr

    @lr.setter
    def lr(self, value: float):
        self.optimizer.lr = value

    @property
    def epochs(self) -> int:
        return self.trainer.epochs

    @epochs.setter
    def epochs(self, value: int):
        self.trainer.epochs = value

    @property
    def update_freq(self) -> int:
        return self.trainer.update_freq

    @update_freq.setter
    def update_freq(self, value: int):
        self.trainer.update_freq = value

    @property
    def codebook_reg(self) -> CodebookRegConfig:
        return self.model.codebook_reg

    @codebook_reg.setter
    def codebook_reg(self, value: CodebookRegConfig):
        self.model.codebook_reg = value

    @property
    def dataset(self) -> str:
        return self.data.dataset

    @dataset.setter
    def dataset(self, value: str):
        self.data.dataset = value