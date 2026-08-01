"""Tests for labram.configs.run_configs.

Covers:
* Default values for each top-level run config match the legacy argparse defaults.
* Nested sub-configs are independent instances per run config.
* to_namespace flattens every leaf field; mutations don't leak back to the config.
* YAML/JSON round-trip preserves nested structure.
* parse_overrides coerces value types and rejects malformed entries.
* Repeated ``--set`` flags accumulate rather than replacing each other.
* PretrainRunConfig.load_config + overrides composes as expected.
"""

import argparse
import importlib
import inspect
from argparse import Namespace

import pytest

from labram.configs import defaults
from labram.configs.run_configs import (
    DataConfig,
    DistributedConfig,
    FinetuneRunConfig,
    OptimizerConfig,
    OutputConfig,
    PretrainModelConfig,
    PretrainRunConfig,
    TokenizerConfig,
    TrainerConfig,
    VQNSPRunConfig,
)
from labram.configs.utils_conf import add_override_arg, parse_overrides


# ============================================================
# Defaults match legacy argparse values
# ============================================================


class TestPretrainDefaults:
    def test_trainer_defaults(self):
        cfg = PretrainRunConfig()
        assert cfg.trainer.batch_size == defaults.DEFAULT_BATCH_SIZE == 64
        assert cfg.trainer.epochs == defaults.DEFAULT_PRETRAIN_EPOCHS == 300
        assert cfg.trainer.start_epoch == 0
        assert cfg.trainer.gradient_accumulation_steps == 1

    def test_optimizer_defaults_match_pretrain_script(self):
        cfg = PretrainRunConfig()
        assert cfg.optimizer.opt == 'adamw'
        assert cfg.optimizer.lr == defaults.DEFAULT_PRETRAIN_LR == 5e-4
        assert cfg.optimizer.weight_decay == 0.05
        assert cfg.optimizer.warmup_epochs == 5
        assert cfg.optimizer.min_lr == defaults.DEFAULT_MIN_LR == 1e-5
        assert cfg.optimizer.opt_eps == 1e-8
        assert cfg.optimizer.opt_betas is None
        assert cfg.optimizer.clip_grad is None

    def test_model_defaults(self):
        cfg = PretrainRunConfig()
        assert cfg.model.model == 'labram_base_patch200_1600_8k_vocab'
        assert cfg.model.input_size == 1600
        assert cfg.model.rel_pos_bias is False
        assert cfg.model.abs_pos_emb is True
        assert cfg.model.layer_scale_init_value == 0.1
        assert cfg.model.drop_path == 0.1

    def test_tokenizer_defaults(self):
        cfg = PretrainRunConfig()
        assert cfg.tokenizer.tokenizer_model == 'vqnsp_encoder_base_decoder_3x200x12'
        assert cfg.tokenizer.tokenizer_weight == ''
        assert cfg.tokenizer.codebook_size == 8192
        assert cfg.tokenizer.quantizer_dim == 32

    def test_distributed_defaults(self):
        cfg = PretrainRunConfig()
        assert cfg.distributed.device == 'cuda'
        assert cfg.distributed.world_size == 1
        assert cfg.distributed.dist_url == 'env://'
        assert cfg.distributed.seed == 0

    def test_data_defaults(self):
        cfg = PretrainRunConfig()
        assert cfg.data.num_workers == 10
        assert cfg.data.pin_mem is True
        assert cfg.data.stride == defaults.DEFAULT_PRETRAIN_STRIDE == 800
        assert cfg.data.time_window == [4, 8]
        assert cfg.data.start_percentage == 0.0
        assert cfg.data.end_percentage == 1.0
        assert cfg.data.datasets_train == []

    def test_output_defaults(self):
        cfg = PretrainRunConfig()
        assert cfg.output.output_dir == ''
        assert cfg.output.resume == ''
        assert cfg.output.auto_resume is True
        assert cfg.output.save_ckpt_freq == defaults.DEFAULT_PRETRAIN_SAVE_CKPT_FREQ == 20


class TestVQNSPDefaults:
    def test_vqnsp_uses_lower_lr_and_wd_than_pretrain(self):
        cfg = VQNSPRunConfig()
        assert cfg.optimizer.lr == defaults.DEFAULT_VQNSP_LR == 5e-5
        assert cfg.optimizer.weight_decay == defaults.DEFAULT_VQNSP_WEIGHT_DECAY == 1e-4

    def test_vqnsp_trainer_epochs(self):
        cfg = VQNSPRunConfig()
        assert cfg.trainer.epochs == defaults.DEFAULT_VQNSP_EPOCHS == 100

    def test_vqnsp_distributed_dist_eval_default(self):
        cfg = VQNSPRunConfig()
        assert cfg.distributed.dist_eval is True

    def test_vqnsp_data_stride(self):
        cfg = VQNSPRunConfig()
        assert cfg.data.stride == defaults.DEFAULT_VQNSP_STRIDE == 200

    def test_vqnsp_model_defaults(self):
        cfg = VQNSPRunConfig()
        assert cfg.model.model == 'vqnsp_encoder_base_decoder_3x200x12'
        assert cfg.model.ema_decay == 0.99


class TestFinetuneDefaults:
    def test_finetune_lr_and_min_lr(self):
        cfg = FinetuneRunConfig()
        assert cfg.optimizer.lr == defaults.DEFAULT_FINETUNE_LR == 5e-4
        assert cfg.optimizer.min_lr == defaults.DEFAULT_FINETUNE_MIN_LR == 1e-6

    def test_finetune_trainer_epochs(self):
        cfg = FinetuneRunConfig()
        assert cfg.trainer.epochs == defaults.DEFAULT_FINETUNE_EPOCHS == 30

    def test_finetune_device_auto(self):
        cfg = FinetuneRunConfig()
        assert cfg.distributed.device == 'auto'

    def test_finetune_model_defaults(self):
        cfg = FinetuneRunConfig()
        assert cfg.model.model == 'labram_base_patch200_200'
        assert cfg.model.qkv_bias is True
        assert cfg.model.rel_pos_bias is True
        assert cfg.model.abs_pos_emb is False
        assert cfg.model.use_mean_pooling is True

    def test_finetune_layer_decay_and_smoothing(self):
        cfg = FinetuneRunConfig()
        assert cfg.layer_decay == 0.9
        assert cfg.smoothing == 0.1


# ============================================================
# Sub-config independence (no shared mutable state)
# ============================================================


class TestClearMLDefaults:
    def test_present_and_disabled_by_default_on_every_phase(self):
        for cfg in (PretrainRunConfig(), VQNSPRunConfig(), FinetuneRunConfig()):
            assert cfg.clearml.enabled is False
            assert cfg.clearml.project_name == 'LaBraM'
            assert cfg.clearml.task_name == ''
            assert cfg.clearml.tags == []
            assert cfg.clearml.offline is False
            assert cfg.clearml.auto_connect_frameworks is True

    def test_independent_tags_list_per_config(self):
        a = PretrainRunConfig()
        b = PretrainRunConfig()
        a.clearml.tags.append('exp-1')
        assert b.clearml.tags == []

    def test_override_via_load_config(self):
        cfg = PretrainRunConfig.load_config(
            None,
            **{'clearml.enabled': True, 'clearml.project_name': 'My EEG'},
        )
        assert cfg.clearml.enabled is True
        assert cfg.clearml.project_name == 'My EEG'


class TestSubConfigIndependence:
    def test_two_pretrain_configs_have_independent_optimizers(self):
        a = PretrainRunConfig()
        b = PretrainRunConfig()
        a.optimizer.lr = 999.0
        assert b.optimizer.lr == defaults.DEFAULT_PRETRAIN_LR

    def test_two_pretrain_configs_have_independent_data_lists(self):
        a = PretrainRunConfig()
        b = PretrainRunConfig()
        a.data.datasets_train.append([['fake.h5']])
        assert b.data.datasets_train == []


# ============================================================
# to_namespace flattens fields without leakage
# ============================================================


class TestToNamespace:
    def test_namespace_contains_all_leaf_fields(self):
        cfg = PretrainRunConfig()
        ns = cfg.to_namespace()
        assert isinstance(ns, Namespace)
        # Pick representative leaf keys from each sub-config
        for key in ('batch_size', 'epochs', 'lr', 'weight_decay', 'model', 'tokenizer_model',
                    'output_dir', 'auto_resume', 'world_size', 'device', 'num_workers',
                    'stride', 'codebook_size', 'drop_path'):
            assert hasattr(ns, key), f"namespace missing leaf {key!r}"

    def test_namespace_mutation_does_not_leak_back(self):
        cfg = PretrainRunConfig()
        ns = cfg.to_namespace()
        ns.lr = 42.0
        ns.epochs = 999
        ns.resume = '/tmp/ckpt.pth'
        assert cfg.optimizer.lr == defaults.DEFAULT_PRETRAIN_LR
        assert cfg.trainer.epochs == defaults.DEFAULT_PRETRAIN_EPOCHS
        assert cfg.output.resume == ''

    def test_vqnsp_namespace_carries_phase_specific_flags(self):
        cfg = VQNSPRunConfig()
        ns = cfg.to_namespace()
        assert ns.disable_eval is False
        assert ns.eval is False
        assert ns.calculate_codebook_usage is False


# ============================================================
# YAML round-trip
# ============================================================


class TestRoundTrip:
    def test_pretrain_yaml_round_trip_preserves_nested_values(self, tmp_path):
        cfg = PretrainRunConfig()
        cfg.optimizer.lr = 1e-3
        cfg.trainer.epochs = 7
        cfg.tokenizer.tokenizer_weight = '/tmp/vqnsp.pth'
        path = str(tmp_path / 'cfg.yaml')
        cfg.save_to(path)

        loaded = PretrainRunConfig.load_from(path)
        assert isinstance(loaded, PretrainRunConfig)
        assert loaded.optimizer.lr == 1e-3
        assert loaded.trainer.epochs == 7
        assert loaded.tokenizer.tokenizer_weight == '/tmp/vqnsp.pth'
        # untouched fields keep defaults
        assert loaded.model.model == 'labram_base_patch200_1600_8k_vocab'

    def test_pretrain_json_round_trip(self, tmp_path):
        cfg = PretrainRunConfig()
        cfg.data.datasets_train = [['/d/a.h5'], ['/d/b.h5', '/d/c.h5']]
        cfg.data.time_window = [4, 8]
        path = str(tmp_path / 'cfg.json')
        cfg.save_to(path)

        loaded = PretrainRunConfig.load_from(path)
        assert loaded.data.datasets_train == [['/d/a.h5'], ['/d/b.h5', '/d/c.h5']]
        assert loaded.data.time_window == [4, 8]


# ============================================================
# CLI override parsing
# ============================================================


class TestParseOverrides:
    def test_int_value_is_coerced(self):
        assert parse_overrides(['trainer.epochs=5']) == {'trainer.epochs': 5}

    def test_float_value_is_coerced(self):
        out = parse_overrides(['optimizer.lr=1e-4'])
        assert out == {'optimizer.lr': 1e-4}

    def test_bool_values_are_coerced_case_insensitive(self):
        out = parse_overrides(['output.auto_resume=false', 'data.pin_mem=True'])
        assert out == {'output.auto_resume': False, 'data.pin_mem': True}

    def test_none_value_is_coerced(self):
        out = parse_overrides(['optimizer.weight_decay_end=none'])
        assert out == {'optimizer.weight_decay_end': None}

    def test_string_value_left_as_is(self):
        out = parse_overrides(['model.model=labram_custom_patch'])
        assert out == {'model.model': 'labram_custom_patch'}

    def test_multiple_overrides(self):
        out = parse_overrides(['a=1', 'b=2.5', 'c=hello', 'd=false'])
        assert out == {'a': 1, 'b': 2.5, 'c': 'hello', 'd': False}

    def test_missing_equals_raises_value_error(self):
        with pytest.raises(ValueError, match='key=value'):
            parse_overrides(['trainer.epochs'])

    def test_empty_list_returns_empty_dict(self):
        assert parse_overrides([]) == {}


class TestAddOverrideArg:
    """``--set`` must accumulate across flags, not keep only the last one."""

    @staticmethod
    def _parser():
        parser = argparse.ArgumentParser()
        add_override_arg(parser)
        return parser

    def test_repeated_flags_accumulate(self):
        cli = self._parser().parse_args([
            '--set', 'data.data_path=/data/TUAB',
            '--set', 'distributed.device=mps',
            '--set', 'trainer.epochs=2',
        ])
        assert parse_overrides(cli.overrides) == {
            'data.data_path': '/data/TUAB',
            'distributed.device': 'mps',
            'trainer.epochs': 2,
        }

    def test_several_values_on_one_flag(self):
        cli = self._parser().parse_args(['--set', 'a=1', 'b=2'])
        assert cli.overrides == ['a=1', 'b=2']

    def test_default_is_empty_and_not_shared_between_parses(self):
        parser = self._parser()
        parser.parse_args(['--set', 'a=1'])
        assert parser.parse_args([]).overrides == []


@pytest.mark.parametrize('module_name', [
    'labram.runs.run_finetune',
    'labram.runs.run_pretrain',
    'labram.runs.run_vqnsp',
    'labram.runs.finetune_cv',
    'labram.runs.sagemaker_entry',
    'labram.runs.submit_sagemaker',
])
def test_entry_points_use_the_shared_override_arg(module_name):
    """Every entry point must go through add_override_arg, so none of them
    regresses to a plain nargs='*' that drops all but the last --set."""
    source = inspect.getsource(importlib.import_module(module_name))
    assert 'add_override_arg(parser)' in source
    assert "add_argument('--set'" not in source


# ============================================================
# load_config + override composition
# ============================================================


class TestLoadConfigWithOverrides:
    def test_load_config_none_applies_overrides(self):
        overrides = parse_overrides(['trainer.epochs=3', 'optimizer.lr=2e-4'])
        cfg = PretrainRunConfig.load_config(None, **overrides)
        assert cfg.trainer.epochs == 3
        assert cfg.optimizer.lr == 2e-4
        # untouched defaults
        assert cfg.trainer.batch_size == defaults.DEFAULT_BATCH_SIZE

    def test_load_config_file_then_override(self, tmp_path):
        seed = PretrainRunConfig()
        seed.trainer.epochs = 50
        seed.optimizer.lr = 1e-3
        path = str(tmp_path / 'cfg.yaml')
        seed.save_to(path)

        overrides = parse_overrides(['optimizer.lr=5e-5'])
        cfg = PretrainRunConfig.load_config(path, **overrides)
        assert cfg.trainer.epochs == 50          # came from file
        assert cfg.optimizer.lr == 5e-5          # CLI override wins


# ============================================================
# Sub-config types are correct after load
# ============================================================


class TestNestedTypes:
    def test_pretrain_sub_config_classes_after_default_construction(self):
        cfg = PretrainRunConfig()
        assert isinstance(cfg.model, PretrainModelConfig)
        assert isinstance(cfg.tokenizer, TokenizerConfig)
        assert isinstance(cfg.optimizer, OptimizerConfig)
        assert isinstance(cfg.trainer, TrainerConfig)
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.output, OutputConfig)
        assert isinstance(cfg.distributed, DistributedConfig)

    def test_pretrain_sub_config_classes_after_yaml_load(self, tmp_path):
        PretrainRunConfig().save_to(str(tmp_path / 'cfg.yaml'))
        cfg = PretrainRunConfig.load_from(str(tmp_path / 'cfg.yaml'))
        assert isinstance(cfg.model, PretrainModelConfig)
        assert isinstance(cfg.tokenizer, TokenizerConfig)
        assert isinstance(cfg.optimizer, OptimizerConfig)
