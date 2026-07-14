# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Assemble the pieces needed to evaluate a trained fine-tune model: rebuild the
# architecture, load its weights, read the run config and data-split record, and
# fetch all of these from either a local output directory or a ClearML experiment.
# ---------------------------------------------------------

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.utils.data
from timm.models import create_model

import labram.models.registry  # noqa: F401  (registers timm factories)
import labram.utils as utils
from labram.configs.run_configs import FinetuneRunConfig
from labram.runs.finetune_setup import enable_window_ids

logger = utils.get_logger(__name__)


# ---------------------------------------------------------------------------
# Config / model / weights
# ---------------------------------------------------------------------------

def load_run_config(path: str) -> FinetuneRunConfig:
    """Load a saved ``FinetuneRunConfig`` (``run_config.yaml`` / ``.json``)."""
    return FinetuneRunConfig.load_config(path)


def build_finetune_model(
    config: FinetuneRunConfig,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """Rebuild the fine-tune model from a run config (plain classifier path).

    Mirrors :func:`labram.runs.run_finetune.get_model` for the non-codebook
    case. For codebook-regularized runs use :func:`build_codebook_classifier`
    directly; here we build the plain :class:`NeuralTransformer` classifier that
    the notebook targets.
    """
    if config.model.codebook_reg.enabled:
        from labram.runs.codebook_setup import build_codebook_classifier
        model = build_codebook_classifier(config)
    else:
        m = config.model
        model = create_model(
            m.model, pretrained=False,
            num_classes=m.nb_classes, drop_rate=m.drop,
            drop_path_rate=m.drop_path, attn_drop_rate=m.attn_drop_rate,
            use_mean_pooling=m.use_mean_pooling, init_scale=m.init_scale,
            use_rel_pos_bias=m.rel_pos_bias, use_abs_pos_emb=m.abs_pos_emb,
            init_values=m.layer_scale_init_value, qkv_bias=m.qkv_bias,
        )
    if device is not None:
        model.to(device)
    return model


def build_model_by_name(
    model_name: str,
    nb_classes: int,
    device: Optional[torch.device] = None,
    **create_kwargs,
) -> torch.nn.Module:
    """Rebuild a fine-tune model by registry name when no run config is at hand.

    Defaults match the TUAB fine-tuning recipe (abs pos-emb on, rel-pos-bias and
    qkv-bias off, mean pooling); override via ``create_kwargs``.
    """
    kwargs = dict(
        num_classes=nb_classes, use_mean_pooling=True,
        use_rel_pos_bias=False, use_abs_pos_emb=True, qkv_bias=False,
        init_values=0.1,
    )
    kwargs.update(create_kwargs)
    model = create_model(model_name, pretrained=False, **kwargs)
    if device is not None:
        model.to(device)
    return model


def load_checkpoint_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    model_key: str = 'model|module',
    prefix: str = '',
) -> torch.nn.Module:
    """Load fine-tuned weights from a checkpoint into ``model`` (in place).

    Handles the ``{'model': ...}`` / ``{'module': ...}`` / bare-state-dict
    layouts written by the trainer and drops ``relative_position_index`` buffers.
    """
    if checkpoint_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    state = None
    for key in model_key.split('|'):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state = checkpoint[key]
            logger.info("Loaded state_dict by model_key=%s", key)
            break
    if state is None:
        state = checkpoint

    for key in list(state.keys()):
        if 'relative_position_index' in key:
            state.pop(key)

    utils.load_state_dict(model, state, prefix=prefix)
    logger.info("Loaded fine-tune weights from %s", checkpoint_path)
    return model


def load_data_split(path: str) -> dict:
    """Read a ``data_split.json`` record (see ``labram.utils.data_split``)."""
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# High-level loaders (local directory / explicit paths)
# ---------------------------------------------------------------------------

@dataclass
class ExperimentAssets:
    """Local paths to the artifacts of a finished fine-tuning experiment."""

    checkpoint: Optional[str] = None
    config: Optional[str] = None
    data_split: Optional[str] = None
    source: str = ''


def find_experiment_assets(output_dir: str) -> ExperimentAssets:
    """Locate a run's checkpoint, config and data-split in a local ``output_dir``.

    Prefers ``checkpoint-best.pth`` then ``checkpoint-final.pth`` then
    ``checkpoint.pth``; reads ``run_config.yaml``/``run_config.json`` and
    ``data_split.json`` when present.
    """
    from labram.utils.clearml_artifacts import resolve_final_checkpoint
    ckpt = resolve_final_checkpoint(output_dir)
    cfg = None
    for name in ('run_config.yaml', 'run_config.json', 'config.yaml', 'config.json'):
        candidate = os.path.join(output_dir, name)
        if os.path.exists(candidate):
            cfg = candidate
            break
    split = os.path.join(output_dir, 'data_split.json')
    split = split if os.path.exists(split) else None
    return ExperimentAssets(checkpoint=ckpt, config=cfg, data_split=split,
                            source=f'local:{output_dir}')


def load_model_from_assets(
    assets: ExperimentAssets,
    device: Optional[torch.device] = None,
    model_name: Optional[str] = None,
    nb_classes: Optional[int] = None,
) -> Tuple[torch.nn.Module, Optional[FinetuneRunConfig], Optional[dict]]:
    """Rebuild + load a model from ``assets``, returning ``(model, config, split)``.

    Uses the saved run config for the architecture when available, otherwise
    falls back to ``model_name`` + ``nb_classes``.
    """
    config = None
    if assets.config:
        try:
            config = load_run_config(assets.config)
        except Exception as exc:
            logger.warning("Could not parse run config %s: %s; falling back to "
                           "model_name/nb_classes", assets.config, exc)
    if config is not None:
        model = build_finetune_model(config, device=device)
    else:
        if model_name is None or nb_classes is None:
            raise ValueError(
                "No run config found; pass model_name and nb_classes to rebuild "
                "the architecture.")
        model = build_model_by_name(model_name, nb_classes, device=device)
    if assets.checkpoint:
        load_checkpoint_weights(model, assets.checkpoint)
    split = load_data_split(assets.data_split) if assets.data_split else None
    return model, config, split


# ---------------------------------------------------------------------------
# ClearML experiment loader (best-effort; requires the optional `clearml` dep)
# ---------------------------------------------------------------------------

def load_clearml_assets(
    task_id: Optional[str] = None,
    task_name: Optional[str] = None,
    project_name: Optional[str] = None,
    checkpoint_artifact: str = 'trained_model',
    config_artifact: str = 'run_config',
    data_split_artifact: str = 'data_split',
) -> ExperimentAssets:
    """Download a ClearML experiment's model / config / data-split artifacts.

    Resolve the task by ``task_id`` or by ``(project_name, task_name)`` and fetch
    local copies of the registered artifacts. The exact artifact names match the
    trainer's uploads (``clearml.artifact_name`` for the model, ``run_config``
    configuration, ``data_split`` artifact). Best-effort: missing artifacts are
    left ``None`` and logged rather than raising.
    """
    from clearml import Task  # optional dependency

    if task_id:
        task = Task.get_task(task_id=task_id)
    else:
        task = Task.get_task(project_name=project_name, task_name=task_name)
    if task is None:
        raise ValueError(
            f"No ClearML task for id={task_id!r} project={project_name!r} name={task_name!r}")

    assets = ExperimentAssets(source=f'clearml:{task.id}')

    # Model checkpoint: prefer a registered OutputModel, fall back to an artifact.
    try:
        output_models = task.models.get('output', []) if hasattr(task, 'models') else []
        if output_models:
            assets.checkpoint = output_models[-1].get_local_copy()
    except Exception as exc:  # pragma: no cover - depends on clearml/S3
        logger.warning("Could not fetch ClearML output model: %s", exc)

    artifacts = getattr(task, 'artifacts', {}) or {}
    if assets.checkpoint is None and checkpoint_artifact in artifacts:
        try:
            assets.checkpoint = artifacts[checkpoint_artifact].get_local_copy()
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not fetch checkpoint artifact %r: %s", checkpoint_artifact, exc)

    if data_split_artifact in artifacts:
        try:
            assets.data_split = artifacts[data_split_artifact].get_local_copy()
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not fetch data-split artifact: %s", exc)

    # The run config is connected as a Configuration object; write it to a file.
    try:
        cfg_text = task.get_configuration_object(name=config_artifact)
        if cfg_text:
            import tempfile
            fd, path = tempfile.mkstemp(suffix='.json', prefix='run_config_')
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                fh.write(cfg_text)
            assets.config = path
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not fetch ClearML run_config configuration: %s", exc)

    logger.info("Resolved ClearML assets: %s", assets)
    return assets


# ---------------------------------------------------------------------------
# Evaluation dataloader (sequential, case-id-aware)
# ---------------------------------------------------------------------------

def build_case_loader(
    dataset,
    batch_size: int = 64,
    agg_case_by: str = 'recording',
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[torch.utils.data.DataLoader, Optional[List[str]]]:
    """Sequential eval loader that yields per-window case ids for aggregation.

    Enables window ids in place (``recording`` / ``subject``) and returns the
    loader plus the window file names in iteration order (or ``None`` if the
    dataset does not expose them), which downstream code attaches to the
    predictions for per-case EEG inspection.
    """
    from labram.eval.inference import ordered_files
    enable_window_ids(dataset, group_by=agg_case_by)
    files = ordered_files(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, sampler=torch.utils.data.SequentialSampler(dataset),
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=False,
    )
    return loader, files
