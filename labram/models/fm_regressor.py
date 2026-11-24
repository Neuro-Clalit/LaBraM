# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Fine-tune model around a third-party pretrained encoder (CBraMod, REVE).
# ---------------------------------------------------------
"""``FoundationRegressor``: a foreign encoder with LaBraM's fine-tune interface.

The model is called exactly like
:class:`labram.models.neural_transformer.NeuralTransformer`, so the training
loops, the channel dropper, bag (MIL) training, the feature cache and the
evaluation scripts run unchanged:

* ``model(x, channel_indices)`` with ``x`` ``[B, C, A, 200]`` in µV / 100 and
  ``channel_indices`` from :func:`labram.data.eeg_constants.get_channel_indices`
  (class-token slot first). The channel *names* are recovered from the indices,
  which is what a foreign encoder needs (a fixed slot layout for CBraMod,
  electrode coordinates for REVE).
* ``return_features=True`` -> ``(output, head-input feature [B, D])``;
  ``bag=...`` -> the recording-level forward of
  :class:`labram.models.bag_forward.BagForwardMixin`.

Input preprocessing is model-owned, as in LaBraM, so training and every
evaluation path see the same signal:

    clip (model.input_clip_uv) -> CAR / band / amplitude (labram_plus)
    -> model.encoder.input_norm -> encoder

Readout (``model.encoder.readout``): ``mean`` averages all channel x patch
tokens; ``attn`` uses the encoder's own pretrained pooling (REVE's query
attention). Both are followed by a LayerNorm (``fc_norm``), then the head.
"""
from typing import Dict, List, Optional, Sequence, Set

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from labram.configs.model_config import MLPHeadConfig, MultiTaskHeadConfig
from labram.data.eeg_constants import standard_1020
from labram.layers import MLPHead, MultiTaskHead
from labram.models.backbones.base import EncoderBackbone
from labram.models.bag_forward import BagForwardMixin

READOUTS = ("mean", "attn")
INPUT_NORMS = ("none", "zscore_window")
HEAD_TYPES = ("linear", "mlp", "multitask")


def zscore_window(x: torch.Tensor, clip_sd: float, eps: float = 1e-6) -> torch.Tensor:
    """Per-channel z-score over the whole window, clipped at ``+-clip_sd``.

    ``x``: ``[B, C, A, T]``. An all-zero (missing) channel stays zero.
    """
    b, c, a, t = x.shape
    flat = x.reshape(b, c, a * t).float()
    flat = (flat - flat.mean(-1, keepdim=True)) / (flat.std(-1, keepdim=True) + eps)
    if clip_sd > 0:
        flat = flat.clamp(-clip_sd, clip_sd)
    return flat.reshape(b, c, a, t)


def _build_head(num_features: int, num_classes: int, head_type: str, mlp_head, multitask_head) -> nn.Module:
    if num_classes <= 0:
        return nn.Identity()
    if head_type == "mlp":
        c = MLPHeadConfig(**mlp_head) if isinstance(mlp_head, dict) else (mlp_head or MLPHeadConfig())
        return MLPHead(num_features, num_classes, hidden_dims=c.hidden_dims, drop=c.drop)
    if head_type == "multitask":
        c = (MultiTaskHeadConfig(**multitask_head) if isinstance(multitask_head, dict)
             else (multitask_head or MultiTaskHeadConfig()))
        return MultiTaskHead(num_features, num_classes, shared_dims=c.shared_dims,
                             age_dims=c.age_dims, abnormal_dims=c.abnormal_dims, drop=c.drop,
                             class_outputs=max(1, len(getattr(c, "class_names", []) or [])))
    if head_type == "linear":
        return nn.Linear(num_features, num_classes)
    raise ValueError(f"head_type={head_type!r} is not supported with a foreign encoder "
                     f"(use one of {HEAD_TYPES})")


def _init_head(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class FoundationRegressor(BagForwardMixin, nn.Module):
    def __init__(self, backbone: EncoderBackbone, *, num_classes: int = 1, head_type: str = "linear",
                 mlp_head=None, multitask_head=None, init_scale: float = 1.0,
                 readout: str = "", input_norm: str = "none", clip_sd: float = 15.0,
                 labram_plus=None, input_clip: float = 0.0):
        super().__init__()
        readout = readout or "mean"
        if readout not in READOUTS:
            raise ValueError(f"model.encoder.readout must be one of {READOUTS}, got {readout!r}")
        if input_norm not in INPUT_NORMS:
            raise ValueError(f"model.encoder.input_norm must be one of {INPUT_NORMS}, got {input_norm!r}")
        self.backbone = backbone
        self.readout = readout
        self.input_norm = input_norm
        self.clip_sd = float(clip_sd)
        self.labram_plus = labram_plus
        # Symmetric amplitude clip in model-input units (uV / 100); 0 = off.
        self.input_clip = float(input_clip or 0.0)
        self.num_features = self.embed_dim = backbone.embed_dim
        self.patch_size = 200
        self.num_classes = num_classes
        self.head_type = head_type

        self.fc_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.head = _build_head(self.num_features, num_classes, head_type, mlp_head, multitask_head)
        self.head.apply(_init_head)
        for out_layer in self._head_output_layers():
            out_layer.weight.data.mul_(init_scale)
            out_layer.bias.data.mul_(init_scale)
        if readout != "attn":
            # Parameters of the encoder's own pooling take no part in a mean readout.
            for p in backbone.pool_parameters():
                p.requires_grad_(False)
        self._names_cache: Dict[tuple, List[str]] = {}

    def _head_output_layers(self):
        if isinstance(self.head, MultiTaskHead):
            return [self.head.age_fc, self.head.abnormal_fc]
        if isinstance(self.head, MLPHead):
            return [self.head.fc]
        return [self.head] if isinstance(self.head, nn.Linear) else []

    # ------------------------------------------------------------ optimizer
    def get_num_layers(self) -> int:
        return self.backbone.num_layers

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return {f"backbone.{n}" for n in self.backbone.no_weight_decay()}

    def layer_id(self, name: str) -> int:
        """Layer index of parameter ``name`` for the layer-wise LR decay: the
        backbone's own (0 = stem, k = layer k), ``num_layers + 1`` (full LR) for
        the readout, the head and the bag pool."""
        if name.startswith("backbone."):
            return self.backbone.layer_id(name[len("backbone."):])
        return self.backbone.num_layers + 1

    def get_classifier(self):
        return self.head

    # ---------------------------------------------------------------- input
    def channel_names(self, channel_indices: Optional[Sequence[int]]) -> List[str]:
        """Channel names of LaBraM-style ``channel_indices`` (class-token slot 0
        first, then ``standard_1020`` index + 1 per channel)."""
        if channel_indices is None:
            raise ValueError(f"{self.backbone.name}: the model needs channel_indices "
                             "(labram.data.eeg_constants.get_channel_indices(ch_names))")
        key = tuple(int(i) for i in channel_indices)
        names = self._names_cache.get(key)
        if names is None:
            if not key or key[0] != 0 or any(not 1 <= i <= len(standard_1020) for i in key[1:]):
                raise ValueError(f"channel_indices must be [0, standard_1020 index + 1, ...], got {list(key)}")
            names = self._names_cache[key] = [standard_1020[i - 1] for i in key[1:]]
        return names

    def maybe_preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_clip > 0:
            x = x.clamp(-self.input_clip, self.input_clip)
        if self.labram_plus is not None and self.labram_plus.preprocesses_input:
            from labram.data.preprocess import apply_labram_plus_preprocess
            x = apply_labram_plus_preprocess(x, self.labram_plus)
        if self.input_norm == "zscore_window":
            x = zscore_window(x, self.clip_sd)
        return x

    # -------------------------------------------------------------- forward
    def forward_tokens(self, x, channel_indices=None):
        """Encoder tokens ``[B, C, P, D]`` of ``x`` ``[B, C, A, 200]``."""
        names = self.channel_names(channel_indices)
        if x.shape[1] != len(names):
            raise ValueError(f"{x.shape[1]} channels in the input, {len(names)} channel indices")
        return self.backbone.encode(self.maybe_preprocess_input(x), names)

    def forward_features(self, x, channel_indices=None, return_patch_tokens=False,
                         return_all_tokens=False, **kwargs):
        tokens = self.forward_tokens(x, channel_indices)
        if return_patch_tokens or return_all_tokens:
            return tokens.flatten(1, 2)
        pooled = self.backbone.pool(tokens) if self.readout == "attn" else tokens.flatten(1, 2).mean(1)
        return self.fc_norm(pooled)

    def forward(self, x, channel_indices=None, **kwargs):
        """``x``: ``[B, C, A, 200]`` (µV / 100). ``bag=`` -> the recording-level
        forward; ``return_features=True`` -> ``(output, head-input feature)``.
        Other keyword arguments of the LaBraM model (``classify_only``, a
        ``condition`` of None) are accepted and ignored."""
        bag = kwargs.pop('bag', None)
        if bag is not None:
            return self.forward_bag(x, channel_indices, bag)
        if kwargs.pop('condition', None) is not None:
            raise ValueError(f"{self.backbone.name}: conditioning inputs are not supported")
        return_features = kwargs.pop('return_features', False)
        feature = self.forward_features(x, channel_indices=channel_indices)
        out = self.head(feature)
        return (out, feature) if return_features else out
