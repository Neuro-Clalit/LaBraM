# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Interface of a third-party pretrained EEG encoder used in place of LaBraM's.
# ---------------------------------------------------------
"""Foreign pretrained encoders behind one interface.

:class:`labram.models.fm_regressor.FoundationRegressor` wraps an
:class:`EncoderBackbone` with the readout, head and MIL pool of a fine-tune
model. A backbone owns the vendored upstream network (``self.net``, upstream
parameter names untouched so the released weights load strictly) and the
adapter that turns our model input into the network's.
"""
from typing import Dict, Iterable, List, Sequence, Set

import torch
import torch.nn as nn


class EncoderBackbone(nn.Module):
    """A pretrained encoder that maps windows to tokens.

    Subclasses set ``embed_dim`` (token width), ``num_layers`` (stacked
    transformer layers, for layer-wise LR decay) and ``name``.
    """

    name: str = ""
    embed_dim: int = 0
    num_layers: int = 0

    def encode(self, x: torch.Tensor, ch_names: Sequence[str]) -> torch.Tensor:
        """``x`` ``[B, C, A, 200]`` in µV / 100 (200 Hz, ``A`` one-second
        patches), ``ch_names`` the ``C`` channel names in order ->
        tokens ``[B, C, P, embed_dim]``."""
        raise NotImplementedError

    def token_grid(self, window_patches: int):
        """``(channels, patches)`` of the fixed token grid :meth:`encode_grid`
        returns for a window of ``window_patches`` one-second patches; only for
        an encoder with a fixed channel layout (``model.encoder.readout=flatten``)."""
        raise NotImplementedError(f"{self.name}: no fixed token grid; use encoder.readout=mean or attn")

    def encode_grid(self, x: torch.Tensor, ch_names: Sequence[str]) -> torch.Tensor:
        """Tokens of the whole fixed layout, in layout order, whatever channels
        and order ``x`` has (a slot without a channel is a flat input)."""
        raise NotImplementedError(f"{self.name}: no fixed token grid; use encoder.readout=mean or attn")

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """The encoder's own pretrained pooling of ``tokens`` ``[B, C, P, D]``
        -> ``[B, D]`` (``model.encoder.readout=attn``); not every encoder has one."""
        raise NotImplementedError(f"{self.name}: no pretrained pooling; use encoder.readout=mean")

    def pool_parameters(self) -> List[nn.Parameter]:
        """Parameters used only by :meth:`pool` (frozen when the readout is not ``attn``)."""
        return []

    def layer_id(self, name: str) -> int:
        """Layer index of parameter ``name`` (relative to this module) for the
        layer-wise LR decay: 0 = input stem, ``k`` = transformer layer ``k``
        (1-based), ``num_layers + 1`` = everything after the last layer."""
        raise NotImplementedError

    def no_weight_decay(self) -> Set[str]:
        """Parameter names (relative to this module) excluded from weight decay."""
        return set()

    def load_pretrained(self, source) -> None:
        """Load the released upstream weights (a path or an already read state
        dict). Raises on any missing or unexpected key."""
        raise NotImplementedError


def read_weight_file(path: str) -> Dict[str, torch.Tensor]:
    """Tensors of a released weight file: ``.safetensors``, or a torch pickle
    read with ``weights_only=True`` (never unpickles code from a download)."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(str(path))
    return torch.load(path, map_location="cpu", weights_only=True)


def load_state_strict(module: nn.Module, state: Dict[str, torch.Tensor], what: str,
                      drop_prefixes: Iterable[str] = ()) -> None:
    """``module.load_state_dict(state)`` that names every mismatch. Keys under
    ``drop_prefixes`` are parts of the release the module does not have."""
    drop = tuple(drop_prefixes)
    state = {k: v for k, v in state.items() if not (drop and k.startswith(drop))}
    own = module.state_dict()
    missing: List[str] = sorted(set(own) - set(state))
    unexpected: List[str] = sorted(set(state) - set(own))
    shape = sorted(k for k in set(own) & set(state) if own[k].shape != state[k].shape)
    if missing or unexpected or shape:
        raise ValueError(f"{what}: the weight file does not match the model "
                         f"(missing {missing[:5]}, unexpected {unexpected[:5]}, shape {shape[:5]}; "
                         f"{len(missing)}/{len(unexpected)}/{len(shape)} keys)")
    module.load_state_dict(state)
