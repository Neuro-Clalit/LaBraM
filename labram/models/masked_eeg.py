# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# ---------------------------------------------------------

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from labram.models.neural_transformer import NeuralTransformerBase
from labram.configs.model_config import NeuralTransformerForMEMConfig


class NeuralTransformerForMaskedEEGModeling(NeuralTransformerBase):
    def __init__(self,
                 config: NeuralTransformerForMEMConfig,
                 *,
                 norm_layer=nn.LayerNorm, qk_norm=None, qk_scale=None,
                 attn_head_dim=None):
        from dataclasses import replace as _replace
        base_cfg = _replace(config, use_norm=True)
        super().__init__(base_cfg, norm_layer=norm_layer, qk_norm=qk_norm, qk_scale=qk_scale, attn_head_dim=attn_head_dim)
        self.num_heads = config.num_heads
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)

        trunc_normal_(self.mask_token, std=config.init_std, a=-config.init_std, b=config.init_std)
        trunc_normal_(self.lm_head.weight, std=config.init_std, a=-config.init_std, b=config.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def forward_features(self, x, channel_indices, bool_masked_pos):
        batch_size, n, a, t = x.shape
        x = self.patch_embed(x)
        seq_len = x.shape[1]

        mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        mask_weight = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - mask_weight) + mask_token * mask_weight

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # _add_pos_time_embeddings slices pos_embed to [:, :n+1] when
        # channel_indices is None, matching the base bugfix and avoiding
        # the old shape mismatch when fewer than 128 electrodes are used.
        x = self._add_pos_time_embeddings(x, batch_size, n, a, t, channel_indices)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        return self.norm(x)

    def forward(self, x, channel_indices=None, bool_masked_pos=None, return_all_tokens=False,
                return_patch_tokens=False, return_all_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)
        x = self.forward_features(x, channel_indices=channel_indices, bool_masked_pos=bool_masked_pos)
        if return_all_patch_tokens:
            return x
        x = x[:, 1:]
        if return_patch_tokens:
            return x
        if return_all_tokens:
            return self.lm_head(x)
        else:
            return self.lm_head(x[bool_masked_pos])

    def forward_return_qkv(self, x, bool_masked_pos=None, split_out_as_qkv=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        mask_weight = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - mask_weight) + mask_token * mask_weight

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            else:
                x, qkv = blk(x, rel_pos_bias=rel_pos_bias, return_qkv=True)

        if split_out_as_qkv:
            x = self.norm(x)
            x = self.lm_head(x)
            q, k, v = x.chunk(3, dim=-1)
            b, n, c = q.shape
            q = q.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            k = k.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            return x, q, k, v
        else:
            x = self.norm(x)
            x = x[:, 1:]
            x = self.lm_head(x[bool_masked_pos])
            q, k, v = qkv[0], qkv[1], qkv[2]

        return x, q, k, v

    def get_last_selfattention(self, x):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            else:
                return blk(x, rel_pos_bias=rel_pos_bias, return_attention=True)


class NeuralTransformerForMEM(nn.Module):
    """Wrapper used by the pre-training runner and registry.

    Owns the masked / unmasked lm_head pair and a projection_head.
    The inner ``NeuralTransformerForMaskedEEGModeling`` (``self.student``)
    acts as the feature backbone; its own lm_head is not used here.
    """

    def __init__(self,
                 config: NeuralTransformerForMEMConfig,
                 *,
                 norm_layer=None, qk_norm=None, qk_scale=None, attn_head_dim=None):
        super().__init__()
        self.patch_size = config.patch_size
        self.student = NeuralTransformerForMaskedEEGModeling(
            config,
            norm_layer=norm_layer or nn.LayerNorm,
            qk_norm=qk_norm,
            qk_scale=qk_scale,
            attn_head_dim=attn_head_dim,
        )
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)
        self.projection_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU()
        )
        trunc_normal_(self.lm_head.weight, std=config.init_std, a=-config.init_std, b=config.init_std)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'student.cls_token', 'student.pos_embed', 'student.time_embed'}

    def forward(self, x, channel_indices=None, bool_masked_pos=None):
        x_masked = self.student(x, channel_indices, bool_masked_pos, return_all_patch_tokens=True)
        x_masked_no_cls = x_masked[:, 1:]
        x_rec = self.lm_head(x_masked_no_cls[bool_masked_pos])

        x_masked_sym = self.student(x, channel_indices, ~bool_masked_pos, return_all_patch_tokens=True)
        x_masked_no_cls_sym = x_masked_sym[:, 1:]
        x_rec_sym = self.lm_head(x_masked_no_cls_sym[~bool_masked_pos])

        return x_rec, x_rec_sym
