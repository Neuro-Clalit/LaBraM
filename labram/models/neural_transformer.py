# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# ---------------------------------------------------------

import math

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from labram.models.att_blocks import Block
from labram.models.layers import PatchEmbed, TemporalConv


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class NeuralTransformerBase(nn.Module):
    """Shared scaffolding for NeuralTransformer (fine-tune) and
    NeuralTransformerForMaskedEEGModeling (pre-train).

    Owns: patch_embed, cls_token, pos_embed, time_embed, pos_drop, blocks, norm.
    Subclasses add task-specific heads (and mask_token / lm_head) and define
    their own forward. The submodule attribute paths are intentionally preserved
    so saved checkpoints (e.g. labram-base.pth) remain loadable across both
    subclasses: stripping the ``student.`` prefix in
    ``runners.finetune_setup.load_finetune_checkpoint`` is enough to map a
    pre-train state_dict onto the fine-tune model.
    """

    def __init__(self,
                 eeg_window_size=1600, patch_size=200, in_chans=1, out_chans=8,
                 embed_dim=200, depth=12, num_heads=10, mlp_ratio=4.,
                 qkv_bias=False, qk_norm=None, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False,
                 init_std=0.02, use_norm=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.init_std = init_std

        # patch embedding: TemporalConv on raw EEG (in_chans=1), else PatchEmbed
        # (decoder path with patch_size=1 used inside VQNSP).
        if in_chans == 1:
            self.patch_embed = TemporalConv(out_chans=out_chans)
        else:
            self.patch_embed = PatchEmbed(
                eeg_window_size=eeg_window_size, patch_size=patch_size,
                in_chans=in_chans, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim))
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.use_rel_pos_bias = use_rel_pos_bias
        window_size = (self.patch_embed.patch_shape
                       if use_rel_pos_bias and hasattr(self.patch_embed, 'patch_shape')
                       else None)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, init_values=init_values,
                window_size=window_size, attn_head_dim=attn_head_dim,
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim) if use_norm else nn.Identity()

        # Init shared params here; subclasses are responsible for initializing
        # their own task-specific params and then calling
        # ``self.apply(self._init_weights)`` and ``self.fix_init_weight()``.
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=init_std)
        trunc_normal_(self.time_embed, std=init_std)
        trunc_normal_(self.cls_token, std=init_std)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def _embed_inputs(self, x, channel_indices=None):
        """patch_embed + cls concat + pos/time embeddings + pos_drop.

        Input  x shape: (B, N, A, T) -- B=batch, N=electrodes, A=patches per
                        electrode, T=patch_size. When T != self.patch_size
                        (decoder path with patch_size=1), the time window is
                        derived from T instead of A.
        Output x shape: (B, N*A + 1, embed_dim) -- CLS token at index 0.
        """
        batch_size, n, a, t = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return self._add_pos_time_embeddings(x, batch_size, n, a, t, channel_indices)

    def _add_pos_time_embeddings(self, x, batch_size, n, a, t, channel_indices=None):
        """Add pos+time embeddings to ``x`` (already prefixed by cls) and apply pos_drop.

        Split out from ``_embed_inputs`` so the masked-modeling forward path
        can do its mask-token replacement between ``patch_embed`` and this step.

        When channel_indices is None the first (n+1) entries of self.pos_embed
        are used (cls slot + first n channels in standard_1020 order). This
        replaces the previous masked-EEG behavior that hard-broadcast the full
        128-row pos_embed table, which silently produced a shape mismatch
        unless the caller supplied 128 electrodes.
        """
        input_time_window = a if t == self.patch_size else t

        if self.pos_embed is not None:
            if channel_indices is not None:
                pos_embed_used = self.pos_embed[:, channel_indices]
            else:
                pos_embed_used = self.pos_embed[:, : n + 1]
            pos_embed = (
                pos_embed_used[:, 1:, :]
                .unsqueeze(2)
                .expand(batch_size, -1, input_time_window, -1)
                .flatten(1, 2)
            )
            pos_embed = torch.cat(
                (pos_embed_used[:, 0:1, :].expand(batch_size, -1, -1), pos_embed),
                dim=1,
            )
            x = x + pos_embed

        if self.time_embed is not None:
            nc = n if t == self.patch_size else a
            time_embed = (
                self.time_embed[:, 0:input_time_window, :]
                .unsqueeze(1)
                .expand(batch_size, nc, -1, -1)
                .flatten(1, 2)
            )
            x[:, 1:, :] += time_embed

        return self.pos_drop(x)


class NeuralTransformer(NeuralTransformerBase):
    def __init__(self, eeg_window_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200, depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001):
        super().__init__(
            eeg_window_size=eeg_window_size, patch_size=patch_size, in_chans=in_chans,
            out_chans=out_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, init_values=init_values, attn_head_dim=None,
            use_abs_pos_emb=use_abs_pos_emb, use_rel_pos_bias=use_rel_pos_bias,
            init_std=0.02, use_norm=not use_mean_pooling,
        )
        self.num_classes = num_classes
        self.time_window = eeg_window_size // patch_size

        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, channel_indices=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        x = self._embed_inputs(x, channel_indices=channel_indices)

        for blk in self.blocks:
            x = blk(x, rel_pos_bias=None)

        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                return self.fc_norm(t.mean(1))
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, x, channel_indices=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        '''
        x: [batch size, number of electrodes, number of patches, patch size]
        For example, for an EEG sample of 4 seconds with 64 electrodes, x will be [batch size, 64, 4, 200]
        '''
        x = self.forward_features(x, channel_indices=channel_indices, return_patch_tokens=return_patch_tokens, return_all_tokens=return_all_tokens, **kwargs)
        x = self.head(x)
        return x

    def forward_intermediate(self, x, channel_indices=None, layer_id=12, norm_output=False):
        x = self._embed_inputs(x, channel_indices=channel_indices)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                # use last norm for all intermediate layers
                if l in layer_id:
                    if norm_output:
                        output_list.append(self.fc_norm(self.norm(x[:, 1:])))
                    else:
                        output_list.append(x[:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f"Not support for layer id is {layer_id} now!")

    def get_intermediate_layers(self, x, channel_indices=None, use_last_norm=False):
        x = self._embed_inputs(x, channel_indices=channel_indices)
        features = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            if use_last_norm:
                features.append(self.norm(x))
            else:
                features.append(x)

        return features
