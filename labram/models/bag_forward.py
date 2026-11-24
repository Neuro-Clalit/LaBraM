# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Recording-level (MIL) forward shared by every fine-tune model.
# ---------------------------------------------------------
"""``forward_bag`` for any model that has ``forward_features``, ``mil`` and ``head``.

The LaBraM :class:`~labram.models.neural_transformer.NeuralTransformer` and the
foreign-encoder :class:`~labram.models.fm_regressor.FoundationRegressor` both
use it, so bag training, the frozen-encoder phase and bag mixup behave the same
whatever encoder produces the window features.
"""
import contextlib

import torch


class BagForwardMixin:
    # Windows encoded per pass when no gradient is needed (evaluation, frozen
    # encoder): a whole long recording is one bag at evaluation, and encoding
    # its thousands of windows at once needs tens of GB. Windows are encoded
    # independently, so chunking does not change the result.
    BAG_ENCODE_CHUNK = 256

    def _encode_windows(self, x, channel_indices):
        n = self.BAG_ENCODE_CHUNK
        if torch.is_grad_enabled() or x.shape[0] <= n:
            return self.forward_features(x, channel_indices=channel_indices)
        return torch.cat([self.forward_features(x[i:i + n], channel_indices=channel_indices)
                          for i in range(0, x.shape[0], n)])

    def forward_bag(self, x, channel_indices, bag):
        """Recording-level MIL forward (``mil.*``; needs ``self.mil``, a
        :class:`labram.layers.bag_pool.MILPooling`).

        ``x`` ``[N, C, A, T]`` holds the real windows of ``B`` bags, flattened in
        bag order (``bag['mask'][b, k]`` marks them). ``bag`` keys: ``mask``
        ``[B, K]``; ``q`` ``[B, K, Q]`` raw descriptors or None; ``t`` ``[B, K]``
        window minutes; ``present`` ``[N, C]`` + ``ci_all`` to encode each
        window on its present channels (the channel-drop path); ``no_grad_encoder``
        (frozen encoder); ``mix`` ``(perm, lam)`` mixes the pooled embeddings of
        bags (mixup). Returns ``(head output [B, ...], alpha [B, K], h [N, D])``.
        """
        mask = bag['mask']
        ctx = torch.no_grad() if bag.get('no_grad_encoder') else contextlib.nullcontext()
        with ctx:
            present = bag.get('present')
            if present is None:
                h = self._encode_windows(x, channel_indices)
            else:
                ci_all = bag['ci_all']
                patterns, inverse = torch.unique(present, dim=0, return_inverse=True)
                parts, rows = [], []
                for p in range(patterns.shape[0]):
                    idx = (inverse == p).nonzero().squeeze(1)
                    chans = patterns[p].nonzero().squeeze(1)
                    if chans.numel() == 0:
                        raise ValueError("a window has no channel with signal")
                    ci = [ci_all[0]] + [ci_all[int(c) + 1] for c in chans]
                    parts.append(self._encode_windows(x[idx][:, chans], ci))
                    rows.append(idx)
                order = torch.argsort(torch.cat(rows))
                h = torch.cat([f.to(parts[0].dtype) for f in parts])[order]
        b, k = mask.shape
        hb = h.new_zeros(b, k, h.shape[-1])
        hb = hb.index_put((mask.nonzero(as_tuple=True)), h)
        z, alpha, _ = self.mil(hb, mask, bag.get('q'), bag.get('t'))
        mix = bag.get('mix')
        if mix is not None:
            perm, lam = mix
            z = lam * z + (1 - lam) * z[perm]
        return self.head(z.to(self.head_dtype())), alpha, h

    def head_dtype(self):
        return next(self.head.parameters()).dtype
