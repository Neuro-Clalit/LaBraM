"""Bag pooling layers (labram/layers/bag_pool.py)."""
import pytest
import torch

from labram.layers.bag_pool import (DescriptorNorm, POOL_KINDS, attention_entropy_gap,
                                    build_bag_pool, masked_softmax)


def _bag(b=3, k=6, d=16, q=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(b, k, d, generator=g)
    qd = torch.randn(b, k, q, generator=g)
    mask = torch.ones(b, k, dtype=torch.bool)
    mask[1, 4:] = False                       # a short bag
    t = torch.arange(k).float().repeat(b, 1) / 6
    return h, qd, mask, t


@pytest.mark.parametrize("kind", POOL_KINDS)
@pytest.mark.parametrize("score_input", ["h", "q", "hq"])
def test_shapes_mask_and_weights_sum_to_one(kind, score_input):
    if kind == "mean" and score_input != "h":
        pytest.skip("mean ignores the score input")
    h, q, mask, t = _bag()
    pool = build_bag_pool(kind, 16, 5, score_input=score_input).eval()
    z, alpha, _ = pool(h, mask, q, t)
    assert z.shape == (3, 16) and alpha.shape == (3, 6)
    assert torch.all(alpha[~mask] == 0)
    assert torch.allclose(alpha.sum(-1), torch.ones(3), atol=1e-5)


@pytest.mark.parametrize("kind", POOL_KINDS)
def test_permutation_invariance(kind):
    h, q, mask, t = _bag()
    mask[:] = True
    pool = build_bag_pool(kind, 16, 5, score_input="h").eval()
    perm = torch.randperm(6)
    z1, a1, _ = pool(h, mask, q, t)
    z2, a2, _ = pool(h[:, perm], mask[:, perm], q[:, perm], t[:, perm])
    assert torch.allclose(z1, z2, atol=1e-5) and torch.allclose(a1[:, perm], a2, atol=1e-5)


def test_mean_pool_equals_masked_mean_and_linear_head_is_weighted_sum():
    h, q, mask, t = _bag()
    z, alpha, _ = build_bag_pool("mean", 16)(h, mask)
    ref = torch.stack([h[i][mask[i]].mean(0) for i in range(3)])
    assert torch.allclose(z, ref, atol=1e-6)
    head = torch.nn.Linear(16, 1)
    pool = build_bag_pool("gated", 16).eval()
    z, alpha, _ = pool(h, mask)
    window_pred = head(h).squeeze(-1)
    assert torch.allclose(head(z).squeeze(-1), (alpha * window_pred).sum(-1), atol=1e-5)


def test_multi_branch_and_topk_masking_train_mode():
    h, q, mask, t = _bag()
    pool = build_bag_pool("gated", 16, 5, score_input="hq", branches=4, attn_dim=8,
                          topk_mask_prob=1.0)
    z, alpha, extra = pool.train()(h, mask, q, t)
    assert extra["alpha_branches"].shape == (3, 4, 6)
    # with every branch's top window hidden, no branch puts weight on its eval-mode top
    pool.eval()
    _, _, ev = pool(h, mask, q, t)
    top = ev["alpha_branches"].argmax(-1)
    assert torch.all(extra["alpha_branches"].gather(-1, top[..., None]) == 0)


def test_entropy_gap_zero_for_uniform():
    mask = torch.ones(2, 5, dtype=torch.bool)
    alpha = masked_softmax(torch.zeros(2, 5), mask)
    assert attention_entropy_gap(alpha, mask).abs() < 1e-6


def test_descriptor_norm():
    q = torch.cat([torch.rand(1000, 1) ** 8 * 1000, torch.randn(1000, 1)], 1)
    q[0, 1] = float("nan")
    norm = DescriptorNorm(2).fit(q)
    assert bool(norm.log_cols[0]) and not bool(norm.log_cols[1])
    out = norm(q)
    assert torch.isfinite(out).all() and out.abs().max() <= 6
    assert out.mean(0).abs().max() < 0.1
