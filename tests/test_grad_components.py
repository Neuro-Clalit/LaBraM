"""Tests for per-loss-component gradient norms (Task 4):
labram.optim_factory.compute_component_grad_norms."""

import torch

from labram.optim_factory import compute_component_grad_norms


def test_component_grad_norms_match_manual():
    torch.manual_seed(0)
    lin = torch.nn.Linear(4, 1)
    x = torch.randn(8, 4)
    out = lin(x)
    comp_a = (out ** 2).mean()
    comp_b = out.abs().mean()

    norms = compute_component_grad_norms(
        {"a": comp_a, "b": comp_b}, lin.parameters())
    assert set(norms) == {"grad_norm_a", "grad_norm_b"}
    assert all(v >= 0 for v in norms.values())

    # Cross-check 'a' against an independent autograd.grad.
    lin.zero_grad()
    grads = torch.autograd.grad(comp_a, list(lin.parameters()), retain_graph=True)
    manual = sum(float(g.norm()) ** 2 for g in grads) ** 0.5
    assert norms["grad_norm_a"] == manual


def test_retains_graph_for_subsequent_backward():
    lin = torch.nn.Linear(3, 1)
    out = lin(torch.randn(5, 3))
    total = out.mean()
    # Computing component norms must not free the graph total still needs.
    compute_component_grad_norms({"x": out.pow(2).mean()}, lin.parameters())
    total.backward()  # would raise "backward through the graph a second time" if freed
    assert lin.weight.grad is not None


def test_allows_unused_parameters():
    a = torch.nn.Linear(3, 1)
    b = torch.nn.Linear(3, 1)  # not part of the loss graph
    out = a(torch.randn(4, 3))
    params = list(a.parameters()) + list(b.parameters())
    norms = compute_component_grad_norms({"only_a": out.mean()}, params)
    assert norms["grad_norm_only_a"] >= 0
