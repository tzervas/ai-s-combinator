"""Tests for BWSK primitive combinators (pure and nn.Module versions)."""

import torch
import torch.nn as nn

from bwsk.classify import OpClass
from bwsk.primitives import (
    B,
    BModule,
    K,
    KModule,
    S,
    SModule,
    W,
    WModule,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inc(x):
    return x + 1


def _double(x):
    return x * 2


def _add(x):
    def _inner(y):
        return x + y

    return _inner


def _identity(x):
    return x


# ---------------------------------------------------------------------------
# Pure combinator tests (Phase 1)
# ---------------------------------------------------------------------------


class TestB:
    def test_composition(self):
        """B f g x = f(g(x))."""
        b = B(_inc, _double)
        assert b(3) == 7  # f(g(3)) = f(6) = 7


class TestW:
    def test_self_application(self):
        """W f x = f(x)(x)."""
        w = W(_add)
        assert w(3) == 6  # f(3)(3) = 3 + 3 = 6


class TestS:
    def test_fan_out_combine(self):
        """S f g x = f(x)(g(x))."""
        s = S(_add, _double)
        assert s(3) == 9  # f(3)(g(3)) = (3 + 6) = 9


class TestK:
    def test_erasure(self):
        """K x y = x (y is discarded)."""
        k = K(_identity)
        assert k(42, "discarded") == 42


# ---------------------------------------------------------------------------
# nn.Module wrapper tests (Phase 2)
# ---------------------------------------------------------------------------


class TestBModule:
    def test_forward_composition(self):
        """BModule(f, g)(x) = f(g(x)) on tensors."""
        f = nn.Linear(10, 5)
        g = nn.Linear(20, 10)
        b = BModule(f, g)
        x = torch.randn(2, 20)
        out = b(x)
        expected = f(g(x))
        assert torch.allclose(out, expected, atol=1e-6)

    def test_classification_is_s(self):
        """Composition is S-type (no information loss)."""
        b = BModule(nn.Linear(10, 5), nn.Linear(20, 10))
        assert b.classification == OpClass.S

    def test_pipeline_operator(self):
        """b1 >> b2 creates BModule — pipeline order."""
        b1 = BModule(nn.Linear(20, 10), nn.Identity())
        b2 = BModule(nn.Linear(10, 5), nn.Identity())
        pipe = b1 >> b2
        assert isinstance(pipe, BModule)


class TestWModule:
    def test_forward_self_application(self):
        """WModule feeds the same input twice to f."""

        class AddInputs(nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        w = WModule(AddInputs())
        x = torch.randn(2, 10)
        out = w(x)
        assert torch.allclose(out, x + x, atol=1e-6)

    def test_classification_is_s(self):
        """Weight sharing is S-type (duplication, not erasure)."""
        w = WModule(nn.Bilinear(10, 10, 5))
        assert w.classification == OpClass.S


class TestSModule:
    def test_forward_fan_out_add(self):
        """SModule fans x to f and g, combines by addition."""
        f = nn.Linear(10, 10)
        g = nn.Linear(10, 10)
        s = SModule(f, g)
        x = torch.randn(2, 10)
        out = s(x)
        expected = f(x) + g(x)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_residual_connection(self):
        """S(identity, block) models a residual connection: x + block(x)."""
        block = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        s = SModule(nn.Identity(), block)
        x = torch.randn(2, 10)
        out = s(x)
        expected = x + block(x)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_custom_combine(self):
        """SModule with custom combiner (multiply instead of add)."""
        f = nn.Linear(10, 10)
        g = nn.Linear(10, 10)
        s = SModule(f, g, combine=torch.mul)
        x = torch.randn(2, 10)
        out = s(x)
        expected = torch.mul(f(x), g(x))
        assert torch.allclose(out, expected, atol=1e-6)

    def test_classification_is_s(self):
        """Fan-out is S-type."""
        s = SModule(nn.Linear(10, 10), nn.Linear(10, 10))
        assert s.classification == OpClass.S


class TestKModule:
    def test_forward_erasure(self):
        """KModule applies f to x, discards y."""
        f = nn.ReLU()
        k = KModule(f)
        x = torch.randn(2, 10)
        y = torch.randn(2, 10)  # should be discarded
        out = k(x, y)
        expected = f(x)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_forward_single_arg(self):
        """KModule works with single argument (y defaults to None)."""
        f = nn.ReLU()
        k = KModule(f)
        x = torch.randn(2, 10)
        out = k(x)
        expected = f(x)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_classification_is_k(self):
        """Erasure is K-type."""
        k = KModule(nn.ReLU())
        assert k.classification == OpClass.K


# ---------------------------------------------------------------------------
# Pipeline operator tests
# ---------------------------------------------------------------------------


class TestPipelineOperator:
    def test_rshift_creates_b_module(self):
        """a >> b creates BModule(b, a)."""
        a = BModule(nn.Identity(), nn.Linear(10, 10))
        b = BModule(nn.Linear(10, 5), nn.Identity())
        pipe = a >> b
        assert isinstance(pipe, BModule)

    def test_pipeline_output_matches_manual_composition(self):
        """Pipeline operator produces same output as manual composition."""
        layer1 = nn.Linear(10, 20)
        layer2 = nn.Linear(20, 5)
        b1 = BModule(layer1, nn.Identity())
        b2 = BModule(layer2, nn.Identity())

        pipe = b1 >> b2
        x = torch.randn(2, 10)
        pipe_out = pipe(x)
        manual_out = layer2(layer1(x))
        assert torch.allclose(pipe_out, manual_out, atol=1e-6)

    def test_triple_pipeline(self):
        """a >> b >> c chains three stages."""
        a = BModule(nn.Linear(10, 20), nn.Identity())
        b = BModule(nn.Linear(20, 15), nn.Identity())
        c = BModule(nn.Linear(15, 5), nn.Identity())
        pipe = a >> b >> c
        x = torch.randn(2, 10)
        out = pipe(x)
        assert out.shape == (2, 5)
