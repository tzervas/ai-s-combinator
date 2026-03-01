"""Tests for BWSK primitive combinators."""

import pytest

from bwsk.primitives import B, K, S, W


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


class TestB:
    @pytest.mark.skip(reason="Not yet implemented")
    def test_composition(self):
        """B f g x = f(g(x))."""
        b = B(_inc, _double)
        assert b(3) == 7  # f(g(3)) = f(6) = 7


class TestW:
    @pytest.mark.skip(reason="Not yet implemented")
    def test_self_application(self):
        """W f x = f(x)(x)."""
        w = W(_add)
        assert w(3) == 6  # f(3)(3) = 3 + 3 = 6


class TestS:
    @pytest.mark.skip(reason="Not yet implemented")
    def test_fan_out_combine(self):
        """S f g x = f(x)(g(x))."""
        s = S(_add, _double)
        assert s(3) == 9  # f(3)(g(3)) = (3 + 6) = 9


class TestK:
    @pytest.mark.skip(reason="Not yet implemented")
    def test_erasure(self):
        """K x y = x (y is discarded)."""
        k = K(_identity)
        assert k(42, "discarded") == 42
