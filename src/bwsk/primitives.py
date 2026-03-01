"""BWSK primitive combinator implementations.

Defines the four fundamental combinators as composable building blocks.
These are pure Python callables (not nn.Modules) implementing the core
combinator reduction rules. nn.Module wrappers come in Phase 2.

- B (compose): B f g x = f(g(x))
- W (share):   W f x = f(x)(x)
- S (fan-out): S f g x = f(x)(g(x))
- K (erase):   K f x y = f(x)   (y discarded)

Why these four? Together, {B, W, S, K} form a complete basis for
combinatory logic. B handles sequential composition, W enables
self-reference/sharing, S provides fan-out with recombination,
and K provides explicit erasure. Every neural network architecture
can be described as a composition of these primitives.
"""

from collections.abc import Callable
from typing import Any


class B:
    """Composition combinator: B f g x = f(g(x)).

    Maps to sequential layer stacking in neural networks. This is the
    most common combinator — any feed-forward pipeline is a chain of B.

    Args:
        f: Outer function to apply second.
        g: Inner function to apply first.
    """

    def __init__(self, f: Callable[..., Any], g: Callable[..., Any]) -> None:
        self.f = f
        self.g = g

    def __call__(self, x: Any) -> Any:
        """Apply g then f: f(g(x))."""
        return self.f(self.g(x))


class W:
    """Self-application / weight sharing combinator: W f x = f(x)(x).

    Maps to weight sharing and self-attention in neural networks. The key
    insight is that the input x is used twice: once to produce a function
    via f(x), and again as the argument to that function. This models
    self-attention where Q and K both derive from the same input.

    Args:
        f: A function that, given x, returns another function to apply to x.
    """

    def __init__(self, f: Callable[..., Any]) -> None:
        self.f = f

    def __call__(self, x: Any) -> Any:
        """Apply f(x) to x: f(x)(x)."""
        return self.f(x)(x)


class S:
    """Fan-out and combine combinator: S f g x = f(x)(g(x)).

    Maps to multi-head attention and residual connections in neural networks.
    The input fans out to both f and g, then f's result (a function) is
    applied to g's result. This captures the pattern of computing two
    derived representations and combining them.

    Args:
        f: Function whose output is applied to g(x). Must return a callable.
        g: Function applied to x to produce the argument for f(x).
    """

    def __init__(self, f: Callable[..., Any], g: Callable[..., Any]) -> None:
        self.f = f
        self.g = g

    def __call__(self, x: Any) -> Any:
        """Fan out x to f and g, then combine: f(x)(g(x))."""
        return self.f(x)(self.g(x))


class K:
    """Erasure / projection combinator: K f x y = f(x).

    Maps to masking, dropout, pooling, and activation clipping in neural
    networks. The second argument y is explicitly discarded — this makes
    information loss visible in the combinator expression. Every K-boundary
    in a BWSK expression is a point where provenance tracking must record
    what was erased.

    Args:
        f: Function to apply to the kept argument.
    """

    def __init__(self, f: Callable[..., Any]) -> None:
        self.f = f

    def __call__(self, x: Any, y: Any = None) -> Any:
        """Apply f to x, discarding y: f(x)."""
        return self.f(x)
