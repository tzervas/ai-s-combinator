"""BWSK primitive combinator implementations.

Two layers of abstraction:
1. **Pure combinators** (B, W, S, K): Plain Python callables implementing
   the core reduction rules. Used for symbolic reasoning and testing.
2. **nn.Module wrappers** (BModule, WModule, SModule, KModule): PyTorch
   modules that apply combinator logic to tensors. Used in actual neural
   network construction and training.

The nn.Module versions add:
- forward() for torch autograd integration
- classification property self-reporting S/K type
- Shape-aware tensor operations
- Support for >> operator (pipeline composition)

Why separate pure and Module versions? Pure combinators are useful for
testing combinator laws (e.g., B f (B g h) = B (B f g) h) without
torch overhead. Module versions are needed for gradient computation
and integration with PyTorch training loops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from bwsk.classify import OpClass

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Pure combinators (Phase 1 — plain Python callables)
# ---------------------------------------------------------------------------


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
    via f(x), and again as the argument to that function.

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
    applied to g's result.

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
    networks. The second argument y is explicitly discarded.

    Args:
        f: Function to apply to the kept argument.
    """

    def __init__(self, f: Callable[..., Any]) -> None:
        self.f = f

    def __call__(self, x: Any, y: Any = None) -> Any:
        """Apply f to x, discarding y: f(x)."""
        return self.f(x)


# ---------------------------------------------------------------------------
# nn.Module wrappers (Phase 2 — torch-integrated)
# ---------------------------------------------------------------------------


class _BWSKModuleBase(nn.Module):
    """Base class for BWSK nn.Module primitives.

    Provides the >> operator for pipeline composition and the
    classification property for self-reporting S/K type.
    """

    _classification: OpClass = OpClass.GRAY

    @property
    def classification(self) -> OpClass:
        """Self-reported S/K classification of this combinator."""
        return self._classification

    def __rshift__(self, other: nn.Module) -> BModule:
        """Pipeline composition: self >> other = B(other, self).

        Reads left-to-right: data flows through self first, then other.
        This matches Unix pipe semantics and is more intuitive than
        mathematical composition order.

        Args:
            other: Module to apply after self.

        Returns:
            A BModule composing other(self(x)).
        """
        return BModule(other, self)


class BModule(_BWSKModuleBase):
    """Composition combinator as nn.Module: B f g x = f(g(x)).

    Sequential layer composition. S-type because no information is lost
    when composing two functions — the output of g feeds directly into f.

    Args:
        f: Outer module applied second.
        g: Inner module applied first.
    """

    _classification = OpClass.S

    def __init__(self, f: nn.Module, g: nn.Module) -> None:
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply g then f: f(g(x))."""
        return self.f(self.g(x))


class WModule(_BWSKModuleBase):
    """Weight sharing combinator as nn.Module: W f x = f(x, x).

    For nn.Modules, W feeds the same input to both arguments of a
    two-input module. This models weight sharing and self-attention
    where the same representation serves as both query and key.

    S-type because the input is duplicated, not erased.

    Args:
        f: Module that accepts two inputs (e.g., via a custom forward).
    """

    _classification = OpClass.S

    def __init__(self, f: nn.Module) -> None:
        super().__init__()
        self.f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply f with x as both arguments: f(x, x)."""
        return self.f(x, x)


class SModule(_BWSKModuleBase):
    """Fan-out and combine combinator as nn.Module: S f g x = f(x) + g(x).

    For nn.Modules, S fans the input to two branches and combines
    their outputs by addition. This models residual connections
    (where f=identity, g=residual block) and multi-head patterns.

    The combination operation is addition by default, matching the
    residual connection pattern that dominates modern architectures.
    Custom combiners can be provided for other patterns.

    S-type because the fan-out preserves all information from x
    and addition is invertible given one operand.

    Args:
        f: First branch module.
        g: Second branch module.
        combine: How to combine f(x) and g(x). Defaults to addition.
    """

    _classification = OpClass.S

    def __init__(
        self,
        f: nn.Module,
        g: nn.Module,
        combine: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.f = f
        self.g = g
        self._combine = combine or torch.add

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fan out x to f and g, combine results: combine(f(x), g(x))."""
        return self._combine(self.f(x), self.g(x))


class KModule(_BWSKModuleBase):
    """Erasure combinator as nn.Module: K f x y = f(x).

    Applies f to x, explicitly discarding y. When used with a single
    argument, acts as a marker that f is an information-erasing operation
    (e.g., ReLU, pooling). This makes K-boundaries visible in the
    combinator expression.

    K-type because information is explicitly discarded.

    Args:
        f: Module to apply to the kept input.
    """

    _classification = OpClass.K

    def __init__(self, f: nn.Module) -> None:
        super().__init__()
        self.f = f

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Apply f to x, discarding y: f(x)."""
        return self.f(x)
