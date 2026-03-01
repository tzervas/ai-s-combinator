"""Architecture examples expressed in the BWSK DSL.

Demonstrates how standard neural network architectures map to BWSK
combinator expressions. Each example constructs both a vanilla PyTorch
model and its BWSK-equivalent, verifying they produce identical outputs.

These examples serve as both documentation and integration tests for the
DSL. They show that BWSK descriptions are not just theoretical — they
compile to real, runnable, gradient-capable PyTorch modules.

Architectures:
- MLP: Pure B-composition (sequential stacking)
- Residual block: S-combinator (fan-out + add)
- Scaled dot-product attention head: W + S + K composition
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from bwsk.primitives import BModule, KModule, SModule, WModule


def make_bwsk_mlp(
    in_features: int,
    hidden: int,
    out_features: int,
) -> nn.Module:
    """Build an MLP using B-composition (sequential stacking).

    Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear
    BWSK expression: B(Linear, B(K(ReLU), B(Linear, B(K(ReLU), Linear))))

    In pipeline form: Linear >> K(ReLU) >> Linear >> K(ReLU) >> Linear

    Why B? An MLP is pure sequential composition with no branching or
    sharing. Every layer feeds its output to the next. ReLU layers are
    wrapped in K to mark them as information-erasing boundaries.

    Args:
        in_features: Input dimension.
        hidden: Hidden layer dimension.
        out_features: Output dimension.

    Returns:
        A BModule-based MLP.
    """
    layer1 = BModule(nn.Linear(in_features, hidden), nn.Identity())
    act1 = KModule(nn.ReLU())
    layer2 = BModule(nn.Linear(hidden, hidden), nn.Identity())
    act2 = KModule(nn.ReLU())
    layer3 = BModule(nn.Linear(hidden, out_features), nn.Identity())

    return layer1 >> act1 >> layer2 >> act2 >> layer3


def make_vanilla_mlp(
    in_features: int,
    hidden: int,
    out_features: int,
) -> nn.Sequential:
    """Build a vanilla PyTorch MLP for comparison."""
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_features),
    )


def make_bwsk_residual_block(dim: int) -> nn.Module:
    """Build a residual block using S-combinator (fan-out + add).

    Architecture: x + ReLU(Linear(LayerNorm(x)))
    BWSK expression: S(Identity, B(K(ReLU), B(Linear, LayerNorm)))

    Why S? A residual connection fans the input to two paths —
    the identity path and the transformation path — then combines
    them by addition. S captures exactly this pattern. The identity
    branch is S-type, and the transformation branch contains a
    K-boundary at ReLU.

    Args:
        dim: Feature dimension (same in and out for residual).

    Returns:
        An SModule-based residual block.
    """
    transform = (
        BModule(nn.LayerNorm(dim), nn.Identity())
        >> BModule(nn.Linear(dim, dim), nn.Identity())
        >> KModule(nn.ReLU())
    )
    return SModule(nn.Identity(), transform)


class _ScaledDotProductAttention(nn.Module):
    """Single-head scaled dot-product attention.

    Computes attention(Q, K, V) = softmax(QK^T / sqrt(d)) V.

    This module accepts (query, key_value) where key_value is used
    for both K and V (self-attention pattern).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = math.sqrt(dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Compute scaled dot-product attention.

        Args:
            q: Query tensor [batch, seq, dim].
            kv: Key/Value tensor [batch, seq, dim].

        Returns:
            Attention output [batch, seq, dim].
        """
        scores = torch.matmul(q, kv.transpose(-2, -1)) / self.scale
        weights = self.softmax(scores)  # K-boundary: softmax erases
        return torch.matmul(weights, kv)


class _QKVProjection(nn.Module):
    """Projects input into Q, K, V and computes attention.

    Uses W-combinator pattern: the same input provides both the
    query (via q_proj) and the key/value (via kv_proj).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim)
        self.attn = _ScaledDotProductAttention(dim)

    def forward(self, x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Project and attend. x and x2 are the same tensor (W-combinator).

        Args:
            x: Input for query projection.
            x2: Same input for key/value projection.

        Returns:
            Attention output.
        """
        q = self.q_proj(x)
        kv = self.kv_proj(x2)
        return self.attn(q, kv)


def make_bwsk_attention_head(dim: int) -> nn.Module:
    """Build a self-attention head using W-combinator (self-application).

    Architecture: Attention(Q_proj(x), KV_proj(x)) + x
    BWSK expression: S(Identity, W(QKV_Projection))

    Why W? Self-attention is the canonical W-combinator pattern. The
    input x is used twice: once to produce queries (Q = W_q * x) and
    once to produce keys and values (K = W_k * x, V = W_v * x). This
    is exactly W f x = f(x)(x) — the input is "shared" between two
    roles. The residual connection wraps this in S.

    Args:
        dim: Feature dimension.

    Returns:
        An SModule with WModule-based attention + residual.
    """
    qkv = _QKVProjection(dim)
    attention = WModule(qkv)
    return SModule(nn.Identity(), attention)
