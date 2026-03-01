"""Tests for BWSK architecture examples.

Verifies that BWSK DSL constructions produce valid, gradient-capable
modules with correct output shapes.
"""

import torch
import torch.nn as nn

from bwsk.classify import OpClass
from bwsk.examples import (
    make_bwsk_attention_head,
    make_bwsk_mlp,
    make_bwsk_residual_block,
)
from bwsk.primitives import BModule, KModule, SModule, WModule
from bwsk.provenance import ProvenanceTracker


class TestBWSKMLP:
    def test_output_shape(self):
        """BWSK MLP produces correct output shape."""
        mlp = make_bwsk_mlp(10, 20, 5)
        x = torch.randn(4, 10)
        out = mlp(x)
        assert out.shape == (4, 5)

    def test_matches_vanilla_with_shared_weights(self):
        """BWSK MLP with shared weights matches vanilla MLP output."""
        # Create shared linear layers
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 20)
        l3 = nn.Linear(20, 5)

        # Build vanilla
        vanilla = nn.Sequential(l1, nn.ReLU(), l2, nn.ReLU(), l3)

        # Build BWSK using the same layers
        bwsk = (
            BModule(l1, nn.Identity())
            >> KModule(nn.ReLU())
            >> BModule(l2, nn.Identity())
            >> KModule(nn.ReLU())
            >> BModule(l3, nn.Identity())
        )

        x = torch.randn(4, 10)
        bwsk_out = bwsk(x)
        vanilla_out = vanilla(x)
        assert torch.allclose(bwsk_out, vanilla_out, atol=1e-6)

    def test_gradient_flow(self):
        """BWSK MLP supports backpropagation."""
        mlp = make_bwsk_mlp(10, 20, 5)
        x = torch.randn(4, 10)
        out = mlp(x)
        loss = out.sum()
        loss.backward()

        # Verify gradients exist on all linear layers
        for m in mlp.modules():
            if isinstance(m, nn.Linear):
                assert m.weight.grad is not None


class TestBWSKResidualBlock:
    def test_output_shape(self):
        """Residual block preserves input shape."""
        block = make_bwsk_residual_block(10)
        x = torch.randn(4, 10)
        out = block(x)
        assert out.shape == (4, 10)

    def test_residual_adds_identity(self):
        """Output includes the identity (residual) path."""
        block = make_bwsk_residual_block(10)
        x = torch.randn(4, 10)
        out = block(x)
        # Output should differ from x (transformation path adds something)
        # but should be close when transform path is small (random init)
        # Just verify it's not identical (ReLU zeroes some values)
        assert not torch.allclose(out, x, atol=1e-8)

    def test_is_s_module(self):
        """Residual block is an SModule at the top level."""
        block = make_bwsk_residual_block(10)
        assert isinstance(block, SModule)
        assert block.classification == OpClass.S

    def test_gradient_flow(self):
        """Residual block supports backpropagation."""
        block = make_bwsk_residual_block(10)
        x = torch.randn(4, 10)
        out = block(x)
        loss = out.sum()
        loss.backward()

        for m in block.modules():
            if isinstance(m, nn.Linear):
                assert m.weight.grad is not None


class TestBWSKAttentionHead:
    def test_output_shape(self):
        """Attention head preserves sequence shape."""
        head = make_bwsk_attention_head(16)
        x = torch.randn(2, 8, 16)  # batch=2, seq=8, dim=16
        out = head(x)
        assert out.shape == (2, 8, 16)

    def test_is_s_module_with_w_inside(self):
        """Attention head is S(Identity, W(QKV))."""
        head = make_bwsk_attention_head(16)
        assert isinstance(head, SModule)
        # The g branch should be a WModule
        assert isinstance(head.g, WModule)

    def test_gradient_flow(self):
        """Attention head supports backpropagation."""
        head = make_bwsk_attention_head(16)
        x = torch.randn(2, 8, 16)
        out = head(x)
        loss = out.sum()
        loss.backward()

        for m in head.modules():
            if isinstance(m, nn.Linear):
                assert m.weight.grad is not None

    def test_self_attention_pattern(self):
        """Same input serves as both query and key/value (W-combinator)."""
        head = make_bwsk_attention_head(16)
        x = torch.randn(2, 8, 16)
        # Should work — the W-combinator passes x as both arguments
        out = head(x)
        assert out.shape == x.shape


class TestProvenanceWithDSL:
    """Test that provenance tracking works with DSL-constructed models."""

    def test_mlp_provenance(self):
        """Provenance tracker captures MLP forward pass."""
        mlp = make_bwsk_mlp(10, 20, 5)
        tracker = ProvenanceTracker()
        tracker.attach(mlp)

        x = torch.randn(2, 10)
        mlp(x)

        graph = tracker.finalize()
        assert len(graph.nodes) > 0
        assert len(graph.k_boundaries) > 0  # ReLU layers
        assert graph.erasure_budget > 0

    def test_residual_provenance(self):
        """Provenance tracker captures residual block forward pass."""
        block = make_bwsk_residual_block(10)
        tracker = ProvenanceTracker()
        tracker.attach(block)

        x = torch.randn(2, 10)
        block(x)

        graph = tracker.finalize()
        assert len(graph.nodes) > 0
        # Should have S-type nodes (Linear, LayerNorm, Identity)
        s_nodes = [n for n in graph.nodes.values() if n.classification == "S"]
        assert len(s_nodes) > 0
