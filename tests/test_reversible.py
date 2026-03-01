"""Tests for reversible backprop through S-phases."""

import torch
import torch.nn as nn

from bwsk.reversible import (
    ReversibleSequence,
    analyze_memory_profile,
    checkpoint_k_boundaries,
)


class TestReversibleSequence:
    def test_output_matches_sequential(self):
        """ReversibleSequence produces same output as plain Sequential."""
        layers = [nn.Linear(10, 10), nn.LayerNorm(10), nn.ReLU(), nn.Linear(10, 5)]
        rev = ReversibleSequence(layers)
        seq = nn.Sequential(*layers)

        x = torch.randn(4, 10)
        rev_out = rev(x)
        seq_out = seq(x)
        assert torch.allclose(rev_out, seq_out, atol=1e-6)

    def test_gradient_flow(self):
        """Gradients flow correctly through reversible sequence."""
        layers = [nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5)]
        rev = ReversibleSequence(layers)

        x = torch.randn(4, 10, requires_grad=True)
        out = rev(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        for layer in layers:
            if isinstance(layer, nn.Linear):
                assert layer.weight.grad is not None

    def test_gradients_match_sequential(self):
        """Gradients match between reversible and standard sequential."""
        # Use shared layers
        l1 = nn.Linear(10, 10)
        l2 = nn.Linear(10, 5)
        relu = nn.ReLU()

        # Clone weights for fair comparison
        l1_clone = nn.Linear(10, 10)
        l2_clone = nn.Linear(10, 5)
        l1_clone.weight.data.copy_(l1.weight.data)
        l1_clone.bias.data.copy_(l1.bias.data)
        l2_clone.weight.data.copy_(l2.weight.data)
        l2_clone.bias.data.copy_(l2.bias.data)

        rev = ReversibleSequence([l1, relu, l2])
        seq = nn.Sequential(l1_clone, nn.ReLU(), l2_clone)

        x = torch.randn(4, 10)
        x_rev = x.clone().requires_grad_(True)
        x_seq = x.clone().requires_grad_(True)

        rev(x_rev).sum().backward()
        seq(x_seq).sum().backward()

        assert torch.allclose(l1.weight.grad, l1_clone.weight.grad, atol=1e-6)
        assert torch.allclose(l2.weight.grad, l2_clone.weight.grad, atol=1e-6)

    def test_segments_partition_correctly(self):
        """S-type layers are grouped, K-type layers are separate."""
        layers = [
            nn.Linear(10, 10),  # S
            nn.LayerNorm(10),  # S
            nn.ReLU(),  # K
            nn.Linear(10, 10),  # S
            nn.ReLU(),  # K
            nn.Linear(10, 5),  # S
        ]
        rev = ReversibleSequence(layers)

        # Segments: [S,S](checkpoint), [K], [S](checkpoint), [K], [S](checkpoint)
        checkpointed = sum(1 for s in rev._segments if s.use_checkpoint)
        non_checkpointed = sum(1 for s in rev._segments if not s.use_checkpoint)

        assert checkpointed == 3  # Three S-segments
        assert non_checkpointed == 2  # Two K-layers

    def test_memory_savings_estimate(self):
        """Memory savings estimate reflects S/K ratio."""
        layers = [
            nn.Linear(10, 10),  # S
            nn.LayerNorm(10),  # S
            nn.Linear(10, 10),  # S
            nn.ReLU(),  # K
        ]
        rev = ReversibleSequence(layers)
        # 3 S-type out of 4 total = 0.75
        assert abs(rev.memory_savings_estimate - 0.75) < 1e-6

    def test_all_s_type_model(self):
        """Model with all S-type ops: 100% savings estimate."""
        layers = [nn.Linear(10, 10), nn.LayerNorm(10), nn.Linear(10, 5)]
        rev = ReversibleSequence(layers)
        assert abs(rev.memory_savings_estimate - 1.0) < 1e-6

    def test_empty_model(self):
        """Empty model handles gracefully."""
        rev = ReversibleSequence([])
        x = torch.randn(4, 10)
        out = rev(x)
        assert torch.allclose(out, x)


class TestCheckpointKBoundaries:
    def test_converts_sequential(self):
        """checkpoint_k_boundaries wraps Sequential children."""
        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5))
        rev = checkpoint_k_boundaries(model)
        assert isinstance(rev, ReversibleSequence)

        x = torch.randn(4, 10)
        out = rev(x)
        assert out.shape == (4, 5)


class TestAnalyzeMemoryProfile:
    def test_profile_sequential(self):
        """Memory profile reports correct S/K counts."""
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.LayerNorm(10),
            nn.ReLU(),
            nn.Linear(10, 5),
        )
        profile = analyze_memory_profile(model)

        assert profile["total_layers"] == 4
        assert profile["s_count"] == 3  # Linear, LayerNorm, Linear
        assert profile["k_count"] == 1  # ReLU
        assert profile["gray_count"] == 0
        assert abs(profile["estimated_memory_savings"] - 0.75) < 1e-6

    def test_profile_all_k(self):
        """Model with all K-type ops: no savings."""
        model = nn.Sequential(nn.ReLU(), nn.Dropout(0.5))
        profile = analyze_memory_profile(model)
        assert profile["estimated_memory_savings"] == 0.0
