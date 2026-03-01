"""Tests for BWSK training loop."""

import torch
import torch.nn as nn

from bwsk.training import BWSKTrainer


class TestBWSKTrainer:
    def test_train_step_returns_metrics(self):
        trainer = BWSKTrainer(model=None, optimizer=None)
        metrics = trainer.train_step(batch=None)
        assert "loss" in metrics

    def test_train_step_reports_erasure_budget(self):
        trainer = BWSKTrainer(model=None, optimizer=None)
        metrics = trainer.train_step(batch=None)
        assert "erasure_budget" in metrics


class TestBWSKTrainerWithModel:
    def _make_model_and_trainer(self, use_reversible: bool = False):
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = BWSKTrainer(model, optimizer, use_reversible=use_reversible)
        return trainer

    def test_real_training_step(self):
        """Train step with a real model produces valid metrics."""
        trainer = self._make_model_and_trainer()
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        metrics = trainer.train_step((x, y))

        assert metrics["loss"] > 0
        assert 0 <= metrics["erasure_budget"] <= 1.0

    def test_reversible_training_step(self):
        """Reversible mode produces same-quality training."""
        trainer = self._make_model_and_trainer(use_reversible=True)
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        metrics = trainer.train_step((x, y))

        assert metrics["loss"] > 0
        assert "erasure_budget" in metrics

    def test_analysis_summary(self):
        """Analysis summary contains all expected sections."""
        trainer = self._make_model_and_trainer()
        summary = trainer.analysis_summary

        assert "erasure_budget" in summary
        assert "memory_profile" in summary
        assert "calm_analysis" in summary
        assert summary["erasure_budget"] > 0

    def test_memory_savings_reported(self):
        """Memory savings estimate is included in metrics."""
        trainer = self._make_model_and_trainer()
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        metrics = trainer.train_step((x, y))
        assert "estimated_memory_savings" in metrics

    def test_parallelism_ratio_reported(self):
        """CALM parallelism ratio is included in metrics."""
        trainer = self._make_model_and_trainer()
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        metrics = trainer.train_step((x, y))
        assert "parallelism_ratio" in metrics

    def test_multiple_steps(self):
        """Multiple training steps reduce loss."""
        trainer = self._make_model_and_trainer()
        x = torch.randn(8, 10)
        y = torch.randn(8, 5)

        losses = []
        for _ in range(20):
            metrics = trainer.train_step((x, y))
            losses.append(metrics["loss"])

        # Loss should decrease over training
        assert losses[-1] < losses[0]

    def test_custom_loss_fn(self):
        """Custom loss function is used."""
        model = nn.Sequential(nn.Linear(10, 5))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = BWSKTrainer(model, optimizer, loss_fn=nn.L1Loss())

        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        metrics = trainer.train_step((x, y))
        assert metrics["loss"] > 0
