"""Tests for BWSK training loop."""

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
