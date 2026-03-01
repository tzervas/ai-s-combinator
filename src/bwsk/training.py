"""Training loop with BWSK analysis.

Provides a training wrapper that:
- Classifies operations as S/K during initialization
- Reports erasure budgets with each training step
- Wraps a standard forward/loss/backward/step loop

Why wrap training? The BWSK framework needs to know the erasure budget
of a model to exploit S-phase reversibility for memory-efficient backprop
(Phase 3). By classifying once at init and reporting per-step, we establish
the infrastructure for future optimizations without changing the training
API.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from bwsk.classify import classify_model


class BWSKTrainer:
    """Training loop wrapper with BWSK-aware optimizations.

    Analyzes the model's computation graph for S/K phase boundaries
    at initialization time. Each training step reports the erasure budget
    alongside standard metrics like loss.

    Args:
        model: The PyTorch model to train. If None, operates in stub mode
            (returns zero metrics — useful for testing the API contract).
        optimizer: The optimizer. If None, skips parameter updates.
    """

    def __init__(self, model: Any, optimizer: Any) -> None:
        self.model = model
        self.optimizer = optimizer
        self._erasure_budget = 0.0

        # Classify model once at init to avoid repeated tracing
        if model is not None and isinstance(model, nn.Module):
            try:
                report = classify_model(model)
                self._erasure_budget = report.erasure_score
            except Exception:
                # Classification may fail for models that can't be traced
                self._erasure_budget = 0.0

    def train_step(self, batch: Any) -> dict[str, float]:
        """Execute a single training step with BWSK analysis.

        Runs the standard forward/loss/backward/step loop and returns
        metrics including the model's erasure budget. When model is None
        (stub mode), returns zero-valued metrics.

        Args:
            batch: A training batch. Expected to be a tuple of
                (inputs, targets) for real training, or any value
                in stub mode.

        Returns:
            Dictionary with at minimum "loss" and "erasure_budget" keys.
        """
        if self.model is None:
            return {"loss": 0.0, "erasure_budget": self._erasure_budget}

        inputs, targets = batch
        self.model.train()

        # Forward pass
        outputs = self.model(inputs)

        # Compute loss (MSE as default)
        loss_fn = nn.MSELoss()
        loss = loss_fn(outputs, targets)

        # Backward pass
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        if self.optimizer is not None:
            self.optimizer.step()

        return {
            "loss": loss.item(),
            "erasure_budget": self._erasure_budget,
        }
