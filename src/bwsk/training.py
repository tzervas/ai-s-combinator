"""Training loop with BWSK analysis and memory optimization.

Provides a training wrapper that:
- Classifies operations as S/K during initialization
- Reports erasure budgets with each training step
- Optionally uses reversible S-phase checkpointing for memory efficiency
- Reports CALM parallelism metrics

Why wrap training? The BWSK framework needs to know the erasure budget
of a model to exploit S-phase reversibility for memory-efficient backprop.
By classifying once at init and reporting per-step, we enable:
1. Memory savings via S-phase activation recomputation
2. Distributed training guidance via CALM analysis
3. Per-epoch erasure budget tracking for architecture insights
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch.nn as nn

from bwsk.calm import analyze_calm
from bwsk.classify import classify_model
from bwsk.reversible import ReversibleSequence, analyze_memory_profile


class BWSKTrainer:
    """Training loop wrapper with BWSK-aware optimizations.

    Analyzes the model's computation graph for S/K phase boundaries
    at initialization time. Each training step reports the erasure budget
    alongside standard metrics like loss.

    Args:
        model: The PyTorch model to train. If None, operates in stub mode
            (returns zero metrics — useful for testing the API contract).
        optimizer: The optimizer. If None, skips parameter updates.
        use_reversible: If True, wrap sequential models in
            ReversibleSequence for memory-efficient backprop.
        loss_fn: Loss function. Defaults to nn.MSELoss().
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        use_reversible: bool = False,
        loss_fn: nn.Module | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.loss_fn = loss_fn or nn.MSELoss()
        self._erasure_budget = 0.0
        self._memory_profile: dict[str, Any] = {}
        self._calm_report: dict[str, Any] = {}

        # Classify and optionally wrap model
        if model is not None and isinstance(model, nn.Module):
            try:
                report = classify_model(model)
                self._erasure_budget = report.erasure_score
            except Exception:
                self._erasure_budget = 0.0

            with contextlib.suppress(Exception):
                self._memory_profile = analyze_memory_profile(model)

            with contextlib.suppress(Exception):
                calm = analyze_calm(model)
                self._calm_report = calm.to_dict()

            if use_reversible:
                children = list(model.children())
                if children:
                    self.model: Any = ReversibleSequence(children)
                else:
                    self.model = model
            else:
                self.model = model
        else:
            self.model = model

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
            Dictionary with "loss", "erasure_budget", and optionally
            "memory_savings" keys.
        """
        if self.model is None:
            return {"loss": 0.0, "erasure_budget": self._erasure_budget}

        inputs, targets = batch
        self.model.train()

        # Forward pass
        outputs = self.model(inputs)

        # Compute loss
        loss = self.loss_fn(outputs, targets)

        # Backward pass
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        if self.optimizer is not None:
            self.optimizer.step()

        metrics: dict[str, float] = {
            "loss": loss.item(),
            "erasure_budget": self._erasure_budget,
        }

        if self._memory_profile:
            metrics["estimated_memory_savings"] = self._memory_profile.get(
                "estimated_memory_savings", 0.0
            )

        if self._calm_report:
            metrics["parallelism_ratio"] = self._calm_report.get("parallelism_ratio", 0.0)

        return metrics

    @property
    def analysis_summary(self) -> dict[str, Any]:
        """Return a summary of the BWSK analysis for this model.

        Includes erasure budget, memory profile, and CALM analysis.
        """
        return {
            "erasure_budget": self._erasure_budget,
            "memory_profile": self._memory_profile,
            "calm_analysis": self._calm_report,
        }
