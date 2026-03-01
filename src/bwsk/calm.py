"""CALM (Consistency As Logical Monotonicity) analysis for BWSK models.

The CALM theorem (Hellerstein, 2010) establishes that monotone computations
can execute coordination-free in distributed settings. In the BWSK framework,
S-type operations are monotone — they preserve information and can be
distributed without synchronization barriers. K-type operations require
coordination because they erase information (irreversible state changes).

This module provides:
- `CALMAnalysis`: Analyzes a model's S/K decomposition to identify
  coordination-free (parallelizable) segments.
- `CALMReport`: Report on which segments can be distributed.
- `partition_for_distribution`: Partition a model into coordination-free
  segments separated by synchronization points (K-boundaries).

Why this matters: A model that is 75% S-type can theoretically distribute
75% of its computation without any inter-device synchronization. The K-type
operations (softmax, pooling, ReLU) are the only points requiring all-reduce
or barrier operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from bwsk.classify import OpClass, classify_operation

if TYPE_CHECKING:
    import torch.nn as nn


@dataclass
class CALMSegment:
    """A segment of the model's computation graph.

    Either coordination-free (all S-type, can be distributed) or
    a synchronization point (contains K-type, requires coordination).
    """

    modules: list[nn.Module]
    module_names: list[str]
    is_monotone: bool  # True = coordination-free (S-type)
    classifications: list[OpClass] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Number of modules in this segment."""
        return len(self.modules)


@dataclass
class CALMReport:
    """CALM analysis report for a model.

    Summarizes which segments can be distributed coordination-free
    and which require synchronization.
    """

    model_name: str
    total_modules: int
    monotone_count: int  # Modules in coordination-free segments
    sync_count: int  # Modules requiring synchronization
    segments: list[CALMSegment] = field(default_factory=list)

    @property
    def parallelism_ratio(self) -> float:
        """Fraction of computation that can be distributed.

        A higher ratio means more of the model can be parallelized
        without inter-device synchronization.
        """
        if self.total_modules == 0:
            return 0.0
        return self.monotone_count / self.total_modules

    @property
    def num_sync_barriers(self) -> int:
        """Number of synchronization barriers needed.

        Each non-monotone segment requires one synchronization barrier.
        Fewer barriers = better distributed performance.
        """
        return sum(1 for s in self.segments if not s.is_monotone)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "model_name": self.model_name,
            "total_modules": self.total_modules,
            "monotone_count": self.monotone_count,
            "sync_count": self.sync_count,
            "parallelism_ratio": self.parallelism_ratio,
            "num_sync_barriers": self.num_sync_barriers,
            "segments": [
                {
                    "module_names": s.module_names,
                    "is_monotone": s.is_monotone,
                    "size": s.size,
                    "classifications": [c.value for c in s.classifications],
                }
                for s in self.segments
            ],
        }


def analyze_calm(model: nn.Module) -> CALMReport:
    """Analyze a model for CALM-theorem parallelism opportunities.

    Partitions the model's leaf modules into monotone (S-type,
    coordination-free) and non-monotone (K-type, synchronization
    required) segments. Consecutive S-type modules form a single
    coordination-free segment.

    Args:
        model: The PyTorch model to analyze.

    Returns:
        A CALMReport with segment decomposition and parallelism metrics.
    """
    segments: list[CALMSegment] = []
    current_s_modules: list[nn.Module] = []
    current_s_names: list[str] = []
    current_s_classes: list[OpClass] = []

    named_children = list(model.named_children())
    if not named_children:
        # Leaf module
        named_children = [("self", model)]

    for name, child in named_children:
        result = classify_operation(child)

        if result.classification == OpClass.S:
            current_s_modules.append(child)
            current_s_names.append(name)
            current_s_classes.append(result.classification)
        else:
            # Flush S-segment
            if current_s_modules:
                segments.append(
                    CALMSegment(
                        modules=list(current_s_modules),
                        module_names=list(current_s_names),
                        is_monotone=True,
                        classifications=list(current_s_classes),
                    )
                )
                current_s_modules = []
                current_s_names = []
                current_s_classes = []

            # K/GRAY segment (synchronization point)
            segments.append(
                CALMSegment(
                    modules=[child],
                    module_names=[name],
                    is_monotone=False,
                    classifications=[result.classification],
                )
            )

    # Flush trailing S-segment
    if current_s_modules:
        segments.append(
            CALMSegment(
                modules=list(current_s_modules),
                module_names=list(current_s_names),
                is_monotone=True,
                classifications=list(current_s_classes),
            )
        )

    monotone_count = sum(s.size for s in segments if s.is_monotone)
    sync_count = sum(s.size for s in segments if not s.is_monotone)
    total = monotone_count + sync_count

    return CALMReport(
        model_name=type(model).__name__,
        total_modules=total,
        monotone_count=monotone_count,
        sync_count=sync_count,
        segments=segments,
    )


def partition_for_distribution(
    model: nn.Module,
    num_devices: int = 2,
) -> list[list[str]]:
    """Partition model into device assignments minimizing sync barriers.

    Assigns consecutive coordination-free segments to the same device
    when possible. K-type sync points are placed at device boundaries
    since they require all-reduce anyway.

    This is a greedy partitioning heuristic — it tries to balance
    computation across devices while minimizing the number of
    cross-device synchronization points.

    Args:
        model: The model to partition.
        num_devices: Number of target devices.

    Returns:
        List of lists of module names, one per device.
    """
    report = analyze_calm(model)

    if num_devices <= 1 or not report.segments:
        all_names = []
        for s in report.segments:
            all_names.extend(s.module_names)
        return [all_names]

    # Greedy: distribute segments round-robin, preferring to keep
    # monotone segments together and placing sync points at boundaries
    devices: list[list[str]] = [[] for _ in range(num_devices)]
    current_device = 0

    for segment in report.segments:
        devices[current_device].extend(segment.module_names)

        # Move to next device after a sync point (K-boundary)
        if not segment.is_monotone:
            current_device = (current_device + 1) % num_devices

    return devices
