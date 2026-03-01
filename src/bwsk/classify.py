"""S/K operation classifier for neural network operations.

Classifies each operation in a neural network as:
- S-type: information-preserving, reversible, coordination-free
- K-type: information-erasing, synchronization point
- Gray: context-dependent, requires manual classification
"""

from enum import Enum
from typing import Any


class OpClass(Enum):
    """Classification of a neural network operation."""

    S = "S"  # Information-preserving
    K = "K"  # Information-erasing
    GRAY = "GRAY"  # Context-dependent


def classify_operation(op: Any) -> OpClass:
    """Classify a neural network operation as S-type, K-type, or Gray.

    Args:
        op: A neural network operation (e.g., torch.nn.Module, torch.fx.Node).

    Returns:
        The S/K classification of the operation.
    """
    raise NotImplementedError("S/K classifier not yet implemented")
