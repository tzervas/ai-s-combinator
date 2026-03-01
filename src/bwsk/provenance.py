"""Provenance tracking through S-phases.

Tracks information flow through neural network operations, preserving
complete provenance for S-type operations and annotating K-type boundaries
with what information was discarded.

Why provenance matters: In a BWSK-described network, S-phases preserve
all information, meaning any output can be traced back to its inputs
through S-type ops. K-boundaries are the only points where information
is lost. By tracking provenance, we can answer "which input features
contributed to this output?" — and the answer is precise because we
know exactly where information was erased.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProvenanceNode:
    """A node in the provenance graph.

    Each node represents one operation in the forward pass, recording
    its type, classification, and connections to other nodes.
    """

    id: str
    op_type: str  # "B", "W", "S", "K", or "tensor_op"
    classification: str  # "S", "K", or "GRAY"
    input_ids: list[str] = field(default_factory=list)
    output_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    erasure_description: str | None = None
    erasure_fraction: float | None = None


@dataclass
class ProvenanceGraph:
    """Complete provenance graph for a computation."""

    nodes: dict[str, ProvenanceNode] = field(default_factory=dict)
    s_phases: list[list[str]] = field(default_factory=list)
    k_boundaries: list[str] = field(default_factory=list)
    erasure_budget: float = 0.0


class ProvenanceTracker:
    """Tracks provenance through forward passes.

    During inference, records which input features contribute to each output
    through S-phases, and annotates K-boundaries with discarded information.

    Nodes are assigned sequential IDs ("node_0", "node_1", ...) in the order
    they are tracked. Each call to track() creates one ProvenanceNode and
    adds it to the graph.
    """

    def __init__(self) -> None:
        self.graph = ProvenanceGraph()
        self.enabled = True
        self._next_id = 0

    def track(self, op: Any, inputs: Any, output: Any) -> ProvenanceNode:
        """Record a provenance event for an operation.

        Creates a ProvenanceNode with a sequential ID and adds it to the
        graph. The op_type is derived from the operation's class name, and
        classification defaults to "GRAY" (can be refined later by the
        classifier).

        Args:
            op: The operation that was applied (nn.Module, function, or None).
            inputs: The inputs to the operation.
            output: The output of the operation.

        Returns:
            The created ProvenanceNode.
        """
        node_id = f"node_{self._next_id}"
        self._next_id += 1

        # Derive op_type from the operation
        if op is None:
            op_type = "unknown"
        elif hasattr(op, "__class__"):
            op_type = type(op).__name__
        else:
            op_type = str(op)

        node = ProvenanceNode(
            id=node_id,
            op_type=op_type,
            classification="GRAY",
        )
        self.graph.nodes[node_id] = node
        return node

    def get_graph(self) -> ProvenanceGraph:
        """Return the current provenance graph."""
        return self.graph
