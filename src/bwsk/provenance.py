"""Provenance tracking through S-phases.

Tracks information flow through neural network operations, preserving
complete provenance for S-type operations and annotating K-type boundaries
with what information was discarded.

Two tracking modes:
1. **Manual tracking**: Call tracker.track(op, inputs, output) explicitly.
   Used in tests and custom training loops.
2. **Forward hook tracking**: Call tracker.attach(model) to register PyTorch
   forward hooks on all modules. The hooks automatically record provenance
   during the forward pass.

Why provenance matters: In a BWSK-described network, S-phases preserve
all information, meaning any output can be traced back to its inputs
through S-type ops. K-boundaries are the only points where information
is lost. By tracking provenance, we can answer "which input features
contributed to this output?" — and the answer is precise because we
know exactly where information was erased.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from bwsk.classify import classify_operation


@dataclass
class ProvenanceNode:
    """A node in the provenance graph.

    Each node represents one operation in the forward pass, recording
    its type, classification, and connections to other nodes.
    """

    id: str
    op_type: str  # "B", "W", "S", "K", or module class name
    classification: str  # "S", "K", or "GRAY"
    input_ids: list[str] = field(default_factory=list)
    output_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    erasure_description: str | None = None
    erasure_fraction: float | None = None


@dataclass
class ProvenanceGraph:
    """Complete provenance graph for a computation.

    Nodes represent operations, s_phases groups consecutive S-type ops
    into reversible segments, and k_boundaries lists the IDs of
    K-type nodes that mark information loss points.
    """

    nodes: dict[str, ProvenanceNode] = field(default_factory=dict)
    s_phases: list[list[str]] = field(default_factory=list)
    k_boundaries: list[str] = field(default_factory=list)
    erasure_budget: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to a plain dict."""
        return {
            "nodes": {
                nid: {
                    "id": n.id,
                    "op_type": n.op_type,
                    "classification": n.classification,
                    "input_ids": n.input_ids,
                    "output_ids": n.output_ids,
                    "metadata": n.metadata,
                    "erasure_description": n.erasure_description,
                    "erasure_fraction": n.erasure_fraction,
                }
                for nid, n in self.nodes.items()
            },
            "s_phases": self.s_phases,
            "k_boundaries": self.k_boundaries,
            "erasure_budget": self.erasure_budget,
        }

    def to_json(self) -> str:
        """Serialize graph to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_graphviz(self) -> str:
        """Render graph as a Graphviz DOT string.

        S-type nodes are green, K-type are red, GRAY are gray.
        This provides a quick visual of where information flows
        freely (green) vs. where it's erased (red).
        """
        colors = {"S": "green", "K": "red", "GRAY": "gray"}
        lines = ["digraph provenance {", "  rankdir=TB;"]
        for nid, node in self.nodes.items():
            color = colors.get(node.classification, "gray")
            label = f"{node.op_type}\\n({node.classification})"
            lines.append(
                f'  "{nid}" [label="{label}", style=filled, fillcolor={color}, fontcolor=white];'
            )
            for out_id in node.output_ids:
                lines.append(f'  "{nid}" -> "{out_id}";')
        lines.append("}")
        return "\n".join(lines)


class ProvenanceTracker:
    """Tracks provenance through forward passes.

    During inference, records which input features contribute to each output
    through S-phases, and annotates K-boundaries with discarded information.

    Nodes are assigned sequential IDs ("node_0", "node_1", ...) in the order
    they are tracked. Each call to track() creates one ProvenanceNode and
    adds it to the graph.

    For automatic tracking, call attach(model) to register forward hooks
    on all leaf modules. Call detach() to remove hooks.
    """

    def __init__(self) -> None:
        self.graph = ProvenanceGraph()
        self.enabled = True
        self._next_id = 0
        self._hooks: list[Any] = []
        self._current_s_phase: list[str] = []

    def track(self, op: Any, inputs: Any, output: Any) -> ProvenanceNode:
        """Record a provenance event for an operation.

        Creates a ProvenanceNode with a sequential ID and adds it to the
        graph. The op_type is derived from the operation's class name, and
        classification is determined by the S/K classifier when the op is
        an nn.Module, otherwise defaults to "GRAY".

        Args:
            op: The operation that was applied (nn.Module, function, or None).
            inputs: The inputs to the operation.
            output: The output of the operation.

        Returns:
            The created ProvenanceNode.
        """
        if not self.enabled:
            return ProvenanceNode(id="disabled", op_type="", classification="")

        node_id = f"node_{self._next_id}"
        self._next_id += 1

        # Derive op_type and classification
        if op is None:
            op_type = "unknown"
            classification = "GRAY"
        elif isinstance(op, nn.Module):
            op_type = type(op).__name__
            result = classify_operation(op)
            classification = result.classification.value
        elif hasattr(op, "__class__"):
            op_type = type(op).__name__
            classification = "GRAY"
        else:
            op_type = str(op)
            classification = "GRAY"

        node = ProvenanceNode(
            id=node_id,
            op_type=op_type,
            classification=classification,
        )
        self.graph.nodes[node_id] = node

        # Track S-phases and K-boundaries
        if classification == "K":
            self.graph.k_boundaries.append(node_id)
            # Close current S-phase if any
            if self._current_s_phase:
                self.graph.s_phases.append(list(self._current_s_phase))
                self._current_s_phase = []
        elif classification == "S":
            self._current_s_phase.append(node_id)

        return node

    def attach(self, model: nn.Module) -> None:
        """Register forward hooks on all leaf modules of the model.

        After calling attach(), every forward pass through the model will
        automatically record provenance for each leaf module.

        Args:
            model: The model to instrument.
        """
        self.detach()  # Remove any existing hooks

        for name, module in model.named_modules():
            # Only hook leaf modules (no children) to avoid double-counting
            children = list(module.children())
            if len(children) > 0:
                continue

            def _make_hook(
                mod_name: str,
            ) -> Any:
                def hook(mod: nn.Module, inp: Any, out: Any) -> None:
                    node = self.track(mod, inp, out)
                    node.metadata["module_name"] = mod_name

                return hook

            handle = module.register_forward_hook(_make_hook(name))
            self._hooks.append(handle)

    def detach(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def finalize(self) -> ProvenanceGraph:
        """Finalize tracking and return the completed graph.

        Closes any open S-phase and computes the erasure budget.

        Returns:
            The finalized ProvenanceGraph.
        """
        # Close trailing S-phase
        if self._current_s_phase:
            self.graph.s_phases.append(list(self._current_s_phase))
            self._current_s_phase = []

        # Compute erasure budget
        total = len(self.graph.nodes)
        k_count = len(self.graph.k_boundaries)
        self.graph.erasure_budget = k_count / total if total > 0 else 0.0

        return self.graph

    def reset(self) -> None:
        """Clear the graph and reset ID counter."""
        self.graph = ProvenanceGraph()
        self._next_id = 0
        self._current_s_phase = []

    def get_graph(self) -> ProvenanceGraph:
        """Return the current provenance graph."""
        return self.graph
