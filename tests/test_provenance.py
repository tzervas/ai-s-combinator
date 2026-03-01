"""Tests for provenance tracking."""

import json

import torch
import torch.nn as nn

from bwsk.provenance import ProvenanceGraph, ProvenanceNode, ProvenanceTracker


class TestProvenanceNode:
    def test_create_node(self):
        node = ProvenanceNode(id="n1", op_type="S", classification="S")
        assert node.id == "n1"
        assert node.op_type == "S"
        assert node.classification == "S"
        assert node.input_ids == []
        assert node.output_ids == []

    def test_k_node_has_erasure_info(self):
        node = ProvenanceNode(
            id="n2",
            op_type="K",
            classification="K",
            erasure_description="ReLU zeroes negative values",
            erasure_fraction=0.5,
        )
        assert node.erasure_fraction == 0.5


class TestProvenanceGraph:
    def test_empty_graph(self):
        graph = ProvenanceGraph()
        assert len(graph.nodes) == 0
        assert graph.erasure_budget == 0.0

    def test_to_dict(self):
        graph = ProvenanceGraph()
        graph.nodes["n1"] = ProvenanceNode(id="n1", op_type="Linear", classification="S")
        d = graph.to_dict()
        assert "nodes" in d
        assert "n1" in d["nodes"]
        assert d["nodes"]["n1"]["classification"] == "S"

    def test_to_json(self):
        graph = ProvenanceGraph()
        graph.nodes["n1"] = ProvenanceNode(id="n1", op_type="ReLU", classification="K")
        j = graph.to_json()
        parsed = json.loads(j)
        assert parsed["nodes"]["n1"]["op_type"] == "ReLU"

    def test_to_graphviz(self):
        graph = ProvenanceGraph()
        graph.nodes["n1"] = ProvenanceNode(
            id="n1",
            op_type="Linear",
            classification="S",
            output_ids=["n2"],
        )
        graph.nodes["n2"] = ProvenanceNode(id="n2", op_type="ReLU", classification="K")
        dot = graph.to_graphviz()
        assert "digraph provenance" in dot
        assert "Linear" in dot
        assert "ReLU" in dot
        assert '"n1" -> "n2"' in dot
        assert "green" in dot  # S-type
        assert "red" in dot  # K-type


class TestProvenanceTracker:
    def test_tracker_starts_enabled(self):
        tracker = ProvenanceTracker()
        assert tracker.enabled is True

    def test_track_records_event(self):
        tracker = ProvenanceTracker()
        tracker.track(op=None, inputs=None, output=None)
        assert len(tracker.get_graph().nodes) > 0

    def test_track_sequential_ids(self):
        """Nodes get sequential IDs: node_0, node_1, ..."""
        tracker = ProvenanceTracker()
        n0 = tracker.track(op=None, inputs=None, output=None)
        n1 = tracker.track(op=None, inputs=None, output=None)
        assert n0.id == "node_0"
        assert n1.id == "node_1"

    def test_track_nn_module_classifies(self):
        """Tracking an nn.Module automatically classifies it."""
        tracker = ProvenanceTracker()
        node = tracker.track(op=nn.ReLU(), inputs=None, output=None)
        assert node.classification == "K"
        assert node.op_type == "ReLU"

    def test_track_s_type_module(self):
        tracker = ProvenanceTracker()
        node = tracker.track(op=nn.Linear(10, 5), inputs=None, output=None)
        assert node.classification == "S"

    def test_k_boundaries_tracked(self):
        """K-type nodes are added to k_boundaries list."""
        tracker = ProvenanceTracker()
        tracker.track(op=nn.Linear(10, 10), inputs=None, output=None)
        tracker.track(op=nn.ReLU(), inputs=None, output=None)
        tracker.track(op=nn.Linear(10, 5), inputs=None, output=None)

        graph = tracker.get_graph()
        assert len(graph.k_boundaries) == 1
        assert graph.k_boundaries[0] == "node_1"

    def test_s_phases_grouped(self):
        """Consecutive S-type ops are grouped into s_phases."""
        tracker = ProvenanceTracker()
        tracker.track(op=nn.Linear(10, 10), inputs=None, output=None)
        tracker.track(op=nn.LayerNorm(10), inputs=None, output=None)
        tracker.track(op=nn.ReLU(), inputs=None, output=None)  # K breaks S-phase
        tracker.track(op=nn.Linear(10, 5), inputs=None, output=None)
        tracker.finalize()

        graph = tracker.get_graph()
        # First S-phase: [node_0, node_1], closed by K at node_2
        # Second S-phase: [node_3], closed by finalize
        assert len(graph.s_phases) == 2
        assert graph.s_phases[0] == ["node_0", "node_1"]
        assert graph.s_phases[1] == ["node_3"]

    def test_finalize_computes_erasure_budget(self):
        tracker = ProvenanceTracker()
        tracker.track(op=nn.Linear(10, 10), inputs=None, output=None)
        tracker.track(op=nn.ReLU(), inputs=None, output=None)
        tracker.track(op=nn.Linear(10, 5), inputs=None, output=None)
        graph = tracker.finalize()
        # 1 K out of 3 total = 1/3
        assert abs(graph.erasure_budget - 1 / 3) < 1e-6

    def test_reset_clears_state(self):
        tracker = ProvenanceTracker()
        tracker.track(op=None, inputs=None, output=None)
        tracker.reset()
        assert len(tracker.get_graph().nodes) == 0

    def test_disabled_tracker_skips_recording(self):
        tracker = ProvenanceTracker()
        tracker.enabled = False
        node = tracker.track(op=None, inputs=None, output=None)
        assert node.id == "disabled"
        assert len(tracker.get_graph().nodes) == 0


class TestProvenanceTrackerHooks:
    """Test automatic forward hook-based tracking."""

    def test_attach_records_forward_pass(self):
        """Attaching hooks to a model records provenance during forward."""

        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 10)
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.relu(self.fc(x))

        model = SimpleModel()
        tracker = ProvenanceTracker()
        tracker.attach(model)

        x = torch.randn(2, 10)
        model(x)

        graph = tracker.get_graph()
        assert len(graph.nodes) == 2
        # Should have module names in metadata
        names = [n.metadata.get("module_name") for n in graph.nodes.values()]
        assert "fc" in names
        assert "relu" in names

    def test_detach_removes_hooks(self):
        """After detach, forward passes no longer record provenance."""

        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        model = SimpleModel()
        tracker = ProvenanceTracker()
        tracker.attach(model)
        model(torch.randn(2, 10))
        assert len(tracker.get_graph().nodes) == 1

        tracker.detach()
        tracker.reset()
        model(torch.randn(2, 10))
        assert len(tracker.get_graph().nodes) == 0

    def test_hook_classifies_operations(self):
        """Hooks should classify each module as S/K/GRAY."""

        class MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(10, 10)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(10, 5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc2(self.relu(self.fc1(x)))

        model = MLP()
        tracker = ProvenanceTracker()
        tracker.attach(model)

        model(torch.randn(2, 10))
        graph = tracker.finalize()

        classifications = [n.classification for n in graph.nodes.values()]
        assert "S" in classifications  # Linear
        assert "K" in classifications  # ReLU
        assert len(graph.k_boundaries) == 1
        assert graph.erasure_budget > 0
