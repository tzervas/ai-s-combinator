"""Tests for provenance tracking."""

import pytest

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


class TestProvenanceTracker:
    def test_tracker_starts_enabled(self):
        tracker = ProvenanceTracker()
        assert tracker.enabled is True

    @pytest.mark.skip(reason="Not yet implemented")
    def test_track_records_event(self):
        tracker = ProvenanceTracker()
        tracker.track(op=None, inputs=None, output=None)
        assert len(tracker.get_graph().nodes) > 0
