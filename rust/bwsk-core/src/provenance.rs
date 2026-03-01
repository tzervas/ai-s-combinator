//! Provenance tracking for BWSK computation graphs.
//!
//! Tracks information flow through operations, recording S-phases
//! (reversible segments) and K-boundaries (information loss points).

use std::collections::HashMap;

use crate::classify::OpClass;

/// A node in the provenance graph.
#[derive(Debug, Clone)]
pub struct ProvenanceNode {
    /// Unique identifier.
    pub id: String,
    /// Operation type (e.g., "Linear", "ReLU").
    pub op_type: String,
    /// S/K/Gray classification.
    pub classification: OpClass,
    /// IDs of input nodes.
    pub input_ids: Vec<String>,
    /// IDs of output nodes.
    pub output_ids: Vec<String>,
    /// Optional description of what information was erased (K-type only).
    pub erasure_description: Option<String>,
}

/// Complete provenance graph for a computation.
#[derive(Debug, Clone, Default)]
pub struct ProvenanceGraph {
    /// All nodes in the graph.
    pub nodes: HashMap<String, ProvenanceNode>,
    /// Groups of consecutive S-type node IDs (reversible segments).
    pub s_phases: Vec<Vec<String>>,
    /// IDs of K-type nodes (information loss points).
    pub k_boundaries: Vec<String>,
    /// Fraction of K-type nodes.
    pub erasure_budget: f64,
}

/// Tracks provenance during computation.
pub struct ProvenanceTracker {
    graph: ProvenanceGraph,
    next_id: usize,
    current_s_phase: Vec<String>,
    enabled: bool,
}

impl ProvenanceTracker {
    /// Create a new enabled tracker.
    pub fn new() -> Self {
        Self {
            graph: ProvenanceGraph::default(),
            next_id: 0,
            current_s_phase: Vec::new(),
            enabled: true,
        }
    }

    /// Record a provenance event.
    pub fn track(&mut self, op_type: &str, classification: OpClass) -> String {
        if !self.enabled {
            return "disabled".to_string();
        }

        let node_id = format!("node_{}", self.next_id);
        self.next_id += 1;

        let node = ProvenanceNode {
            id: node_id.clone(),
            op_type: op_type.to_string(),
            classification,
            input_ids: Vec::new(),
            output_ids: Vec::new(),
            erasure_description: None,
        };

        self.graph.nodes.insert(node_id.clone(), node);

        match classification {
            OpClass::K => {
                self.graph.k_boundaries.push(node_id.clone());
                if !self.current_s_phase.is_empty() {
                    self.graph.s_phases.push(self.current_s_phase.clone());
                    self.current_s_phase.clear();
                }
            }
            OpClass::S => {
                self.current_s_phase.push(node_id.clone());
            }
            OpClass::Gray => {}
        }

        node_id
    }

    /// Finalize tracking: close open S-phase and compute erasure budget.
    pub fn finalize(&mut self) -> &ProvenanceGraph {
        if !self.current_s_phase.is_empty() {
            self.graph.s_phases.push(self.current_s_phase.clone());
            self.current_s_phase.clear();
        }

        let total = self.graph.nodes.len();
        let k_count = self.graph.k_boundaries.len();
        self.graph.erasure_budget = if total > 0 {
            k_count as f64 / total as f64
        } else {
            0.0
        };

        &self.graph
    }

    /// Get the current graph (without finalizing).
    pub fn graph(&self) -> &ProvenanceGraph {
        &self.graph
    }

    /// Reset the tracker.
    pub fn reset(&mut self) {
        self.graph = ProvenanceGraph::default();
        self.next_id = 0;
        self.current_s_phase.clear();
    }

    /// Set enabled/disabled state.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl Default for ProvenanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_track_creates_nodes() {
        let mut tracker = ProvenanceTracker::new();
        let id = tracker.track("Linear", OpClass::S);
        assert_eq!(id, "node_0");
        assert_eq!(tracker.graph().nodes.len(), 1);
    }

    #[test]
    fn test_sequential_ids() {
        let mut tracker = ProvenanceTracker::new();
        let id0 = tracker.track("Linear", OpClass::S);
        let id1 = tracker.track("ReLU", OpClass::K);
        assert_eq!(id0, "node_0");
        assert_eq!(id1, "node_1");
    }

    #[test]
    fn test_k_boundaries_tracked() {
        let mut tracker = ProvenanceTracker::new();
        tracker.track("Linear", OpClass::S);
        tracker.track("ReLU", OpClass::K);
        tracker.track("Linear", OpClass::S);

        assert_eq!(tracker.graph().k_boundaries.len(), 1);
        assert_eq!(tracker.graph().k_boundaries[0], "node_1");
    }

    #[test]
    fn test_s_phases_grouped() {
        let mut tracker = ProvenanceTracker::new();
        tracker.track("Linear", OpClass::S);
        tracker.track("LayerNorm", OpClass::S);
        tracker.track("ReLU", OpClass::K);
        tracker.track("Linear", OpClass::S);
        tracker.finalize();

        let graph = tracker.graph();
        assert_eq!(graph.s_phases.len(), 2);
        assert_eq!(graph.s_phases[0], vec!["node_0", "node_1"]);
        assert_eq!(graph.s_phases[1], vec!["node_3"]);
    }

    #[test]
    fn test_finalize_computes_erasure_budget() {
        let mut tracker = ProvenanceTracker::new();
        tracker.track("Linear", OpClass::S);
        tracker.track("ReLU", OpClass::K);
        tracker.track("Linear", OpClass::S);
        tracker.finalize();

        assert!((tracker.graph().erasure_budget - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_disabled_tracker() {
        let mut tracker = ProvenanceTracker::new();
        tracker.set_enabled(false);
        let id = tracker.track("Linear", OpClass::S);
        assert_eq!(id, "disabled");
        assert_eq!(tracker.graph().nodes.len(), 0);
    }

    #[test]
    fn test_reset() {
        let mut tracker = ProvenanceTracker::new();
        tracker.track("Linear", OpClass::S);
        tracker.reset();
        assert_eq!(tracker.graph().nodes.len(), 0);
    }
}
