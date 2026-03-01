# SPEC-003: Provenance Tracking System

## Goal

Track information flow through neural network forward passes, providing complete provenance for S-phases and annotating K-boundaries with discarded information.

## Success Criteria

| Criterion | Target |
|-----------|--------|
| S-phase completeness | 100% of S-phase operations tracked |
| K-boundary annotation | Every K-boundary has erasure description |
| Overhead (enabled) | < 5% inference slowdown |
| Overhead (disabled) | Zero overhead |
| Output formats | JSON, HTML visualization, Graphviz |

## User Stories

- **US-7**: Trace predictions through S-phases without SHAP/Captum
- **US-8**: Identify which K-boundaries discard most information
- **US-9**: Produce regulatory audit trail of data processing

## API Sketch

```python
from bwsk.provenance import ProvenanceTracker

tracker = ProvenanceTracker()
# ... attach to model forward hooks ...
output = model(input)
graph = tracker.get_graph()
graph.to_json("provenance.json")
```

## Test Plan

- Unit tests for ProvenanceNode and ProvenanceGraph
- Integration test: trace through a 2-layer network
- Performance benchmark: overhead with/without tracking
