# US-08: Identify Erasure Hotspots

**As** a data scientist,
**I want** to identify which K-boundaries in my model discard the most information,
**So that** I can decide whether to replace them with S-type alternatives.

## Acceptance Criteria

- [ ] Erasure fraction quantified per K-operation
- [ ] Operations ranked by information loss
- [ ] Suggestions for S-type alternatives where applicable

## Spec Reference

- [SPEC-001: S/K Operation Classifier](../specs/SPEC_001_sk_classifier.md)
- [SPEC-003: Provenance Tracking](../specs/SPEC_003_provenance.md)
