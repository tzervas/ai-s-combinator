# US-01: Classify Model via CLI

**As** an ML engineer,
**I want** to run `bwsk classify my_model.pt` and see which layers erase information,
**So that** I can understand my model's information flow.

## Acceptance Criteria

- [ ] CLI accepts a saved PyTorch model path
- [ ] Output is a JSON erasure budget report
- [ ] Each layer classified as S, K, or Gray
- [ ] Report includes per-layer summary and whole-model erasure score

## Spec Reference

- [SPEC-001: S/K Operation Classifier](../specs/SPEC_001_sk_classifier.md)
