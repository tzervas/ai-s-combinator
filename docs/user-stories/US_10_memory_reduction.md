# US-10: Reduce Activation Memory via S-Phase Reversibility

**As** an ML engineer training large models,
**I want** to reduce activation memory by 50%+ without changing my architecture or losing accuracy, by exploiting the reversibility of S-phases,
**So that** I can train larger models on the same hardware.

## Acceptance Criteria

- [ ] S-phase activations reconstructed during backprop (not stored)
- [ ] Only K-boundary outputs checkpointed
- [ ] >= 50% peak activation memory reduction
- [ ] < 2% compute overhead
- [ ] < 0.1% accuracy difference from standard training

## Spec Reference

- [SPEC-004: Architecture DSL](../specs/SPEC_004_architecture_dsl.md)
