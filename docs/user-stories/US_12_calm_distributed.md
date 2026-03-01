# US-12: CALM-Monotone Distributed Training

**As** an engineer running distributed training on 8 GPUs,
**I want** to reduce AllReduce overhead by using CRDT gradient accumulation for monotone operations, with a tunable staleness parameter,
**So that** I can train faster without sacrificing convergence.

## Acceptance Criteria

- [ ] CRDT gradient accumulator replaces AllReduce for S-phase gradients
- [ ] Staleness parameter k is tunable (k=1 sync, k=inf fully async)
- [ ] Final accuracy within 0.5% of synchronous at k=4
- [ ] >= 20% wall-clock reduction vs AllReduce at 8 GPUs

## Spec Reference

- [SPEC-004: Architecture DSL](../specs/SPEC_004_architecture_dsl.md)
