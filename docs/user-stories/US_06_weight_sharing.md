# US-06: Express Weight Sharing with W Primitive

**As** an architect,
**I want** to express weight sharing explicitly using `W(shared_block, depth=6)` for a Universal Transformer and have the framework handle parameter tying correctly,
**So that** shared-weight architectures are first-class citizens.

## Acceptance Criteria

- [ ] W primitive ties parameters across applications
- [ ] `W(block, depth=N)` applies block N times with shared weights
- [ ] Gradient flows correctly through shared parameters

## Spec Reference

- [SPEC-002: BWSK Primitives](../specs/SPEC_002_bwsk_primitives.md)
