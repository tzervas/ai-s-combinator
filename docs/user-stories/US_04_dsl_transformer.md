# US-04: Define Transformer Block in BWSK DSL

**As** an ML engineer,
**I want** to define a transformer block using `S(linear_Q, S(linear_K, linear_V)) >> K_softmax() >> S_residual()` and have the DSL compile it to PyTorch with shape validation,
**So that** I get compile-time guarantees about my architecture.

## Acceptance Criteria

- [ ] DSL syntax compiles to a valid `nn.Module`
- [ ] Shape mismatches caught at definition time, not runtime
- [ ] Compiled module produces identical outputs to hand-written PyTorch (atol=1e-6)
- [ ] S/K classification automatically applied to each DSL node

## Spec Reference

- [SPEC-002: BWSK Primitives](../specs/SPEC_002_bwsk_primitives.md)
- [SPEC-004: Architecture DSL](../specs/SPEC_004_architecture_dsl.md)
