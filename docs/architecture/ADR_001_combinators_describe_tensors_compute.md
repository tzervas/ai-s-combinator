# ADR-001: Combinators Describe, Tensors Compute

**Date**: 2026-03-01
**Status**: Accepted

## Context

The naive approach would use combinator graph reduction as the neural network execution engine. HVM2 benchmarks show this is 100-4500x slower than cuBLAS for dense linear algebra. Graph reduction cannot compete with tensor cores for matmul.

## Decision

Use combinator logic as the **architectural description language** (compile-time) and tensor operations as the **execution substrate** (runtime). The combinator structure is metadata that provides type-level guarantees, not the execution engine.

## Consequences

- **Easier**: Zero runtime overhead from combinator framework (it's compile-time only). Compatible with existing PyTorch/JAX ecosystems. Provenance and reversibility are free from the type system.
- **Harder**: Cannot exploit interaction net parallelism at runtime. Must maintain two representations (combinator and tensor).

## Alternatives Considered

1. **Pure graph reduction runtime** (HVM2-style): Rejected — 100-4500x slower than tensor cores for the operations that dominate neural network compute.
2. **Hybrid runtime** (graph reduction for control flow, tensors for compute): Considered for future work but adds complexity without clear benefit for Phase 1.
