# ADR-002: Python-First, Rust Port Later

**Date**: 2026-03-01
**Status**: Accepted

## Context

The framework needs both rapid prototyping (research iteration) and high-performance deployment. Python dominates ML tooling; Rust provides safety and performance for production.

## Decision

Implement Phases 1-3 in Python (uv + ruff, PyTorch integration). Port to Rust (Burn + CubeCL) in Phase 4 after the logic is proven and APIs are stable.

## Consequences

- **Easier**: Fast iteration on DSL design, immediate PyTorch ecosystem access, lower barrier for ML researcher contributions.
- **Harder**: Rust port requires reimplementation, not just binding. Two codebases to maintain during transition.

## Alternatives Considered

1. **Rust-first**: Rejected — Burn ecosystem not mature enough for rapid research iteration. Would slow Phase 1-3 significantly.
2. **Python-only**: Rejected — misses deployment benefits (single binary, no Python runtime, CubeCL GPU kernels).
