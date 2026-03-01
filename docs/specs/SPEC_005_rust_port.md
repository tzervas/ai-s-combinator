# SPEC-005: Rust Port Strategy

## Goal

Port the BWSK core type system and S/K classification to Rust, with Burn framework integration for GPU-accelerated training via CubeCL.

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Rust/Python parity | Identical classification results |
| Burn integration | Full BWSK model trains on GPU via CubeCL |
| Binary size | < 50MB for inference deployment |
| Cross-validation | Same model produces identical results in both languages |

## User Stories

- **US-14**: Define architecture using BWSK combinators in Rust
- **US-15**: Deploy as single binary with ms startup

## Crate Structure

- `bwsk-core`: Pure computation (no ML deps) — primitives, classify, provenance
- `bwsk-burn`: Burn backend — module compilation, reversible backprop, CALM training

## Test Plan

- Cross-validation: Python and Rust produce identical S/K classifications
- Burn module compiles and runs forward pass on GPU
- Inference binary benchmark (startup time, size)
