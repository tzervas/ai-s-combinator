# US-15: Deploy as Single Binary

**As** an infrastructure engineer,
**I want** to deploy a BWSK-defined model as a single binary with millisecond startup, using Burn's inference capabilities,
**So that** I avoid Python runtime dependencies in production.

## Acceptance Criteria

- [ ] Single binary < 50MB for inference
- [ ] Startup time < 100ms
- [ ] No Python runtime required
- [ ] Model weights embedded or loaded from file

## Spec Reference

- [SPEC-005: Rust Port Strategy](../specs/SPEC_005_rust_port.md)
