# US-11: Compare Training Memory Strategies

**As** a researcher,
**I want** to compare memory/compute tradeoffs between standard training, gradient checkpointing, RevNet, and S-phase reversible training on the same model,
**So that** I can quantify the benefit of S-phase awareness.

## Acceptance Criteria

- [ ] Benchmark on at least GPT-2-124M
- [ ] Metrics: peak memory, FLOPs, wall-clock time, final accuracy
- [ ] Fair comparison (same hyperparameters, same data)

## Spec Reference

- [SPEC-004: Architecture DSL](../specs/SPEC_004_architecture_dsl.md)
