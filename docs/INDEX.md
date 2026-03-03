# Documentation Index

> **This file is the master index of all project documentation.**
> It must be kept current. When you create, rename, or delete a doc, update this index.
> When you significantly change a doc's scope, update its description here.

Last updated: 2026-03-01

---

## How to Use This Index

**Starting a new session?** Read this file first to orient. Then read only what's relevant to your task.

**Looking for something specific?** Use the cross-reference tables and "When to Read" guidance below.

**Created or changed a doc?** Update this index before committing.

---

## Quick Lookup: What Do I Read For...?

| I need to...                                    | Read these                                      |
|-------------------------------------------------|-------------------------------------------------|
| Understand the project vision                   | `CLAUDE.md`, `research/COMBINATOR_AI_FRAMEWORK.md` |
| Implement the S/K classifier                    | `specs/SPEC_001_sk_classifier.md`, `research/TORCH_FX_RESEARCH.md` |
| Implement BWSK primitives                       | `specs/SPEC_002_bwsk_primitives.md`             |
| Implement provenance tracking                   | `specs/SPEC_003_provenance.md`                  |
| Implement the architecture DSL                  | `specs/SPEC_004_architecture_dsl.md`            |
| Plan the Rust port                              | `specs/SPEC_005_rust_port.md`                   |
| Understand a design decision                    | `architecture/ADR_00N_*.md`                     |
| Understand prior art and what's novel            | `research/PRIOR_ART_AND_CALM_RESEARCH.md`       |
| Know the theoretical foundations                 | `research/DEEP_ANALYSIS_PURE_S.md`, `research/SYNTHESIS_REPORT.md` |
| Know what acceptance criteria to hit             | `user-stories/US_NN_*.md`                       |
| Know how to write code that passes CI            | `CLAUDE.md` > "Development Best Practices"      |
| Pick up where the last session left off          | `CLAUDE.md` > "Continuation Plan"               |

---

## Specifications (`docs/specs/`)

| File | Title | Phase | Status | Depends On | Key Contents |
|------|-------|-------|--------|------------|--------------|
| `SPEC_001_sk_classifier.md` | S/K Operation Classifier | 1 | **Implementing** | None | Classification database (70+ ops), algorithm pseudocode, edge cases, output format, test plan |
| `SPEC_002_bwsk_primitives.md` | BWSK Primitive Implementations | 1-2 | Spec ready | SPEC-001 | B/W/S/K combinator rules, nn.Module wrapping, composition |
| `SPEC_003_provenance.md` | Provenance Tracking System | 2 | Spec ready | SPEC-001, SPEC-002 | ProvenanceNode/Graph dataclasses, tracker hooks, output formats |
| `SPEC_004_architecture_dsl.md` | Architecture DSL | 2 | Spec ready | SPEC-002 | `>>` operator, shape validation, compile to nn.Module |
| `SPEC_005_rust_port.md` | Rust Port Strategy | 4 | Spec ready | SPEC-001..004 | bwsk-core + bwsk-burn crate structure, cross-validation |

---

## User Stories (`docs/user-stories/`)

| File | Story | Phase | Status | Spec | Key Acceptance Criteria |
|------|-------|-------|--------|------|------------------------|
| `US_01_classifier_cli.md` | Classify Model via CLI | 1 | **Partially met** | SPEC-001 | CLI, JSON schema, programmatic API, error handling |
| `US_02_compare_architectures.md` | Compare Architecture Erasure Budgets | 1 | Not started | SPEC-001 | Side-by-side comparison, `bwsk compare` command |
| `US_03_compliance_audit.md` | Audit K-Boundaries for Compliance | 1 | Not started | SPEC-001 | Compliance document format, `bwsk audit` command |
| `US_04_dsl_transformer.md` | Define Transformer Block in DSL | 2 | Not started | SPEC-002, SPEC-004 | DSL syntax, shape validation |
| `US_05_composable_modules.md` | Compose Modules with Type-Checked Shapes | 2 | Not started | SPEC-004 | `>>` operator, shape mismatch errors |
| `US_06_weight_sharing.md` | Express Weight Sharing with W | 2 | Not started | SPEC-002 | `W(block, depth=N)`, parameter tying |
| `US_07_provenance_trace.md` | Trace Predictions Through S-Phases | 2 | Not started | SPEC-003 | Forward hooks, S-phase tracing |
| `US_08_erasure_hotspots.md` | Identify Erasure Hotspots | 1-2 | Not started | SPEC-001, SPEC-003 | Ranked K-operations, S-type alternatives |
| `US_09_regulatory_audit_trail.md` | Regulatory Audit Trail | 2 | Not started | SPEC-003 | HTML/PDF provenance export |
| `US_10_memory_reduction.md` | Reduce Activation Memory | 3 | Not started | SPEC-004 | S-phase reversible backprop, 50%+ memory reduction |
| `US_11_training_comparison.md` | Compare Training Memory Strategies | 3 | Not started | SPEC-004 | Benchmark vs standard/checkpointing/RevNet |
| `US_12_calm_distributed.md` | CALM-Monotone Distributed Training | 3 | Not started | SPEC-004 | CRDT gradient accumulator, staleness parameter |
| `US_13_calm_convergence.md` | CALM Convergence Analysis | 3 | Not started | SPEC-004 | Loss landscape, sharpness metrics |
| `US_14_rust_bwsk.md` | Define Architecture in Rust | 4 | Not started | SPEC-005 | Rust BWSK API, Burn modules, CubeCL |
| `US_15_single_binary.md` | Deploy as Single Binary | 4 | Not started | SPEC-005 | <50MB binary, <100ms startup |

---

## Architecture Decision Records (`docs/architecture/`)

| File | Decision | Date | Status | Relevant To |
|------|----------|------|--------|-------------|
| `ADR_001_combinators_describe_tensors_compute.md` | Combinators as description language, not execution engine | 2026-03-01 | Accepted | All phases |
| `ADR_002_python_first_rust_later.md` | Python Phases 1-3, Rust Phase 4 | 2026-03-01 | Accepted | All phases |
| `ADR_003_torch_fx_for_tracing.md` | torch.fx symbolic tracing for graph extraction | 2026-02-28 | Accepted | Phase 1 (classifier) |
| `ADR_004_classification_confidence_scoring.md` | 4-tier confidence: 1.0/0.8/0.5/0.3 | 2026-02-28 | Accepted | Phase 1 (classifier) |
| `ADR_005_nn_module_primitives.md` | BWSK primitives as nn.Module with >> operator | 2026-03-01 | Accepted | Phase 2 (DSL) |
| `ADR_006_reversible_backprop_calm.md` | Reversible backprop via S-phase checkpointing + CALM analysis | 2026-03-01 | Accepted | Phase 3 (training) |
| `ADR_007_rust_port_strategy.md` | Two-crate Rust architecture with Burn ML framework | 2026-03-01 | Accepted | Phase 4 (Rust) |
| `ADR_008_erasure_minimized_nas.md` | Evolutionary NAS with Pareto frontier for erasure vs accuracy | 2026-03-01 | Accepted | Phase 5 (NAS) |
| `ADR_009_extended_benchmarks.md` | Extended benchmarks: 17 models, 4 experiment types | 2026-03-01 | Accepted | All phases |
| `ADR_010_huggingface_publication.md` | HuggingFace publication strategy | 2026-03-01 | Accepted | All phases |
| `ADR_TEMPLATE.md` | Template for new ADRs | — | Template | — |

**Next ADR number**: ADR-011

---

## Research Documents (`docs/research/`)

| File | Topic | When to Read |
|------|-------|-------------|
| `COMBINATOR_AI_FRAMEWORK.md` | Full framework design: 5 innovations, S/K classification, hybrid architecture | Project orientation, vision discussions |
| `BWSK_PROJECT_PLAN.md` | Requirements, deliverables, success criteria, timeline, risk register | Project planning, milestone tracking |
| `DEEP_ANALYSIS_PURE_S.md` | Pure S computational class, substructural logic, {B,W} Turing-completeness | Theoretical questions, paper writing |
| `SYNTHESIS_REPORT.md` | Research synthesis: what S can/cannot compute, AI connections | Theoretical questions, paper writing |
| `TORCH_FX_RESEARCH.md` | torch.fx capabilities, limitations, Dynamo comparison, practical code patterns | Implementing classifier, graph tracing |
| `PRIOR_ART_AND_CALM_RESEARCH.md` | Captum/FrEIA/RevNet prior art, CALM theorem formalization, gradient monotonicity gaps | Novelty claims, CALM training implementation |

---

## Source Code (`src/bwsk/`)

| File | Purpose | Implementation Status | Tests | Spec |
|------|---------|-----------------------|-------|------|
| `__init__.py` | Package init, version | Done | — | — |
| `classify.py` | S/K operation classifier | **Done**: classify_operation, classify_model, DB (70+ ops), torch.fx tracing, ErasureBudgetReport, per_layer_summary | 69 passing | SPEC-001 |
| `primitives.py` | B, W, S, K combinators | **Done**: pure callables + nn.Module wrappers, >> pipeline operator, classification property | 19 passing | SPEC-002 |
| `provenance.py` | Provenance tracking | **Done**: dataclasses, tracker.track(), forward hooks, to_json(), to_graphviz() | 19 passing | SPEC-003 |
| `training.py` | BWSK-aware training loop | **Done**: BWSKTrainer with reversible mode, memory/CALM analysis | 9 passing | — |
| `examples.py` | Architecture examples in BWSK DSL | **Done**: MLP, residual block, attention head | 13 passing | SPEC-002, SPEC-004 |
| `reversible.py` | Reversible backprop via S-phase checkpointing | **Done**: ReversibleSequence, checkpoint_k_boundaries, memory profiling | 10 passing | — |
| `calm.py` | CALM monotone analysis and distribution partitioning | **Done**: analyze_calm, partition_for_distribution | 8 passing | — |
| `nas.py` | Erasure-minimized Neural Architecture Search | **Done**: random + evolutionary search, Pareto frontier, gene encoding | 13 passing | — |

---

## Cross-Reference: Spec → Source → Tests → Stories

| Spec | Source File | Test File | User Stories |
|------|-----------|-----------|-------------|
| SPEC-001 | `classify.py` | `test_classify.py` | US-01, US-02, US-03, US-08 |
| SPEC-002 | `primitives.py`, `examples.py` | `test_primitives.py`, `test_examples.py` | US-04, US-05, US-06 |
| SPEC-003 | `provenance.py` | `test_provenance.py` | US-07, US-08, US-09 |
| SPEC-004 | `reversible.py`, `calm.py` | `test_reversible.py`, `test_calm.py` | US-10, US-11, US-12, US-13 |
| SPEC-005 | `rust/bwsk-core/`, `rust/bwsk-burn/` | Rust `#[cfg(test)]` modules | US-14, US-15 |
| — | `nas.py` | `test_nas.py` | — |

---

## Scripts (`scripts/`)

| File | Purpose | Notes |
|------|---------|-------|
| `ci.sh` | CI runner shell script | Used by `just ci` |
| `publish-ghcr.sh` | GHCR container publishing | — |
| `security-scan.sh` | Security scanning (gitleaks, detect-secrets, pip-audit) | Used by `just security-scan` |
| `validate_and_compare.py` | End-to-end BWSK validation and PyTorch comparison | Generates `validation_results.json` and `docs/VALIDATION_REPORT.md` |
| `gpt2_benchmark.py` | GPT-2 (124M) benchmark: BWSK vs conventional PyTorch | Generates `gpt2_benchmark_results.json` and `docs/GPT2_BENCHMARK_REPORT.md` |
| `multi_model_benchmark.py` | Multi-model benchmark (BERT, GPT-2 Med, T5, OPT, Pythia-410M, Pythia-1B) | Generates per-model JSON + `docs/MULTI_MODEL_BENCHMARK_REPORT.md` |
| `rust_cross_validation.py` | Python vs Rust classification cross-validation and Burn benchmarks | Generates `docs/RUST_CROSS_VALIDATION_REPORT.md` |
| `extended_benchmark.py` | Extended benchmark: 17 models (scale sweep + architecture diversity) | Generates `docs/EXTENDED_BENCHMARK_REPORT.md` |
| `convergence_experiment.py` | Convergence experiment: 1500 steps, 3 seeds, statistical analysis | Generates `docs/CONVERGENCE_REPORT.md` |
| `memory_throughput_profiler.py` | Memory/throughput profiler: detailed breakdown per model | Generates `docs/MEMORY_THROUGHPUT_REPORT.md` |
| `generate_whitepaper_figures.py` | Generate publication-quality figures from experiment data | Outputs to `docs/figures/` |
| `upload_to_huggingface.py` | Upload benchmark datasets and fine-tuned models to HuggingFace | — |
| `bench_utils.py` | Shared utilities: VRAM estimation, memory helpers, calibration | Used by gpu_scheduler and benchmark scripts |
| `gpu_scheduler.py` | VRAM-aware GPU task scheduler with bin-packing | CLI: `--schedule-only`, `--calibrate`, `--all` |
| `full_training_pipeline.py` | Full convergence training: epoch-based, early stopping, HF upload, cleanup | CLI: `--models`, `--experiment`, `--resume`, `--dry-run` |
| `training_utils.py` | Shared training infrastructure: datasets, early stopper, checkpoints, cleanup | Used by full_training_pipeline.py |

## Tests (`tests/`)

| File | Purpose | Tests |
|------|---------|-------|
| `test_classify.py` | S/K classifier tests | 69 |
| `test_primitives.py` | BWSK combinator tests | 19 |
| `test_provenance.py` | Provenance tracking tests | 19 |
| `test_training.py` | Training loop tests | 9 |
| `test_examples.py` | Architecture example tests | 13 |
| `test_reversible.py` | Reversible backprop tests | 10 |
| `test_calm.py` | CALM analysis tests | 8 |
| `test_nas.py` | NAS tests | 13 |
| `test_gpu_scheduler.py` | VRAM estimation and scheduler tests | 20 |
| `test_training_utils.py` | Training utils: datasets, early stopper, checkpoints | 22 |

## Generated Reports

| File | Purpose | Generated By |
|------|---------|-------------|
| `docs/VALIDATION_REPORT.md` | Quantitative BWSK vs PyTorch comparison with tables | `scripts/validate_and_compare.py` |
| `scripts/validation_results.json` | Raw validation data (JSON) | `scripts/validate_and_compare.py` |
| `docs/GPT2_BENCHMARK_REPORT.md` | GPT-2 benchmark: classification, perplexity, fine-tuning, memory, CALM | `scripts/gpt2_benchmark.py` |
| `scripts/gpt2_benchmark_results.json` | Raw GPT-2 benchmark data (JSON) | `scripts/gpt2_benchmark.py` |
| `docs/MULTI_MODEL_BENCHMARK_REPORT.md` | Multi-model benchmark: cross-architecture BWSK comparison | `scripts/multi_model_benchmark.py` |
| `scripts/multi_model_benchmark_results.json` | Combined multi-model benchmark data (JSON) | `scripts/multi_model_benchmark.py` |
| `docs/RUST_CROSS_VALIDATION_REPORT.md` | Python vs Rust classification parity and Burn benchmark results | `scripts/rust_cross_validation.py` |
| `docs/EXTENDED_BENCHMARK_REPORT.md` | Extended benchmark: 17-model cross-architecture BWSK comparison | `scripts/extended_benchmark.py` |
| `docs/CONVERGENCE_REPORT.md` | Statistical convergence analysis with paired t-tests and CIs | `scripts/convergence_experiment.py` |
| `docs/MEMORY_THROUGHPUT_REPORT.md` | Memory breakdown and throughput scaling analysis | `scripts/memory_throughput_profiler.py` |
| `docs/WHITEPAPER.md` | Scientific whitepaper with complete empirical results (17 models, 5 families) | Manual + benchmark data |
| `docs/figures/` | 8 publication-quality figures for whitepaper | `scripts/generate_whitepaper_figures.py` |
| `scripts/vram_calibration.json` | Measured peak VRAM for 16 models (calibrates scheduler) | `scripts/gpu_scheduler.py --calibrate` |
| `scripts/extended_*_results.json` | Per-model extended benchmark results (17 files + combined) | `scripts/extended_benchmark.py` |
| `scripts/convergence_*_results.json` | Per-model convergence results (4 files + combined) | `scripts/convergence_experiment.py` |
| `scripts/memory_throughput_results.json` | Memory profiler results (7 models, 3 modes) | `scripts/memory_throughput_profiler.py` |
| `scripts/profile_*_results.json` | Per-model memory profiler results (7 files) | `scripts/memory_throughput_profiler.py` |
| `docs/FULL_TRAINING_REPORT.md` | Full convergence training results across all models and modes | `scripts/full_training_pipeline.py` |
| `scripts/full_training_results.json` | Combined full training results (JSON) | `scripts/full_training_pipeline.py` |
| `scripts/fulltrain_*_results.json` | Per-model full training results | `scripts/full_training_pipeline.py` |
| `scripts/pipeline_state.json` | Pipeline resume state (completed/failed runs) | `scripts/full_training_pipeline.py` |
