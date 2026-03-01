# CLAUDE.md — BWSK Combinator AI Framework

> **Read this file completely before taking any action.**

## Project Overview

**Name**: BWSK Combinator AI Framework
**Team**: Average Joe's Labs
**Goal**: A framework that uses combinator logic (B, W, S, K) as a typed architectural description language for neural networks, providing compile-time guarantees about information flow, reversibility, and parallelism while executing on standard tensor hardware.

**One-sentence pitch**: "Combinators describe, tensors compute — and the description gives you free provenance, free reversibility, and free parallelism guarantees."

## The BWSK Primitives

| Combinator | Rule | Neural Network Mapping |
|------------|------|----------------------|
| **B** (compose) | `B f g x = f(g(x))` | Sequential layer stacking |
| **W** (share) | `W f x = f(x)(x)` | Weight sharing, self-attention |
| **S** (fan-out) | `S f g x = f(x)(g(x))` | Multi-head attention, residual connections |
| **K** (erase) | `K x y = x` | Masking, dropout, pooling, activation clipping |

### S/K Classification

Every neural network operation is either:
- **S-type** (information-preserving, reversible, coordination-free): linear projection, residual connection, concatenation, layer norm, embedding
- **K-type** (information-erasing, synchronization point): ReLU, dropout, max pooling, masking, loss computation
- **Gray** (context-dependent): softmax, batch norm, attention (mixed)

~75% of transformer computation is S-type. The framework makes this explicit and exploits it.

## Development Workflow

### Setup
```bash
uv sync                    # Install dependencies
uv run pytest              # Run tests
```

### Daily Commands
```bash
just lint                  # ruff check
just format                # ruff format
just test                  # pytest
just typecheck             # pyright (if available)
just security-scan         # detect-secrets + pip-audit
just ci                    # All of the above
just rust-check            # cargo check (Rust workspace)
just rust-test             # cargo test (Rust workspace)
just full-ci               # Python + Rust CI
```

### Before Every Commit
Run `just ci` and ensure all checks pass. GPG signing is enabled.

## Architecture Overview

```
src/bwsk/
├── classify.py       # S/K operation classifier (torch.fx-based)
├── primitives.py     # B, W, S, K primitive nn.Module wrappers
├── provenance.py     # Provenance tracking through S-phases
└── training.py       # Training loop with BWSK analysis

rust/
├── bwsk-core/        # Pure computation primitives (no ML deps)
└── bwsk-burn/        # Burn integration (CubeCL GPU backend)
```

## Coding Standards

### Python
- Python 3.12+, managed by `uv`
- Linting/formatting: `ruff` (replaces black + isort + flake8)
- Type hints required on all public APIs
- Tests: `pytest`, one test file per source module
- Docstrings: Google style

### Rust
- Edition 2021
- `cargo fmt` + `cargo clippy -D warnings` before commit
- Tests in the same file (`#[cfg(test)]` modules)

## Test-Driven Development

1. Write a failing test first
2. Implement the minimum code to pass
3. Refactor if needed
4. All tests must pass before merging

Test files mirror source structure:
- `src/bwsk/classify.py` -> `tests/test_classify.py`
- `src/bwsk/primitives.py` -> `tests/test_primitives.py`

## Specs and User Stories

- Formal specs: `docs/specs/SPEC_NNN_*.md`
- User stories: `docs/user-stories/`
- Architecture decisions: `docs/architecture/`
- Research background: `docs/research/`

Each spec includes: Goal, Success Criteria, User Stories, API Sketch, Test Plan.

## Key Research Documents

| Document | Contents |
|----------|----------|
| `docs/research/COMBINATOR_AI_FRAMEWORK.md` | Full framework design document |
| `docs/research/BWSK_PROJECT_PLAN.md` | Project plan with specs, user stories, success criteria |
| `docs/research/DEEP_ANALYSIS_PURE_S.md` | Deep analysis of pure S capabilities and limitations |
| `docs/research/SYNTHESIS_REPORT.md` | Synthesis of S-combinator research findings |

## Key Theoretical Results

- **Pure S is NOT Turing-complete** (Waldmann 1998 + n_S monotonicity proof)
- **{B,W} IS Turing-complete** (Statman 1986): `theta_4 = B(WW)(BW(BBB))`
- **S bundles B+C+W inseparably**; the limitation is bundling, not missing weakening
- **CALM theorem**: monotone computation = coordination-free distributed execution
- **Landauer**: non-erasing computation approaches thermodynamic minimum

## Implementation Phases

1. **S/K Classifier** (Python/PyTorch, torch.fx) — 2-4 weeks
2. **BWSK DSL + Provenance** — 4-8 weeks
3. **Reversible Backprop + CALM Training** — 4-8 weeks
4. **Rust Crate + Burn Integration** — 8-12 weeks
5. **Erasure-Minimized NAS** — 12+ weeks

## CI/CD

All CI runs locally via `just`. No GitHub Actions configured.
Security scanning: `detect-secrets`, `cargo-audit` (Rust), `pip-audit`.

---
*BWSK Combinator AI Framework — Average Joe's Labs*
