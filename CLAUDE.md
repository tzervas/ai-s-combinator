# CLAUDE.md — BWSK Combinator AI Framework

> **Read this file completely before taking any action.**

## Project Overview

**Name**: BWSK Combinator AI Framework
**Author**: Tyler Zervas
**Goal**: A framework that uses combinator logic (B, W, S, K) as a typed architectural description language for neural networks, providing compile-time guarantees about information flow, reversibility, and parallelism while executing on standard tensor hardware.

**One-sentence pitch**: "Combinators describe, tensors compute — and the description gives you free provenance, free reversibility, and free parallelism guarantees."

---

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

---

## Development Workflow

### Setup
```bash
uv sync --all-extras          # Install all dependencies (including dev)
uv run pytest                 # Run tests
```

### Daily Commands
```bash
just lint                     # ruff check
just format                   # ruff format (check mode)
just format-fix               # ruff format (auto-fix)
just test                     # pytest
just test-cov                 # pytest with coverage
just typecheck                # pyright (if available)
just gitleaks                 # gitleaks secret scan
just security-scan            # gitleaks + detect-secrets + pip-audit
just ci                       # Full Python CI (lint + format + test + security)
just rust-check               # cargo check (Rust workspace)
just rust-test                # cargo test (Rust workspace)
just rust-ci                  # Full Rust CI (fmt + clippy + check + test)
just full-ci                  # Python + Rust CI
```

### Before Every Commit
1. Run `just ci` and ensure all checks pass
2. Pre-commit hook runs automatically: `git-secrets` + `gitleaks protect --staged`
3. GPG signing is enabled — all commits must be signed

---

## Security

### Gitleaks (Pre-Commit)
- Config: `.gitleaks.toml` — allowlists for .venv, __pycache__, target/, lock files
- Pre-commit hook: `.git/hooks/pre-commit` runs `gitleaks protect --staged` on every commit
- Manual scan: `just gitleaks` or `gitleaks detect --config .gitleaks.toml --verbose`
- **Never commit** `.env`, credentials, API keys, or private keys

### Additional Scanning
- `detect-secrets`: Scans for high-entropy strings and known secret patterns
- `pip-audit`: Checks Python dependencies for known vulnerabilities
- `cargo-audit`: Checks Rust dependencies (when Rust crates have deps)

---

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

---

## Coding Standards

### Python
- Python 3.12+, managed by `uv`
- Linting/formatting: `ruff` (replaces black + isort + flake8)
- Type hints required on all public APIs
- Tests: `pytest`, one test file per source module
- Docstrings: Google style
- Imports: use `collections.abc` for `Callable`, `Iterable`, etc. (not `typing`)

### Rust
- Edition 2021
- `cargo fmt` + `cargo clippy -D warnings` before commit
- Tests in the same file (`#[cfg(test)]` modules)

---

## Test-Driven Development — MANDATORY

1. Write a failing test first
2. Implement the minimum code to pass
3. Refactor if needed
4. All tests must pass before merging

Test files mirror source structure:
- `src/bwsk/classify.py` -> `tests/test_classify.py`
- `src/bwsk/primitives.py` -> `tests/test_primitives.py`
- `src/bwsk/provenance.py` -> `tests/test_provenance.py`
- `src/bwsk/training.py` -> `tests/test_training.py`

Skipped tests (`@pytest.mark.skip`) mark unimplemented features. Unskip and implement one at a time.

---

## Documentation Structure

### Specs (`docs/specs/`)
Formal specifications with Goal, Success Criteria, User Stories, API Sketch, Test Plan.

| Spec | Title | Phase |
|------|-------|-------|
| SPEC-001 | S/K Operation Classifier | 1 |
| SPEC-002 | BWSK Primitive Implementations | 1-2 |
| SPEC-003 | Provenance Tracking System | 2 |
| SPEC-004 | Architecture DSL | 2 |
| SPEC-005 | Rust Port Strategy | 4 |

### User Stories (`docs/user-stories/`)
15 user stories (US-01 through US-15) covering all phases. Each has acceptance criteria and spec references.

| Story | Title | Spec |
|-------|-------|------|
| US-01 | Classify Model via CLI | SPEC-001 |
| US-02 | Compare Architecture Erasure Budgets | SPEC-001 |
| US-03 | Audit K-Boundaries for Compliance | SPEC-001 |
| US-04 | Define Transformer Block in DSL | SPEC-002, SPEC-004 |
| US-05 | Compose Modules with Type-Checked Shapes | SPEC-004 |
| US-06 | Express Weight Sharing with W | SPEC-002 |
| US-07 | Trace Predictions Through S-Phases | SPEC-003 |
| US-08 | Identify Erasure Hotspots | SPEC-001, SPEC-003 |
| US-09 | Regulatory Audit Trail | SPEC-003 |
| US-10 | Reduce Activation Memory | SPEC-004 |
| US-11 | Compare Training Memory Strategies | SPEC-004 |
| US-12 | CALM-Monotone Distributed Training | SPEC-004 |
| US-13 | CALM Convergence Analysis | SPEC-004 |
| US-14 | Define Architecture in Rust | SPEC-005 |
| US-15 | Deploy as Single Binary | SPEC-005 |

### Architecture Decisions (`docs/architecture/`)
ADRs (Architecture Decision Records) documenting key design choices.

| ADR | Decision |
|-----|----------|
| ADR-001 | Combinators describe, tensors compute (not graph reduction runtime) |
| ADR-002 | Python-first, Rust port later |

Use `docs/architecture/ADR_TEMPLATE.md` for new ADRs.

### Research (`docs/research/`)
Background research documents from the S-combinator project.

| Document | Contents |
|----------|----------|
| `COMBINATOR_AI_FRAMEWORK.md` | Full framework design: 5 innovations, S/K classification, implementation roadmap |
| `BWSK_PROJECT_PLAN.md` | Project plan: requirements, deliverables, success criteria, timeline |
| `DEEP_ANALYSIS_PURE_S.md` | Pure S computational class, substructural logic, minimal extensions |
| `SYNTHESIS_REPORT.md` | Research synthesis: what S can/cannot compute, AI connections |

---

## Key Theoretical Results

- **Pure S is NOT Turing-complete** (Waldmann 1998 + n_S monotonicity proof)
- **{B,W} IS Turing-complete** (Statman 1986): `theta_4 = B(WW)(BW(BBB))`
- **S bundles B+C+W inseparably**; the limitation is bundling, not missing weakening
- **CALM theorem**: monotone computation = coordination-free distributed execution
- **Landauer**: non-erasing computation approaches thermodynamic minimum
- **~75% of transformer ops are S-type** — the framework makes this explicit

---

## Implementation Phases

| Phase | Deliverable | Stack | Duration |
|-------|-------------|-------|----------|
| 1 | S/K Classifier | Python, torch.fx | 2-4 weeks |
| 2 | BWSK DSL + Provenance | Python, PyTorch | 4-8 weeks |
| 3 | Reversible Backprop + CALM Training | Python, PyTorch | 4-8 weeks |
| 4 | Rust Crate + Burn Integration | Rust, CubeCL | 8-12 weeks |
| 5 | Erasure-Minimized NAS | Python | 12+ weeks |

**Current status**: Phase 1 ready to begin. All specs, user stories, and test stubs in place.

---

## CI/CD

All CI runs locally via `just`. No GitHub Actions — no runners configured.

### Pre-commit hooks (automatic)
- `git-secrets` — AWS credential patterns
- `gitleaks` — broad secret detection via `.gitleaks.toml`

### Manual CI
- `just ci` — lint + format + test + security scan
- `just full-ci` — Python + Rust

---
*BWSK Combinator AI Framework — Tyler Zervas*
