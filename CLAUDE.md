# CLAUDE.md — BWSK Combinator AI Framework

> **Read this file completely before taking any action.**
>
> **Documentation index**: See `docs/INDEX.md` for the master index of all docs, cross-references, and "what to read for any task" lookup table.

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
├── classify.py       # S/K operation classifier (torch.fx-based, 70+ ops)
├── primitives.py     # B, W, S, K combinators (pure + nn.Module wrappers)
├── provenance.py     # Provenance tracking (manual + forward hooks)
├── training.py       # BWSK-aware training loop with memory optimization
├── examples.py       # Architecture examples: MLP, residual, attention
├── reversible.py     # Reversible backprop via S-phase checkpointing
├── calm.py           # CALM monotone analysis + distribution partitioning
└── nas.py            # Erasure-minimized Neural Architecture Search

rust/
├── bwsk-core/        # Pure computation primitives (no ML deps)
└── bwsk-burn/        # Burn 0.20 integration (CubeCL GPU backend)
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

## Development Best Practices (Write Correct Code from the Start)

These patterns prevent the most common lint/format/test failures. Follow them to avoid correction cycles.

### Import Conventions (ruff I001, F401)
- **Import order** (ruff isort): stdlib, then blank line, then third-party, then blank line, then first-party (`from bwsk...`). No blank lines within a group.
- **Remove unused imports**: ruff flags F401 aggressively. If you import `torch` but only use `torch.nn as nn`, remove the bare `import torch`.
- **Use `from __future__ import annotations`** in source modules for forward-reference support.
- **First-party imports**: `from bwsk.classify import OpClass` (not relative imports).

### Line Length (ruff E501)
- **Max 100 characters** (configured in `pyproject.toml`). Break long strings:
  ```python
  # BAD (>100 chars on one line):
  _register("nn.LayerNorm", OpClass.S, 0.95, "Per-sample normalization; invertible given affine params.")

  # GOOD:
  _register(
      "nn.LayerNorm", OpClass.S, 0.95,
      "Per-sample normalization; invertible given affine params.",
  )
  ```

### Formatting (ruff format)
- Always run `just format-fix` after writing code, or write code that already matches ruff's style.
- Trailing commas on multiline structures (ruff will add them).
- No trailing whitespace.
- Single quotes vs double quotes: ruff uses double quotes by default.

### Type Annotations
- Use `collections.abc.Callable`, `collections.abc.Iterable` (not `typing.Callable`).
- Use `X | None` union syntax (not `Optional[X]`).
- Use `list[T]`, `dict[K, V]`, `tuple[T, ...]` lowercase generics (Python 3.12+).
- Annotate all public function signatures. Private helpers can use `Any` when type is genuinely dynamic.

### Test Patterns
- Tests import from the public API: `from bwsk.classify import classify_operation, OpClass`
- Test file names mirror source: `src/bwsk/classify.py` -> `tests/test_classify.py`
- Use `pytest.approx()` for float comparisons, not `==`.
- Use `@pytest.mark.skip(reason="...")` for unimplemented features — never `pass` with no assertion.
- Test class names: `TestClassifyOperation`, `TestClassifyModel`, etc.
- Each test method tests one thing. Name describes expected behavior: `test_relu_is_k_type`.

### PyTorch-Specific Patterns
- `torch` is an optional dependency. Import it inside functions if possible, or gate with `try/except ImportError`.
- Use `torch.fx.symbolic_trace(model)` for graph extraction. Handle `TraceError` gracefully.
- Classify by canonical module name (e.g., `"nn.ReLU"`, not `"torch.nn.modules.activation.ReLU"`).
- When inspecting module attributes (stride, training), always use `getattr` with fallback or `isinstance` check.

### Common Pitfalls to Avoid
- Don't use `typing.Any` import alongside `from __future__ import annotations` — use `Any` from `typing` (this is fine with `from __future__`).
- Don't use bare `assert` in production code — use exceptions. `assert` is only for tests.
- Don't shadow built-in names (`type`, `id`, `input`, `list`). Use `op_type`, `node_id`, etc.
- Don't use mutable default arguments (`def f(x=[])`). Use `None` + `if x is None: x = []`.

---

## Documentation Standards — MANDATORY

**Undocumented code is incomplete code.** Every function, class, and module must have documentation that explains the "why", not just the "what".

### Docstrings (Google Style, Required)

Every public function, class, and module MUST have a Google-style docstring. Include:
1. **Summary line**: What it does (one sentence).
2. **Why it exists**: The reasoning behind this function/class. Why was this approach chosen? What problem does it solve?
3. **Args/Returns/Raises**: Standard Google-style sections.

```python
def classify_operation(
    op: nn.Module,
    custom_rules: dict[str, OpClass] | None = None,
) -> ClassificationResult:
    """Classify a single nn.Module as S-type, K-type, or Gray.

    This is the core classification primitive. It checks user overrides first
    (allowing project-specific customization), then attribute-dependent refinement
    (e.g., Conv stride, BatchNorm train/eval), then the default database, and
    finally falls back to GRAY for unknown operations. This 4-step pipeline
    ensures deterministic classification while remaining extensible.

    Args:
        op: The module to classify. Must be an nn.Module instance.
        custom_rules: Optional dict mapping canonical op names (e.g., "nn.ReLU")
            to OpClass overrides. Overrides always get confidence=1.0 because
            the user has explicitly decided.

    Returns:
        A ClassificationResult with classification, confidence, and rationale.

    Raises:
        TypeError: If op is not an nn.Module.
    """
```

### Documentation Maintenance Protocol

**When you create, rename, move, or delete any file**, update these indexes:
1. **`docs/INDEX.md`**: Master documentation index. Update the relevant table.
2. **`CLAUDE.md`**: If it's a spec, ADR, research doc, or source file, update the relevant section.
3. **Memory files**: Update `~/.claude/projects/.../memory/project-state.md` if implementation status changed.

**When you make a design decision**, document the reasoning:
1. **Non-trivial decisions**: Create an ADR in `docs/architecture/` (next number in `docs/INDEX.md`).
2. **Minor decisions**: Add to memory file `decisions.md` with rationale.
3. **In-code decisions**: Add a comment explaining "why", not "what". The code shows what; the comment explains why this approach was chosen over alternatives.

### Spec and Story Documentation
- Specs MUST include a "Why" or "Goal" section explaining the reasoning behind the feature.
- User stories MUST have acceptance criteria that explain what "done" looks like and why each criterion matters.
- ADRs MUST have "Alternatives Considered" explaining why other approaches were rejected — this is where the most valuable reasoning lives.

### When to Update Documentation
- **Before committing code**: Ensure docstrings are present on all new/changed public APIs.
- **After implementing a feature**: Update the spec's status, the user story's checklist, and `docs/INDEX.md`.
- **After making a design decision**: Document it (ADR or memory file).
- **After creating a new file**: Add it to `docs/INDEX.md`.

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
| ADR-003 | torch.fx symbolic tracing for model graph extraction |
| ADR-004 | Classification confidence scoring (4-tier: 1.0/0.8/0.5/0.3) |
| ADR-005 | BWSK primitives as nn.Module with >> pipeline operator |
| ADR-006 | Reversible backprop via S-phase checkpointing + CALM analysis |
| ADR-007 | Two-crate Rust architecture with Burn ML framework |
| ADR-008 | Erasure-minimized NAS with Pareto frontier |
| ADR-009 | Extended benchmarks: 17 models, 4 experiment types |
| ADR-010 | HuggingFace publication strategy |

Use `docs/architecture/ADR_TEMPLATE.md` for new ADRs.

### Research (`docs/research/`)
Background research documents from the S-combinator project.

| Document | Contents |
|----------|----------|
| `COMBINATOR_AI_FRAMEWORK.md` | Full framework design: 5 innovations, S/K classification, implementation roadmap |
| `BWSK_PROJECT_PLAN.md` | Project plan: requirements, deliverables, success criteria, timeline |
| `DEEP_ANALYSIS_PURE_S.md` | Pure S computational class, substructural logic, minimal extensions |
| `SYNTHESIS_REPORT.md` | Research synthesis: what S can/cannot compute, AI connections |
| `TORCH_FX_RESEARCH.md` | torch.fx capabilities, limitations, Dynamo comparison, practical patterns |
| `PRIOR_ART_AND_CALM_RESEARCH.md` | Captum/FrEIA/RevNet analysis, CALM theorem formalization, gradient monotonicity |

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

**Current status**: All 5 phases complete. Full benchmark pipeline run across 17 models on RTX 5080 + RTX 3090 Ti. Whitepaper written with complete empirical results. VRAM-aware GPU scheduler with calibration. Full convergence training pipeline complete: 96 models trained to convergence (16 architectures x 3 modes x 2 experiments), consolidated into 16 HuggingFace repos (one per model). ~200 Python tests passing, 29 Rust tests passing. Full CI green. GitHub repo public. 8 publication-quality figures generated.

- **Phase 1** (S/K Classifier): classify_operation, classify_model, 70+ op database, torch.fx tracing, per_layer_summary, BWSK primitives, provenance tracker, training loop
- **Phase 2** (DSL + Provenance): nn.Module wrappers (BModule/WModule/SModule/KModule), >> pipeline operator, forward hook provenance, to_json/to_graphviz, architecture examples (MLP, residual, attention)
- **Phase 3** (Reversible + CALM): S-phase checkpointing via ReversibleSequence, CALM monotone analysis, distribution partitioning, enhanced BWSKTrainer with memory optimization
- **Phase 4** (Rust Port): bwsk-core (combinators, classifier, provenance), bwsk-burn (BLinear, SResidual, KRelu, BwskMlp on Burn 0.20)
- **Phase 5** (NAS): Erasure-minimized NAS with random + evolutionary search, Pareto frontier, gene encoding

See ADR-003 through ADR-008 for design decisions.

---

## Continuation Plan (for "continue" prompt)

All 5 implementation phases are **complete**. When the user says "continue", look for:

1. **Unskipped tests or TODOs** in the codebase
2. **User story acceptance criteria** not yet met (CLI tools, comparison commands, etc.)
3. **Performance optimization** opportunities
4. **Documentation gaps** (missing docstrings, outdated specs)

### Work Principles
- **TDD**: Write failing test, implement, refactor. No code without a test.
- **`just ci` before every commit**: lint + format + test + security must all pass.
- **`just format-fix`**: Run after writing code to auto-fix formatting.
- **Docstrings on everything**: Google style, include the "why". Undocumented code is incomplete.
- **Update indexes after changes**: `docs/INDEX.md`, CLAUDE.md status, memory files.
- **Keep CLAUDE.md current**: Update "Current status" and this Continuation Plan after each milestone.
- **Create ADRs** for non-obvious design decisions. Document "why" and "alternatives considered".
- **Commit after each logical unit of work** (e.g., after primitives, after provenance, etc.)

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
