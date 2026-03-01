# BWSK Combinator AI Framework — Task Runner

# Run ruff linter
lint:
    uv run ruff check src/ tests/

# Run ruff formatter (check mode)
format:
    uv run ruff format --check src/ tests/

# Auto-fix formatting
format-fix:
    uv run ruff format src/ tests/

# Run pytest
test:
    uv run pytest

# Run pytest with coverage
test-cov:
    uv run pytest --cov=bwsk --cov-report=term-missing

# Run pyright type checker
typecheck:
    uv run pyright src/

# Run gitleaks scan
gitleaks:
    gitleaks detect --config .gitleaks.toml --verbose

# Run security scanning (gitleaks + detect-secrets + pip-audit)
security-scan: gitleaks
    bash scripts/security-scan.sh

# Full Python CI pipeline
ci: lint format test security-scan

# Check Rust workspace
rust-check:
    cd rust && cargo check --workspace

# Test Rust workspace
rust-test:
    cd rust && cargo test --workspace

# Rust formatting
rust-fmt:
    cd rust && cargo fmt --all

# Rust linting
rust-clippy:
    cd rust && cargo clippy --workspace -- -D warnings

# Full Rust CI
rust-ci: rust-fmt rust-clippy rust-check rust-test

# Full CI (Python + Rust)
full-ci: ci rust-ci

# Run GPT-2 benchmark (BWSK vs conventional PyTorch)
gpt2-benchmark:
    uv run python scripts/gpt2_benchmark.py
