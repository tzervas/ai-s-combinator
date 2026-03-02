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

# Run multi-model benchmark (BERT, GPT-2 Med, T5, OPT, Pythia, Pythia-1B)
multi-model-benchmark:
    uv run python scripts/multi_model_benchmark.py

# Run Burn forward pass benchmark (release build)
rust-bench:
    cd rust && cargo run --example bench_forward --release

# Run Python vs Rust cross-validation
cross-validate:
    uv run python scripts/rust_cross_validation.py

# Run extended benchmark (17 models: scale sweep + architecture diversity)
extended-benchmark:
    uv run python scripts/extended_benchmark.py

# Run convergence experiment (1500 steps, 3 seeds, statistical analysis)
convergence:
    uv run python scripts/convergence_experiment.py

# Run memory/throughput profiler
profiler:
    uv run python scripts/memory_throughput_profiler.py

# Generate whitepaper figures from experiment results
figures:
    uv run python scripts/generate_whitepaper_figures.py

# Upload results and models to HuggingFace
upload-hf:
    uv run python scripts/upload_to_huggingface.py

# Show VRAM schedule (dry run)
schedule:
    uv run python scripts/gpu_scheduler.py --schedule-only --all

# Run extended benchmark with VRAM scheduling
extended-benchmark-scheduled:
    uv run python scripts/gpu_scheduler.py --script extended_benchmark

# Run all benchmarks with VRAM scheduling
benchmark-all:
    uv run python scripts/gpu_scheduler.py --all
