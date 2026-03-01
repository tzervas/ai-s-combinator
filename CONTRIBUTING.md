# Contributing to BWSK

## Development Setup

```bash
git clone git@github.com:tzervas/ai-s-combinator.git
cd ai-s-combinator
uv sync --all-extras
```

## Workflow

1. Create a feature branch from `main`
2. Write tests first (TDD)
3. Implement the feature
4. Run `just ci` — all checks must pass
5. Open a PR using the template

## Code Standards

### Python
- Python 3.12+, type hints on all public APIs
- `ruff` for linting and formatting
- Google-style docstrings
- One test file per source module in `tests/`

### Rust
- Edition 2021, `cargo fmt` + `cargo clippy -D warnings`
- Tests in `#[cfg(test)]` modules

## PR Requirements

- All CI checks pass (`just ci`)
- Tests cover new functionality
- Specs updated if API changes
- GPG-signed commits
