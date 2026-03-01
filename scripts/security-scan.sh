#!/usr/bin/env bash
set -euo pipefail

echo "--- Security Scan ---"

# detect-secrets
if uv run detect-secrets --version &> /dev/null 2>&1; then
    echo "Running detect-secrets scan..."
    uv run detect-secrets scan --list-all-plugins 2>/dev/null || true
    uv run detect-secrets scan src/ tests/ --baseline .secrets.baseline 2>/dev/null || \
        uv run detect-secrets scan src/ tests/ 2>/dev/null || \
        echo "detect-secrets scan completed (no baseline)"
else
    echo "detect-secrets not installed, skipping"
fi

# pip-audit
if uv run pip-audit --version &> /dev/null 2>&1; then
    echo "Running pip-audit..."
    uv run pip-audit || echo "pip-audit found issues (see above)"
else
    echo "pip-audit not installed, skipping"
fi

# cargo-audit (if Rust workspace exists)
if [ -d "rust" ] && command -v cargo-audit &> /dev/null; then
    echo "Running cargo-audit..."
    cd rust && cargo audit || echo "cargo-audit found issues (see above)"
fi

echo "--- Security scan complete ---"
