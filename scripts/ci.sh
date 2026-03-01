#!/usr/bin/env bash
set -euo pipefail

echo "=== BWSK Local CI ==="

echo "--- Lint ---"
uv run ruff check src/ tests/

echo "--- Format Check ---"
uv run ruff format --check src/ tests/

echo "--- Tests ---"
uv run pytest

echo "--- Type Check ---"
if command -v pyright &> /dev/null || uv run pyright --version &> /dev/null 2>&1; then
    uv run pyright src/
else
    echo "pyright not available, skipping type check"
fi

echo "--- Security Scan ---"
bash scripts/security-scan.sh

echo "=== All checks passed ==="
