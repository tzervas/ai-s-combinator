# BWSK: Combinator-Typed AI Primitives

A framework that uses combinator logic (B, W, S, K) as a **typed architectural description language** for neural networks. Combinators describe the architecture; tensors compute the results.

## Why?

~75% of transformer computation is **S-type** (information-preserving). The remaining ~25% is **K-type** (information-erasing). Making this boundary explicit gives you:

- **Automatic provenance** — trace predictions back through S-phases without SHAP/LIME
- **Memory reduction** — S-phases are reversible, no activation storage needed
- **Coordination-free distribution** — S-phases are monotone (CALM theorem)
- **Compositionality** — architectures defined as combinator expressions compose by construction

## Installation

```bash
# Requires Python 3.12+ and uv
uv sync
```

## Quick Start

```python
import bwsk

# Classify operations in an existing model
from bwsk.classify import classify_operation

# Use BWSK primitives
from bwsk.primitives import S, K, B, W
```

## Development

```bash
just ci           # Run full local CI (lint + format + test + security)
just test         # Run tests only
just lint         # Lint only
just rust-check   # Check Rust workspace
just full-ci      # Python + Rust
```

## Project Structure

```
src/bwsk/          # Python package
  classify.py      # S/K operation classifier
  primitives.py    # B, W, S, K primitive implementations
  provenance.py    # Provenance tracking
  training.py      # Training loop with BWSK analysis
rust/               # Rust workspace (scaffold)
  bwsk-core/       # Core primitives
  bwsk-burn/       # Burn ML framework integration
docs/
  research/        # Research documents
  specs/           # Formal specifications
  user-stories/    # User stories
  architecture/    # Architecture decision records
```

## Research Background

This framework emerged from research into the S combinator's computational properties. Key findings:

- Pure S is not Turing-complete (n_S monotonicity barrier)
- {B,W} IS Turing-complete (Statman 1986)
- S's "fan-out and combine" pattern maps directly to attention and residual connections
- The S/K boundary in neural networks separates reversible from irreversible computation

See `docs/research/` for full details.

## License

MIT — see [LICENSE](LICENSE)
