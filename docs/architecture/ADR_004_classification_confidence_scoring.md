# ADR-004: Classification Confidence Scoring
**Date**: 2026-02-28
**Status**: Accepted

## Context

Not all S/K classifications are equally certain. nn.ReLU is unambiguously K-type (maps all negatives to zero). nn.Linear is S-type only when the weight matrix is full-rank, which we don't check at classification time. Softmax is genuinely ambiguous (Gray). We need a way to communicate classification certainty.

## Decision

Each classification includes a confidence score (float 0.0-1.0) following these rules:

- **1.0 (definite)**: Operation is in the default classification database with no caveats. Examples: nn.ReLU → K(1.0), nn.Dropout → K(1.0), residual add → S(1.0), concatenation → S(1.0).
- **0.8 (high)**: Operation is classified by structural analysis with reasonable assumptions. Examples: nn.Linear → S(0.8) because we assume full-rank weights but don't verify.
- **0.5 (medium)**: Operation has context-dependent classification. Examples: nn.BatchNorm → GRAY(0.5), softmax → GRAY(0.5).
- **0.3 (low)**: Operation classified by heuristic or fallback. Examples: unknown custom module → GRAY(0.3).
- **User overrides**: Always confidence 1.0 (user explicitly decided).

The confidence score is informational — it does not change the classification itself. It helps users identify which classifications they should manually verify.

## Consequences

Easier:
- Users know which parts of the report to trust.
- Automated tools can filter by confidence threshold.
- Compliance audits can focus on low-confidence operations.

Harder:
- Must maintain confidence levels alongside classifications.
- More complex classification result type.

## Alternatives Considered

1. **Binary certain/uncertain**: Too coarse — doesn't distinguish "probably S" from "genuinely ambiguous."
2. **No confidence scores**: Simpler but forces users to know which ops are ambiguous. Bad UX for non-experts.
3. **Probabilistic classification (Bayesian)**: Overkill for static analysis. We're not doing statistical inference — we're applying known rules.
