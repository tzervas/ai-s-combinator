# US-07: Trace Predictions Through S-Phases

**As** an AI safety researcher,
**I want** to trace a model's prediction back through S-phases to see which input tokens contributed most, without running a separate SHAP/Captum computation,
**So that** I get built-in explainability from the architecture itself.

## Acceptance Criteria

- [ ] Provenance tracker hooks into forward pass
- [ ] S-phase contributions traced input-to-output
- [ ] K-boundaries annotated with erasure description
- [ ] Output as JSON or visualization

## Spec Reference

- [SPEC-003: Provenance Tracking](../specs/SPEC_003_provenance.md)
