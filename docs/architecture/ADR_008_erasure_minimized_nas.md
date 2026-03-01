# ADR-008: Erasure-Minimized Neural Architecture Search

## Status
Accepted

## Context
Standard NAS optimizes for accuracy and latency. Our framework classifies operations as S-type (information-preserving) or K-type (information-erasing). We need a NAS approach that additionally optimizes for minimal information erasure, finding architectures that maintain performance while pushing the S-type fraction higher.

## Decision

### Gene encoding
An architecture is encoded as `ArchitectureGene`: a list of operation names plus dimension parameters. The first and last operations are always `Linear` to ensure correct input/output dimensions. Interior positions can be any S-type or K-type operation.

### Search space
- **S-type ops**: Linear, LeakyReLU, Softplus, LayerNorm, Identity
- **K-type ops**: ReLU, GELU, Sigmoid, Dropout
- Architecture: variable-depth sequential models with operator choices per position

### Multi-objective optimization
Two objectives optimized simultaneously:
1. **Minimize erasure score** (fraction of K-type ops) — primary
2. **Maximize task accuracy** (proxy via training loss) — secondary

Results are ranked using Pareto frontier computation rather than a single scalar objective, preserving the full accuracy-erasure tradeoff for the user to inspect.

### Search algorithms
1. **Random search** (baseline): generates N random architectures, evaluates each, returns Pareto frontier
2. **Evolutionary search**: tournament selection with single-point mutation, breeds toward the Pareto frontier of accuracy vs. erasure

### Evaluation
Each architecture is built, classified via `classify_model()`, and briefly trained on synthetic data. The erasure score comes from the classifier; the accuracy proxy comes from `1 / (1 + final_loss)` after a few training steps.

## Alternatives Considered

1. **Reinforcement learning controller** (NASNet-style): Rejected for Phase 5 scope. RL controllers add significant complexity (policy network, reward shaping) for uncertain benefit in this small search space. Evolutionary search is simpler and well-suited to multi-objective optimization.

2. **Differentiable NAS** (DARTS-style): Rejected because DARTS relaxes the search space into continuous weights, making it hard to enforce discrete S/K classification constraints during search. Our discrete gene encoding naturally respects S/K boundaries.

3. **Single scalar objective** (weighted sum of accuracy and erasure): Rejected in favor of Pareto frontier. A weighted sum forces the user to choose the tradeoff weight upfront, while Pareto analysis presents all non-dominated solutions for post-hoc selection.
