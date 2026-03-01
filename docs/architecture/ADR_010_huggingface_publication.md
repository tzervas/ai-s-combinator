# ADR-010: HuggingFace Publication Strategy

**Date**: 2026-03-01
**Status**: Accepted

## Context

The BWSK framework has produced benchmark results across 17 models with classification,
training, memory, and convergence data. To maximize reproducibility and community
engagement, we want to publish both the raw data and fine-tuned model checkpoints.

## Decision

### Public repositories under `tzervas` on HuggingFace Hub:

**Dataset repos** (benchmark results as structured JSON + CSV):
- `tzervas/bwsk-benchmark-scale-sweep` — 10 transformer models, 70M→2.7B
- `tzervas/bwsk-benchmark-architecture-diversity` — CNN, ViT, SSM, MoE results
- `tzervas/bwsk-benchmark-convergence` — 1500-step convergence with 3 seeds
- `tzervas/bwsk-benchmark-combined` — All results combined

**Model repos** (fine-tuned checkpoints with model cards):
- `tzervas/bwsk-{model}-{mode}` for representative models
- 6 repos: Pythia-70M, GPT-2 Medium, ResNet-50 × {conventional, reversible}

### Naming convention
- Dataset repos: `bwsk-benchmark-{experiment-name}`
- Model repos: `bwsk-{model-slug}-{training-mode}`
- All repos tagged: `ai-s-combinator`, `bwsk`, `information-preservation`

### All repos are public
The research is open. No private repos for any benchmark data or models.

### Model cards
Auto-generated from benchmark JSON. Include: base model, training config, S/K
classification summary, memory savings, loss curves, link to paper/repo.

## Consequences

**Easier:**
- Anyone can reproduce results by downloading datasets
- Fine-tuned models enable quick comparison without re-training
- HuggingFace discovery via tags and model cards

**Harder:**
- Must maintain consistency between local JSON and uploaded data
- Model checkpoints are large (hundreds of MB each)
- Upload script must handle auth gracefully

## Alternatives Considered

1. **Weights & Biases instead of HuggingFace**: W&B is better for experiment tracking
   but HuggingFace is the standard for model/dataset sharing. Chose HF for broader reach.

2. **Private repos with access requests**: Would limit reproducibility. Open science
   is more valuable than gatekeeping.

3. **Only upload datasets, not models**: Models enable quick validation without
   re-training. The storage cost is worth the convenience.

4. **Upload to both HF and Zenodo**: Zenodo provides DOIs for citation. Could add
   later, but HuggingFace alone is sufficient for initial publication.
