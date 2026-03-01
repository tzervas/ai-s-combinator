# ADR-009: Extended Benchmarks — 17 Models Across 4 Experiment Types

**Date**: 2026-03-01
**Status**: Accepted

## Context

The initial BWSK benchmarks covered 6 transformer models (BERT, GPT-2 Med, T5, OPT,
Pythia-410M, Pythia-1B) on WikiText-2. While this validated the framework across encoder,
decoder, and encoder-decoder architectures, it left several questions unanswered:

1. How does S/K ratio scale with model size within the same architecture family?
2. Do non-transformer architectures (CNNs, SSMs, MoE) have fundamentally different S/K profiles?
3. Is BWSK-reversible training statistically equivalent to conventional training?
4. How do memory savings and throughput overhead scale with model size?

Answering these requires a broader benchmark suite with statistical rigor.

## Decision

We extend the benchmark to 17 models across 4 experiment types:

### Scale Sweep (10 transformer models, 70M→2.7B)
Pythia-70M, T5-small, BERT-base, GPT-2 Small, Pythia-160M, OPT-350M, GPT-2 Medium,
Pythia-410M, Pythia-1B, Phi-2. The Pythia family (70M→1B) enables within-family
size-dependent analysis. Cross-family models test universality.

### Architecture Diversity (7 non-transformer models)
- **CNNs**: ResNet-50 (25M), EfficientNet-B0 (5M), MobileNetV2 (3.4M) on CIFAR-10
- **ViT**: ViT-base (86M) on CIFAR-10
- **SSM/Mamba**: Mamba-130M, Mamba-370M on WikiText-2
- **MoE**: Switch-Base-8 (220M) on WikiText-2

### Convergence (1500 steps × 3 seeds × 4 models)
Pythia-70M, GPT-2 Medium, Mamba-130M, ResNet-50. Three random seeds (42, 123, 456)
enable paired t-tests and 95% confidence intervals.

### Memory/Throughput Profiling
All 17 models with detailed memory breakdown (params, grads, optimizer, activations)
and forward/backward timing.

### Why these specific models?

- **Pythia family**: Same architecture, different sizes → isolates size effect
- **ResNet/EfficientNet/MobileNet**: Diverse CNN designs with different K-type profiles
- **Mamba**: State-space models — a fundamentally different information flow pattern
- **Switch-Base-8**: MoE routing as an extreme K-type operation (argmax selection)
- **Phi-2**: Largest model to test scaling limits on consumer GPU (16GB)

### Why 4 experiment types?

Each answers a different hypothesis:
- Scale sweep → H1 (75% hypothesis), H2 (size-independence)
- Architecture diversity → H5 (CNN vs transformer), H6 (MoE routing)
- Convergence → H3 (statistical equivalence)
- Memory/throughput → H4 (memory savings)

## Consequences

**Easier:**
- Statistical claims about BWSK-reversible training quality
- Cross-architecture S/K profile comparison
- Publication-quality figures with error bars

**Harder:**
- Full experiment suite takes hours on a single GPU
- Phi-2 (2.7B) may require classification-only mode without fine-tuning
- mamba-ssm dependency is optional and may not install cleanly

## Alternatives Considered

1. **More models, fewer experiments**: Could benchmark 50+ models with classification-only.
   Rejected because training comparison is the key claim.

2. **Fewer seeds**: 2 seeds instead of 3. Rejected because 3 is the minimum for
   meaningful confidence intervals.

3. **Longer training (5000 steps)**: More steps would give better convergence data
   but would make the full suite impractical on consumer hardware. 1500 steps is a
   pragmatic compromise.

4. **Include RWKV**: RWKV is another interesting architecture but lacks stable
   HuggingFace integration. Deferred to future work.
