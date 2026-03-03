# BWSK: Combinator-Typed AI Primitives

A framework that uses combinator logic (B, W, S, K) as a **typed architectural description language** for neural networks. Combinators describe the architecture; tensors compute the results.

**Paper**: [docs/WHITEPAPER.md](docs/WHITEPAPER.md) | **Results**: [docs/FULL_TRAINING_REPORT.md](docs/FULL_TRAINING_REPORT.md) | **Trained Models**: [HuggingFace Collection](#huggingface-models)

## Why?

Every neural network operation is either **information-preserving** (S-type) or **information-erasing** (K-type). Making this boundary explicit gives you:

- **Automatic provenance** — trace predictions back through S-phases without SHAP/LIME
- **Memory reduction** — S-phases are reversible, no activation storage needed (up to 44% savings)
- **Coordination-free distribution** — S-phases are monotone (CALM theorem)
- **Compositionality** — architectures defined as combinator expressions compose by construction

## Key Results

Validated across **17 models** spanning 5 architecture families on an NVIDIA RTX 5080. **72 trained models** published to [HuggingFace](#huggingface-models).

| Family | Models | S-type Ratio | Memory Savings (Reversible) |
|--------|--------|-------------|----------------------------|
| SSM/Mamba | Mamba-130M, 370M | **86%** (0% K-type) | 0% (GRAY-interrupted) |
| Transformer | GPT-2, Pythia, BERT, T5 | **60-89%** | 16-37% |
| Vision Transformer | ViT-base | **72%** | 37% |
| Mixture-of-Experts | Switch-Base-8 | **53%** | N/A (16GB VRAM limit) |
| CNN | ResNet-50, EfficientNet, MobileNet | **33-37%** | ~0% |

**Statistical equivalence**: All BWSK training modes produce equivalent convergence for fine-tuning (<1% delta for LMs, p > 0.05 across 36 statistical runs). 72 models trained to convergence with epoch-based early stopping.

## The BWSK Primitives

| Combinator | Rule | Neural Network Mapping |
|------------|------|----------------------|
| **B** (compose) | `B f g x = f(g(x))` | Sequential layer stacking |
| **W** (share) | `W f x = f(x)(x)` | Weight sharing, self-attention |
| **S** (fan-out) | `S f g x = f(x)(g(x))` | Multi-head attention, residual connections |
| **K** (erase) | `K x y = x` | Masking, dropout, pooling, activation clipping |

## Installation

```bash
# Requires Python 3.12+ and uv
uv sync
```

## Quick Start

```python
from bwsk.classify import classify_operation, classify_model, OpClass
from bwsk.primitives import S, K, B, W

# Classify a single operation
import torch.nn as nn
result = classify_operation(nn.ReLU())
print(result)  # K-type, confidence=0.5, "Zeroes negative values..."

# Classify an entire model
from transformers import AutoModel
model = AutoModel.from_pretrained("EleutherAI/pythia-70m")
summary = classify_model(model)
print(f"S-type: {summary.s_ratio:.1%}, K-type: {summary.k_ratio:.1%}")
# S-type: 67.8%, K-type: 32.2%

# BWSK-aware training with reversible backprop
from bwsk.training import BWSKTrainer
trainer = BWSKTrainer(model, mode="bwsk_reversible")
trainer.train(dataloader)  # Up to 44% less VRAM
```

## Benchmarks and Training

```bash
# Quick benchmarks (300 steps)
just extended-benchmark       # 17 models, S/K classification + training
just convergence              # 4 models x 3 seeds, statistical analysis

# Full convergence training (epoch-based, early stopping)
just full-train               # All 16 models x 3 modes x 2 experiments
just full-train-finetune      # Fine-tune only
just full-train-scratch       # From-scratch only

# Profiling and analysis
just profiler                 # Memory/throughput breakdown
just figures                  # Generate whitepaper figures
just schedule                 # Show VRAM schedule (dry run)
```

## Development

```bash
just ci           # Full Python CI (lint + format + test + security)
just test         # Run tests only (200+ tests)
just lint         # Ruff linter
just rust-ci      # Full Rust CI (fmt + clippy + check + test)
just full-ci      # Python + Rust
```

## Project Structure

```
src/bwsk/              # Python package
  classify.py          # S/K operation classifier (70+ ops, torch.fx)
  primitives.py        # B, W, S, K combinators (pure + nn.Module)
  provenance.py        # Provenance tracking (forward hooks)
  training.py          # BWSK-aware training with memory optimization
  reversible.py        # Reversible backprop via S-phase checkpointing
  calm.py              # CALM monotone analysis + distribution
  nas.py               # Erasure-minimized Neural Architecture Search

rust/                  # Rust workspace (exact classification parity)
  bwsk-core/           # Core primitives, classifier, provenance
  bwsk-burn/           # Burn 0.20 integration (CubeCL GPU backend)

scripts/               # Benchmark and training infrastructure
  extended_benchmark.py       # 17-model benchmark suite
  full_training_pipeline.py   # Full convergence training pipeline
  training_utils.py           # Shared training utilities
  convergence_experiment.py   # Statistical convergence analysis
  memory_throughput_profiler.py
  gpu_scheduler.py            # VRAM-aware bin-packing scheduler
  generate_whitepaper_figures.py

docs/
  WHITEPAPER.md        # Full research paper with results
  FULL_TRAINING_REPORT.md  # Full training results report
  research/            # Background research documents
  specs/               # Formal specifications (SPEC-001 to 005)
  user-stories/        # User stories (US-01 to US-15)
  architecture/        # Architecture decision records (ADR-001 to 010)
  figures/             # 8 publication-quality figures
```

## HuggingFace Models

All trained models are published to HuggingFace under the `tzervas` organization. Each model is trained in three BWSK modes (conventional, bwsk_analyzed, bwsk_reversible) across two experiments (fine-tune, from-scratch).

**72 trained models** across 12 architectures, all publicly available. Each model is trained in three BWSK modes across two experiments. Models with NaN training issues (OPT-350M, Pythia-410M, Pythia-1B conv/analyzed) and Switch-Base-8 (16GB VRAM exceeded) are excluded.

**Naming convention**: `tzervas/bwsk-{model}-{experiment}-{mode}`

| Model | Family | Params | Fine-tune Models | From-Scratch Models | Results |
|-------|--------|--------|-----------------|--------------------| --------|
| Pythia-70M | Transformer | 70M | [conv](https://huggingface.co/tzervas/bwsk-pythia-70m-finetune-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-pythia-70m-finetune-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-pythia-70m-finetune-bwsk_reversible) | [conv](https://huggingface.co/tzervas/bwsk-pythia-70m-scratch-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-pythia-70m-scratch-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-pythia-70m-scratch-bwsk_reversible) | [JSON](https://huggingface.co/tzervas/bwsk-pythia-70m-full-training-results) |
| Pythia-160M | Transformer | 160M | [conv](https://huggingface.co/tzervas/bwsk-pythia-160m-finetune-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-pythia-160m-finetune-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-pythia-160m-finetune-bwsk_reversible) | [conv](https://huggingface.co/tzervas/bwsk-pythia-160m-scratch-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-pythia-160m-scratch-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-pythia-160m-scratch-bwsk_reversible) | [JSON](https://huggingface.co/tzervas/bwsk-pythia-160m-full-training-results) |
| Pythia-410M | Transformer | 410M | [conv](https://huggingface.co/tzervas/bwsk-pythia-410m-finetune-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-pythia-410m-finetune-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-pythia-410m-finetune-bwsk_reversible) | [conv](https://huggingface.co/tzervas/bwsk-pythia-410m-scratch-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-pythia-410m-scratch-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-pythia-410m-scratch-bwsk_reversible) | [JSON](https://huggingface.co/tzervas/bwsk-pythia-410m-full-training-results) |
| Pythia-1B | Transformer | 1B | [conv](https://huggingface.co/tzervas/bwsk-pythia-1b-finetune-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-pythia-1b-finetune-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-pythia-1b-finetune-bwsk_reversible) | [conv](https://huggingface.co/tzervas/bwsk-pythia-1b-scratch-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-pythia-1b-scratch-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-pythia-1b-scratch-bwsk_reversible) | [JSON](https://huggingface.co/tzervas/bwsk-pythia-1b-full-training-results) |
| GPT-2 Small | Transformer | 124M | [conv](https://huggingface.co/tzervas/bwsk-gpt2-small-finetune-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-gpt2-small-finetune-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-gpt2-small-finetune-bwsk_reversible) | [conv](https://huggingface.co/tzervas/bwsk-gpt2-small-scratch-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-gpt2-small-scratch-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-gpt2-small-scratch-bwsk_reversible) | [JSON](https://huggingface.co/tzervas/bwsk-gpt2-small-full-training-results) |
| GPT-2 Medium | Transformer | 345M | [conv](https://huggingface.co/tzervas/bwsk-gpt2-medium-finetune-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-gpt2-medium-finetune-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-gpt2-medium-finetune-bwsk_reversible) | [conv](https://huggingface.co/tzervas/bwsk-gpt2-medium-scratch-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-gpt2-medium-scratch-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-gpt2-medium-scratch-bwsk_reversible) | [JSON](https://huggingface.co/tzervas/bwsk-gpt2-medium-full-training-results) |
| BERT-base | Transformer | 110M | [conv](https://huggingface.co/tzervas/bwsk-bert-base-finetune-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-bert-base-finetune-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-bert-base-finetune-bwsk_reversible) | [conv](https://huggingface.co/tzervas/bwsk-bert-base-scratch-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-bert-base-scratch-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-bert-base-scratch-bwsk_reversible) | [JSON](https://huggingface.co/tzervas/bwsk-bert-base-full-training-results) |
| OPT-125M | Transformer | 125M | [conv](https://huggingface.co/tzervas/bwsk-opt-125m-finetune-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-opt-125m-finetune-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-opt-125m-finetune-bwsk_reversible) | [conv](https://huggingface.co/tzervas/bwsk-opt-125m-scratch-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-opt-125m-scratch-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-opt-125m-scratch-bwsk_reversible) | [JSON](https://huggingface.co/tzervas/bwsk-opt-125m-full-training-results) |
| OPT-350M | Transformer | 331M | [conv](https://huggingface.co/tzervas/bwsk-opt-350m-finetune-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-opt-350m-finetune-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-opt-350m-finetune-bwsk_reversible) | [conv](https://huggingface.co/tzervas/bwsk-opt-350m-scratch-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-opt-350m-scratch-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-opt-350m-scratch-bwsk_reversible) | [JSON](https://huggingface.co/tzervas/bwsk-opt-350m-full-training-results) |
| T5-small | Transformer | 60M | [conv](https://huggingface.co/tzervas/bwsk-t5-small-finetune-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-t5-small-finetune-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-t5-small-finetune-bwsk_reversible) | [conv](https://huggingface.co/tzervas/bwsk-t5-small-scratch-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-t5-small-scratch-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-t5-small-scratch-bwsk_reversible) | [JSON](https://huggingface.co/tzervas/bwsk-t5-small-full-training-results) |
| ResNet-50 | CNN | 25M | [conv](https://huggingface.co/tzervas/bwsk-resnet50-finetune-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-resnet50-finetune-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-resnet50-finetune-bwsk_reversible) | [conv](https://huggingface.co/tzervas/bwsk-resnet50-scratch-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-resnet50-scratch-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-resnet50-scratch-bwsk_reversible) | [JSON](https://huggingface.co/tzervas/bwsk-resnet50-full-training-results) |
| EfficientNet-B0 | CNN | 5M | [conv](https://huggingface.co/tzervas/bwsk-efficientnet-b0-finetune-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-efficientnet-b0-finetune-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-efficientnet-b0-finetune-bwsk_reversible) | [conv](https://huggingface.co/tzervas/bwsk-efficientnet-b0-scratch-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-efficientnet-b0-scratch-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-efficientnet-b0-scratch-bwsk_reversible) | [JSON](https://huggingface.co/tzervas/bwsk-efficientnet-b0-full-training-results) |
| MobileNetV2 | CNN | 3M | [conv](https://huggingface.co/tzervas/bwsk-mobilenetv2-finetune-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-mobilenetv2-finetune-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-mobilenetv2-finetune-bwsk_reversible) | [conv](https://huggingface.co/tzervas/bwsk-mobilenetv2-scratch-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-mobilenetv2-scratch-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-mobilenetv2-scratch-bwsk_reversible) | [JSON](https://huggingface.co/tzervas/bwsk-mobilenetv2-full-training-results) |
| ViT-base | Vision Transformer | 86M | [conv](https://huggingface.co/tzervas/bwsk-vit-base-finetune-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-vit-base-finetune-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-vit-base-finetune-bwsk_reversible) | [conv](https://huggingface.co/tzervas/bwsk-vit-base-scratch-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-vit-base-scratch-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-vit-base-scratch-bwsk_reversible) | [JSON](https://huggingface.co/tzervas/bwsk-vit-base-full-training-results) |
| Mamba-130M | SSM | 130M | [conv](https://huggingface.co/tzervas/bwsk-mamba-130m-finetune-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-mamba-130m-finetune-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-mamba-130m-finetune-bwsk_reversible) | [conv](https://huggingface.co/tzervas/bwsk-mamba-130m-scratch-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-mamba-130m-scratch-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-mamba-130m-scratch-bwsk_reversible) | [JSON](https://huggingface.co/tzervas/bwsk-mamba-130m-full-training-results) |
| Mamba-370M | SSM | 370M | [conv](https://huggingface.co/tzervas/bwsk-mamba-370m-finetune-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-mamba-370m-finetune-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-mamba-370m-finetune-bwsk_reversible) | [conv](https://huggingface.co/tzervas/bwsk-mamba-370m-scratch-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-mamba-370m-scratch-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-mamba-370m-scratch-bwsk_reversible) | [JSON](https://huggingface.co/tzervas/bwsk-mamba-370m-full-training-results) |
| Switch-Base-8 | MoE | 220M | [conv](https://huggingface.co/tzervas/bwsk-switch-base-8-finetune-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-switch-base-8-finetune-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-switch-base-8-finetune-bwsk_reversible) | [conv](https://huggingface.co/tzervas/bwsk-switch-base-8-scratch-conventional) / [analyzed](https://huggingface.co/tzervas/bwsk-switch-base-8-scratch-bwsk_analyzed) / [reversible](https://huggingface.co/tzervas/bwsk-switch-base-8-scratch-bwsk_reversible) | [JSON](https://huggingface.co/tzervas/bwsk-switch-base-8-full-training-results) |

## Research Background

This framework emerged from research into the S combinator's computational properties:

- **Pure S is not Turing-complete** (Waldmann 1998, n_S monotonicity barrier)
- **{B,W} IS Turing-complete** (Statman 1986): `theta_4 = B(WW)(BW(BBB))`
- S's "fan-out and combine" pattern maps directly to **attention and residual connections**
- The **S/K boundary** separates reversible from irreversible computation
- **CALM theorem**: S-type operations are monotone = coordination-free distributed execution

See [docs/WHITEPAPER.md](docs/WHITEPAPER.md) for the full paper and [docs/research/](docs/research/) for background.

## License

MIT — see [LICENSE](LICENSE)
