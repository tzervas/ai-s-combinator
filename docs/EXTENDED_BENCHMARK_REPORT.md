# Extended Benchmark Report: BWSK Analysis Across 17 Models

This report covers the extended BWSK benchmark across 10 transformer models (scale sweep) and 7 non-transformer models (architecture diversity).

**Device**: NVIDIA GeForce RTX 5080
**Date**: 2026-03-01 17:19:56
**Fine-tuning steps**: 300 per mode

# Part 1: Scale Sweep (Transformers)

## BERT-base (110M, transformer/masked_lm)

**Source**: google-bert/bert-base-uncased | **Dataset**: wikitext
**Batch size**: 4, **Seq/Img len**: 512

### A. S/K Classification

| Metric | Value |
|--------|-------|
| Total leaf modules | 153 |
| S-type | 103 (67.3%) |
| K-type | 50 (32.7%) |
| GRAY | 0 (0.0%) |

### B. Evaluation (pseudo-perplexity)

| Baseline | With Provenance |
|----------|-----------------|
| 11.4833 | 11.7408 |

Provenance overhead: 0.3s

### C/D. Fine-tuning & Memory (300 steps)

| Metric | Conventional | BWSK-Analyzed | BWSK-Reversible |
|--------|-------------|---------------|-----------------|
| Final loss | 2.4500 | 2.1483 | 1.8810 |
| Eval metric | 6.7775 | 6.8438 | 6.7871 |
| Peak memory (MB) | 3214 | 3221 | 2113 |
| Wall time (s) | 22.9 | 23.0 | 28.6 |
| NaN steps | 0 | 0 | 0 |

**Memory savings**: 1101MB (34.3%)

### E. CALM Analysis
Average parallelism ratio: 0.000

---

## GPT-2 Medium (345M, transformer/causal_lm)

**Source**: openai-community/gpt2-medium | **Dataset**: wikitext
**Batch size**: 2, **Seq/Img len**: 512

### A. S/K Classification

| Metric | Value |
|--------|-------|
| Total leaf modules | 245 |
| S-type | 148 (60.4%) |
| K-type | 97 (39.6%) |
| GRAY | 0 (0.0%) |

### B. Evaluation (perplexity)

| Baseline | With Provenance |
|----------|-----------------|
| 18.4790 | 18.4790 |

Provenance overhead: 0.3s

### C/D. Fine-tuning & Memory (300 steps)

| Metric | Conventional | BWSK-Analyzed | BWSK-Reversible |
|--------|-------------|---------------|-----------------|
| Final loss | 2.6042 | 2.6093 | 2.6326 |
| Eval metric | 15.7095 | 15.6695 | 15.7146 |
| Peak memory (MB) | 7511 | 7512 | 6786 |
| Wall time (s) | 28.3 | 28.1 | 32.1 |
| NaN steps | 0 | 0 | 0 |

**Memory savings**: 725MB (9.7%)

### E. CALM Analysis
Average parallelism ratio: 0.500

---

## GPT-2 Small (124M, transformer/causal_lm)

**Source**: openai-community/gpt2 | **Dataset**: wikitext
**Batch size**: 4, **Seq/Img len**: 512

### A. S/K Classification

| Metric | Value |
|--------|-------|
| Total leaf modules | 125 |
| S-type | 76 (60.8%) |
| K-type | 49 (39.2%) |
| GRAY | 0 (0.0%) |

### B. Evaluation (perplexity)

| Baseline | With Provenance |
|----------|-----------------|
| 25.1927 | 25.1927 |

Provenance overhead: 0.1s

### C/D. Fine-tuning & Memory (300 steps)

| Metric | Conventional | BWSK-Analyzed | BWSK-Reversible |
|--------|-------------|---------------|-----------------|
| Final loss | 3.5428 | 3.5537 | 3.5661 |
| Eval metric | 19.7108 | 19.6759 | 19.7401 |
| Peak memory (MB) | 4840 | 4836 | 2709 |
| Wall time (s) | 26.4 | 26.5 | 31.6 |
| NaN steps | 0 | 0 | 0 |

**Memory savings**: 2131MB (44.0%)

### E. CALM Analysis
Average parallelism ratio: 0.500

---

## OPT-350M (331M, transformer/causal_lm)

**Source**: facebook/opt-350m | **Dataset**: wikitext
**Batch size**: 2, **Seq/Img len**: 512

### A. S/K Classification

| Metric | Value |
|--------|-------|
| Total leaf modules | 221 |
| S-type | 197 (89.1%) |
| K-type | 24 (10.9%) |
| GRAY | 0 (0.0%) |

### B. Evaluation (perplexity)

| Baseline | With Provenance |
|----------|-----------------|
| 20.8872 | 20.8872 |

Provenance overhead: 0.3s

### C/D. Fine-tuning & Memory (300 steps)

| Metric | Conventional | BWSK-Analyzed | BWSK-Reversible |
|--------|-------------|---------------|-----------------|
| Final loss | 3.7056 | 3.7162 | 3.6961 |
| Eval metric | nan | nan | nan |
| Peak memory (MB) | 5502 | 5502 | 3178 |
| Wall time (s) | 5.1 | 5.1 | 5.3 |
| NaN steps | 299 | 299 | 299 |

**Memory savings**: 2324MB (42.2%)

### E. CALM Analysis
Average parallelism ratio: 0.667

---

## Phi-2 (2700M, transformer/causal_lm)

**Source**: microsoft/phi-2 | **Dataset**: wikitext
**Batch size**: 1, **Seq/Img len**: 256

### A. S/K Classification

| Metric | Value |
|--------|-------|
| Total leaf modules | 293 |
| S-type | 228 (77.8%) |
| K-type | 65 (22.2%) |
| GRAY | 0 (0.0%) |

### B. Evaluation (perplexity)

| Baseline | With Provenance |
|----------|-----------------|
| 9.1289 | 9.1289 |

Provenance overhead: 0.5s

### C/D. Fine-tuning: SKIPPED
Reason: Classification-only mode (OOM risk)

### E. CALM Analysis
Average parallelism ratio: 0.250

---

## Pythia-160M (160M, transformer/causal_lm)

**Source**: EleutherAI/pythia-160m | **Dataset**: wikitext
**Batch size**: 4, **Seq/Img len**: 512

### A. S/K Classification

| Metric | Value |
|--------|-------|
| Total leaf modules | 113 |
| S-type | 76 (67.3%) |
| K-type | 37 (32.7%) |
| GRAY | 0 (0.0%) |

### B. Evaluation (perplexity)

| Baseline | With Provenance |
|----------|-----------------|
| 25.6522 | 25.6522 |

Provenance overhead: 0.1s

### C/D. Fine-tuning & Memory (300 steps)

| Metric | Conventional | BWSK-Analyzed | BWSK-Reversible |
|--------|-------------|---------------|-----------------|
| Final loss | 3.8312 | 3.8312 | 3.8312 |
| Eval metric | nan | nan | nan |
| Peak memory (MB) | 3507 | 3512 | 2426 |
| Wall time (s) | 2.8 | 2.8 | 2.7 |
| NaN steps | 299 | 299 | 299 |

**Memory savings**: 1081MB (30.8%)

### E. CALM Analysis
Average parallelism ratio: 0.333

---

## Pythia-1B (1010M, transformer/causal_lm)

**Source**: EleutherAI/pythia-1b | **Dataset**: wikitext
**Batch size**: 1, **Seq/Img len**: 512

### A. S/K Classification

| Metric | Value |
|--------|-------|
| Total leaf modules | 149 |
| S-type | 100 (67.1%) |
| K-type | 49 (32.9%) |
| GRAY | 0 (0.0%) |

### B. Evaluation (perplexity)

| Baseline | With Provenance |
|----------|-----------------|
| 12.5179 | 12.5179 |

Provenance overhead: 0.2s

### C/D. Fine-tuning & Memory (300 steps)

| Metric | Conventional | BWSK-Analyzed | BWSK-Reversible |
|--------|-------------|---------------|-----------------|
| Final loss | 2.6795 | 2.6795 | 2.6795 |
| Eval metric | nan | nan | nan |
| Peak memory (MB) | 10860 | 10860 | 9667 |
| Wall time (s) | 6.1 | 6.1 | 6.2 |
| NaN steps | 299 | 299 | 299 |

**Memory savings**: 1192MB (11.0%)

### E. CALM Analysis
Average parallelism ratio: 0.333

---

## Pythia-410M (405M, transformer/causal_lm)

**Source**: EleutherAI/pythia-410m | **Dataset**: wikitext
**Batch size**: 2, **Seq/Img len**: 512

### A. S/K Classification

| Metric | Value |
|--------|-------|
| Total leaf modules | 221 |
| S-type | 148 (67.0%) |
| K-type | 73 (33.0%) |
| GRAY | 0 (0.0%) |

### B. Evaluation (perplexity)

| Baseline | With Provenance |
|----------|-----------------|
| 15.0926 | 15.0926 |

Provenance overhead: 0.3s

### C/D. Fine-tuning & Memory (300 steps)

| Metric | Conventional | BWSK-Analyzed | BWSK-Reversible |
|--------|-------------|---------------|-----------------|
| Final loss | 3.3223 | 3.3223 | 3.3223 |
| Eval metric | nan | nan | nan |
| Peak memory (MB) | 6227 | 6227 | 3884 |
| Wall time (s) | 5.5 | 5.5 | 5.4 |
| NaN steps | 299 | 299 | 299 |

**Memory savings**: 2343MB (37.6%)

### E. CALM Analysis
Average parallelism ratio: 0.333

---

## Pythia-70M (70M, transformer/causal_lm)

**Source**: EleutherAI/pythia-70m | **Dataset**: wikitext
**Batch size**: 8, **Seq/Img len**: 512

### A. S/K Classification

| Metric | Value |
|--------|-------|
| Total leaf modules | 59 |
| S-type | 40 (67.8%) |
| K-type | 19 (32.2%) |
| GRAY | 0 (0.0%) |

### B. Evaluation (perplexity)

| Baseline | With Provenance |
|----------|-----------------|
| 44.6527 | 44.6527 |

Provenance overhead: 0.0s

### C/D. Fine-tuning & Memory (300 steps)

| Metric | Conventional | BWSK-Analyzed | BWSK-Reversible |
|--------|-------------|---------------|-----------------|
| Final loss | 4.3485 | 4.3485 | 4.3485 |
| Eval metric | nan | nan | nan |
| Peak memory (MB) | 3967 | 3967 | 3243 |
| Wall time (s) | 2.7 | 2.7 | 2.7 |
| NaN steps | 299 | 299 | 299 |

**Memory savings**: 724MB (18.2%)

### E. CALM Analysis
Average parallelism ratio: 0.333

---

## T5-small (60M, transformer/seq2seq)

**Source**: google-t5/t5-small | **Dataset**: wikitext
**Batch size**: 4, **Seq/Img len**: 512

### A. S/K Classification

| Metric | Value |
|--------|-------|
| Total leaf modules | 190 |
| S-type | 134 (70.5%) |
| K-type | 56 (29.5%) |
| GRAY | 0 (0.0%) |

### B. Evaluation (perplexity)

| Baseline | With Provenance |
|----------|-----------------|
| 31336.8652 | 31336.8652 |

Provenance overhead: 4.2s

### C/D. Fine-tuning & Memory (300 steps)

| Metric | Conventional | BWSK-Analyzed | BWSK-Reversible |
|--------|-------------|---------------|-----------------|
| Final loss | 4.0069 | 4.0329 | 4.0428 |
| Eval metric | 41.8872 | 41.8996 | 41.7200 |
| Peak memory (MB) | 1983 | 1985 | 1177 |
| Wall time (s) | 9.6 | 9.6 | 12.1 |
| NaN steps | 0 | 0 | 0 |

**Memory savings**: 806MB (40.6%)

### E. CALM Analysis
Average parallelism ratio: 1.000

---

# Part 2: Architecture Diversity

## EfficientNet-B0 (5M, cnn/image_cls)

**Source**: efficientnet_b0 | **Dataset**: cifar10
**Batch size**: 32, **Seq/Img len**: 224

### A. S/K Classification

| Metric | Value |
|--------|-------|
| Total leaf modules | 230 |
| S-type | 77 (33.5%) |
| K-type | 137 (59.6%) |
| GRAY | 16 (7.0%) |

### B. Evaluation (accuracy)

| Baseline | With Provenance |
|----------|-----------------|
| 0.1019 | 0.0887 |

Provenance overhead: -0.1s

### C/D. Fine-tuning & Memory (300 steps)

| Metric | Conventional | BWSK-Analyzed | BWSK-Reversible |
|--------|-------------|---------------|-----------------|
| Final loss | 1.6503 | 1.8249 | 1.8513 |
| Eval metric | 0.3519 | 0.2719 | 0.3544 |
| Peak memory (MB) | 2792 | 2792 | 2792 |
| Wall time (s) | 10.0 | 10.0 | 10.0 |
| NaN steps | 0 | 0 | 0 |

**Memory savings**: 0MB (0.0%)

### E. CALM Analysis
Average parallelism ratio: 0.111

---

## Mamba-130M (130M, ssm/ssm_lm)

**Source**: state-spaces/mamba-130m-hf | **Dataset**: wikitext
**Batch size**: 2, **Seq/Img len**: 256

### A. S/K Classification

| Metric | Value |
|--------|-------|
| Total leaf modules | 171 |
| S-type | 147 (86.0%) |
| K-type | 0 (0.0%) |
| GRAY | 24 (14.0%) |

### B. Evaluation (perplexity)

| Baseline | With Provenance |
|----------|-----------------|
| 19.5634 | 19.5634 |

Provenance overhead: 0.0s

### C/D. Fine-tuning & Memory (300 steps)

| Metric | Conventional | BWSK-Analyzed | BWSK-Reversible |
|--------|-------------|---------------|-----------------|
| Final loss | 2.6193 | 2.6193 | 2.6193 |
| Eval metric | 16.7947 | 16.7947 | 16.7947 |
| Peak memory (MB) | 2486 | 2487 | 2487 |
| Wall time (s) | 17.3 | 17.0 | 17.0 |
| NaN steps | 0 | 0 | 0 |

**Memory savings**: -2MB (-0.1%)

### E. CALM Analysis
Average parallelism ratio: 0.000

---

## Mamba-370M (370M, ssm/ssm_lm)

**Source**: state-spaces/mamba-370m-hf | **Dataset**: wikitext
**Batch size**: 1, **Seq/Img len**: 256

### A. S/K Classification

| Metric | Value |
|--------|-------|
| Total leaf modules | 339 |
| S-type | 291 (85.8%) |
| K-type | 0 (0.0%) |
| GRAY | 48 (14.2%) |

### B. Evaluation (perplexity)

| Baseline | With Provenance |
|----------|-----------------|
| 13.6040 | 13.6040 |

Provenance overhead: 0.2s

### C/D. Fine-tuning & Memory (300 steps)

| Metric | Conventional | BWSK-Analyzed | BWSK-Reversible |
|--------|-------------|---------------|-----------------|
| Final loss | 3.0559 | 3.0566 | 3.0567 |
| Eval metric | 12.2860 | 12.2871 | 12.2849 |
| Peak memory (MB) | 7102 | 7102 | 7102 |
| Wall time (s) | 37.8 | 37.4 | 36.7 |
| NaN steps | 0 | 0 | 0 |

**Memory savings**: 0MB (0.0%)

### E. CALM Analysis
Average parallelism ratio: 0.000

---

## MobileNetV2 (3M, cnn/image_cls)

**Source**: mobilenet_v2 | **Dataset**: cifar10
**Batch size**: 32, **Seq/Img len**: 224

### A. S/K Classification

| Metric | Value |
|--------|-------|
| Total leaf modules | 141 |
| S-type | 48 (34.0%) |
| K-type | 93 (66.0%) |
| GRAY | 0 (0.0%) |

### B. Evaluation (accuracy)

| Baseline | With Provenance |
|----------|-----------------|
| 0.0950 | 0.0956 |

Provenance overhead: -0.0s

### C/D. Fine-tuning & Memory (300 steps)

| Metric | Conventional | BWSK-Analyzed | BWSK-Reversible |
|--------|-------------|---------------|-----------------|
| Final loss | 1.5789 | 1.6542 | 1.7327 |
| Eval metric | 0.3350 | 0.3894 | 0.3550 |
| Peak memory (MB) | 2463 | 2463 | 2463 |
| Wall time (s) | 7.9 | 7.9 | 7.9 |
| NaN steps | 0 | 0 | 0 |

**Memory savings**: 0MB (0.0%)

### E. CALM Analysis
Average parallelism ratio: 0.947

---

## ResNet-50 (25M, cnn/image_cls)

**Source**: resnet50 | **Dataset**: cifar10
**Batch size**: 32, **Seq/Img len**: 224

### A. S/K Classification

| Metric | Value |
|--------|-------|
| Total leaf modules | 126 |
| S-type | 47 (37.3%) |
| K-type | 79 (62.7%) |
| GRAY | 0 (0.0%) |

### B. Evaluation (accuracy)

| Baseline | With Provenance |
|----------|-----------------|
| 0.1231 | 0.1113 |

Provenance overhead: -0.2s

### C/D. Fine-tuning & Memory (300 steps)

| Metric | Conventional | BWSK-Analyzed | BWSK-Reversible |
|--------|-------------|---------------|-----------------|
| Final loss | 2.0824 | 1.8902 | 2.3571 |
| Eval metric | 0.2000 | 0.3081 | 0.2550 |
| Peak memory (MB) | 2974 | 2978 | 2978 |
| Wall time (s) | 16.1 | 16.1 | 16.1 |
| NaN steps | 0 | 0 | 0 |

**Memory savings**: -4MB (-0.1%)

### E. CALM Analysis
Average parallelism ratio: 0.838

---

## Switch-Base-8 (220M, moe/seq2seq)

**Source**: google/switch-base-8 | **Dataset**: wikitext
**Batch size**: 1, **Seq/Img len**: 256

### A. S/K Classification

| Metric | Value |
|--------|-------|
| Total leaf modules | 718 |
| S-type | 378 (52.6%) |
| K-type | 278 (38.7%) |
| GRAY | 62 (8.6%) |

### B. Evaluation (perplexity)

| Baseline | With Provenance |
|----------|-----------------|
| 112118.2969 | 112118.2969 |

Provenance overhead: 7.6s

### C/D. Fine-tuning & Memory (300 steps)

| Metric | Conventional | BWSK-Analyzed | BWSK-Reversible |
|--------|-------------|---------------|-----------------|
| Final loss | 4.6595 | 4.5476 | 4.5059 |
| Eval metric | 97.1412 | 93.1430 | 97.0109 |
| Peak memory (MB) | 12869 | 12867 | 12918 |
| Wall time (s) | 42.6 | 41.2 | 54.8 |
| NaN steps | 0 | 0 | 0 |

**Memory savings**: -49MB (-0.4%)

### E. CALM Analysis
Average parallelism ratio: 1.000

---

## ViT-base (86M, vit/image_cls)

**Source**: google/vit-base-patch16-224 | **Dataset**: cifar10
**Batch size**: 16, **Seq/Img len**: 224

### A. S/K Classification

| Metric | Value |
|--------|-------|
| Total leaf modules | 136 |
| S-type | 98 (72.1%) |
| K-type | 38 (27.9%) |
| GRAY | 0 (0.0%) |

### B. Evaluation (accuracy)

| Baseline | With Provenance |
|----------|-----------------|
| 0.0544 | 0.0537 |

Provenance overhead: 0.5s

### C/D. Fine-tuning & Memory (300 steps)

| Metric | Conventional | BWSK-Analyzed | BWSK-Reversible |
|--------|-------------|---------------|-----------------|
| Final loss | 0.0115 | 0.0142 | 0.0310 |
| Eval metric | 0.9788 | 0.9681 | 0.9694 |
| Peak memory (MB) | 4712 | 4713 | 1718 |
| Wall time (s) | 69.5 | 65.2 | 76.9 |
| NaN steps | 0 | 0 | 0 |

**Memory savings**: 2995MB (63.6%)

### E. CALM Analysis
Average parallelism ratio: 0.400

---

## Cross-Model Comparison

### S/K Ratios by Model

| Model | Family | Params (M) | S-ratio | K-ratio | GRAY-ratio |
|-------|--------|-----------|---------|---------|------------|
| BERT-base | transformer | 110 | 0.673 | 0.327 | 0.000 |
| EfficientNet-B0 | cnn | 5 | 0.335 | 0.596 | 0.070 |
| GPT-2 Medium | transformer | 345 | 0.604 | 0.396 | 0.000 |
| GPT-2 Small | transformer | 124 | 0.608 | 0.392 | 0.000 |
| Mamba-130M | ssm | 130 | 0.860 | 0.000 | 0.140 |
| Mamba-370M | ssm | 370 | 0.858 | 0.000 | 0.142 |
| MobileNetV2 | cnn | 3 | 0.340 | 0.660 | 0.000 |
| OPT-350M | transformer | 331 | 0.891 | 0.109 | 0.000 |
| Phi-2 | transformer | 2700 | 0.778 | 0.222 | 0.000 |
| Pythia-160M | transformer | 160 | 0.673 | 0.327 | 0.000 |
| Pythia-1B | transformer | 1010 | 0.671 | 0.329 | 0.000 |
| Pythia-410M | transformer | 405 | 0.670 | 0.330 | 0.000 |
| Pythia-70M | transformer | 70 | 0.678 | 0.322 | 0.000 |
| ResNet-50 | cnn | 25 | 0.373 | 0.627 | 0.000 |
| Switch-Base-8 | moe | 220 | 0.526 | 0.387 | 0.086 |
| T5-small | transformer | 60 | 0.705 | 0.295 | 0.000 |
| ViT-base | vit | 86 | 0.721 | 0.279 | 0.000 |

### Memory Comparison

| Model | Conventional (MB) | Reversible (MB) | Savings (%) |
|-------|-------------------|-----------------|-------------|
| BERT-base | 3214 | 2113 | 34.3% |
| EfficientNet-B0 | 2792 | 2792 | 0.0% |
| GPT-2 Medium | 7511 | 6786 | 9.7% |
| GPT-2 Small | 4840 | 2709 | 44.0% |
| Mamba-130M | 2486 | 2487 | -0.1% |
| Mamba-370M | 7102 | 7102 | 0.0% |
| MobileNetV2 | 2463 | 2463 | 0.0% |
| OPT-350M | 5502 | 3178 | 42.2% |
| Pythia-160M | 3507 | 2426 | 30.8% |
| Pythia-1B | 10860 | 9667 | 11.0% |
| Pythia-410M | 6227 | 3884 | 37.6% |
| Pythia-70M | 3967 | 3243 | 18.2% |
| ResNet-50 | 2974 | 2978 | -0.1% |
| Switch-Base-8 | 12869 | 12918 | -0.4% |
| T5-small | 1983 | 1177 | 40.6% |
| ViT-base | 4712 | 1718 | 63.6% |

### CALM Parallelism by Architecture

| Model | Family | Avg Parallelism Ratio |
|-------|--------|-----------------------|
| BERT-base | transformer | 0.000 |
| EfficientNet-B0 | cnn | 0.111 |
| GPT-2 Medium | transformer | 0.500 |
| GPT-2 Small | transformer | 0.500 |
| Mamba-130M | ssm | 0.000 |
| Mamba-370M | ssm | 0.000 |
| MobileNetV2 | cnn | 0.947 |
| OPT-350M | transformer | 0.667 |
| Phi-2 | transformer | 0.250 |
| Pythia-160M | transformer | 0.333 |
| Pythia-1B | transformer | 0.333 |
| Pythia-410M | transformer | 0.333 |
| Pythia-70M | transformer | 0.333 |
| ResNet-50 | cnn | 0.838 |
| Switch-Base-8 | moe | 1.000 |
| T5-small | transformer | 1.000 |
| ViT-base | vit | 0.400 |
