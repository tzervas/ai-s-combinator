# Memory & Throughput Profiling Report

**Device**: NVIDIA GeForce RTX 5080
**Date**: 2026-03-01 23:32:16

## Pythia-70M (70M params)

### Memory Breakdown (MB)

| Component | conventional | bwsk_analyzed | bwsk_reversible |
|-----------|---|---|---|
| Params | 268.7 | 268.7 | 268.7 |
| Grads | 0.0 | 0.0 | 0.0 |
| Optimizer | 537.3 | 537.3 | 537.3 |
| Activations | 3906.9 | 3903.4 | 3228.3 |
| Total Peak | 4712.9 | 4709.4 | 4034.3 |

### Timing Breakdown (seconds per step)

| Phase | conventional | bwsk_analyzed | bwsk_reversible |
|-------|---|---|---|
| Forward | 0.0212 | 0.0199 | 0.0197 |
| Backward | 0.0359 | 0.0358 | 0.0426 |
| Optimizertep | 0.0093 | 0.0090 | 0.0090 |
| Totaltep | 0.0664 | 0.0647 | 0.0713 |

### Throughput

- **conventional**: 61674 tokens/sec
- **bwsk_analyzed**: 63322 tokens/sec
- **bwsk_reversible**: 57418 tokens/sec

**Memory savings (reversible vs conventional)**: 678.6 MB (14.4%)

---

## Pythia-160M (160M params)

### Memory Breakdown (MB)

| Component | conventional | bwsk_analyzed | bwsk_reversible |
|-----------|---|---|---|
| Params | 619.2 | 619.2 | 619.2 |
| Grads | 0.0 | 0.0 | 0.0 |
| Optimizer | 1238.4 | 1238.4 | 1238.4 |
| Activations | 2695.0 | 2697.6 | 1681.9 |
| Total Peak | 4552.6 | 4555.2 | 3539.5 |

### Timing Breakdown (seconds per step)

| Phase | conventional | bwsk_analyzed | bwsk_reversible |
|-------|---|---|---|
| Forward | 0.0286 | 0.0286 | 0.0279 |
| Backward | 0.0475 | 0.0475 | 0.0620 |
| Optimizertep | 0.0214 | 0.0214 | 0.0214 |
| Totaltep | 0.0976 | 0.0975 | 0.1113 |

### Throughput

- **conventional**: 20988 tokens/sec
- **bwsk_analyzed**: 21008 tokens/sec
- **bwsk_reversible**: 18405 tokens/sec

**Memory savings (reversible vs conventional)**: 1013.1 MB (22.3%)

---

## GPT-2 Medium (345M params)

### Memory Breakdown (MB)

| Component | conventional | bwsk_analyzed | bwsk_reversible |
|-----------|---|---|---|
| Params | 1353.5 | 1353.5 | 1353.5 |
| Grads | 0.0 | 0.0 | 0.0 |
| Optimizer | 2707.1 | 2707.1 | 2707.1 |
| Activations | 3550.9 | 3550.9 | 2823.5 |
| Total Peak | 7611.5 | 7611.5 | 6884.1 |

### Timing Breakdown (seconds per step)

| Phase | conventional | bwsk_analyzed | bwsk_reversible |
|-------|---|---|---|
| Forward | 0.0221 | 0.0207 | 0.0196 |
| Backward | 0.0394 | 0.0394 | 0.0548 |
| Optimizertep | 0.0473 | 0.0474 | 0.0473 |
| Totaltep | 0.1089 | 0.1075 | 0.1217 |

### Throughput

- **conventional**: 9406 tokens/sec
- **bwsk_analyzed**: 9528 tokens/sec
- **bwsk_reversible**: 8412 tokens/sec

**Memory savings (reversible vs conventional)**: 727.4 MB (9.6%)

---

## Pythia-410M (405M params)

### Memory Breakdown (MB)

| Component | conventional | bwsk_analyzed | bwsk_reversible |
|-----------|---|---|---|
| Params | 773.1 | 773.1 | 773.1 |
| Grads | 0.0 | 0.0 | 0.0 |
| Optimizer | 1546.2 | 1546.2 | 1546.2 |
| Activations | 4197.4 | 4197.4 | 1662.7 |
| Total Peak | 6516.8 | 6516.8 | 3982.1 |

### Timing Breakdown (seconds per step)

| Phase | conventional | bwsk_analyzed | bwsk_reversible |
|-------|---|---|---|
| Forward | 0.0195 | 0.0193 | 0.0184 |
| Backward | 0.0300 | 0.0305 | 0.0430 |
| Optimizertep | 0.0290 | 0.0295 | 0.0295 |
| Totaltep | 0.0785 | 0.0793 | 0.0909 |

### Throughput

- **conventional**: 992 tokens/sec
- **bwsk_analyzed**: 998 tokens/sec
- **bwsk_reversible**: 1031 tokens/sec

**Memory savings (reversible vs conventional)**: 2534.7 MB (38.9%)

---

## Pythia-1B (1010M params)

### Memory Breakdown (MB)

| Component | conventional | bwsk_analyzed | bwsk_reversible |
|-----------|---|---|---|
| Params | 1929.8 | 1929.8 | 1929.8 |
| Grads | 0.0 | 0.0 | 0.0 |
| Optimizer | 3859.6 | 3859.6 | 3859.6 |
| Activations | 5248.1 | 5249.0 | 3927.9 |
| Total Peak | 11037.6 | 11038.4 | 9717.4 |

### Timing Breakdown (seconds per step)

| Phase | conventional | bwsk_analyzed | bwsk_reversible |
|-------|---|---|---|
| Forward | 0.0210 | 0.0210 | 0.0207 |
| Backward | 0.0333 | 0.0335 | 0.0469 |
| Optimizertep | 0.0744 | 0.0739 | 0.0744 |
| Totaltep | 0.1287 | 0.1284 | 0.1421 |

### Throughput

- **conventional**: 442 tokens/sec
- **bwsk_analyzed**: 442 tokens/sec
- **bwsk_reversible**: 443 tokens/sec

**Memory savings (reversible vs conventional)**: 1320.2 MB (12.0%)

---

## ResNet-50 (25M params)

### Memory Breakdown (MB)

| Component | conventional | bwsk_analyzed | bwsk_reversible |
|-----------|---|---|---|
| Params | 89.8 | 89.8 | 89.8 |
| Grads | 0.0 | 0.0 | 0.0 |
| Optimizer | 179.5 | 179.5 | 179.5 |
| Activations | 2712.8 | 2712.8 | 2712.8 |
| Total Peak | 2982.0 | 2982.0 | 2982.0 |

### Timing Breakdown (seconds per step)

| Phase | conventional | bwsk_analyzed | bwsk_reversible |
|-------|---|---|---|
| Forward | 0.0219 | 0.0183 | 0.0184 |
| Backward | 0.0377 | 0.0364 | 0.0365 |
| Optimizertep | 0.0032 | 0.0032 | 0.0033 |
| Totaltep | 0.0628 | 0.0579 | 0.0582 |

### Throughput

- **conventional**: 510 images/sec
- **bwsk_analyzed**: 553 images/sec
- **bwsk_reversible**: 550 images/sec

**Memory savings (reversible vs conventional)**: 0.0 MB (0.0%)

---

## Mamba-130M (130M params)

### Memory Breakdown (MB)

| Component | conventional | bwsk_analyzed | bwsk_reversible |
|-----------|---|---|---|
| Params | 492.6 | 492.6 | 492.6 |
| Grads | 0.0 | 0.0 | 0.0 |
| Optimizer | 985.2 | 985.2 | 985.2 |
| Activations | 2946.4 | 2946.4 | 1751.1 |
| Total Peak | 4424.2 | 4424.2 | 3228.9 |

### Timing Breakdown (seconds per step)

| Phase | conventional | bwsk_analyzed | bwsk_reversible |
|-------|---|---|---|
| Forward | 0.0367 | 0.0364 | 0.0365 |
| Backward | 0.0667 | 0.0636 | 0.0911 |
| Optimizertep | 0.0167 | 0.0166 | 0.0166 |
| Totaltep | 0.1201 | 0.1167 | 0.1442 |

### Throughput

- **conventional**: 17058 tokens/sec
- **bwsk_analyzed**: 17554 tokens/sec
- **bwsk_reversible**: 14203 tokens/sec

**Memory savings (reversible vs conventional)**: 1195.3 MB (27.0%)

---

## Scaling Analysis

### Peak Memory vs Model Size

| Model | Params (M) | Conv (MB) | Rev (MB) | Savings (%) |
|-------|-----------|-----------|----------|-------------|
| Pythia-70M | 70 | 4713 | 4034 | 14.4% |
| Pythia-160M | 160 | 4553 | 3540 | 22.3% |
| GPT-2 Medium | 345 | 7612 | 6884 | 9.6% |
| Pythia-410M | 405 | 6517 | 3982 | 38.9% |
| Pythia-1B | 1010 | 11038 | 9717 | 12.0% |
| ResNet-50 | 25 | 2982 | 2982 | 0.0% |
| Mamba-130M | 130 | 4424 | 3229 | 27.0% |

### Throughput vs Model Size

| Model | Params (M) | Conv (items/s) | Rev (items/s) | Overhead (%) |
|-------|-----------|----------------|---------------|--------------|
| Pythia-70M | 70 | 61674 | 57418 | 6.9% |
| Pythia-160M | 160 | 20988 | 18405 | 12.3% |
| GPT-2 Medium | 345 | 9406 | 8412 | 10.6% |
| Pythia-410M | 405 | 992 | 1031 | -4.0% |
| Pythia-1B | 1010 | 442 | 443 | -0.1% |
| ResNet-50 | 25 | 510 | 550 | -7.9% |
| Mamba-130M | 130 | 17058 | 14203 | 16.7% |
