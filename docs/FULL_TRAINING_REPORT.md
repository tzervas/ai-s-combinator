# Full Training Pipeline Report

Epoch-based training to convergence with early stopping.

**Date**: 2026-03-03 22:58:21
**Device**: NVIDIA GeForce RTX 3090 Ti

## Pythia-1B (1010M, transformer)

S-ratio: 0.671, K-ratio: 0.329

### Finetune

| Mode | Best Val | Test | Train Loss | Epochs | Steps | Time (s) | Memory (MB) |
|------|---------|------|-----------|--------|-------|----------|-------------|
| conventional | 11.0050 | 0.0000 | 2.6745 | 2 | 1186 | 1917 | 21209 |
| bwsk_analyzed | 10.9944 | 0.0000 | 2.3196 | 2 | 1186 | 1917 | 21209 |
| bwsk_reversible | 10.9773 | 0.0000 | 2.2676 | 2 | 1186 | 1917 | 21209 |

### Scratch

| Mode | Best Val | Test | Train Loss | Epochs | Steps | Time (s) | Memory (MB) |
|------|---------|------|-----------|--------|-------|----------|-------------|
| conventional | 205.6999 | 0.0000 | 4.3119 | 3 | 1779 | 3022 | 21209 |
| bwsk_analyzed | 204.0505 | 0.0000 | 3.9609 | 3 | 1779 | 3018 | 21209 |
| bwsk_reversible | 204.1432 | 0.0000 | 4.7875 | 3 | 1779 | 3016 | 21209 |

---

## Switch-Base-8 (220M, moe)

S-ratio: 0.526, K-ratio: 0.387

### Finetune

| Mode | Best Val | Test | Train Loss | Epochs | Steps | Time (s) | Memory (MB) |
|------|---------|------|-----------|--------|-------|----------|-------------|
| conventional | 29.0179 | 27.7215 | 2.9923 | 5 | 13850 | 5363 | 15563 |
| bwsk_analyzed | 29.9891 | 28.6584 | 3.1352 | 4 | 10000 | 6458 | 15563 |
| bwsk_reversible | 29.2447 | 27.9624 | 3.2770 | 5 | 13850 | 8953 | 15563 |

### Scratch

| Mode | Best Val | Test | Train Loss | Epochs | Steps | Time (s) | Memory (MB) |
|------|---------|------|-----------|--------|-------|----------|-------------|
| conventional | 289.2626 | 290.6109 | 5.5342 | 5 | 13850 | 6347 | 14527 |
| bwsk_analyzed | 288.6744 | 288.1153 | 5.2518 | 5 | 13850 | 6491 | 14491 |
| bwsk_reversible | 297.6718 | 299.3535 | 5.0745 | 5 | 13850 | 6378 | 14455 |

---

## Cross-Model Summary

| Model | Params (M) | S-ratio | Best Mode (FT) | Best Val (FT) | Best Mode (Scratch) | Best Val (Scratch) |
|-------|-----------|---------|----------------|---------------|---------------------|-------------------|
| Pythia-1B | 1010 | 0.671 | bwsk_reversible | 10.9773 | bwsk_analyzed | 204.0505 |
| Switch-Base-8 | 220 | 0.526 | conventional | 29.0179 | bwsk_analyzed | 288.6744 |
