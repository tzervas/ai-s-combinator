# Convergence Experiment Report

**Steps**: 1500 | **Seeds**: [42, 123, 456] | **Modes**: conventional, bwsk_analyzed, bwsk_reversible
**Log interval**: every 100 steps

## Mamba-130M

### Training Runs

| Mode | Seed | Final Loss | Wall Time (s) | Peak Mem (MB) | 90% Conv Step | NaN |
|------|------|-----------|---------------|---------------|---------------|-----|
| conventional | 42 | 2.8779 | 97.9 | 2583 | 24 | 0 |
| conventional | 123 | 2.8781 | 103.4 | 2584 | 24 | 0 |
| conventional | 456 | 2.8779 | 103.6 | 2584 | 24 | 0 |
| bwsk_analyzed | 42 | 2.8777 | 103.3 | 2584 | 24 | 0 |
| bwsk_analyzed | 123 | 2.8780 | 103.5 | 2584 | 24 | 0 |
| bwsk_analyzed | 456 | 2.8778 | 103.7 | 2584 | 24 | 0 |
| bwsk_reversible | 42 | 2.8778 | 103.5 | 2584 | 24 | 0 |
| bwsk_reversible | 123 | 2.8780 | 104.3 | 2584 | 24 | 0 |
| bwsk_reversible | 456 | 2.8779 | 103.9 | 2584 | 24 | 0 |

### Statistical Analysis

| Comparison | Mean A ± Std | Mean B ± Std | t-stat | p-value | Cohen's d | 95% CI (diff) | Significant |
|------------|-------------|-------------|--------|---------|-----------|---------------|-------------|
| conventional vs bwsk_analyzed | 2.8780±0.0001 | 2.8778±0.0002 | 2.982 | 0.0965 | 1.721 | [-0.0001, 0.0004] | trending |
| conventional vs bwsk_reversible | 2.8780±0.0001 | 2.8779±0.0001 | 1.675 | 0.2360 | 0.967 | [-0.0001, 0.0003] | no |

---

## Hypothesis Testing Summary

### H3: BWSK-reversible produces statistically equivalent quality

| Model | p-value | Cohen's d | Verdict |
|-------|---------|-----------|---------|
| Mamba-130M | 0.2360 | 0.967 | Equivalent (fail to reject H0) |
