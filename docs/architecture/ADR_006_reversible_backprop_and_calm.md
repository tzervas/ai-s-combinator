# ADR-006: Reversible Backprop and CALM Analysis

## Status
Accepted

## Context
Phase 3 adds two memory/compute optimizations based on the S/K classification:
1. Reversible backprop: avoid storing S-phase activations during backward pass.
2. CALM analysis: identify coordination-free segments for distributed training.

## Decision

### Reversible Backprop (reversible.py)
- Use `torch.utils.checkpoint` (gradient checkpointing) for S-phase segments.
- Partition model into segments: consecutive S-type layers form checkpointed segments, K-type layers store activations normally.
- `ReversibleSequence` wraps a list of modules with automatic segment partitioning.
- Memory savings proportional to S-type ratio (~75% for transformers).

We chose torch.checkpoint over true invertible functions (like RevNet's additive coupling) because:
1. It works with any S-type module, not just specially structured ones.
2. It's a standard PyTorch primitive with good backward compatibility.
3. The cost (one extra forward pass per S-segment) is acceptable.

### CALM Analysis (calm.py)
- `analyze_calm()` partitions model into monotone (S-type, coordination-free) and non-monotone (K-type, sync required) segments.
- `partition_for_distribution()` provides greedy device assignment.
- Reports include `parallelism_ratio` and `num_sync_barriers`.

The connection to CALM (Hellerstein 2010) is: S-type operations are monotone in the lattice-theoretic sense (they don't erase information), so by CALM they can execute coordination-free. K-type operations are non-monotone and require synchronization.

### Training Integration
- `BWSKTrainer` reports `estimated_memory_savings` and `parallelism_ratio` per step.
- Optional `use_reversible=True` wraps Sequential models in ReversibleSequence.
- Analysis is done once at init, metrics reported each step.

## Alternatives Considered

1. **True invertible layers (RevNet-style)**: Would require all S-type layers to have explicit inverse functions. Too restrictive — not all S-type ops have easy-to-compute inverses (e.g., full-rank matrix multiply is invertible but inverse is expensive). torch.checkpoint is simpler and works universally.

2. **Custom autograd Functions**: Could implement backward through S-phases without storing any activations. More complex to maintain and debug. torch.checkpoint provides the same benefit with standard PyTorch infrastructure.

3. **Graph-level optimization (torch.compile)**: Could potentially identify reversible subgraphs automatically. But torch.compile's optimization passes don't know about S/K classification. Our approach provides explicit, inspectable partitioning.
