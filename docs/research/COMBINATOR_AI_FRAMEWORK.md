# Combinator AI Framework: Design Document

**Date**: 2026-03-01
**Status**: Draft — design for deliberation
**Depends on**: `research/DEEP_ANALYSIS_PURE_S.md`, `research/SYNTHESIS_REPORT.md`

---

## Executive Summary

We propose a **phase-separated combinator AI framework** that exploits the structural correspondence between the S combinator and neural network operations. The framework does NOT replace tensor operations with combinator graph reduction (which would be 100-500x slower based on HVM2 benchmarks). Instead, it uses **combinator logic as the architectural description language** and **tensor operations as the execution substrate**, gaining:

1. **Automatic provenance** through S-phase tracking
2. **Coordination-free distributed training** via CALM-monotone gradient accumulation
3. **Reduced memory** through reversibility of S-phases (no activation storage)
4. **Energy savings** by minimizing erasure operations (fewer non-linearities)
5. **Compositionality** — architectures defined as combinator expressions compose by construction

The key insight: **use combinators to DESCRIBE, tensors to COMPUTE.**

---

## Part 1: Why NOT Pure Graph Reduction

The HVM2 benchmarks kill the naive approach:

| Metric | HVM2 (RTX 4090) | cuBLAS (RTX 4090) | Ratio |
|--------|-----------------|-------------------|-------|
| Peak throughput | 74 GIPS | 330 TFLOPS (fp16) | 1:4,500 |
| Memory pattern | Pointer-chasing | Coalesced stride-1 | Irregular vs optimal |
| Tensor core use | None | Full | 0% vs 100% |
| Single-thread vs GHC | 5x slower | N/A | Interpretation overhead |
| Practical speedup vs OCaml | Needs full 4090 to match | N/A | — |

**Graph reduction cannot compete with tensor cores for dense linear algebra.** Period. Any framework that tries to execute matmul via S-reduction will be thousands of times slower.

But that's the wrong comparison. The value of combinators is not in replacing GEMM — it's in replacing the **architectural plumbing** that connects GEMMs.

---

## Part 2: The Hybrid Architecture

### The Two Levels

```
LEVEL 1: COMBINATOR ARCHITECTURE LAYER (compile-time)
┌──────────────────────────────────────────────────┐
│  Describe the neural network as a combinator      │
│  expression. S-phases and K-phases are explicit.  │
│  Compile to tensor operation graph.               │
│                                                    │
│  Example: Transformer attention head =            │
│    S (linear_Q) (S (linear_K) (linear_V))         │
│    followed by K-mask, then S-residual            │
└──────────────────────────────────────────────────┘
              │ compiles to
              ▼
LEVEL 2: TENSOR EXECUTION LAYER (runtime)
┌──────────────────────────────────────────────────┐
│  Standard tensor operations: matmul, softmax,     │
│  elementwise ops. Executed via cuBLAS/cuDNN.      │
│  The combinator structure is metadata, not the    │
│  execution engine.                                │
└──────────────────────────────────────────────────┘
```

The combinator layer is a **type system** and **composition framework**, not an interpreter. It provides compile-time guarantees about information flow while the runtime uses standard tensor operations.

### What This Buys

**Compile-time guarantees from S-phase analysis:**
- Information flow tracking: which inputs contribute to which outputs (automatic XAI)
- Reversibility certification: S-phases are provably invertible → no activation storage needed
- Parallelism certification: S-phases are provably coordination-free → automatic sharding
- Monotonicity certification: S-phases produce monotone functions → CALM applies

**Runtime benefits from phase separation:**
- Fewer synchronization barriers in distributed training
- Reduced memory from reversible S-phases
- Energy profiling: measure erasure cost per K-boundary

---

## Part 3: The S/K Phase Classification

### Neural Operations Classified

Every operation in a neural network is either S-shaped (fan-out-transform-combine, non-erasing) or K-shaped (information-discarding). Here is the complete classification:

#### Pure S Operations (non-erasing, reversible, coordination-free)

| Operation | S-pattern | Why non-erasing |
|-----------|-----------|-----------------|
| Linear projection (`Wx + b`) | `S W_apply b_add x` | Invertible when W is full-rank |
| Residual connection (`x + F(x)`) | `S F id x → (F(x), x) → F(x) + x` | Invertible: `x = out - F(x)` |
| Multi-head split | `S head₁ (S head₂ ... headₙ) x` | Fan-out, each head sees full x |
| Layer normalization | `S stats_compute normalize x` | Compute μ,σ from x, apply to x |
| Attention scores (`Q·Kᵀ`) | `S Q_proj K_proj x → Q(x) · K(x)ᵀ` | Both Q,K derived from same input |
| Concatenation | `S f g x → [f(x), g(x)]` | No information lost |
| Skip connections | `S id F x → (x, F(x))` | Original preserved |
| Embedding lookup | `x → E[x]` | Invertible (discrete input preserved) |
| Softplus activation | `x → log(1 + eˣ)` | Invertible: `x = log(eʸ - 1)` |
| SiLU/Swish activation | `x → x · σ(x)` | Nearly invertible (monotone for x>-1.28) |
| GELU activation | `x → x · Φ(x)` | Nearly invertible (monotone for x>-0.68) |

#### K Operations (erasing, synchronization points, irreversible)

| Operation | K-pattern | What is erased |
|-----------|-----------|----------------|
| ReLU | `K-select(x, 0, x>0)` | All negative values → zero (50% erasure) |
| Hard attention mask | `K-mask(scores, mask)` | Padded positions → -∞ |
| Max pooling | `K-select(max, rest)` | All but maximum value |
| Average pooling | `K-reduce(mean, seq)` | Individual values → aggregate |
| Dropout | `K-mask(x, random_mask)` | Random 10-50% of values |
| Argmax / top-k | `K-select(top, rest)` | All but selected elements |
| Loss computation | `K-reduce(pred, label → scalar)` | Entire structure → single number |
| Weight update | `K-overwrite(θ_old, θ_new)` | Old parameters discarded |
| Gradient clipping | `K-clamp(grad, max)` | Gradient magnitude above threshold |
| Quantization | `K-round(x, bits)` | Sub-bit precision |

#### Gray Zone (partially erasing, depends on implementation)

| Operation | S or K? | Analysis |
|-----------|---------|----------|
| Softmax | **S** if denominator preserved, **K** if only ratios kept | Standard impl is K (loses scale) |
| Batch normalization | **S** if per-sample stats kept, **K** if batch stats only | Usually K (batch mean/var discard individual info) |
| Attention (full) | **S** for score computation, **K** for masking, **S** for value weighting | Mixed |
| GRU/LSTM gates | **S** for gate computation, **K** for gate application (sigmoid ∈ [0,1] partially erases) | Mixed |

### The Erasure Budget

A standard transformer layer has this erasure profile:

```
TRANSFORMER LAYER ERASURE BUDGET:
  Self-attention:
    Q, K, V projections .............. S (0% erasure)
    Score computation ................ S (0%)
    Attention masking ................ K (~5-20% of positions masked)
    Softmax .......................... K (loses absolute scale)
    Value weighting .................. S (0%)
    Output projection ................ S (0%)
    Residual + LayerNorm ............. S (0%)

  FFN:
    Up-projection (W₁x + b₁) ........ S (0%)
    Activation (GELU/ReLU) ........... K (50% erasure for ReLU, ~15% for GELU)
    Down-projection (W₂x + b₂) ...... S (0%)
    Residual + LayerNorm ............. S (0%)

  TOTAL: ~3-4 K-operations per layer out of ~12 operations
         = ~70-75% of operations are S-type
```

**Three-quarters of transformer computation is already S-shaped.** The framework makes this explicit and exploits it.

---

## Part 4: Five Concrete Innovations

### Innovation 1: S-Phase Reversible Training (Memory)

**Problem**: Training a large transformer stores activations for every layer (O(L) memory in depth L). Activation checkpointing trades 30% compute for 80% memory reduction. RevNets trade 50-100% compute overhead for O(1) memory.

**S-Phase solution**: Since S-operations are invertible by construction, S-phase activations never need storage. Only K-boundary outputs need checkpointing.

```
Standard:   Store activations at every layer boundary    O(L) memory
Checkpoint: Store at every √L layers, recompute rest    O(√L) memory, 1.3x compute
RevNet:     Coupled invertible blocks                    O(1) memory, 1.5-2x compute
S-Phase:    Store only at K-boundaries                   O(K) memory, 1.0x compute
            where K = number of K-operations (3-4 per layer)
```

For a 96-layer GPT: standard stores 96 activation tensors. S-Phase stores ~4 per layer × 96 = 384, but each is small (just the K-boundary outputs: post-mask scores, post-activation FFN values). The bulk activations (Q, K, V projections, pre-activation FFN, residuals) are reconstructed on the fly during backprop by inverting the S-operations.

**Estimated memory savings**: 50-70% activation memory reduction with ~0% compute overhead (S-inversions are cheap: subtract for residuals, solve linear system for projections if W is cached).

**Difference from RevNet**: RevNet requires specially designed coupled blocks. S-Phase works with ANY architecture where operations are classified — no architectural modification needed.

### Innovation 2: CALM-Monotone Gradient Accumulation (Distributed Training)

**Problem**: Distributed training requires AllReduce to synchronize gradients across devices. This accounts for 30-60% of training time at scale.

**Insight**: Gradient accumulation is a monotone operation (adding gradient contributions only increases the accumulated sum). By the CALM theorem, monotone operations don't need coordination.

**Design**:
```
TRADITIONAL DISTRIBUTED TRAINING:
  Each device: forward → backward → local gradients
  AllReduce: ∑ gradients across all devices (SYNCHRONOUS)
  Each device: apply averaged gradient

CALM-MONOTONE TRAINING:
  Each device: forward → backward → local gradients
  CRDT-merge: contribute gradients to shared monotone accumulator
              (no synchronization barrier — eventually consistent)
  Each device: apply gradient when local accumulator is "fresh enough"
              (staleness bounded by lattice diameter)
```

This is essentially **asynchronous SGD with formal monotonicity guarantees**. Async SGD has been studied extensively (Hogwild!, elastic averaging) but without the formal CALM framework. The key difference: CALM tells you exactly WHICH operations are safe to desynchronize (the monotone ones = S-phases and gradient sums) and which MUST synchronize (the K-operations = weight updates with non-monotone learning rate schedules).

**Practical implementation**: Use a CRDT-like gradient buffer where:
- Each device writes `(device_id, step, gradient_chunk)` tuples
- The merge function is `sum` (monotone, commutative, associative)
- Devices read the current sum and apply when the sum includes "enough" contributions
- No barrier, no coordinator, no AllReduce

**Risk**: Gradient staleness can hurt convergence. Mitigation: bound staleness to k steps (where k is tunable). With k=1 this degenerates to synchronous; with k=∞ this is fully async. The sweet spot depends on the model and dataset.

### Innovation 3: S-Typed Architecture DSL (Compositionality)

**Problem**: Neural architectures are described imperatively (PyTorch `forward()` methods) with no formal guarantees about information flow, composability, or properties.

**Design**: A domain-specific language where architectures are combinator expressions:

```python
# Traditional PyTorch
class Attention(nn.Module):
    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.masked_fill(mask == 0, -1e9)   # K-operation!
        attn = F.softmax(scores, dim=-1)                # K-operation!
        attn = F.dropout(attn, p=0.1)                   # K-operation!
        return torch.matmul(attn, v)

# Combinator DSL
attention = (
    S(linear("Q"), S(linear("K"), linear("V")))    # S-phase: fan-out to Q,K,V
    >> S(dot_product, scale(sqrt_dk))               # S-phase: compute scores
    >> K_mask(padding_mask)                          # K-boundary: mask padding
    >> K_softmax()                                   # K-boundary: normalize
    >> K_dropout(0.1)                                # K-boundary: regularize
    >> S(matmul, identity)                           # S-phase: apply to values
)

# The DSL compiler:
# 1. Validates S/K classification (type-checks information flow)
# 2. Identifies reversible segments (all S-phases between K-boundaries)
# 3. Generates standard PyTorch code with activation checkpointing at K-boundaries
# 4. Annotates operations for CALM-safe distribution
# 5. Produces provenance graph for XAI
```

**What the type system catches at compile time:**
- "You applied dropout inside what you claimed was an S-phase" → type error
- "This residual connection's inverse requires W to be full-rank but W is 768×3072" → type warning
- "This S-phase can be distributed without coordination" → annotation
- "This K-boundary discards 50% of information (ReLU)" → erasure audit

### Innovation 4: Provenance-Preserving Inference (XAI)

**Problem**: Explaining why a model made a prediction requires post-hoc methods (SHAP, LIME, attention visualization) that are approximate and sometimes misleading.

**S-Phase solution**: S-operations preserve complete provenance by construction. Every output element can be traced back through the S-reduction chain to identify exactly which inputs contributed and via which transformation path.

```
INPUT:  "The cat sat on the mat"
        [t₁,  t₂, t₃, t₄, t₅, t₆]

S-PHASE TRACE (attention):
  Q(t₃) · K(t₂)ᵀ = 0.7   ← "sat" attended to "cat"
  Q(t₃) · K(t₆)ᵀ = 0.2   ← "sat" attended to "mat"

  The trace is the S-reduction graph:
  S(Q_proj, K_proj)(t₃) → (Q(t₃), K(t₃))
  S(Q_proj, K_proj)(t₂) → (Q(t₂), K(t₂))
  ...complete fan-out and combination history preserved

K-BOUNDARY (softmax):
  Provenance note: "absolute scores normalized; relative ordering preserved"
  Information lost: scale of raw scores

K-BOUNDARY (attention mask):
  Provenance note: "positions 7-512 masked (padding)"
  Information lost: attention to padding tokens
```

The S-reduction graph IS the explanation. No post-hoc approximation needed for S-phases. K-boundaries are annotated with what information was discarded and why.

### Innovation 5: Erasure-Minimized Architecture Search (Energy)

**Problem**: Current NAS (Neural Architecture Search) optimizes for accuracy, latency, and model size. Energy efficiency is not directly optimized.

**S-Phase solution**: Add an "erasure budget" objective to NAS. Minimize K-operations while maintaining accuracy.

```
ERASURE-AWARE NAS OBJECTIVES:
  1. Accuracy (standard)
  2. Latency (standard)
  3. Erasure score = Σ(information_erased at each K-boundary)
     - ReLU: 0.50 (erases 50% of values)
     - GELU: 0.15 (erases ~15%)
     - Softplus: 0.00 (erases nothing)
     - Dropout(p): p (erases p fraction)
     - Max pooling: (n-1)/n (erases all but 1 of n values)
     - Average pooling: (n-1)/n (erases individual values, keeps mean)

SEARCH SPACE MODIFICATIONS:
  - Allow activation function choice: {ReLU, GELU, SiLU, Softplus, Identity}
  - Allow pooling choice: {MaxPool, AvgPool, StridedConv (S-type), None}
  - Penalize K-operations proportional to their erasure score
```

**Hypothesis**: Architectures with lower erasure scores will be:
- More energy-efficient (fewer Landauer-limit operations)
- More interpretable (more of the computation is traceable)
- Require less activation memory (more reversible phases)
- Potentially more accurate for tasks requiring fine-grained input-output correspondence

**Counter-hypothesis**: Erasure is regularization. ReLU's 50% erasure prevents overfitting. Reducing erasure might require other regularization (larger weight decay, more data augmentation) to compensate.

---

## Part 5: The {B,W} Alternative

Our research revealed that {B,W} (composition + self-application) is Turing-complete without explicit K. This suggests an alternative primitive set:

```
B-PRIMITIVE: Composition
  B f g x = f(g(x))
  "Chain two transformations sequentially"
  Maps to: sequential layer stacking, function composition

W-PRIMITIVE: Self-application / weight sharing
  W f x = f(x)(x)
  "Apply a function that examines its input twice"
  Maps to: weight sharing, self-attention (Q=K case),
           recurrent connections

S-PRIMITIVE: Fan-out + combine
  S f g x = f(x)(g(x))
  "Two perspectives on the same data, combined"
  Maps to: multi-head attention, residual connections,
           feature pyramid networks
```

The {B,W,S} set (without K) gives us a rich non-erasing vocabulary:
- B for sequential composition (layer stacking)
- W for self-referential computation (recurrence, self-attention)
- S for parallel perspectives (multi-head, skip connections)

K is introduced only at explicit erasure boundaries, as a distinct architectural decision with documented justification.

This is the **relevant logic approach to neural architecture**: every input must be used at least once, erasure is a deliberate exception.

---

## Part 6: Implementation Roadmap

### Phase A: Proof of Concept (2-4 weeks)

1. **S/K classifier for PyTorch ops**: A static analysis tool that takes a `nn.Module` and classifies each operation as S or K, producing an erasure budget report.

2. **S-phase reversible backprop**: For a single transformer block, implement backward pass that reconstructs S-phase activations instead of storing them. Measure memory savings vs. standard and vs. activation checkpointing.

3. **Benchmark**: Compare memory, compute, and accuracy on a small model (GPT-2 124M) with:
   - Standard training (store all activations)
   - Activation checkpointing
   - S-phase reversible training

### Phase B: DSL Prototype (4-8 weeks)

1. **Combinator DSL in Python**: A small library where architectures are expressed as combinator compositions. Compiles to PyTorch `nn.Module`.

2. **Type checker**: Validates S/K classification, identifies reversible segments, generates provenance metadata.

3. **Provenance demo**: Visualize the S-reduction trace for a trained attention head, showing input-to-output information flow.

### Phase C: Distributed Training (8-12 weeks)

1. **CRDT gradient accumulator**: Implement monotone gradient merging without AllReduce.

2. **Benchmark**: Compare convergence and wall-clock time vs. synchronous distributed training on 4-8 GPUs.

3. **Staleness analysis**: Measure how gradient staleness affects convergence as a function of the staleness bound k.

### Phase D: Erasure-Minimized NAS (12+ weeks)

1. **Erasure-aware search**: Add erasure budget to a NAS framework (e.g., NNI, AutoML).

2. **Compare architectures**: Do low-erasure architectures achieve comparable accuracy?

3. **Energy measurement**: Measure actual energy consumption (GPU power draw) for matched-accuracy architectures with different erasure budgets.

---

## Part 7: What This Is NOT

To be clear about scope and avoid overclaiming:

1. **NOT a graph reduction runtime**: We do not propose executing neural networks via combinator graph reduction. Tensor operations remain the execution substrate. Combinators are the architectural description language.

2. **NOT a replacement for PyTorch/JAX**: The DSL compiles TO PyTorch/JAX. It's a higher-level abstraction, not a new runtime.

3. **NOT thermodynamic computing**: The Landauer argument motivates erasure minimization but current hardware is 7 orders of magnitude above the limit. The practical benefits are memory reduction and interpretability, not energy savings at the Landauer scale.

4. **NOT a proof that S alone suffices for AI**: Pure S is NOT Turing-complete (Waldmann 1998, our n_S proof). The framework uses S as a structural primitive alongside K. The innovation is in making the S/K boundary explicit and exploiting S-phase properties.

5. **NOT a claim that fewer non-linearities is always better**: ReLU's erasure serves as implicit regularization. Removing it requires compensating with other regularization. The hypothesis that low-erasure architectures can match high-erasure ones needs experimental validation.

---

## Key References

### Foundational
- Waldmann, J. (1998). "Normalization of S-terms is decidable." RTA 1998.
- Statman, R. (1986). {B,W} fixed-point combinator.
- Girard, J.-Y. (1987). "Linear Logic." TCS 50.
- Hellerstein, J. et al. "Keeping CALM." arXiv:1901.01930.

### Neural Architecture
- Gomez et al. (2017). "The Reversible Residual Network." NeurIPS 2017.
- Gavranovic et al. (2024). "Categorical Deep Learning." ICML 2024.
- Cruttwell et al. (2024). "Deep Learning with Parametric Lenses." arXiv:2404.00408.

### GPU Graph Reduction
- HVM2 / HigherOrderCO. ICFP 2024 FProPer workshop.
- Lafont, Y. (1997). "Interaction Combinators." I&C 137.

### Monotone Neural Networks
- Runje & Shankaranarayana (2023). "Constrained Monotonic NNs." ICML 2023.
- "Scalable Monotonic Neural Networks." ICLR 2024.

### Thermodynamic Computing
- Landauer, R. (1961). "Irreversibility and Heat Generation."
- "Thermodynamic Bounds on DNN Energy." arXiv:2503.09980.

### This Project
- n_S monotonicity proof: `hypothesis/proofs/non_universality/S_non_universality_v3.md`
- Deep analysis: `research/DEEP_ANALYSIS_PURE_S.md`
- Synthesis report: `research/SYNTHESIS_REPORT.md`

---

*Draft 2026-03-01 — for deliberation*
*Project: S-Combinator Research & Prize Competition*
