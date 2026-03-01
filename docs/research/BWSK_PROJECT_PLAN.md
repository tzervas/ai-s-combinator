# BWSK: Combinator-Typed AI Primitives
## Project Plan — Average Joe's Labs

**Version**: 1.0 Draft
**Date**: 2026-03-01
**Authors**: [Average Joe's Labs team]
**Status**: Research complete, ready for implementation planning

---

## 1. Vision

A framework that uses combinator logic (B, W, S, K) as a **typed architectural description language** for neural networks, providing compile-time guarantees about information flow, reversibility, and parallelism while executing on standard tensor hardware.

**One-sentence pitch**: "Combinators describe, tensors compute — and the description gives you free provenance, free reversibility, and free parallelism guarantees."

---

## 2. Novelty Assessment (from research)

| Component | Prior Art | Our Novelty |
|-----------|-----------|-------------|
| S/K operation classifier | None exists | **First tool to classify NN ops by information preservation** |
| CALM + gradient accumulation | Not in literature | **First formal connection between CALM monotonicity and gradient sync** |
| B/W/S/K typed primitive set | W and K have no categorical formalization in ML | **First unified combinator algebra for all NN patterns** |
| Attention as S-combinator | "Attention as Hypernetwork" (ICLR 2025) touches it | **First explicit formalization** |
| Computation-graph provenance | Only post-hoc (SHAP, Captum) | **First built-in provenance from architecture description** |
| Erasure-minimized NAS | Not studied | **First erasure-budget objective for architecture search** |

---

## 3. Requirements

### 3.1 Functional Requirements

**FR-1**: S/K Classifier
- FR-1.1: Accept any PyTorch `nn.Module` as input
- FR-1.2: Trace the computation graph via `torch.fx`
- FR-1.3: Classify each graph node as S-type, K-type, or Gray (with confidence)
- FR-1.4: Output an "erasure budget" report: per-node classification, per-layer summary, whole-model erasure score
- FR-1.5: Support custom classification rules (user can override defaults)

**FR-2**: BWSK Architecture DSL
- FR-2.1: Define primitive combinators B (compose), W (share), S (fan-out), K (erase)
- FR-2.2: Architectures are expressions built from BWSK primitives + tensor ops
- FR-2.3: Type-check information flow at definition time (catch erasure in S-phase, shape mismatches)
- FR-2.4: Compile DSL expressions to standard PyTorch `nn.Module`
- FR-2.5: Generate provenance metadata from the combinator expression graph

**FR-3**: Provenance Engine
- FR-3.1: During inference, track which input features contribute to each output through S-phases
- FR-3.2: At K-boundaries, annotate what information was discarded and why
- FR-3.3: Produce a provenance graph viewable as JSON, HTML visualization, or Graphviz
- FR-3.4: Provenance tracking must be toggleable (zero overhead when disabled)

**FR-4**: S-Phase Reversible Backprop
- FR-4.1: Identify maximal reversible (S-only) segments between K-boundaries
- FR-4.2: During backward pass, reconstruct S-phase activations from outputs instead of stored tensors
- FR-4.3: Only checkpoint activations at K-boundaries
- FR-4.4: Measure and report memory savings vs standard and vs activation checkpointing

**FR-5**: CALM-Monotone Distributed Training
- FR-5.1: Identify which operations in the training loop are monotone (gradient accumulation) vs non-monotone (LR scheduling, weight update)
- FR-5.2: Implement CRDT-like gradient accumulation buffer
- FR-5.3: Allow staleness parameter k (k=1 → synchronous, k=∞ → fully async)
- FR-5.4: Report convergence metrics compared to synchronous training

### 3.2 Non-Functional Requirements

**NFR-1**: Performance
- S/K classifier runs in < 5 seconds for models up to 70B parameters
- DSL compilation adds < 1% overhead to model initialization
- Provenance tracking adds < 5% overhead when enabled
- S-phase reversible backprop achieves >= 50% activation memory reduction with < 2% compute overhead

**NFR-2**: Compatibility
- PyTorch >= 2.0 (initial target)
- Python >= 3.10
- Rust edition 2021 (for crate development)
- Burn >= 0.20 (for Rust ML backend)
- CUDA, ROCm, Metal backends via CubeCL (Rust) / standard PyTorch (Python)

**NFR-3**: Extensibility
- Custom combinator primitives (user-defined beyond BWSK)
- Custom S/K classification rules
- Plugin architecture for provenance visualization backends

---

## 4. Deliverables

### Phase 1: S/K Classifier (Python/PyTorch) — 2-4 weeks

| # | Deliverable | Description |
|---|-------------|-------------|
| D1.1 | `bwsk-classify` Python package | PyPI-installable S/K classifier for PyTorch models |
| D1.2 | CLI tool | `bwsk classify model.pt` → erasure budget JSON report |
| D1.3 | Jupyter integration | `bwsk.classify(model)` → inline visualization |
| D1.4 | Classification database | Default S/K rules for all `torch.nn` modules and `torch` ops |
| D1.5 | Test suite | Classification tests for ResNet, GPT-2, ViT, BERT architectures |
| D1.6 | Blog post / paper draft | "Where Does Information Go? An S/K Erasure Budget for Neural Networks" |

### Phase 2: BWSK DSL + Provenance (Python/PyTorch) — 4-8 weeks

| # | Deliverable | Description |
|---|-------------|-------------|
| D2.1 | `bwsk` Python package | Core DSL: B, W, S, K primitives, composition, type checking |
| D2.2 | Architecture library | GPT-2, ResNet-50, ViT-B/16 expressed in BWSK DSL |
| D2.3 | Provenance engine | Input-to-output tracking through S-phases, K-boundary annotations |
| D2.4 | Visualization | Interactive HTML provenance graph (d3.js or similar) |
| D2.5 | Benchmark | Train GPT-2-124M in BWSK DSL vs vanilla PyTorch: accuracy, speed, memory |
| D2.6 | Paper draft | "BWSK: Combinator-Typed Neural Network Primitives" |

### Phase 3: Reversible Backprop + CALM Training (Python/PyTorch) — 4-8 weeks

| # | Deliverable | Description |
|---|-------------|-------------|
| D3.1 | S-phase checkpoint optimizer | Automatic reversible segment detection + checkpoint placement |
| D3.2 | Memory benchmark | GPT-2, LLaMA-7B: memory savings vs standard, vs gradient checkpointing |
| D3.3 | CRDT gradient accumulator | `bwsk.distributed.CRDTAccumulator` |
| D3.4 | Multi-GPU benchmark | 4-8 GPU convergence comparison: sync vs CALM-async (k=1,2,4,8,∞) |
| D3.5 | Convergence analysis | Learning curves, final accuracy, wall-clock time per approach |

### Phase 4: Rust Crate + Burn Integration — 8-12 weeks

| # | Deliverable | Description |
|---|-------------|-------------|
| D4.1 | `bwsk-core` Rust crate | Core BWSK type system, combinator primitives, S/K classification |
| D4.2 | `bwsk-burn` Rust crate | Burn backend integration: BWSK DSL compiles to Burn modules |
| D4.3 | CubeCL GPU kernels | GPU-accelerated S-phase operations via Burn/CubeCL |
| D4.4 | Python bindings | PyO3 bindings for Rust core (optional performance backend for Python package) |
| D4.5 | Cross-validation | Same model in Python BWSK and Rust BWSK produces identical results |

### Phase 5: Erasure-Minimized NAS — 12+ weeks

| # | Deliverable | Description |
|---|-------------|-------------|
| D5.1 | Erasure-aware NAS | Architecture search with erasure budget objective |
| D5.2 | Activation function comparison | Softplus vs SiLU vs GELU vs ReLU accuracy/memory/energy at scale |
| D5.3 | Energy measurement | GPU power draw comparison for matched-accuracy architectures |
| D5.4 | Paper | "Erasure-Minimized Neural Architectures: Less Information Loss, Same Accuracy" |

---

## 5. User Stories

### S/K Classifier

**US-1**: As an ML engineer, I want to run `bwsk classify my_model.pt` and see which layers erase information, so I can understand my model's information flow.

**US-2**: As a researcher, I want to compare the erasure budgets of ResNet vs ViT vs GPT-2 to understand how different architectures handle information.

**US-3**: As a compliance officer, I want to verify that a model's K-boundaries (information erasure points) are documented, so I can audit the model for regulatory purposes.

### BWSK DSL

**US-4**: As an ML engineer, I want to define a transformer block using `S(linear_Q, S(linear_K, linear_V)) >> K_softmax() >> S_residual()` and have the DSL compile it to PyTorch with shape validation.

**US-5**: As a researcher, I want to compose two BWSK-defined modules (`encoder >> decoder`) and have the type system verify shape compatibility at definition time, not at runtime.

**US-6**: As an architect, I want to express weight sharing explicitly using W (`W(shared_block, depth=6)` for a Universal Transformer) and have the framework handle parameter tying correctly.

### Provenance

**US-7**: As an AI safety researcher, I want to trace a model's prediction back through S-phases to see which input tokens contributed most, without running a separate SHAP/Captum computation.

**US-8**: As a data scientist, I want to identify which K-boundaries (masking, dropout, pooling) in my model are discarding the most information, so I can decide whether to replace them with S-type alternatives.

**US-9**: As a product manager, I want to show regulators an audit trail of how our model processes customer data, with explicit documentation of where data is combined (S) vs discarded (K).

### Reversible Backprop

**US-10**: As an ML engineer training large models, I want to reduce activation memory by 50%+ without changing my architecture or losing accuracy, by exploiting the reversibility of S-phases.

**US-11**: As a researcher, I want to compare memory/compute tradeoffs between standard training, gradient checkpointing, RevNet, and S-phase reversible training on the same model.

### CALM Training

**US-12**: As an engineer running distributed training on 8 GPUs, I want to reduce AllReduce overhead by using CRDT gradient accumulation for monotone operations, with a tunable staleness parameter.

**US-13**: As a systems researcher, I want to measure whether CALM-monotone training converges to flatter minima (better generalization) than synchronous training, as theory predicts.

### Rust/Burn

**US-14**: As a Rust developer, I want to define a neural network architecture using BWSK combinators in Rust and have it compile to optimized Burn modules with GPU support via CubeCL.

**US-15**: As an infrastructure engineer, I want to deploy a BWSK-defined model as a single binary with millisecond startup, using Burn's inference capabilities.

---

## 6. Success Criteria

### Phase 1 (S/K Classifier)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Classification accuracy | 95%+ agreement with manual expert classification | Test on 10 architectures, expert review |
| Performance | < 5s for 70B parameter model graph | Wall-clock benchmark |
| Coverage | All `torch.nn` modules classified | Count of unclassified ops → 0 |
| Usability | < 3 lines of code to classify a model | API simplicity test |

### Phase 2 (BWSK DSL)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Architecture expressiveness | GPT-2, ResNet-50, ViT-B/16 fully expressible | Implement and verify |
| Compilation correctness | BWSK model produces identical outputs to vanilla PyTorch | Numerical comparison (atol=1e-6) |
| Type checking catches errors | 100% of shape mismatches caught at definition time | Test suite with intentional errors |
| Provenance overhead | < 5% inference slowdown when enabled | Benchmark on GPU |

### Phase 3 (Reversible Backprop + CALM)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Memory reduction (reversible) | >= 50% peak activation memory reduction | `torch.cuda.max_memory_allocated()` |
| Compute overhead (reversible) | < 2% additional FLOPs | FLOP counter comparison |
| Accuracy preservation (reversible) | < 0.1% accuracy difference from standard | Training run comparison |
| CALM convergence | Final accuracy within 0.5% of synchronous at k=4 | Multi-GPU benchmark |
| CALM speedup | >= 20% wall-clock reduction vs AllReduce at 8 GPUs | Distributed training benchmark |

### Phase 4 (Rust)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Rust/Python parity | Identical classification results | Cross-validation test |
| Burn integration | Full BWSK model trains on GPU via CubeCL | End-to-end training benchmark |
| Binary size | < 50MB for inference-only deployment | Build and measure |

### Phase 5 (NAS)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Low-erasure accuracy | Within 1% of high-erasure baseline (ReLU) | Matched architecture comparison |
| Memory improvement | >= 30% activation memory reduction from fewer K-ops | Memory profiling |
| Energy measurement | Measurable GPU power reduction for low-erasure model | `nvidia-smi` power draw comparison |

---

## 7. Technical Specification

### 7.1 S/K Classification Rules

The classifier assigns each operation one of three labels:

```
S-TYPE (information-preserving):
  - Invertible: output uniquely determines input
  - Examples: linear projection (full-rank), residual addition,
    concatenation, layer norm, softplus, SiLU, embedding lookup

K-TYPE (information-erasing):
  - Non-invertible: some inputs map to the same output
  - Examples: ReLU (all negatives → 0), dropout (random zeroing),
    max pooling (discards non-max), masking, argmax, loss computation

GRAY (context-dependent):
  - May be S or K depending on parameters or runtime values
  - Examples: softmax (S if denominator preserved, K if only ratios),
    batch norm (K if batch stats discard per-sample info),
    convolution with stride > 1 (K if information lost in downsampling)
```

**Classification algorithm**:
```python
def classify_node(node: fx.Node) -> Classification:
    # 1. Check explicit override table
    if node.target in USER_OVERRIDES:
        return USER_OVERRIDES[node.target]

    # 2. Check default classification database
    if node.target in DEFAULT_CLASSIFICATIONS:
        return DEFAULT_CLASSIFICATIONS[node.target]

    # 3. Structural analysis
    if is_linear_layer(node) and is_full_rank(node):
        return S_TYPE  # invertible linear map
    if is_elementwise(node) and is_monotone_bijection(node):
        return S_TYPE  # invertible activation
    if has_zero_outputs(node):  # e.g., ReLU zeros negatives
        return K_TYPE

    # 4. Default: GRAY with warning
    return GRAY
```

### 7.2 BWSK Primitive Definitions

```python
class B(Primitive):
    """Composition: B f g x = f(g(x))"""
    """Maps to: sequential layer stacking"""
    def forward(self, x):
        return self.f(self.g(x))

    @property
    def classification(self):
        # B is S-type if both f and g are S-type
        # B is K-type if either f or g is K-type
        return combine_classifications(self.f, self.g)

class W(Primitive):
    """Self-application / Weight sharing: W f x = f(x)(x)"""
    """Maps to: shared weights applied to same input, recurrence"""
    def forward(self, x):
        return self.f(x, x)  # f receives x twice (weight sharing)

    @property
    def classification(self):
        return S_TYPE  # W duplicates, never erases

class S(Primitive):
    """Fan-out and combine: S f g x = f(x)(g(x))"""
    """Maps to: multi-head attention, residual connections"""
    def forward(self, x):
        return self.combine(self.f(x), self.g(x))

    @property
    def classification(self):
        return S_TYPE  # S fans out and combines, never erases

class K(Primitive):
    """Erasure / Projection: K x y = x"""
    """Maps to: masking, dropout, pooling, activation clipping"""
    def forward(self, x, y=None):
        # y is explicitly discarded
        return self.select(x)

    @property
    def classification(self):
        return K_TYPE  # K always erases
```

### 7.3 Provenance Data Model

```python
@dataclass
class ProvenanceNode:
    id: str
    op_type: Literal["B", "W", "S", "K", "tensor_op"]
    classification: Literal["S", "K", "GRAY"]
    input_ids: List[str]      # which nodes feed into this one
    output_ids: List[str]     # which nodes this feeds into
    metadata: Dict[str, Any]  # op-specific info

    # K-type specific
    erasure_description: Optional[str]   # what was discarded
    erasure_fraction: Optional[float]    # estimated fraction of info lost

@dataclass
class ProvenanceGraph:
    nodes: Dict[str, ProvenanceNode]
    s_phases: List[List[str]]    # maximal reversible segments
    k_boundaries: List[str]      # K-type node ids
    erasure_budget: float         # total erasure score
```

### 7.4 Architecture Examples in BWSK DSL

```python
# Transformer attention head
attention_head = (
    S(linear("Q", d_model, d_k),                    # S-phase: fan-out to Q
      S(linear("K", d_model, d_k),                   # S-phase: fan-out to K
        linear("V", d_model, d_v)))                   # S-phase: fan-out to V
    >> B(scale(1/sqrt(d_k)), dot_product("QK"))       # S-phase: score computation
    >> K(mask(padding_mask), label="attention_mask")   # K-boundary: mask padding
    >> K(softmax(dim=-1), label="score_normalize")     # K-boundary: normalize
    >> K(dropout(0.1), label="regularize")             # K-boundary: dropout
    >> B(matmul("attn_V"), linear("out", d_v, d_model))  # S-phase: value weighting
)

# Transformer block with residual
transformer_block = (
    S(identity(),                                      # S-phase: skip connection
      B(attention_head, layer_norm()))                  # S-phase: attend + norm
    >> add()                                            # S-phase: residual add
    >> S(identity(),                                    # S-phase: skip connection
         B(linear("ffn_up", d_model, d_ff),            # S-phase: up-project
           K(gelu(), label="activation"),               # K-boundary: non-linearity
           linear("ffn_down", d_ff, d_model),           # S-phase: down-project
           layer_norm()))                                # S-phase: normalize
    >> add()                                            # S-phase: residual add
)

# Universal Transformer (weight-shared depth via W)
universal_transformer = W(transformer_block, depth=6)  # W: same block, 6 iterations

# Full model
gpt2 = B(
    embedding("tokens", vocab_size, d_model),          # S-phase: embed
    positional_encoding(max_seq_len, d_model),          # S-phase: position
    *[transformer_block for _ in range(12)],            # 12 blocks
    layer_norm(),                                       # S-phase: final norm
    K(linear("lm_head", d_model, vocab_size),           # K-ish: project to vocab
      label="output_projection")
)
```

### 7.5 Rust Crate Structure

```
bwsk/
├── bwsk-core/              # Core type system, classification, provenance
│   ├── src/
│   │   ├── primitives.rs   # B, W, S, K trait definitions
│   │   ├── classify.rs     # S/K classification engine
│   │   ├── provenance.rs   # Provenance graph data structures
│   │   ├── compose.rs      # Combinator composition + type checking
│   │   └── lib.rs
│   └── Cargo.toml
│
├── bwsk-burn/              # Burn backend integration
│   ├── src/
│   │   ├── module.rs       # BWSK → Burn module compilation
│   │   ├── backward.rs     # S-phase reversible backprop
│   │   ├── distributed.rs  # CALM-monotone gradient accumulator
│   │   └── lib.rs
│   └── Cargo.toml
│
├── bwsk-python/            # PyO3 Python bindings
│   ├── src/
│   │   └── lib.rs
│   └── Cargo.toml
│
└── bwsk-cli/               # CLI tool
    ├── src/
    │   └── main.rs         # `bwsk classify`, `bwsk trace`, `bwsk benchmark`
    └── Cargo.toml
```

---

## 8. Risk Register

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| S-phase reversibility has numerical precision issues at depth | Accumulated floating-point error makes reconstruction inaccurate | Medium | Use mixed-precision checkpointing at S-phase boundaries every N layers |
| CALM async training doesn't converge as well as sync | Gradient staleness causes divergence | Medium | Tunable staleness bound k; fall back to sync at k=1 |
| Erasure minimization hurts accuracy | Less regularization from fewer K-ops | Medium | Compensate with weight decay, data augmentation; benchmark thoroughly |
| FX tracing doesn't capture all ops (dynamic control flow) | Classifier misses K-operations in branches | Low-Medium | Support `torch.compile` / Dynamo as alternative tracer |
| Burn/CubeCL not mature enough for training at scale | Missing ops, bugs in gradient computation | Medium | Start with PyTorch; Rust is Phase 4 (later) |
| Adoption: ML practitioners resist new DSL | "Just let me write PyTorch" | High | Phase 1 (classifier) requires zero DSL adoption — it analyzes existing models |

---

## 9. Competitive Landscape

| Tool | What It Does | How BWSK Differs |
|------|-------------|------------------|
| Captum (Meta) | Post-hoc feature attribution (SHAP, IG) | BWSK: built-in provenance from architecture description, zero runtime cost for S-phases |
| FrEIA | Invertible NN construction | BWSK: classifies ANY existing model, not just specially constructed invertible ones |
| RevNet / iRevNet | Reversible architectures for memory savings | BWSK: exploits S-phases in standard architectures, no architectural changes needed |
| Equinox (JAX) | Functional NN library | BWSK: adds S/K type system and provenance on top of functional composition |
| Gavranovic CDL | Categorical theory of architectures | BWSK: practical implementation of combinator-typed primitives (CDL is pure theory) |
| Hogwild! / Local SGD | Async gradient methods | BWSK: formal CALM monotonicity guarantees + CRDT semantics, not just lock-free overwrites |

---

## 10. Research Outputs

### Papers (target venues)

1. **"Where Does Information Go? S/K Erasure Budgets for Neural Networks"**
   - Venue: NeurIPS / ICML workshop on interpretability
   - Content: S/K classification framework, erasure budgets for 20+ architectures, information flow analysis

2. **"BWSK: Combinator-Typed Neural Network Primitives"**
   - Venue: ICML / ICLR (main track)
   - Content: Full framework: DSL, type system, provenance, reversible backprop, CALM training
   - Theory: Formal connection between combinator logic structural rules and neural network information flow

3. **"Erasure-Minimized Neural Architectures"**
   - Venue: NeurIPS
   - Content: NAS with erasure budget, activation function comparison, energy measurement
   - Key result: low-erasure architectures match accuracy with better memory/provenance properties

### Open Source

- `bwsk` Python package (PyPI)
- `bwsk-core`, `bwsk-burn` Rust crates (crates.io)
- Documentation, tutorials, example architectures

---

## 11. Timeline (Estimated)

```
Month 1-2:  Phase 1 — S/K Classifier (Python/PyTorch)
            Ship D1.1-D1.5, draft blog post

Month 2-4:  Phase 2 — BWSK DSL + Provenance
            Ship D2.1-D2.4, begin benchmark D2.5

Month 4-6:  Phase 3 — Reversible Backprop + CALM Training
            Ship D3.1-D3.4, convergence analysis D3.5
            Submit Paper 1 (S/K Erasure Budgets)

Month 6-9:  Phase 4 — Rust Crate + Burn Integration
            Ship D4.1-D4.4, cross-validation D4.5
            Submit Paper 2 (BWSK Framework)

Month 9-12: Phase 5 — Erasure-Minimized NAS
            Ship D5.1-D5.3
            Submit Paper 3 (Erasure-Minimized Architectures)
```

---

## 12. Open Research Questions

1. **Does the S/K boundary predict generalization?** Hypothesis: models with a "right" S/K ratio generalize better than those that are too S-heavy (underfit, no compression) or too K-heavy (overfit to spurious correlations that get erased).

2. **Can CALM monotonicity formally guarantee convergence for async gradient accumulation?** Need to prove that gradient summation satisfies the CALM monotonicity conditions precisely, not just approximately.

3. **What is the optimal erasure budget per layer?** Is there a principle (like the information bottleneck) that predicts where K-operations should be placed for maximum performance?

4. **Does the BWSK decomposition reveal architectural patterns invisible to other analyses?** E.g., do models that are "W-heavy" (lots of weight sharing) generalize differently from "S-heavy" (lots of fan-out) models?

5. **Can S-phase provenance replace or improve upon attention-based explanations?** The attention-as-explanation debate (Jain & Wallace 2019 vs Wiegreffe & Pinter 2019) might be resolved by S-phase provenance, which tracks actual information flow rather than attention weights.

---

## 13. Key References

### Foundational Theory
- Waldmann, J. (1998). "Normalization of S-terms is decidable." RTA 1998.
- Statman, R. (1986). {B,W} fixed-point combinator Θ₄.
- Hellerstein, J. et al. "Keeping CALM." arXiv:1901.01930.
- Gavranovic et al. (2024). "Categorical Deep Learning." ICML 2024, arXiv:2402.15332.
- Fong, Spivak, Tuyeras (2019). "Backprop as Functor." arXiv:1711.10455.

### Information Flow in Neural Networks
- Tishby & Schwartz-Ziv (2015). "Deep Learning and the Information Bottleneck." arXiv:1503.02406.
- Saxe et al. (2018). "On the Information Bottleneck Theory." ICLR 2018.
- Jain & Wallace (2019). "Attention is not Explanation." arXiv:1902.10186.

### Activation Functions and Erasure
- "ReLU Strikes Back." ICLR 2024. (< 1% accuracy loss vs GELU at 1B params)
- "The Resurrection of the ReLU." arXiv:2505.22074.
- GELU Analysis: arXiv:2305.12073.

### Async SGD
- Recht et al. (2011). "Hogwild!" arXiv:1106.5730.
- Maranjyan et al. (2025). "Ringmaster ASGD." arXiv:2501.16168.

### Reversible Networks
- Gomez et al. (2017). "The Reversible Residual Network." NeurIPS 2017.
- Jacobsen et al. (2018). "i-RevNet." arXiv:1802.07088.

### Rust ML Ecosystem
- Burn (Tracel AI): 14.5K stars, CubeCL GPU backend, pure Rust.
- Candle (HuggingFace): 19.5K stars, inference-focused.

### This Project
- S-combinator research: `/home/kang/Documents/combinator/`
- Deep analysis: `research/DEEP_ANALYSIS_PURE_S.md`
- Framework design: `ai-s-combinator/research/COMBINATOR_AI_FRAMEWORK.md`

---

*Project Plan v1.0 — Average Joe's Labs*
*2026-03-01*
