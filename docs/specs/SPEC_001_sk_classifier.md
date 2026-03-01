# SPEC-001: S/K Operation Classifier

**Phase**: 1 (Foundation)
**Status**: Ready to implement
**Depends on**: None
**References**: ADR-001, US-01, US-02, US-03, US-08

---

## 1. Goal

Classify every operation in a PyTorch neural network as **S-type** (information-preserving), **K-type** (information-erasing), or **Gray** (context-dependent). The classifier operates on a `torch.fx` graph representation of the model, walking each node and assigning an `OpClass` based on a classification database, user overrides, and structural heuristics.

The output is an **erasure budget report** that quantifies how much of the model's computation is information-preserving vs. information-erasing. This report is the foundation for all downstream BWSK analysis: provenance tracking (SPEC-003), reversible backprop, CALM distributed training, and erasure-minimized architecture search.

### Why This Matters

- **Provenance**: S-type phases preserve a lossless audit trail. K-type boundaries are the exact points where provenance degrades.
- **Reversibility**: S-type chains can be inverted for activation recomputation, saving memory during training.
- **Parallelism**: S-type (monotone) operations are coordination-free under CALM. The erasure budget tells you how much of the model can run without synchronization.
- **Architecture comparison**: Erasure budgets let researchers compare architectures (ResNet vs ViT vs GPT-2) on an information-theoretic axis that is independent of parameter count or FLOP count.

---

## 2. Success Criteria

| Criterion | Target |
|-----------|--------|
| Classification accuracy | 95%+ agreement with manual expert classification |
| Performance | < 5s for 70B parameter model graph |
| Coverage | All `torch.nn` modules classified |
| Usability | < 3 lines of code to classify a model |
| Extensibility | User can override any classification with a single dict entry |
| Determinism | Same model always produces same report |

---

## 3. User Stories

- **US-01**: `bwsk classify my_model.pt` produces an erasure budget JSON report
- **US-02**: Compare erasure budgets of ResNet vs ViT vs GPT-2
- **US-03**: Audit model K-boundaries for regulatory compliance
- **US-08**: Identify erasure hotspots in a model (layers with disproportionate K-count)

---

## 4. Classification Database

The classification database is the core lookup table. Every recognized operation maps to an `OpClass` with a confidence level and rationale. Confidence is 1.0 for operations whose classification is unambiguous by definition, and lower for operations that depend on configuration or runtime state.

### 4.1 Linear Layers

| Operation | Class | Confidence | Rationale |
|-----------|-------|------------|-----------|
| `nn.Linear` | S | 0.9 | Affine transform `y = xW^T + b`. Full-rank weight matrix is invertible (S-type). Rank-deficient weight is K-type, but we cannot determine rank statically; default S with note. |
| `nn.LazyLinear` | S | 0.9 | Same as `nn.Linear` once materialized. |
| `nn.Bilinear` | S | 0.8 | `y = x_1^T A x_2 + b`. Bilinear map preserves information from both inputs given the other is fixed and the weight tensor is full-rank, but information from one input can mask the other. Conservatively S. |

### 4.2 Convolutions

| Operation | Class | Confidence | Rationale |
|-----------|-------|------------|-----------|
| `nn.Conv1d` (stride=1) | S | 0.9 | Convolution with stride 1 is a linear map with shared weights. No spatial downsampling; information is preserved. |
| `nn.Conv1d` (stride>1) | K | 0.9 | Stride > 1 downsamples the spatial dimension; some input positions are never read. Information is erased. |
| `nn.Conv2d` (stride=1) | S | 0.9 | Same reasoning as Conv1d stride=1. |
| `nn.Conv2d` (stride>1) | K | 0.9 | Same reasoning as Conv1d stride>1. |
| `nn.Conv3d` (stride=1) | S | 0.9 | Same reasoning as Conv1d stride=1. |
| `nn.Conv3d` (stride>1) | K | 0.9 | Same reasoning as Conv1d stride>1. |
| `nn.ConvTranspose1d` | S | 0.9 | Transposed convolution upsamples; no information loss. |
| `nn.ConvTranspose2d` | S | 0.9 | Transposed convolution upsamples; no information loss. |
| `nn.ConvTranspose3d` | S | 0.9 | Transposed convolution upsamples; no information loss. |

**Note on convolution stride**: The classifier must inspect the `stride` attribute of convolution modules at classification time. If stride is `(1,)`, `(1,1)`, or `(1,1,1)`, classify as S. Otherwise classify as K.

### 4.3 Normalization

| Operation | Class | Confidence | Rationale |
|-----------|-------|------------|-----------|
| `nn.LayerNorm` | S | 0.95 | Normalizes across features within a single sample. Invertible given the learned affine parameters (scale and shift). No cross-sample dependency. |
| `nn.RMSNorm` | S | 0.95 | Simpler variant of LayerNorm (no mean subtraction). Invertible given scale. |
| `nn.BatchNorm1d` | GRAY | 0.7 | **Train mode**: computes running statistics across the batch (cross-sample dependency, not coordination-free, K-type). **Eval mode**: uses frozen statistics (per-sample, invertible, S-type). Classification depends on `module.training`. |
| `nn.BatchNorm2d` | GRAY | 0.7 | Same as BatchNorm1d. |
| `nn.BatchNorm3d` | GRAY | 0.7 | Same as BatchNorm1d. |
| `nn.GroupNorm` | S | 0.9 | Normalizes within groups of channels per sample. No cross-sample dependency. Invertible given affine parameters. |
| `nn.InstanceNorm1d` | S | 0.9 | Normalizes each channel per sample independently. No cross-sample dependency. |
| `nn.InstanceNorm2d` | S | 0.9 | Same as InstanceNorm1d. |
| `nn.InstanceNorm3d` | S | 0.9 | Same as InstanceNorm1d. |

### 4.4 Activations

| Operation | Class | Confidence | Rationale |
|-----------|-------|------------|-----------|
| `nn.ReLU` | K | 1.0 | `max(0, x)` erases all negative values. The canonical K-type operation. |
| `nn.LeakyReLU` | S | 0.95 | `max(alpha*x, x)` with alpha > 0. Bijective (invertible) for any nonzero alpha. No information loss. |
| `nn.PReLU` | S | 0.95 | Parametric version of LeakyReLU. Bijective for alpha != 0. |
| `nn.ELU` | K | 0.9 | `elu(x) = x if x>0, alpha*(exp(x)-1) if x<=0`. The negative branch saturates toward -alpha, collapsing distinct negative inputs. Not injective on the negative half. |
| `nn.GELU` | K | 0.9 | Non-monotonic near zero; the region around x ~ -0.5 maps distinct inputs to the same output. Not injective. |
| `nn.SiLU` (Swish) | K | 0.9 | `x * sigmoid(x)` is non-monotonic for x < 0. Not injective. |
| `nn.Mish` | K | 0.9 | `x * tanh(softplus(x))` is non-monotonic for large negative x. Not injective. |
| `nn.Softplus` | S | 0.95 | `log(1 + exp(x))` is strictly monotonic and invertible. S-type. |
| `nn.Sigmoid` | K | 0.95 | Maps R to (0,1). Strictly monotonic but saturates: distinct large-magnitude inputs map to values indistinguishable at finite precision. Practically K-type. |
| `nn.Tanh` | K | 0.95 | Same saturation argument as Sigmoid. Practically K-type. |
| `nn.Softmax` | K | 1.0 | Reduces dimensionality by 1 (outputs sum to 1). Erases one degree of freedom. K-type. |
| `nn.LogSoftmax` | K | 1.0 | `log(softmax(x))`. Same dimensionality reduction as Softmax. |
| `nn.Hardswish` | K | 0.9 | Piecewise linear but clips to 0 for x <= -3. Erases information in the clipped region. |
| `nn.Hardsigmoid` | K | 0.9 | Piecewise linear but clips to 0 and 1 at the boundaries. Information loss in saturated regions. |

### 4.5 Pooling

| Operation | Class | Confidence | Rationale |
|-----------|-------|------------|-----------|
| `nn.MaxPool1d` | K | 1.0 | Selects maximum from each window; all other values are erased. |
| `nn.MaxPool2d` | K | 1.0 | Same as MaxPool1d. |
| `nn.MaxPool3d` | K | 1.0 | Same as MaxPool1d. |
| `nn.AvgPool1d` | K | 1.0 | Averages values within each window; individual values are irrecoverable. |
| `nn.AvgPool2d` | K | 1.0 | Same as AvgPool1d. |
| `nn.AvgPool3d` | K | 1.0 | Same as AvgPool1d. |
| `nn.AdaptiveAvgPool1d` | K | 1.0 | Adaptive variant; same erasure property as AvgPool. |
| `nn.AdaptiveAvgPool2d` | K | 1.0 | Same as AdaptiveAvgPool1d. |
| `nn.AdaptiveAvgPool3d` | K | 1.0 | Same as AdaptiveAvgPool1d. |
| `nn.AdaptiveMaxPool1d` | K | 1.0 | Same as MaxPool. |
| `nn.AdaptiveMaxPool2d` | K | 1.0 | Same as MaxPool. |
| `nn.AdaptiveMaxPool3d` | K | 1.0 | Same as MaxPool. |

### 4.6 Dropout

| Operation | Class | Confidence | Rationale |
|-----------|-------|------------|-----------|
| `nn.Dropout` | K | 1.0 | Randomly zeroes elements during training. Erases information stochastically. Identity at eval, but classified by worst-case (training) behavior. |
| `nn.Dropout2d` | K | 1.0 | Zeroes entire channels. Same reasoning. |
| `nn.Dropout3d` | K | 1.0 | Same reasoning. |
| `nn.AlphaDropout` | K | 1.0 | Designed for SELU networks; still randomly erases activations. |

### 4.7 Recurrent Layers

| Operation | Class | Confidence | Rationale |
|-----------|-------|------------|-----------|
| `nn.RNN` | GRAY | 0.6 | Contains both linear transforms (S-type) and tanh activations (K-type) internally. Mixed. The overall module is Gray because the K-type activation is integral. |
| `nn.LSTM` | GRAY | 0.6 | Contains sigmoid gates (K-type) and tanh activations (K-type) alongside linear transforms (S-type). The forget gate is explicitly an erasure mechanism. |
| `nn.GRU` | GRAY | 0.6 | Contains sigmoid reset/update gates (K-type) and tanh (K-type) alongside linear transforms. Similar to LSTM. |

### 4.8 Embedding

| Operation | Class | Confidence | Rationale |
|-----------|-------|------------|-----------|
| `nn.Embedding` | S | 1.0 | Lookup table. Each input index maps to a unique vector. Injective (1-to-1). No information loss. |
| `nn.EmbeddingBag` | K | 0.9 | Sums or averages embeddings for a bag of indices. Multiple distinct bags can produce the same output. Information loss from aggregation. |

### 4.9 Attention

| Operation | Class | Confidence | Rationale |
|-----------|-------|------------|-----------|
| `nn.MultiheadAttention` | GRAY | 0.7 | Internally mixed: Q/K/V projections are S-type (linear), attention weights use softmax (K-type), weighted sum is S-type, output projection is S-type. The module as a whole is Gray because K-type softmax is integral. |

### 4.10 Loss Functions

All loss functions are K-type with confidence 1.0. A loss function reduces an entire output tensor to a scalar; this is maximal information erasure.

| Operation | Class | Confidence | Rationale |
|-----------|-------|------------|-----------|
| `nn.CrossEntropyLoss` | K | 1.0 | Reduces to scalar. |
| `nn.NLLLoss` | K | 1.0 | Reduces to scalar. |
| `nn.MSELoss` | K | 1.0 | Reduces to scalar. |
| `nn.L1Loss` | K | 1.0 | Reduces to scalar. |
| `nn.SmoothL1Loss` | K | 1.0 | Reduces to scalar. |
| `nn.BCELoss` | K | 1.0 | Reduces to scalar. |
| `nn.BCEWithLogitsLoss` | K | 1.0 | Reduces to scalar. |
| `nn.KLDivLoss` | K | 1.0 | Reduces to scalar. |
| `nn.HuberLoss` | K | 1.0 | Reduces to scalar. |
| `nn.CosineEmbeddingLoss` | K | 1.0 | Reduces to scalar. |
| `nn.TripletMarginLoss` | K | 1.0 | Reduces to scalar. |
| `nn.CTCLoss` | K | 1.0 | Reduces to scalar. |

### 4.11 Functional Operations (torch / torch.nn.functional)

| Operation | Class | Confidence | Rationale |
|-----------|-------|------------|-----------|
| `torch.matmul` | S | 0.9 | Matrix multiply. S-type given full-rank operands. |
| `torch.bmm` | S | 0.9 | Batched matmul. Same as matmul. |
| `torch.add` | S | 1.0 | Elementwise addition. Invertible given one operand. |
| `torch.sub` | S | 1.0 | Elementwise subtraction. Invertible. |
| `torch.mul` | S | 0.9 | Elementwise multiplication. Invertible if no zeros in multiplier. Confidence < 1.0 because zero multiplier erases. |
| `torch.div` | S | 0.9 | Elementwise division. Invertible if divisor is nonzero. |
| `torch.cat` | S | 1.0 | Concatenation. Strictly information-preserving; all inputs recoverable via split. |
| `torch.stack` | S | 1.0 | Stacking. Same as cat with a new dimension. |
| `torch.chunk` | S | 1.0 | Splits tensor into chunks. No information loss; cat is the inverse. |
| `torch.split` | S | 1.0 | Same as chunk. |
| `torch.reshape` | S | 1.0 | View change. Bijection on element ordering. |
| `torch.permute` | S | 1.0 | Dimension reordering. Bijection. |
| `torch.transpose` | S | 1.0 | Dimension swap. Bijection. |
| `torch.flatten` | S | 1.0 | Reshape to 1D. Invertible given original shape. |
| `torch.unsqueeze` | S | 1.0 | Adds a dimension of size 1. Invertible via squeeze. |
| `torch.squeeze` | S | 1.0 | Removes dimensions of size 1. Invertible via unsqueeze. |
| `torch.clamp` | K | 1.0 | Clips values to [min, max]. Distinct values outside range map to boundary. |
| `torch.abs` | K | 1.0 | Erases sign information. |
| `torch.sum` (reducing) | K | 1.0 | Reduces dimension(s) to summed scalar/vector. |
| `torch.mean` (reducing) | K | 1.0 | Same as sum. |
| `torch.max` (reducing) | K | 1.0 | Selects maximum; all other values erased. |
| `torch.min` (reducing) | K | 1.0 | Selects minimum; all other values erased. |
| `torch.argmax` | K | 1.0 | Returns index of maximum; entire tensor reduced to index. |
| `torch.argmin` | K | 1.0 | Same as argmax. |
| `F.scaled_dot_product_attention` | GRAY | 0.7 | Contains S-type matmul and K-type softmax internally. Mixed, same as `nn.MultiheadAttention`. |
| `F.relu` | K | 1.0 | Functional version of nn.ReLU. |
| `F.gelu` | K | 0.9 | Functional version of nn.GELU. |
| `F.silu` | K | 0.9 | Functional version of nn.SiLU. |
| `F.dropout` | K | 1.0 | Functional version of nn.Dropout. |
| `F.layer_norm` | S | 0.95 | Functional version of nn.LayerNorm. |
| `F.softmax` | K | 1.0 | Functional version of nn.Softmax. |
| `F.log_softmax` | K | 1.0 | Functional version of nn.LogSoftmax. |
| `F.cross_entropy` | K | 1.0 | Functional version of nn.CrossEntropyLoss. |

---

## 5. Classification Algorithm

The classifier processes a `torch.fx.GraphModule` and assigns an `OpClass` to every node. The algorithm is a four-step pipeline applied to each node:

```
CLASSIFY(node, model, user_overrides, database) -> ClassificationResult:

    Step 1: USER OVERRIDE
        key = canonical_name(node)
        if key in user_overrides:
            return ClassificationResult(
                classification = user_overrides[key],
                confidence = 1.0,
                rationale = "User override"
            )

    Step 2: DATABASE LOOKUP
        key = canonical_name(node)
        if key in database:
            entry = database[key]
            # For configurable ops, refine based on attributes
            if entry.requires_attribute_check:
                return refine_by_attributes(node, entry, model)
            return ClassificationResult(
                classification = entry.op_class,
                confidence = entry.confidence,
                rationale = entry.rationale
            )

    Step 3: STRUCTURAL ANALYSIS (for call_module nodes only)
        module = get_module(model, node.target)
        if module is not None:
            # Check if the module is a known container (Sequential, ModuleList)
            if is_container(module):
                return ClassificationResult(GRAY, 0.5,
                    "Container module; classify children individually")

            # Check if the module wraps a single recognized op
            if has_single_child_op(module):
                return CLASSIFY(single_child, model, user_overrides, database)

            # Default for unknown modules
            return ClassificationResult(GRAY, 0.3,
                "Unknown module type; defaulting to GRAY")

    Step 4: DEFAULT
        emit_warning(f"Unrecognized operation: {node.target}")
        return ClassificationResult(GRAY, 0.0,
            "Unrecognized operation; defaulting to GRAY")
```

### 5.1 Canonical Name Resolution

The canonical name maps a `torch.fx.Node` to a string key for database lookup:

- `call_module` nodes: resolve to the module's class name (e.g., `"nn.Linear"`, `"nn.Conv2d"`).
- `call_function` nodes: resolve to the function's qualified name (e.g., `"torch.add"`, `"F.relu"`).
- `call_method` nodes: resolve to the method name on the tensor (e.g., `"Tensor.view"`, `"Tensor.reshape"`).
- `placeholder` nodes: input placeholders. Classified as S with confidence 1.0 (no operation).
- `get_attr` nodes: parameter/buffer access. Classified as S with confidence 1.0 (no transformation).
- `output` nodes: graph output. Classified as S with confidence 1.0 (no transformation).

### 5.2 Attribute-Dependent Refinement

Some operations have a base classification that is refined by inspecting module attributes:

```
refine_by_attributes(node, entry, model):
    module = get_module(model, node.target)

    # Convolution stride check
    if isinstance(module, (Conv1d, Conv2d, Conv3d)):
        if all(s == 1 for s in module.stride):
            return ClassificationResult(S, 0.9, "Convolution with stride=1")
        else:
            return ClassificationResult(K, 0.9,
                f"Convolution with stride={module.stride}; spatial downsampling erases information")

    # BatchNorm training mode check
    if isinstance(module, (BatchNorm1d, BatchNorm2d, BatchNorm3d)):
        if module.training:
            return ClassificationResult(K, 0.8,
                "BatchNorm in training mode; cross-sample statistics are K-type")
        else:
            return ClassificationResult(S, 0.8,
                "BatchNorm in eval mode; frozen statistics, per-sample invertible")

    # Fallback to entry default
    return ClassificationResult(entry.op_class, entry.confidence, entry.rationale)
```

### 5.3 Graph Traversal

The classifier walks the `torch.fx.Graph` in topological order. Each node is classified independently (classification does not depend on neighboring nodes). The full pipeline:

```
classify_model(model, custom_rules=None):
    graph_module = torch.fx.symbolic_trace(model)
    database = load_default_database()
    user_overrides = custom_rules or {}
    results = []

    for node in graph_module.graph.nodes:
        result = CLASSIFY(node, graph_module, user_overrides, database)
        result.op_name = node.name
        result.op_type = canonical_name(node)
        results.append(result)

    return build_report(model, results)
```

---

## 6. Edge Cases and Ambiguities

### 6.1 nn.Linear with Rank-Deficient Weight

A `nn.Linear` layer with a non-square weight matrix (e.g., projecting from 768 to 64) is technically K-type because the mapping is not surjective in the input space. However, this is an extremely common operation that users expect to see as S-type (it is a linear projection, not an activation or pooling). We classify it as **S with confidence 0.9** and document the caveat. The rationale: the information is preserved in the output subspace even though the ambient dimension is reduced. A truly rank-deficient square weight matrix is a degenerate case that cannot be detected statically.

### 6.2 Convolutions with Stride > 1

Stride > 1 skips input positions, which is a form of spatial downsampling. The skipped positions are never read, so their information is erased. This is classified as **K**. If the user knows their strided convolution is paired with a corresponding transposed convolution (encoder-decoder), they can override to S.

### 6.3 BatchNorm in Train vs. Eval Mode

`nn.BatchNorm` computes running mean and variance across the batch during training. This cross-sample dependency makes it K-type in training mode (it is not coordination-free). In eval mode, it uses frozen statistics and operates per-sample, making it S-type. The classifier checks `module.training` at classification time and reports the current mode. The report should note that the classification may change if the model switches modes.

### 6.4 Softmax

Softmax maps R^n to the (n-1)-simplex. It erases one degree of freedom (the outputs sum to 1, so any uniform shift of the input is invisible). This makes it K-type with confidence 1.0. Some formulations preserve scale information via temperature; these are still K-type because the simplex constraint is the fundamental issue.

### 6.5 Attention Modules

`nn.MultiheadAttention` and `F.scaled_dot_product_attention` are internally mixed. They contain S-type linear projections (Q, K, V, output), a K-type softmax over attention weights, and S-type weighted sums. The module as a whole is classified GRAY. For finer-grained analysis, users should decompose attention into its constituent operations (which the provenance tracker in SPEC-003 will do automatically).

### 6.6 Custom User Modules

Any `nn.Module` subclass not in the classification database is classified as **GRAY with confidence 0.0**. A warning is emitted. Users are expected to provide overrides for their custom modules via the `custom_rules` parameter.

### 6.7 Elementwise Multiplication by Zero

`torch.mul` is S-type in general, but multiplication by a zero tensor erases information. Since the multiplier is a runtime value, this cannot be detected statically. We classify `torch.mul` as **S with confidence 0.9** and document the caveat.

### 6.8 Reducing vs. Non-Reducing Operations

Operations like `torch.sum`, `torch.mean`, `torch.max` can be called with or without a `dim` argument. With a `dim` argument, they reduce along one axis (still K-type, fewer values than inputs). Without a `dim` argument, they reduce the entire tensor to a scalar (maximal K-type). Both cases are K-type; the only difference is severity.

### 6.9 In-Place Operations

In-place operations (e.g., `relu_`, `add_`) have the same classification as their out-of-place counterparts. The classifier normalizes in-place variants to their base operation before lookup.

---

## 7. Error Handling

| Scenario | Behavior |
|----------|----------|
| Unrecognized operation | Classify as GRAY with confidence 0.0. Emit a warning listing the operation name and type. |
| `torch.fx.symbolic_trace` fails (dynamic control flow, unsupported Python constructs) | Raise `ClassificationError` with the `torch.fx` error message. Suggest the user provide a concrete `torch.fx.GraphModule` or use `torch.fx.Tracer` with `concrete_args`. Do not silently return partial results. |
| Empty model (no operations in graph) | Return a valid `ErasureBudgetReport` with `total_ops=0`, `erasure_score=0.0`, and empty `per_node` list. |
| Model with only placeholder/output nodes | Same as empty model. Placeholders and outputs are classified S but do not count toward the erasure score. |
| User override specifies an invalid `OpClass` value | Raise `ValueError` at the start of classification, before processing any nodes. |
| Module attribute inspection fails (e.g., missing `.stride`) | Fall back to the database default classification. Emit a warning. |

---

## 8. Output Format

### 8.1 Data Classes

```python
from dataclasses import dataclass, field
from enum import Enum


class OpClass(Enum):
    S = "S"       # Information-preserving
    K = "K"       # Information-erasing
    GRAY = "GRAY" # Context-dependent


@dataclass
class ClassificationResult:
    """Classification of a single operation in the model graph."""
    op_name: str                # Node name in the fx graph (e.g., "linear_0")
    op_type: str                # Canonical type (e.g., "nn.Linear", "torch.add")
    classification: OpClass     # S, K, or GRAY
    confidence: float           # 0.0 to 1.0
    rationale: str              # Human-readable explanation


@dataclass
class ErasureBudgetReport:
    """Complete classification report for a model."""
    model_name: str
    total_ops: int              # Excludes placeholder, get_attr, output nodes
    s_count: int
    k_count: int
    gray_count: int
    erasure_score: float        # k_count / total_ops (0.0 = fully S, 1.0 = fully K)
    per_node: list[ClassificationResult] = field(default_factory=list)

    @property
    def s_ratio(self) -> float:
        """Fraction of operations that are S-type."""
        return self.s_count / self.total_ops if self.total_ops > 0 else 0.0

    @property
    def k_ratio(self) -> float:
        """Fraction of operations that are K-type."""
        return self.k_count / self.total_ops if self.total_ops > 0 else 0.0

    def per_layer_summary(self) -> dict[str, dict]:
        """Group classifications by layer prefix.

        Returns:
            Dict mapping layer prefix to {s_count, k_count, gray_count, erasure_score}.
        """
        ...

    def to_dict(self) -> dict:
        """Serialize the full report to a plain dict."""
        ...

    def to_json(self) -> str:
        """Serialize the full report to a JSON string."""
        ...
```

### 8.2 Erasure Score

The erasure score is defined as:

```
erasure_score = k_count / total_ops
```

Where `total_ops` counts only computational nodes (excludes `placeholder`, `get_attr`, and `output` nodes). GRAY nodes are excluded from both numerator and denominator for the erasure score but are reported separately.

If there are GRAY nodes, the report also provides bounds:

```
erasure_score_lower = k_count / (total_ops)                    # assume all GRAY are S
erasure_score_upper = (k_count + gray_count) / (total_ops)     # assume all GRAY are K
```

### 8.3 JSON Output Schema

```json
{
  "model_name": "GPT2",
  "total_ops": 847,
  "s_count": 635,
  "k_count": 148,
  "gray_count": 64,
  "erasure_score": 0.175,
  "erasure_score_lower": 0.175,
  "erasure_score_upper": 0.250,
  "per_node": [
    {
      "op_name": "transformer.h.0.attn.c_attn",
      "op_type": "nn.Linear",
      "classification": "S",
      "confidence": 0.9,
      "rationale": "Affine transform; full-rank weight matrix is invertible."
    }
  ],
  "per_layer_summary": {
    "transformer.h.0": {
      "s_count": 18,
      "k_count": 4,
      "gray_count": 2,
      "erasure_score": 0.167
    }
  }
}
```

---

## 9. API Sketch

```python
from bwsk.classify import classify_operation, classify_model, OpClass

# Classify a single module
result = classify_operation(nn.ReLU())
assert result.classification == OpClass.K

# Classify a full model via torch.fx
report = classify_model(model)
print(report.erasure_score)         # 0.175
print(report.s_ratio)               # 0.75
print(report.per_layer_summary())   # dict of layer -> {s_count, k_count, ...}

# Classify with custom overrides
report = classify_model(model, custom_rules={
    "nn.MyCustomNorm": OpClass.S,
    "nn.Softmax": OpClass.S,       # Override default K classification
})

# Serialize
json_str = report.to_json()
plain_dict = report.to_dict()

# CLI usage
# $ bwsk classify model.pt --output report.json
# $ bwsk classify model.pt --format table
```

### 9.1 classify_operation

```python
def classify_operation(
    op: nn.Module,
    custom_rules: dict[str, OpClass] | None = None,
) -> ClassificationResult:
    """Classify a single nn.Module instance.

    Args:
        op: The module to classify.
        custom_rules: Optional dict mapping canonical op names to OpClass overrides.

    Returns:
        A ClassificationResult for this operation.

    Raises:
        TypeError: If op is not an nn.Module.
    """
```

### 9.2 classify_model

```python
def classify_model(
    model: nn.Module,
    custom_rules: dict[str, OpClass] | None = None,
    concrete_args: dict | None = None,
) -> ErasureBudgetReport:
    """Classify all operations in a model via torch.fx tracing.

    Args:
        model: The PyTorch model to classify.
        custom_rules: Optional dict mapping canonical op names to OpClass overrides.
        concrete_args: Optional concrete args for torch.fx.symbolic_trace
            (needed for models with dynamic control flow).

    Returns:
        An ErasureBudgetReport with per-node and aggregate classifications.

    Raises:
        ClassificationError: If torch.fx tracing fails.
        ValueError: If custom_rules contains invalid OpClass values.
    """
```

---

## 10. Test Plan

### 10.1 Unit Tests: Classification Database

Every entry in the classification database must have a corresponding unit test. Test structure:

```python
class TestClassificationDatabase:
    # Linear layers
    def test_linear_is_s(self): ...
    def test_lazy_linear_is_s(self): ...
    def test_bilinear_is_s(self): ...

    # Convolutions
    def test_conv2d_stride1_is_s(self): ...
    def test_conv2d_stride2_is_k(self): ...
    def test_conv_transpose2d_is_s(self): ...

    # Normalization
    def test_layer_norm_is_s(self): ...
    def test_batch_norm_train_is_k(self): ...
    def test_batch_norm_eval_is_s(self): ...
    def test_group_norm_is_s(self): ...
    def test_instance_norm_is_s(self): ...

    # Activations
    def test_relu_is_k(self): ...
    def test_leaky_relu_is_s(self): ...
    def test_gelu_is_k(self): ...
    def test_silu_is_k(self): ...
    def test_softplus_is_s(self): ...
    def test_sigmoid_is_k(self): ...
    def test_tanh_is_k(self): ...
    def test_softmax_is_k(self): ...
    def test_elu_is_k(self): ...
    def test_mish_is_k(self): ...
    def test_prelu_is_s(self): ...
    def test_hardswish_is_k(self): ...
    def test_hardsigmoid_is_k(self): ...

    # Pooling
    def test_max_pool2d_is_k(self): ...
    def test_avg_pool2d_is_k(self): ...
    def test_adaptive_avg_pool2d_is_k(self): ...

    # Dropout
    def test_dropout_is_k(self): ...
    def test_alpha_dropout_is_k(self): ...

    # Recurrent
    def test_lstm_is_gray(self): ...
    def test_gru_is_gray(self): ...

    # Embedding
    def test_embedding_is_s(self): ...
    def test_embedding_bag_is_k(self): ...

    # Attention
    def test_multihead_attention_is_gray(self): ...

    # Loss
    def test_cross_entropy_loss_is_k(self): ...
    def test_mse_loss_is_k(self): ...
```

### 10.2 Edge Case Tests

```python
class TestEdgeCases:
    def test_conv2d_stride1_vs_stride2(self): ...
    def test_batch_norm_train_vs_eval(self): ...
    def test_custom_module_defaults_to_gray(self): ...
    def test_custom_module_with_override(self): ...
    def test_empty_model(self): ...
    def test_model_with_only_linear_layers(self): ...
    def test_inplace_relu_same_as_relu(self): ...
```

### 10.3 Integration Tests

```python
class TestIntegration:
    def test_classify_simple_mlp(self):
        """MLP: Linear -> ReLU -> Linear -> ReLU -> Linear.
        Expected: 3 S (Linear), 2 K (ReLU), erasure_score ~ 0.4."""

    def test_classify_simple_cnn(self):
        """CNN: Conv2d -> ReLU -> MaxPool -> Conv2d -> ReLU -> AdaptiveAvgPool -> Linear.
        Expected: mix of S and K, erasure_score depends on strides."""

    def test_classify_residual_block(self):
        """ResBlock: x + F(x) where F = Conv -> BN -> ReLU -> Conv -> BN.
        Tests that the residual addition is S-type."""

    def test_classify_transformer_block(self):
        """Single transformer block with self-attention and FFN.
        Tests GRAY classification for attention, S for linear, K for activations."""
```

### 10.4 Performance Tests

```python
class TestPerformance:
    @pytest.mark.slow
    def test_classification_time_small_model(self):
        """Classification of a 10M parameter model should complete in < 1s."""

    @pytest.mark.slow
    def test_classification_time_large_model(self):
        """Classification of a 1B parameter model graph should complete in < 5s.
        Note: we trace the graph only, not the weights."""
```

### 10.5 Custom Rules Tests

```python
class TestCustomRules:
    def test_override_relu_to_s(self):
        """User overrides ReLU to S-type. Verify it takes effect."""

    def test_override_custom_module(self):
        """User provides classification for an unknown custom module."""

    def test_invalid_override_raises(self):
        """Invalid OpClass value in custom_rules raises ValueError."""

    def test_override_takes_priority_over_database(self):
        """User override should supersede the default database entry."""
```

### 10.6 Serialization Tests

```python
class TestSerialization:
    def test_to_json_roundtrip(self):
        """Report serialized to JSON and deserialized should be equivalent."""

    def test_to_dict_structure(self):
        """Report.to_dict() should have the expected keys and types."""

    def test_per_layer_summary_grouping(self):
        """per_layer_summary should correctly group by layer prefix."""
```

---

## Appendix A: Classification Decision Tree (Visual Summary)

```
Is there a user override?
├── YES → Use override (confidence 1.0)
└── NO → Is the op in the default database?
    ├── YES → Does it need attribute refinement?
    │   ├── YES (Conv stride, BN mode) → Refine and return
    │   └── NO → Return database entry
    └── NO → Is it a known container module?
        ├── YES → GRAY (classify children individually)
        └── NO → Is it an unknown nn.Module?
            ├── YES → GRAY (confidence 0.0, emit warning)
            └── NO → GRAY (confidence 0.0, emit warning)
```

## Appendix B: Theoretical Basis for S/K Classification

An operation `f: X -> Y` is:

- **S-type (injective)**: `f(a) = f(b)` implies `a = b`. No two distinct inputs produce the same output. Information is preserved.
- **K-type (non-injective)**: there exist `a != b` such that `f(a) = f(b)`. Some inputs are conflated. Information is erased.

In practice, we relax "injective" to "injective up to floating-point precision" and "practically invertible given the learned parameters." The confidence score reflects this relaxation:

- **1.0**: Classification is a mathematical certainty (e.g., ReLU erases negatives, pooling discards non-selected values).
- **0.9-0.95**: Classification holds for typical configurations but has theoretical edge cases (e.g., Linear with rank-deficient weights, mul by zero).
- **0.7-0.8**: Classification depends on runtime state or is a composite of S and K sub-operations (e.g., BatchNorm, attention).
- **0.0-0.5**: Unknown or unrecognized operation; default GRAY.

The connection to the BWSK combinator system: S-type operations correspond to the **B** (composition) and **S** (fan-out) combinators, which preserve information flow. K-type operations correspond to the **K** (erase) combinator, which discards inputs. The **W** (share) combinator reuses information and is S-type. The classifier's job is to label each node so the combinator-level analysis knows where information boundaries lie.
