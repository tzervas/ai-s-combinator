"""S/K operation classifier for neural network operations.

Classifies each operation in a neural network as:
- S-type: information-preserving, reversible, coordination-free
- K-type: information-erasing, synchronization point
- Gray: context-dependent, requires manual classification

Usage:
    from bwsk.classify import classify_operation, classify_model, OpClass

    result = classify_operation(nn.ReLU())   # -> ClassificationResult(K, 1.0, ...)
    report = classify_model(model)           # -> ErasureBudgetReport
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch.nn as nn


class OpClass(Enum):
    """Classification of a neural network operation."""

    S = "S"  # Information-preserving
    K = "K"  # Information-erasing
    GRAY = "GRAY"  # Context-dependent


@dataclass
class ClassificationResult:
    """Classification of a single operation in the model graph."""

    op_name: str
    op_type: str
    classification: OpClass
    confidence: float
    rationale: str


@dataclass
class ErasureBudgetReport:
    """Complete classification report for a model."""

    model_name: str
    total_ops: int
    s_count: int
    k_count: int
    gray_count: int
    erasure_score: float
    per_node: list[ClassificationResult] = field(default_factory=list)

    @property
    def s_ratio(self) -> float:
        """Fraction of operations that are S-type."""
        return self.s_count / self.total_ops if self.total_ops > 0 else 0.0

    @property
    def k_ratio(self) -> float:
        """Fraction of operations that are K-type."""
        return self.k_count / self.total_ops if self.total_ops > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full report to a plain dict."""
        return {
            "model_name": self.model_name,
            "total_ops": self.total_ops,
            "s_count": self.s_count,
            "k_count": self.k_count,
            "gray_count": self.gray_count,
            "erasure_score": self.erasure_score,
            "per_node": [
                {
                    "op_name": r.op_name,
                    "op_type": r.op_type,
                    "classification": r.classification.value,
                    "confidence": r.confidence,
                    "rationale": r.rationale,
                }
                for r in self.per_node
            ],
        }

    def per_layer_summary(self) -> dict[str, dict[str, Any]]:
        """Group per-node results by top-level layer prefix.

        Extracts the first dotted component of each op_name (e.g., "layer1"
        from "layer1_0") to group nodes by logical layer. This enables
        per-layer erasure analysis to identify which layers are K-heavy.

        Returns:
            Dict mapping layer prefix to {s_count, k_count, gray_count,
            erasure_score}.
        """
        groups: dict[str, list[ClassificationResult]] = {}
        for r in self.per_node:
            # torch.fx names use underscores: "layer1_0", "layer1_1"
            # Extract prefix by splitting on underscore and taking first part
            # that looks like a layer name. For names like "layer1_0_1",
            # the prefix is "layer1".
            parts = r.op_name.split("_")
            prefix = parts[0] if parts else r.op_name
            groups.setdefault(prefix, []).append(r)

        summary: dict[str, dict[str, Any]] = {}
        for prefix, results in groups.items():
            s = sum(1 for r in results if r.classification == OpClass.S)
            k = sum(1 for r in results if r.classification == OpClass.K)
            g = sum(1 for r in results if r.classification == OpClass.GRAY)
            total = len(results)
            summary[prefix] = {
                "s_count": s,
                "k_count": k,
                "gray_count": g,
                "erasure_score": k / total if total > 0 else 0.0,
            }
        return summary

    def to_json(self) -> str:
        """Serialize the full report to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Classification Database
# ---------------------------------------------------------------------------

# Each entry: (OpClass, confidence, rationale)
_ClassEntry = tuple[OpClass, float, str]

# Default classification database mapping module class names to (OpClass, confidence, rationale).
_DEFAULT_DB: dict[str, _ClassEntry] = {}


def _register(names: str | list[str], cls: OpClass, confidence: float, rationale: str) -> None:
    """Register one or more operation names in the classification database."""
    if isinstance(names, str):
        names = [names]
    for name in names:
        _DEFAULT_DB[name] = (cls, confidence, rationale)


# --- Linear layers ---
_register("nn.Linear", OpClass.S, 0.9, "Affine transform; invertible given full-rank weight.")
_register("nn.LazyLinear", OpClass.S, 0.9, "Same as nn.Linear once materialized.")
_register("nn.Bilinear", OpClass.S, 0.8, "Bilinear map; preserves info from both inputs.")

# --- Convolutions (stride=1 default; stride>1 refined at classification time) ---
_register(
    ["nn.Conv1d", "nn.Conv2d", "nn.Conv3d"],
    OpClass.S,
    0.9,
    "Convolution with stride=1; linear map with shared weights.",
)
_register(
    ["nn.ConvTranspose1d", "nn.ConvTranspose2d", "nn.ConvTranspose3d"],
    OpClass.S,
    0.9,
    "Transposed convolution; upsamples, no information loss.",
)

# --- Normalization ---
_register(
    "nn.LayerNorm", OpClass.S, 0.95, "Per-sample normalization; invertible given affine params."
)
_register("nn.RMSNorm", OpClass.S, 0.95, "Simpler LayerNorm variant; invertible given scale.")
_register("nn.GroupNorm", OpClass.S, 0.9, "Per-group normalization; no cross-sample dependency.")
_register(
    ["nn.InstanceNorm1d", "nn.InstanceNorm2d", "nn.InstanceNorm3d"],
    OpClass.S,
    0.9,
    "Per-channel per-sample normalization; no cross-sample dependency.",
)
_register(
    ["nn.BatchNorm1d", "nn.BatchNorm2d", "nn.BatchNorm3d"],
    OpClass.GRAY,
    0.7,
    "BatchNorm: K in train mode (cross-sample stats), S in eval mode (frozen stats).",
)

# --- Activations ---
_register("nn.ReLU", OpClass.K, 1.0, "max(0,x) erases all negative values.")
_register("nn.ReLU6", OpClass.K, 1.0, "Clips to [0,6]; erases negatives and values > 6.")
_register("nn.LeakyReLU", OpClass.S, 0.95, "max(alpha*x, x) with alpha>0; bijective, no info loss.")
_register("nn.PReLU", OpClass.S, 0.95, "Parametric LeakyReLU; bijective for alpha != 0.")
_register("nn.ELU", OpClass.K, 0.9, "Saturates toward -alpha for negatives; not injective.")
_register("nn.GELU", OpClass.K, 0.9, "Non-monotonic near zero; not injective.")
_register("nn.SiLU", OpClass.K, 0.9, "x*sigmoid(x) is non-monotonic for x<0; not injective.")
_register("nn.Mish", OpClass.K, 0.9, "x*tanh(softplus(x)); non-monotonic for large negative x.")
_register("nn.Softplus", OpClass.S, 0.95, "log(1+exp(x)); strictly monotonic and invertible.")
_register("nn.Sigmoid", OpClass.K, 0.95, "Maps R to (0,1); saturates, practically K-type.")
_register("nn.Tanh", OpClass.K, 0.95, "Maps R to (-1,1); saturates, practically K-type.")
_register("nn.Softmax", OpClass.K, 1.0, "Reduces dimensionality by 1 (outputs sum to 1).")
_register("nn.LogSoftmax", OpClass.K, 1.0, "log(softmax(x)); same dimensionality reduction.")
_register("nn.Hardswish", OpClass.K, 0.9, "Clips to 0 for x<=-3; erases in clipped region.")
_register("nn.Hardsigmoid", OpClass.K, 0.9, "Clips at boundaries; info loss in saturated regions.")
_register("nn.SELU", OpClass.K, 0.9, "Saturates for negatives like ELU; not injective.")
_register("nn.CELU", OpClass.K, 0.9, "Saturates for negatives; not injective.")

# --- Pooling ---
_register(
    ["nn.MaxPool1d", "nn.MaxPool2d", "nn.MaxPool3d"],
    OpClass.K,
    1.0,
    "Selects maximum from window; all other values erased.",
)
_register(
    ["nn.AvgPool1d", "nn.AvgPool2d", "nn.AvgPool3d"],
    OpClass.K,
    1.0,
    "Averages window; individual values irrecoverable.",
)
_register(
    ["nn.AdaptiveAvgPool1d", "nn.AdaptiveAvgPool2d", "nn.AdaptiveAvgPool3d"],
    OpClass.K,
    1.0,
    "Adaptive average pooling; same erasure as AvgPool.",
)
_register(
    ["nn.AdaptiveMaxPool1d", "nn.AdaptiveMaxPool2d", "nn.AdaptiveMaxPool3d"],
    OpClass.K,
    1.0,
    "Adaptive max pooling; same erasure as MaxPool.",
)

# --- Dropout ---
_register(
    ["nn.Dropout", "nn.Dropout1d", "nn.Dropout2d", "nn.Dropout3d"],
    OpClass.K,
    1.0,
    "Randomly zeroes elements; erases information stochastically.",
)
_register(
    "nn.AlphaDropout",
    OpClass.K,
    1.0,
    "SELU-compatible dropout; still randomly erases activations.",
)

# --- Recurrent ---
_register("nn.RNN", OpClass.GRAY, 0.6, "Mixed: linear transforms (S) + tanh (K) internally.")
_register(
    "nn.LSTM",
    OpClass.GRAY,
    0.6,
    "Mixed: sigmoid gates (K) + tanh (K) + linear (S). Forget gate is explicit erasure.",
)
_register(
    "nn.GRU",
    OpClass.GRAY,
    0.6,
    "Mixed: sigmoid gates (K) + tanh (K) + linear (S).",
)

# --- Embedding ---
_register("nn.Embedding", OpClass.S, 1.0, "Lookup table; injective mapping, no info loss.")
_register(
    "nn.EmbeddingBag",
    OpClass.K,
    0.9,
    "Aggregates embeddings (sum/mean); individual values lost.",
)

# --- Attention ---
_register(
    "nn.MultiheadAttention",
    OpClass.GRAY,
    0.7,
    "Mixed: Q/K/V projections (S), softmax (K), weighted sum (S).",
)

# --- Loss functions ---
_LOSS_NAMES = [
    "nn.CrossEntropyLoss",
    "nn.NLLLoss",
    "nn.MSELoss",
    "nn.L1Loss",
    "nn.SmoothL1Loss",
    "nn.BCELoss",
    "nn.BCEWithLogitsLoss",
    "nn.KLDivLoss",
    "nn.HuberLoss",
    "nn.CosineEmbeddingLoss",
    "nn.TripletMarginLoss",
    "nn.CTCLoss",
    "nn.MarginRankingLoss",
    "nn.MultiMarginLoss",
    "nn.MultiLabelMarginLoss",
    "nn.MultiLabelSoftMarginLoss",
    "nn.HingeEmbeddingLoss",
    "nn.PoissonNLLLoss",
    "nn.GaussianNLLLoss",
]
_register(_LOSS_NAMES, OpClass.K, 1.0, "Loss function; reduces to scalar, maximal erasure.")

# --- Container modules (pass-through, classify children individually) ---
_register(
    ["nn.Sequential", "nn.ModuleList", "nn.ModuleDict"],
    OpClass.S,
    0.5,
    "Container module; classification depends on children.",
)

# --- Misc ---
_register("nn.Identity", OpClass.S, 1.0, "Identity function; no transformation.")
_register("nn.Flatten", OpClass.S, 1.0, "Reshape; bijection on element ordering.")
_register("nn.Unflatten", OpClass.S, 1.0, "Reshape; bijection on element ordering.")


# ---------------------------------------------------------------------------
# Functional operations database (for torch.fx call_function nodes)
# ---------------------------------------------------------------------------

_FUNCTIONAL_DB: dict[str, _ClassEntry] = {}


def _register_fn(names: str | list[str], cls: OpClass, confidence: float, rationale: str) -> None:
    if isinstance(names, str):
        names = [names]
    for name in names:
        _FUNCTIONAL_DB[name] = (cls, confidence, rationale)


_register_fn(
    ["torch.add", "torch.sub"],
    OpClass.S,
    1.0,
    "Elementwise arithmetic; invertible given one operand.",
)
_register_fn("torch.mul", OpClass.S, 0.9, "Elementwise mul; invertible if no zeros in multiplier.")
_register_fn("torch.div", OpClass.S, 0.9, "Elementwise div; invertible if divisor nonzero.")
_register_fn(
    ["torch.matmul", "torch.bmm", "torch.mm"],
    OpClass.S,
    0.9,
    "Matrix multiply; S-type given full-rank operands.",
)
_register_fn(
    ["torch.cat", "torch.stack"],
    OpClass.S,
    1.0,
    "Concatenation/stacking; all inputs recoverable.",
)
_register_fn(
    ["torch.chunk", "torch.split", "torch.tensor_split"],
    OpClass.S,
    1.0,
    "Splits tensor; no information loss.",
)
_register_fn(
    [
        "torch.reshape",
        "torch.permute",
        "torch.transpose",
        "torch.flatten",
        "torch.unsqueeze",
        "torch.squeeze",
        "torch.contiguous",
    ],
    OpClass.S,
    1.0,
    "Shape manipulation; bijection on elements.",
)
_register_fn("torch.neg", OpClass.S, 1.0, "Negation; bijective.")
_register_fn("torch.exp", OpClass.S, 1.0, "Exponential; strictly monotonic, invertible.")
_register_fn("torch.log", OpClass.S, 0.95, "Log; invertible for positive inputs.")
_register_fn("torch.clamp", OpClass.K, 1.0, "Clips values; distinct inputs map to boundary.")
_register_fn("torch.abs", OpClass.K, 1.0, "Erases sign information.")
_register_fn(
    ["torch.sum", "torch.mean", "torch.prod"],
    OpClass.K,
    1.0,
    "Reduces dimension(s); individual values irrecoverable.",
)
_register_fn(
    ["torch.max", "torch.min", "torch.amax", "torch.amin"],
    OpClass.K,
    1.0,
    "Selects extremum; all other values erased.",
)
_register_fn(
    ["torch.argmax", "torch.argmin"],
    OpClass.K,
    1.0,
    "Returns index only; entire tensor reduced.",
)

# torch.nn.functional
_register_fn("torch.nn.functional.relu", OpClass.K, 1.0, "Functional ReLU; erases negatives.")
_register_fn("torch.nn.functional.gelu", OpClass.K, 0.9, "Functional GELU; non-monotonic.")
_register_fn("torch.nn.functional.silu", OpClass.K, 0.9, "Functional SiLU; non-monotonic.")
_register_fn("torch.nn.functional.leaky_relu", OpClass.S, 0.95, "Functional LeakyReLU; bijective.")
_register_fn("torch.nn.functional.softplus", OpClass.S, 0.95, "Functional Softplus; invertible.")
_register_fn("torch.nn.functional.dropout", OpClass.K, 1.0, "Functional dropout; erases randomly.")
_register_fn("torch.nn.functional.layer_norm", OpClass.S, 0.95, "Functional LayerNorm; invertible.")
_register_fn("torch.nn.functional.softmax", OpClass.K, 1.0, "Functional softmax; reduces dim.")
_register_fn("torch.nn.functional.log_softmax", OpClass.K, 1.0, "Functional log_softmax.")
_register_fn("torch.nn.functional.cross_entropy", OpClass.K, 1.0, "Reduces to scalar.")
_register_fn(
    "torch.nn.functional.scaled_dot_product_attention",
    OpClass.GRAY,
    0.7,
    "Mixed S and K internally.",
)
_register_fn("torch.nn.functional.linear", OpClass.S, 0.9, "Linear transform; invertible.")
_register_fn("torch.nn.functional.embedding", OpClass.S, 1.0, "Lookup; injective.")

# Operator module (torch.fx represents Python operators via operator module)
import operator  # noqa: E402

_OPERATOR_DB: dict[Any, _ClassEntry] = {
    operator.add: (OpClass.S, 1.0, "Addition; invertible."),
    operator.sub: (OpClass.S, 1.0, "Subtraction; invertible."),
    operator.mul: (OpClass.S, 0.9, "Multiplication; invertible if nonzero."),
    operator.truediv: (OpClass.S, 0.9, "Division; invertible if nonzero."),
    operator.floordiv: (OpClass.K, 0.9, "Floor division; erases fractional part."),
    operator.mod: (OpClass.K, 0.9, "Modulo; erases quotient."),
    operator.neg: (OpClass.S, 1.0, "Negation; bijective."),
    operator.getitem: (OpClass.S, 1.0, "Indexing; selects subset, info preserved in context."),
}


# ---------------------------------------------------------------------------
# Method-level database (for torch.fx call_method nodes)
# ---------------------------------------------------------------------------

_METHOD_DB: dict[str, _ClassEntry] = {
    "view": (OpClass.S, 1.0, "View/reshape; bijection."),
    "reshape": (OpClass.S, 1.0, "Reshape; bijection."),
    "permute": (OpClass.S, 1.0, "Permute dims; bijection."),
    "transpose": (OpClass.S, 1.0, "Transpose; bijection."),
    "contiguous": (OpClass.S, 1.0, "Memory layout; no data change."),
    "flatten": (OpClass.S, 1.0, "Flatten; bijection."),
    "unflatten": (OpClass.S, 1.0, "Unflatten; bijection."),
    "unsqueeze": (OpClass.S, 1.0, "Add dim; bijection."),
    "squeeze": (OpClass.S, 1.0, "Remove dim; bijection."),
    "expand": (OpClass.S, 1.0, "Broadcast; no data loss."),
    "repeat": (OpClass.S, 0.9, "Repeat; data duplicated, original recoverable."),
    "chunk": (OpClass.S, 1.0, "Split; no data loss."),
    "split": (OpClass.S, 1.0, "Split; no data loss."),
    "sum": (OpClass.K, 1.0, "Reduction; individual values lost."),
    "mean": (OpClass.K, 1.0, "Reduction; individual values lost."),
    "max": (OpClass.K, 1.0, "Selects max; others erased."),
    "min": (OpClass.K, 1.0, "Selects min; others erased."),
    "clamp": (OpClass.K, 1.0, "Clips values."),
    "abs": (OpClass.K, 1.0, "Erases sign."),
    "relu": (OpClass.K, 1.0, "Erases negatives."),
    "sigmoid": (OpClass.K, 0.95, "Saturates."),
    "tanh": (OpClass.K, 0.95, "Saturates."),
    "softmax": (OpClass.K, 1.0, "Reduces dim."),
    "log_softmax": (OpClass.K, 1.0, "Reduces dim."),
    "add": (OpClass.S, 1.0, "Addition; invertible."),
    "sub": (OpClass.S, 1.0, "Subtraction; invertible."),
    "mul": (OpClass.S, 0.9, "Multiplication; invertible if nonzero."),
    "div": (OpClass.S, 0.9, "Division; invertible if nonzero."),
    "matmul": (OpClass.S, 0.9, "Matrix multiply; S given full-rank."),
}


# ---------------------------------------------------------------------------
# Convolution stride refinement
# ---------------------------------------------------------------------------

_CONV_TYPES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
)

_BATCH_NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
)


def _refine_by_attributes(module: nn.Module) -> _ClassEntry | None:
    """Refine classification based on module attributes.

    Returns None if no refinement is needed (use default).
    """
    if isinstance(module, _CONV_TYPES):
        if all(s == 1 for s in module.stride):
            return (OpClass.S, 0.9, "Convolution with stride=1; no spatial downsampling.")
        return (
            OpClass.K,
            0.9,
            f"Convolution with stride={module.stride}; spatial downsampling erases information.",
        )

    if isinstance(module, _BATCH_NORM_TYPES):
        if module.training:
            return (OpClass.K, 0.8, "BatchNorm in train mode; cross-sample stats are K-type.")
        return (OpClass.S, 0.8, "BatchNorm in eval mode; frozen stats, per-sample invertible.")

    return None


# ---------------------------------------------------------------------------
# Canonical name resolution
# ---------------------------------------------------------------------------


def _canonical_name_for_module(module: nn.Module) -> str:
    """Get canonical name for an nn.Module (e.g., 'nn.Linear')."""
    cls_name = type(module).__name__
    module_path = type(module).__module__
    if module_path.startswith("torch.nn"):
        return f"nn.{cls_name}"
    return cls_name


def _canonical_name_for_function(target: Any) -> str:
    """Get canonical name for a function target.

    torch.fx may resolve functions to internal names like
    ``torch._VariableFunctionsClass.add`` instead of ``torch.add``.
    We normalize these to the public API name so the functional DB
    lookup succeeds.
    """
    if hasattr(target, "__module__") and hasattr(target, "__qualname__"):
        module = getattr(target, "__module__", "")
        qualname = getattr(target, "__qualname__", "")
        if module.startswith("torch.nn.functional"):
            return f"torch.nn.functional.{qualname}"
        if module.startswith("torch"):
            # Normalize internal class method names like
            # "_VariableFunctionsClass.add" -> "add"
            func_name = qualname.rsplit(".", 1)[-1]
            canonical = f"torch.{func_name}"
            # Only use the simplified name if it's in our DB
            if canonical in _FUNCTIONAL_DB:
                return canonical
            return f"torch.{qualname}"
    return str(target)


# ---------------------------------------------------------------------------
# Public API: classify_operation
# ---------------------------------------------------------------------------


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
    """
    canonical = _canonical_name_for_module(op)

    # Step 1: User override
    if custom_rules:
        for key, override_class in custom_rules.items():
            if key == canonical or key == type(op).__name__:
                return ClassificationResult(
                    op_name="",
                    op_type=canonical,
                    classification=override_class,
                    confidence=1.0,
                    rationale="User override",
                )

    # Step 2: Attribute-dependent refinement
    refined = _refine_by_attributes(op)
    if refined is not None:
        cls, conf, rationale = refined
        return ClassificationResult(
            op_name="",
            op_type=canonical,
            classification=cls,
            confidence=conf,
            rationale=rationale,
        )

    # Step 3: Database lookup
    if canonical in _DEFAULT_DB:
        cls, conf, rationale = _DEFAULT_DB[canonical]
        return ClassificationResult(
            op_name="",
            op_type=canonical,
            classification=cls,
            confidence=conf,
            rationale=rationale,
        )

    # Step 4: Default to GRAY
    warnings.warn(
        f"Unrecognized operation: {canonical}. Defaulting to GRAY.",
        stacklevel=2,
    )
    return ClassificationResult(
        op_name="",
        op_type=canonical,
        classification=OpClass.GRAY,
        confidence=0.0,
        rationale=f"Unrecognized operation: {canonical}; defaulting to GRAY.",
    )


# ---------------------------------------------------------------------------
# Public API: classify_model
# ---------------------------------------------------------------------------


class ClassificationError(Exception):
    """Raised when torch.fx tracing fails."""


def classify_model(
    model: nn.Module,
    custom_rules: dict[str, OpClass] | None = None,
    concrete_args: dict[str, Any] | None = None,
) -> ErasureBudgetReport:
    """Classify all operations in a model via torch.fx tracing.

    Args:
        model: The PyTorch model to classify.
        custom_rules: Optional dict mapping canonical op names to OpClass overrides.
        concrete_args: Optional concrete args for torch.fx.symbolic_trace.

    Returns:
        An ErasureBudgetReport with per-node and aggregate classifications.

    Raises:
        ClassificationError: If torch.fx tracing fails.
    """
    import torch.fx

    try:
        if concrete_args:
            graph_module = torch.fx.symbolic_trace(model, concrete_args=concrete_args)
        else:
            graph_module = torch.fx.symbolic_trace(model)
    except Exception as e:
        raise ClassificationError(f"torch.fx tracing failed: {e}") from e

    results: list[ClassificationResult] = []

    for node in graph_module.graph.nodes:
        # Skip structural nodes (placeholder, get_attr, output)
        if node.op in ("placeholder", "get_attr", "output"):
            continue

        result = _classify_node(node, graph_module, custom_rules)
        result.op_name = node.name
        results.append(result)

    s_count = sum(1 for r in results if r.classification == OpClass.S)
    k_count = sum(1 for r in results if r.classification == OpClass.K)
    gray_count = sum(1 for r in results if r.classification == OpClass.GRAY)
    total_ops = len(results)
    erasure_score = k_count / total_ops if total_ops > 0 else 0.0

    model_name = type(model).__name__

    return ErasureBudgetReport(
        model_name=model_name,
        total_ops=total_ops,
        s_count=s_count,
        k_count=k_count,
        gray_count=gray_count,
        erasure_score=erasure_score,
        per_node=results,
    )


def _classify_node(
    node: Any,  # torch.fx.Node
    graph_module: Any,  # torch.fx.GraphModule
    custom_rules: dict[str, OpClass] | None,
) -> ClassificationResult:
    """Classify a single torch.fx.Node."""
    if node.op == "call_module":
        return _classify_call_module(node, graph_module, custom_rules)
    elif node.op == "call_function":
        return _classify_call_function(node, custom_rules)
    elif node.op == "call_method":
        return _classify_call_method(node, custom_rules)
    else:
        return ClassificationResult(
            op_name=node.name,
            op_type=str(node.op),
            classification=OpClass.GRAY,
            confidence=0.0,
            rationale=f"Unknown node op: {node.op}",
        )


def _classify_call_module(
    node: Any,
    graph_module: Any,
    custom_rules: dict[str, OpClass] | None,
) -> ClassificationResult:
    """Classify a call_module node."""
    module = graph_module.get_submodule(node.target)
    return classify_operation(module, custom_rules=custom_rules)


def _classify_call_function(
    node: Any,
    custom_rules: dict[str, OpClass] | None,
) -> ClassificationResult:
    """Classify a call_function node."""
    target = node.target
    canonical = _canonical_name_for_function(target)

    # Step 1: User override
    if custom_rules and canonical in custom_rules:
        return ClassificationResult(
            op_name=node.name,
            op_type=canonical,
            classification=custom_rules[canonical],
            confidence=1.0,
            rationale="User override",
        )

    # Step 2: Check operator DB (for Python operator module functions)
    if target in _OPERATOR_DB:
        cls, conf, rationale = _OPERATOR_DB[target]
        return ClassificationResult(
            op_name=node.name,
            op_type=canonical,
            classification=cls,
            confidence=conf,
            rationale=rationale,
        )

    # Step 3: Check functional DB
    if canonical in _FUNCTIONAL_DB:
        cls, conf, rationale = _FUNCTIONAL_DB[canonical]
        return ClassificationResult(
            op_name=node.name,
            op_type=canonical,
            classification=cls,
            confidence=conf,
            rationale=rationale,
        )

    # Step 4: Default
    warnings.warn(
        f"Unrecognized function: {canonical}. Defaulting to GRAY.",
        stacklevel=2,
    )
    return ClassificationResult(
        op_name=node.name,
        op_type=canonical,
        classification=OpClass.GRAY,
        confidence=0.0,
        rationale=f"Unrecognized function: {canonical}; defaulting to GRAY.",
    )


def _classify_call_method(
    node: Any,
    custom_rules: dict[str, OpClass] | None,
) -> ClassificationResult:
    """Classify a call_method node."""
    method_name = node.target  # e.g., "view", "reshape", "add"

    # Step 1: User override
    if custom_rules and method_name in custom_rules:
        return ClassificationResult(
            op_name=node.name,
            op_type=f"Tensor.{method_name}",
            classification=custom_rules[method_name],
            confidence=1.0,
            rationale="User override",
        )

    # Step 2: Method DB
    if method_name in _METHOD_DB:
        cls, conf, rationale = _METHOD_DB[method_name]
        return ClassificationResult(
            op_name=node.name,
            op_type=f"Tensor.{method_name}",
            classification=cls,
            confidence=conf,
            rationale=rationale,
        )

    # Step 3: Default
    warnings.warn(
        f"Unrecognized method: {method_name}. Defaulting to GRAY.",
        stacklevel=2,
    )
    return ClassificationResult(
        op_name=node.name,
        op_type=f"Tensor.{method_name}",
        classification=OpClass.GRAY,
        confidence=0.0,
        rationale=f"Unrecognized method: {method_name}; defaulting to GRAY.",
    )
