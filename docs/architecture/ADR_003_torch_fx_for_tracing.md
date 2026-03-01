# ADR-003: Use torch.fx Symbolic Tracing for S/K Classifier Graph Analysis
**Date**: 2026-02-28
**Status**: Accepted

## Context

The S/K classifier (SPEC-001) must inspect the computation graph of arbitrary PyTorch models and classify each operation as S-type (information-preserving), K-type (information-erasing), or Gray (context-dependent).

To do this, the classifier needs access to a structured representation of the model's computation graph — one that exposes individual operations, their types, and how they are connected.

Several mechanisms exist for obtaining this representation from a PyTorch model:

- `torch.fx` symbolic tracing
- `torch.compile` / TorchDynamo
- Manual forward hook inspection
- Static source analysis (AST parsing)
- ONNX export

Each mechanism has different tradeoffs in terms of API stability, granularity of the resulting graph, and ability to handle the range of models the classifier will encounter.

## Decision

Use `torch.fx` symbolic tracing as the primary mechanism for analyzing PyTorch model computation graphs in the S/K classifier.

When `torch.fx` tracing succeeds, the classifier operates on the resulting `fx.Graph` at the individual node level. Each `fx.Node` provides:

- `opcode`: one of `call_module`, `call_function`, `call_method`, `placeholder`, `get_attr`, `output`
- `target`: the actual function, method, or module being called
- `args` and `kwargs`: the node's inputs, expressed as references to other nodes

This node-level representation is the correct granularity for S/K classification — it exposes individual operations such as `F.relu`, `torch.matmul`, and `nn.LayerNorm` as distinct nodes, allowing each to be classified independently.

`torch.fx` is the chosen primary mechanism because:

- It produces a clean, well-defined intermediate representation (`fx.Graph` with `Node` objects) that is directly traversable
- It works with most standard PyTorch modules without modification
- It has been a stable, well-documented API since PyTorch 2.0
- The resulting graph is inspectable at the operation level, which is exactly what S/K classification requires

**Fallback strategy** for models that fail `torch.fx` tracing:

1. If the entire model fails tracing with `torch.fx.proxy.TraceError`, fall back to module-level classification: iterate over `model.named_modules()` and classify each `nn.Module` submodule by type, without graph-level analysis.
2. For individual operations that are untraceable within an otherwise-traceable model, classify those nodes as Gray and emit a warning.
3. Future versions may adopt `torch.compile` / TorchDynamo as a more robust fallback for control-flow-heavy models.

## Consequences

**Easier**:

- The `fx.Graph` representation is straightforward to traverse; classifying each node requires only inspecting `node.op` and `node.target`.
- Standard transformer models (attention, layer norm, linear, residual connections) trace cleanly with `torch.fx`, covering the primary use cases in Phase 1.
- The node-level graph makes it possible to compute precise S/K ratios, identify K-boundaries, and track provenance through S-phases (SPEC-003).
- `torch.fx` is already a dependency of PyTorch — no additional dependency is introduced.

**Harder or limited**:

- Models with data-dependent control flow (e.g., `if x > 0:`, early exit, dynamic sequence lengths that branch on values) cannot be traced symbolically and require the module-level fallback.
- Models with dynamic shapes that depend on input values at trace time may fail or produce incorrect graphs.
- Some modules with complex `__init__` or non-standard `forward` implementations may raise `TraceError`.
- The module-level fallback produces coarser-grained classification than node-level analysis and may miss functional operations applied inline.

## Alternatives Considered

1. **torch.compile / TorchDynamo**: Dynamo is more robust than `torch.fx` symbolic tracing — it handles data-dependent control flow by inserting graph breaks and tracing each subgraph separately. However, it targets `torch.ops.aten`-level operations (low-level decompositions), which are less directly mappable to user-facing S/K categories than the higher-level ops exposed by `torch.fx`. The API is also more complex and continues to evolve. Dynamo is a strong candidate for a future fallback or replacement once the classifier's classification logic is stable.

2. **Forward hooks only**: PyTorch's `register_forward_hook` API can observe the inputs and outputs of each `nn.Module` at runtime, but it cannot see functional operations applied outside of a named module (e.g., `F.relu(x)`, `torch.matmul(q, k)`, or arithmetic expressions in `forward`). This makes it insufficient for fine-grained S/K classification, which must distinguish individual operations within a module's `forward` method.

3. **Static source analysis (AST parsing)**: Parsing the Python source of a model's `forward` method is fragile — it cannot handle dynamically constructed modules, closures, inherited methods, or operations defined in C extensions. It also cannot produce the actual computation graph, only a syntactic approximation of it.

4. **ONNX export**: ONNX represents computation as a portable graph of standard operators. While this provides a structured graph, it converts PyTorch-specific constructs into an external format and loses information relevant to S/K classification (e.g., whether an operation is a PyTorch `nn.Module` or a functional call). ONNX export is also significantly more heavyweight than `torch.fx` tracing and is designed for interoperability rather than analysis.
