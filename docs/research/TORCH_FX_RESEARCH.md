# torch.fx Research: Symbolic Tracing for S/K Classification

> Research document for the BWSK Combinator AI Framework.
> Covers torch.fx capabilities, limitations, and practical patterns
> for implementing SPEC-001 (S/K Operation Classifier).

---

## 1. Capabilities: What torch.fx Symbolic Tracing Captures

### 1.1 Overview

`torch.fx` is a Python-to-Python transformation toolkit for `nn.Module` instances.
It consists of three main components:

1. **Symbolic Tracer** -- executes the module's `forward()` with `Proxy` objects
   instead of real tensors, recording every operation into an FX `Graph`.
2. **Intermediate Representation (IR)** -- the `Graph`, a doubly-linked list of
   `Node` objects representing operations and data dependencies.
3. **Python Code Generation** -- a `GraphModule` that holds the IR and can
   regenerate a valid Python `forward()` method from it.

Basic usage:

```python
import torch
from torch.fx import symbolic_trace

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x).clamp(min=0.0, max=1.0)

traced = symbolic_trace(MyModel())
print(traced.graph)     # The IR
print(traced.code)      # Generated Python code
```

### 1.2 The Six Node Types (Opcodes)

Every `Node` in the graph has an `op` property that determines its semantics.
The six opcodes are:

| Opcode | Semantics | `target` | `args` / `kwargs` |
|--------|-----------|----------|-------------------|
| `placeholder` | Function input (one per forward argument) | Parameter name (string) | Empty, or `args[0]` is default value |
| `get_attr` | Retrieves a parameter/buffer from the module hierarchy | Fully-qualified dotted name (e.g. `"linear.weight"`) | Empty |
| `call_function` | Calls a free function (e.g. `torch.relu`, `operator.add`) | The Python callable itself | Python calling convention |
| `call_method` | Calls a method on a value (e.g. `.view()`, `.clamp()`) | Method name as string (e.g. `"clamp"`) | `args[0]` is `self`; rest follow Python calling convention |
| `call_module` | Invokes an `nn.Module`'s `forward()` method | Fully-qualified module name (e.g. `"encoder.layer1"`) | Python calling convention (excluding `self`) |
| `output` | The return value of the traced function | `"output"` | `args[0]` is the return value (may be a tuple) |

Key properties on each `Node`:
- `node.op` -- the opcode string
- `node.name` -- unique name within the graph
- `node.target` -- what is being called/accessed (varies by opcode)
- `node.args`, `node.kwargs` -- arguments (may contain other `Node` references)
- `node.users` -- dict of nodes that consume this node's output
- `node.all_input_nodes` -- list of `Node` objects this node depends on

### 1.3 Graph Structure

The `Graph` is a doubly-linked list of `Node` objects. It starts with `placeholder`
nodes (inputs), ends with a single `output` node, and contains operation nodes
in between. The data-dependency edges are encoded via `Node` references in
`args`/`kwargs`.

```python
for node in traced.graph.nodes:
    print(f"{node.op:15s} | {node.name:20s} | target={node.target}")
```

Example output for the model above:
```
placeholder     | x                    | target=x
call_module     | linear               | target=linear
call_method     | clamp                | target=clamp
output          | output               | target=output
```

### 1.4 Leaf Modules

By default, standard `torch.nn` modules (Linear, Conv2d, BatchNorm2d, etc.) are
treated as **leaf modules** -- they appear as single `call_module` nodes rather
than being traced through. This is controlled by `Tracer.is_leaf_module()`.

You can customize which modules are leaves by subclassing `Tracer`:

```python
class CustomTracer(torch.fx.Tracer):
    def is_leaf_module(self, m, module_qualified_name):
        # Trace inside everything except nn.Linear
        if isinstance(m, torch.nn.Linear):
            return True
        return False

graph = CustomTracer().trace(model)
```

---

## 2. Limitations of torch.fx Symbolic Tracing

### 2.1 Data-Dependent Control Flow

**This is the primary limitation.** Symbolic tracing executes forward() with
Proxy objects, not real tensors. If the code branches on a tensor value,
the tracer cannot determine which path to follow.

```python
class BadModel(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:    # ERROR: Proxy has no concrete value
            return x + 1
        else:
            return x - 1
```

Error: `"symbolically traced variables cannot be used as inputs to control flow"`

**Workaround**: Use `concrete_args` to specialize, but this only traces one path:

```python
traced = symbolic_trace(BadModel(), concrete_args={"x": torch.ones(3)})
# Only captures the x.sum() > 0 branch -- INCORRECT for negative inputs
```

**Impact on BWSK**: Most standard model architectures (transformers, CNNs, etc.)
do not use data-dependent control flow in their forward pass, so this limitation
is unlikely to block our S/K classifier for common models.

### 2.2 Dynamic Shapes

Shapes are not tracked symbolically in `torch.fx.symbolic_trace`. If forward()
uses `x.shape[0]` in control flow, it will fail. If shapes are used only as
arguments to operations (e.g., `x.view(batch, -1)`), they may trace successfully
but get baked in as constants.

### 2.3 torch.autograd.Function Subclasses

Custom `autograd.Function` subclasses with user-defined `forward()` and
`backward()` methods are **not traced through** by `symbolic_trace`. The tracer
cannot see inside the custom backward logic.

**Impact on BWSK**: Models using custom autograd functions will appear as opaque
`call_function` nodes. We should classify these as `GRAY` by default and allow
user annotation.

### 2.4 nn.ModuleList / nn.ModuleDict Iteration

Iterating over `nn.ModuleList` works if the iteration is static (fixed number
of modules). The tracer unrolls the loop and records each module call:

```python
class StackedLayers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(10, 10) for _ in range(3)
        ])

    def forward(self, x):
        for layer in self.layers:   # OK: static iteration
            x = layer(x)
        return x
```

This traces successfully. However, if the number of iterations depends on
input data, it will fail.

### 2.5 Tensor Creation Inside forward()

Tensor constructors behave differently:

| Constructor | Behavior | Problem? |
|-------------|----------|----------|
| `torch.zeros(3, 4)` | Value embedded as constant in trace | Only if shape is dynamic |
| `torch.ones(3, 4)` | Value embedded as constant in trace | Only if shape is dynamic |
| `torch.zeros_like(x)` | Traced correctly | No |
| `torch.ones_like(x)` | Traced correctly | No |
| `torch.rand(3, 4)` | **Single random value baked in** | **Yes -- incorrect semantics** |
| `torch.randn(3, 4)` | **Single random value baked in** | **Yes -- incorrect semantics** |

**Workaround for non-deterministic constructors**: Use `torch.fx.wrap()`:

```python
@torch.fx.wrap
def create_noise(shape):
    return torch.randn(shape)

class NoisyModel(torch.nn.Module):
    def forward(self, x):
        noise = create_noise(x.shape)  # Appears as call_function node
        return x + noise
```

### 2.6 In-Place Operations

In-place operations (`x.add_(1)`, `x += 1`, `x[:, 0] = 0`) are problematic.
They may trace but can produce incorrect gradients or violate the functional
semantics assumed by graph transformations. The PyTorch team discourages
in-place operations in traced code.

**Impact on BWSK**: In-place ops are semantically K-type (they destroy the
original value). If we encounter them, classify as K-type.

### 2.7 Python Built-ins

- `print()`: Executes during tracing, then disappears from the graph.
- `len()` on a Proxy: Fails unless the length is known at trace time.
- `isinstance()`: Evaluated at trace time with Proxy objects, not tensors.
- List/dict comprehensions over Proxies: May fail.

### 2.8 Summary of Tracing Failure Modes

| Pattern | Fails? | Workaround |
|---------|--------|------------|
| Data-dependent `if/else` | Yes | `concrete_args`, refactor, or use `torch.compile` |
| Data-dependent loops | Yes | Refactor or use `torch.compile` |
| `x.shape` in control flow | Yes | `concrete_args` |
| `torch.autograd.Function` | Partially (opaque) | Wrap with `torch.fx.wrap` or classify as GRAY |
| Static `ModuleList` iteration | No | -- |
| Dynamic `ModuleList` iteration | Yes | Refactor |
| `torch.zeros(static_shape)` | No (constant) | -- |
| `torch.randn(static_shape)` | Incorrect | Use `torch.fx.wrap` |
| In-place operations | Risky | Avoid or rewrite |
| `print()`, `len()` | Side effects / failure | Remove or wrap |

---

## 3. torch.compile / Dynamo vs torch.fx.symbolic_trace

### 3.1 Architecture Comparison

| Aspect | `torch.fx.symbolic_trace` | `torch.compile` (Dynamo) | `torch.export` |
|--------|--------------------------|--------------------------|----------------|
| **Tracing level** | Python operator overloading (`__torch_function__`) | Python bytecode interception | Python bytecode (Dynamo) |
| **Control flow** | Fails on data-dependent flow | Graph breaks, falls back to eager | Requires `torch.cond()` for dynamic flow |
| **IR** | FX Graph (high-level: `torch.*` ops) | FX Graph (lowered to ATen ops) | FX Graph (ATen ops, stricter) |
| **Dynamic shapes** | No support | Yes, with symbolic shapes | Yes, with constraints |
| **Graph breaks** | Fatal error | Graceful fallback to Python | Not allowed (whole-graph) |
| **Use case** | Analysis, simple transforms | JIT optimization | Model export, deployment |
| **Overhead** | Minimal | JIT compilation cost | One-time export cost |

### 3.2 How Dynamo Works

TorchDynamo intercepts Python bytecode at the frame level. When it encounters
untraceable code (data-dependent control flow, unsupported Python), it performs
a **graph break**: it compiles the graph captured so far, falls back to Python
for the untraceable section, then resumes capturing a new graph.

```python
# torch.compile handles this automatically:
@torch.compile
def func(x):
    if x.sum() > 0:     # Graph break here
        return x + 10
    else:
        return x - 10
# Result: 2 graphs, 1 graph break -- but correct results
```

For full-graph mode (no breaks), use `torch.cond()`:

```python
@torch.compile(fullgraph=True)
def func(x):
    return torch.cond(
        torch.all(x >= 0),
        lambda x: x + 10,
        lambda x: x - 10,
        (x,)
    )
# Result: 1 graph, 0 graph breaks
```

### 3.3 Is Dynamo Strictly Better?

**For optimization**: Yes, Dynamo/torch.compile is strictly more capable.

**For our S/K classification use case**: Not necessarily. We need:

1. A graph we can iterate over and inspect node-by-node.
2. Node targets that map to identifiable operations (nn.Module types, torch functions).
3. The ability to process the graph offline (not at JIT time).

`torch.fx.symbolic_trace` gives us a **high-level graph** where `call_module`
nodes reference `nn.Linear`, `nn.ReLU`, etc. directly. This is ideal for S/K
classification because our classification table maps directly to these
module types.

Dynamo's ATen-level IR decomposes these into lower-level operators (e.g.,
`aten.mm`, `aten.add`, `aten.relu`), which is harder to classify at the
architectural level.

**Recommendation**: Use `torch.fx.symbolic_trace` as the primary mechanism for
SPEC-001. Fall back to `torch.export` for models that fail tracing. Consider
Dynamo only if we need to handle highly dynamic models in later phases.

### 3.4 torch.export as Middle Ground

`torch.export` uses Dynamo under the hood but produces a single whole-program
graph (no graph breaks). It operates at the ATen op level but provides richer
metadata (tensor shapes, dtypes, constraints).

```python
import torch.export

exported = torch.export.export(model, (example_input,))
# exported.graph_module is a GraphModule with ATen-level ops
```

This could serve as a fallback when `symbolic_trace` fails, though the ATen-level
ops require a different classification mapping.

---

## 4. Practical Patterns for S/K Classification

### 4.1 Basic Graph Iteration and Classification

This is the core pattern for our classifier:

```python
import torch
import torch.nn as nn
from torch.fx import symbolic_trace

# Classification tables
S_MODULES = {
    nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.LayerNorm, nn.GroupNorm, nn.Embedding,
    nn.Identity,
}

K_MODULES = {
    nn.ReLU, nn.GELU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh,
    nn.Dropout, nn.Dropout2d,
    nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
}

S_FUNCTIONS = {
    torch.add, torch.sub, torch.mul, torch.matmul,
    torch.cat, torch.stack,
    torch.transpose, torch.permute,
}

K_FUNCTIONS = {
    torch.relu, torch.clamp, torch.where,
    torch.max, torch.min, torch.argmax,
}


def classify_graph(model: nn.Module, example_input: torch.Tensor):
    """Trace model and classify each node as S-type, K-type, or Gray."""
    traced = symbolic_trace(model)
    results = []

    for node in traced.graph.nodes:
        if node.op == "placeholder" or node.op == "output":
            continue  # Skip I/O nodes

        if node.op == "get_attr":
            classification = "S"  # Parameter access is information-preserving

        elif node.op == "call_module":
            # Look up the actual module instance
            module = traced.get_submodule(node.target)
            mod_type = type(module)
            if mod_type in S_MODULES:
                classification = "S"
            elif mod_type in K_MODULES:
                classification = "K"
            else:
                classification = "GRAY"

        elif node.op == "call_function":
            if node.target in S_FUNCTIONS:
                classification = "S"
            elif node.target in K_FUNCTIONS:
                classification = "K"
            else:
                classification = "GRAY"

        elif node.op == "call_method":
            method_name = node.target
            if method_name in {"view", "reshape", "permute", "transpose",
                               "contiguous", "expand", "repeat", "unsqueeze",
                               "squeeze", "flatten", "unflatten"}:
                classification = "S"  # Shape ops preserve information
            elif method_name in {"clamp", "clamp_"}:
                classification = "K"
            else:
                classification = "GRAY"

        else:
            classification = "GRAY"

        results.append({
            "name": node.name,
            "op": node.op,
            "target": str(node.target),
            "classification": classification,
        })

    return results
```

### 4.2 Accessing Module Instances from call_module Nodes

When a node has `op == "call_module"`, its `target` is a string like
`"encoder.layers.0.self_attn"`. To get the actual module:

```python
module_instance = traced_model.get_submodule(node.target)
isinstance(module_instance, nn.Linear)  # True/False
```

This is essential for type-based classification.

### 4.3 Using the Transformer API for Graph Rewriting

The `torch.fx.Transformer` class enables one-to-one node transformations:

```python
from torch.fx import Transformer

class SKAnnotator(Transformer):
    """Annotate each node with its S/K classification."""

    def call_module(self, target, args, kwargs):
        module = self.fetch_attr(target)
        result = super().call_module(target, args, kwargs)
        # Attach metadata to the current node
        self.current_node.meta["sk_class"] = classify_module(module)
        return result

    def call_function(self, target, args, kwargs):
        result = super().call_function(target, args, kwargs)
        self.current_node.meta["sk_class"] = classify_function(target)
        return result

annotated = SKAnnotator(traced).transform()
```

### 4.4 Using node.meta for Metadata

Each `Node` has a `meta` dictionary for attaching arbitrary metadata.
This is the recommended way to store classification results:

```python
for node in traced.graph.nodes:
    node.meta["sk_class"] = classify(node)

# Later retrieval:
for node in traced.graph.nodes:
    print(f"{node.name}: {node.meta.get('sk_class', 'unclassified')}")
```

### 4.5 Pattern Matching with replace_pattern

For finding and replacing subgraph patterns (useful for later phases):

```python
from torch.fx import subgraph_rewriter

def residual_pattern(x, weight):
    """Match: y = linear(x) + x"""
    y = torch.nn.functional.linear(x, weight)
    return y + x

def annotated_residual(x, weight):
    """Replace with annotated version."""
    y = torch.nn.functional.linear(x, weight)
    return y + x  # Same computation, but now matched

matches = subgraph_rewriter.replace_pattern(traced, residual_pattern, annotated_residual)
```

### 4.6 Handling the operator Module

Many arithmetic operations show up as nodes with targets from the `operator`
module, not `torch`:

```python
import operator

# x + y  -->  call_function, target=operator.add
# x * y  -->  call_function, target=operator.mul
# x[0]   -->  call_function, target=operator.getitem

S_OPERATORS = {
    operator.add, operator.mul, operator.sub,
    operator.truediv, operator.floordiv,
    operator.getitem,
}
```

This is a common gotcha -- always check for both `torch.*` and `operator.*`
targets when classifying `call_function` nodes.

---

## 5. Known Issues with PyTorch 2.x

### 5.1 TorchScript Deprecation (PyTorch 2.10+)

TorchScript (`torch.jit.script`, `torch.jit.trace`) is officially deprecated
as of PyTorch 2.10. The recommended replacement is `torch.export`. This does
**not** affect `torch.fx`, which remains actively maintained and is the
foundation of `torch.compile`.

### 5.2 torch.fx Stability

`torch.fx` is the IR underlying `torch.compile`, `torch.export`, and the
Inductor compiler. It is a core part of the PyTorch 2.x stack and is not
deprecated. The API has been stable since PyTorch 1.9.

### 5.3 ATen IR vs Torch IR

PyTorch 2.x graph transformations increasingly target the ATen operator set
rather than the high-level `torch.*` functions. For our S/K classifier:

- `torch.fx.symbolic_trace` produces **Torch-level IR** (nn.Module calls,
  torch.* functions) -- better for architectural classification.
- `torch.compile` / `torch.export` produce **ATen-level IR** (aten.mm,
  aten.relu, etc.) -- better for optimization but harder to classify
  architecturally.

### 5.4 Graph Breaks and torch.compile

If we ever move to Dynamo-based tracing, we must handle graph breaks. A model
with graph breaks produces multiple disjoint FX graphs. The
`torch._dynamo.explain()` utility helps diagnose them:

```python
explanation = torch._dynamo.explain(model)(example_input)
print(explanation.graph_count)
print(explanation.break_reasons)
```

### 5.5 Backward Compatibility Considerations

- `torch.fx.symbolic_trace` API has been stable across PyTorch 1.9 through 2.10.
- The `Node` opcode set (the six types) has not changed.
- `GraphModule.code` generation is stable.
- `node.meta` dict is stable and widely used by PyTorch internals.

The main risk is not breakage but rather the ecosystem moving toward
`torch.export` and ATen IR for new tooling. For Phase 1 (S/K Classifier),
`symbolic_trace` remains the right choice.

---

## 6. Recommendations for BWSK Phase 1

### 6.1 Primary Approach

Use `torch.fx.symbolic_trace` as the tracing backend for SPEC-001. It provides:
- High-level IR that maps directly to our S/K classification tables.
- `call_module` nodes that reference concrete `nn.Module` types.
- A simple iteration API (`for node in graph.nodes`).
- The `node.meta` dict for attaching classification results.
- Stable API with no anticipated deprecation.

### 6.2 Fallback Strategy

For models that fail `symbolic_trace`, implement a graceful fallback:

```python
def trace_model(model, example_input):
    try:
        return symbolic_trace(model)
    except Exception as e:
        # Fallback: try torch.export (ATen-level, needs different classification map)
        try:
            exported = torch.export.export(model, (example_input,))
            return exported.graph_module
        except Exception:
            raise RuntimeError(
                f"Model cannot be traced by symbolic_trace or torch.export: {e}"
            )
```

### 6.3 Classification Architecture

```
Input: nn.Module
  |
  v
torch.fx.symbolic_trace()
  |
  v
Iterate graph.nodes
  |
  +---> placeholder / output --> skip
  +---> get_attr              --> S (parameter access)
  +---> call_module           --> lookup type(module) in classification table
  +---> call_function         --> lookup target in function classification table
  +---> call_method           --> lookup method name in method classification table
  |
  v
Output: List of (node_name, opcode, target, S|K|GRAY)
```

### 6.4 Testing Strategy

Test with these model families:
1. Simple feedforward (Linear + ReLU) -- should fully trace.
2. ResNet-style (residual connections) -- tests `operator.add`.
3. Transformer (MultiheadAttention) -- tests leaf module behavior.
4. Models with `ModuleList` -- tests static iteration.
5. Models with dropout -- tests K-type classification.

### 6.5 Things to Avoid in Phase 1

- Do **not** use `torch.compile` or Dynamo -- unnecessary complexity.
- Do **not** try to trace inside leaf modules -- classify them at the module level.
- Do **not** handle `torch.autograd.Function` specially -- classify as GRAY.
- Do **not** support data-dependent control flow -- document it as unsupported
  and raise a clear error.

---

## Sources

- [torch.fx -- PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/fx.html)
- [torch.fx README (GitHub)](https://github.com/pytorch/pytorch/blob/main/torch/fx/README.md)
- [torch.fx Node source (GitHub)](https://github.com/pytorch/pytorch/blob/main/torch/fx/node.py)
- [torch.export -- PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html)
- [PyTorch Graphs Three Ways: Data-Dependent Control Flow (Thomas J. Fan, 2025)](https://thomasjpfan.com/2025/03/pytorch-graphs-three-ways-data-dependent-control-flow/)
- [Dynamo Deep-Dive -- PyTorch documentation](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)
- [TORCH.FX: Practical Program Capture and Transformation (Reed et al., 2021)](https://arxiv.org/pdf/2112.08429)
- [Writing Graph Transformations on ATen IR -- PyTorch documentation](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_transformations.html)
- [torch.fx subgraph_rewriter source (GitHub)](https://github.com/pytorch/pytorch/blob/main/torch/fx/subgraph_rewriter.py)
- [Torch FX Transformation and Pipeline Parallelism (Insu Jang, 2023)](https://insujang.github.io/2023-04-22/torch-fx-transformation-and-pipeline-parallelism/)
- [fx: unable to symbolically trace model with torch.zeroes(Proxy) (GitHub Issue #44887)](https://github.com/pytorch/pytorch/issues/44887)
- [How can Torch.FX work with autograd.Function? (PyTorch Forums)](https://discuss.pytorch.org/t/how-can-torch-fx-work-with-autograd-function/145922)
- [GraphMend: Code Transformations for Fixing Graph Breaks in PyTorch 2](https://arxiv.org/html/2509.16248v1)
- [PyTorch 2.9 Release Blog](https://pytorch.org/blog/pytorch-2-9/)
