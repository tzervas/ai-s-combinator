# ADR-005: nn.Module Primitive Wrappers

## Status
Accepted

## Context
Phase 1 implemented BWSK combinators as pure Python callables. Phase 2 needs them to work with PyTorch tensors, gradients, and the training loop. We needed to decide how to bridge pure combinator logic and PyTorch's nn.Module system.

## Decision
Keep both layers: pure callables (B, W, S, K) for symbolic reasoning and testing, plus nn.Module subclasses (BModule, WModule, SModule, KModule) for tensor computation. All Module versions inherit from `_BWSKModuleBase`, which provides:
- `classification` property for self-reporting S/K type
- `__rshift__` operator for pipeline composition (`a >> b`)

### Key design choices:

1. **SModule uses addition as default combiner** — residual connections (`x + f(x)`) are the dominant use of S in modern architectures. Custom combiners are supported via the `combine` parameter.

2. **WModule passes the same input twice** — `W(f)(x) = f(x, x)` for nn.Modules, rather than the pure combinator's `f(x)(x)` (which requires currying). This is more natural for PyTorch modules that accept two arguments.

3. **KModule accepts optional second argument** — `K(f)(x, y=None)` allows K to work both as a binary erasure combinator and as a unary wrapper marking an operation as K-type.

4. **Pipeline operator `>>` only works on BWSK modules** — we chose not to monkey-patch `nn.Module.__rshift__` to avoid surprising behavior. Users wrap vanilla modules in BModule to use the pipeline syntax.

## Alternatives Considered

1. **Single class hierarchy** (pure and Module merged): Rejected because pure combinators are useful for testing combinator laws without torch, and the nn.Module overhead (parameter registration, hooks) is unnecessary for symbolic manipulation.

2. **Monkey-patch nn.Module.__rshift__**: Would allow `nn.Linear(10,5) >> nn.ReLU()` directly. Rejected because modifying stdlib classes is fragile and can conflict with other libraries.

3. **Functional API only** (no Module subclasses): Would use `compose(f, g)` functions instead of classes. Rejected because nn.Module integration is essential for parameter management, serialization, and hook-based provenance tracking.
