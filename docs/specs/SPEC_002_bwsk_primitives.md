# SPEC-002: BWSK Primitive Implementations

## Goal

Implement B (compose), W (share), S (fan-out), K (erase) as composable `nn.Module` wrappers with automatic S/K classification.

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Correctness | Each primitive satisfies its combinator reduction rule |
| Composability | Primitives can be arbitrarily nested and composed |
| PyTorch compatibility | Compiles to standard `nn.Module` with autograd support |
| Classification | Each primitive self-reports its S/K classification |

## User Stories

- **US-4**: Define transformer block using BWSK DSL with shape validation
- **US-5**: Compose BWSK modules with type-checked shape compatibility
- **US-6**: Express weight sharing with W primitive

## API Sketch

```python
from bwsk.primitives import B, W, S, K

# Composition
layer = B(linear_up, gelu)  # B f g x = f(g(x))

# Fan-out + combine
attn = S(q_proj, S(k_proj, v_proj))  # S f g x = f(x)(g(x))
```

## Test Plan

- Unit tests for each combinator reduction rule
- Composition tests (nested primitives)
- Gradient flow tests through composed primitives
