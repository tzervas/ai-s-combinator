# SPEC-004: Architecture DSL

## Goal

A domain-specific language for defining neural architectures as combinator expressions, with compile-time type checking and automatic S/K phase identification.

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Expressiveness | GPT-2, ResNet-50, ViT-B/16 fully expressible |
| Compilation correctness | Identical outputs to vanilla PyTorch (atol=1e-6) |
| Type checking | 100% of shape mismatches caught at definition time |
| Compilation overhead | < 1% of model initialization time |

## User Stories

- **US-4**: Define transformer block using BWSK DSL
- **US-5**: Compose modules with shape validation at definition time

## API Sketch

```python
from bwsk.dsl import pipeline

attention = (
    S(linear("Q"), S(linear("K"), linear("V")))
    >> K_mask(padding_mask)
    >> K_softmax()
    >> S(matmul, identity)
)

model = attention.compile()  # -> nn.Module
```

## Test Plan

- Define standard architectures in DSL and compare outputs
- Intentional type errors caught at definition time
- Benchmark compilation overhead
