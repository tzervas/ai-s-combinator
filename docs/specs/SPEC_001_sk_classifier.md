# SPEC-001: S/K Operation Classifier

## Goal

Classify every operation in a PyTorch neural network as S-type (information-preserving), K-type (information-erasing), or Gray (context-dependent).

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Classification accuracy | 95%+ agreement with manual expert classification |
| Performance | < 5s for 70B parameter model graph |
| Coverage | All `torch.nn` modules classified |
| Usability | < 3 lines of code to classify a model |

## User Stories

- **US-1**: `bwsk classify my_model.pt` -> erasure budget JSON report
- **US-2**: Compare erasure budgets of ResNet vs ViT vs GPT-2
- **US-3**: Audit model K-boundaries for regulatory compliance

## API Sketch

```python
from bwsk.classify import classify_operation, classify_model, OpClass

# Single operation
result = classify_operation(nn.ReLU())  # -> OpClass.K

# Full model
report = classify_model(model)
print(report.erasure_budget)
print(report.per_layer_summary())
```

## Test Plan

- Unit tests for each `torch.nn` module classification
- Integration test with ResNet-50, GPT-2, ViT-B/16
- Performance benchmark with large model graph
