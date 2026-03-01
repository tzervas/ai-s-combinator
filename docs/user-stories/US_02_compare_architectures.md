# US-02: Compare Architecture Erasure Budgets

**As** a researcher,
**I want** to compare the erasure budgets of ResNet vs ViT vs GPT-2,
**So that** I can understand how different architectures handle information.

## Acceptance Criteria

### What "Comparable" Means
- [ ] Every report produced by `bwsk classify` uses the same JSON schema version (field `"bwsk_version"` matches across all reports being compared)
- [ ] Classification labels (`"S"`, `"K"`, `"Gray"`) are assigned by the same rule table regardless of architecture, so the same operation type (e.g., `ReLU`) always receives the same label
- [ ] `erasure_score` is normalized to `[0.0, 1.0]` for all architectures so scores are directly comparable across models of different depths
- [ ] Layer counts (`s_count`, `k_count`, `gray_count`) are expressed both as raw integers and as percentages of `total_layers` in every report

### CLI Comparison Command
- [ ] `bwsk compare <report1.json> <report2.json> [<report3.json> ...]` accepts two or more pre-generated JSON reports as input
- [ ] `--output text` (default) prints a side-by-side table; `--output json` emits a machine-readable diff report
- [ ] Exits with code 1 if any input file is missing or has an incompatible schema version
- [ ] Accepts `--sort-by erasure_score|s_pct|k_pct` to order columns in the comparison table (default: input order)

### Side-by-Side Comparison Presentation
- [ ] Text output includes one column per model, identified by the model's filename (or `--label` override)
- [ ] Rows cover: `total_layers`, `s_count`, `k_pct` (%), `k_count`, `k_pct` (%), `gray_count`, `gray_pct` (%), and `erasure_score`
- [ ] A "winner" annotation marks the model with the lowest `erasure_score` on the summary row
- [ ] Layer-type breakdown shows which `module_type` values contribute most to K-count for each model
- [ ] Gray layers are listed per model with their counts so the researcher can assess classification ambiguity

### JSON Comparison Report Schema
- [ ] JSON output includes a top-level `"models"` array, one entry per input report, each containing at minimum: `model_path`, `bwsk_version`, `total_layers`, `s_count`, `k_count`, `gray_count`, `erasure_score`, `s_pct`, `k_pct`, `gray_pct`
- [ ] Top-level `"summary"` object includes: `lowest_erasure_model` (filename), `highest_erasure_model` (filename), `erasure_score_range` (max minus min, rounded to 4 decimal places)
- [ ] Top-level `"top_k_types"` object maps each model filename to a list of `{ "type": "<ModuleType>", "k_count": <int> }` sorted descending by `k_count`

### Programmatic API
- [ ] Public function `bwsk.classify.compare_reports(reports: list[ClassificationReport]) -> ComparisonReport` is importable and callable without invoking the CLI
- [ ] `ComparisonReport.to_json() -> str` and `ComparisonReport.to_text() -> str` mirror the CLI output modes
- [ ] Passing fewer than two reports raises `ValueError` with a clear message

## Example Output

### Text (`bwsk compare resnet18.json vit_b16.json gpt2.json`)
```
BWSK Architecture Comparison — 2026-02-28T12:00:00Z

Metric              ResNet-18     ViT-B/16      GPT-2
------------------------------------------------------
Total layers        60            72            148
S count             42 (70.0%)    58 (80.6%)    118 (79.7%)
K count             15 (25.0%)     8 (11.1%)     22 (14.9%)
Gray count           3  (5.0%)     6  (8.3%)      8  (5.4%)
Erasure score       0.2500        0.1111 *      0.1486

* lowest erasure score

Top K-type contributors:
  ResNet-18:  ReLU (12), MaxPool2d (2), Dropout (1)
  ViT-B/16:   Dropout (5), GELU (3)
  GPT-2:      Dropout (14), GELU (8)
```

### JSON (`bwsk compare resnet18.json vit_b16.json gpt2.json --output json`)
```json
{
  "bwsk_version": "0.1.0",
  "timestamp": "2026-02-28T12:00:00Z",
  "models": [
    {
      "model_path": "/home/user/resnet18.pt",
      "total_layers": 60,
      "s_count": 42, "s_pct": 70.0,
      "k_count": 15, "k_pct": 25.0,
      "gray_count": 3, "gray_pct": 5.0,
      "erasure_score": 0.25
    },
    {
      "model_path": "/home/user/vit_b16.pt",
      "total_layers": 72,
      "s_count": 58, "s_pct": 80.6,
      "k_count": 8,  "k_pct": 11.1,
      "gray_count": 6, "gray_pct": 8.3,
      "erasure_score": 0.1111
    }
  ],
  "summary": {
    "lowest_erasure_model": "vit_b16.json",
    "highest_erasure_model": "resnet18.json",
    "erasure_score_range": 0.1389
  },
  "top_k_types": {
    "resnet18.json": [
      { "type": "ReLU", "k_count": 12 },
      { "type": "MaxPool2d", "k_count": 2 },
      { "type": "Dropout", "k_count": 1 }
    ],
    "vit_b16.json": [
      { "type": "Dropout", "k_count": 5 },
      { "type": "GELU", "k_count": 3 }
    ]
  }
}
```

## Spec Reference

- [SPEC-001: S/K Operation Classifier](../specs/SPEC_001_sk_classifier.md)
