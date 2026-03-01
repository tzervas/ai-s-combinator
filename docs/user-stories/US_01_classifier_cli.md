# US-01: Classify Model via CLI

**As** an ML engineer,
**I want** to run `bwsk classify my_model.pt` and see which layers erase information,
**So that** I can understand my model's information flow.

## Acceptance Criteria

### CLI Interface
- [ ] CLI accepts a saved PyTorch model path as a positional argument: `bwsk classify <model_path>`
- [ ] `--output json` flag emits a machine-readable JSON report to stdout
- [ ] `--output text` flag (default) emits a human-readable summary table to stdout
- [ ] `--output-file <path>` flag writes the report to the specified file instead of stdout
- [ ] `--help` prints usage, all flags, and at least one example invocation

### Error Handling
- [ ] Non-existent model path exits with code 1 and prints `Error: file not found: <path>` to stderr
- [ ] File that is not a valid PyTorch checkpoint exits with code 2 and prints `Error: not a valid PyTorch file: <path>` to stderr
- [ ] Model containing unsupported layer types exits with code 0 but includes a `"warnings"` field in the JSON report listing unclassified modules by name and type
- [ ] Corrupt file (valid path, but unreadable by `torch.load`) exits with code 2 with a descriptive error message

### Classification Output
- [ ] Each layer is classified as exactly one of: `"S"`, `"K"`, or `"Gray"`
- [ ] Report includes a per-layer table with columns: `layer_name`, `module_type`, `classification`, `erasure_fraction`
- [ ] Report includes a whole-model erasure score in `[0.0, 1.0]` computed as `k_count / total_count`
- [ ] Gray layers are listed separately in a `"gray_layers"` section with a note explaining the ambiguity

### JSON Report Schema
- [ ] JSON output validates against the following schema (all fields required):
  ```json
  {
    "model_path": "<absolute path>",
    "bwsk_version": "<semver string>",
    "timestamp": "<ISO 8601 UTC>",
    "total_layers": <integer>,
    "s_count": <integer>,
    "k_count": <integer>,
    "gray_count": <integer>,
    "erasure_score": <float in [0.0, 1.0]>,
    "layers": [
      {
        "name": "<fully-qualified module name>",
        "type": "<Python class name>",
        "classification": "S" | "K" | "Gray",
        "erasure_fraction": <float in [0.0, 1.0]>
      }
    ],
    "warnings": ["<string>"]
  }
  ```
- [ ] `erasure_score` equals `k_count / total_layers` (rounded to 4 decimal places)
- [ ] `layers` list is ordered by depth-first traversal of the module tree

### Human-Readable Summary (text output)
- [ ] Summary header shows model path, timestamp, and bwsk version
- [ ] Per-layer table is printed with aligned columns and a header row
- [ ] Footer line shows: `S: <n>  K: <n>  Gray: <n>  Erasure score: <0.NN>`
- [ ] Gray layers are annotated with `(Gray)` and an explanatory parenthetical

### Programmatic API
- [ ] Public function `bwsk.classify.classify_model(model: nn.Module) -> ClassificationReport` is importable and callable from Python without invoking the CLI
- [ ] `ClassificationReport` is a dataclass (or Pydantic model) whose fields match the JSON schema above
- [ ] `ClassificationReport.to_json() -> str` serializes to the canonical JSON format
- [ ] `ClassificationReport.to_text() -> str` produces the same human-readable output as the CLI `--output text` mode
- [ ] The CLI is a thin wrapper over `classify_model`; no classification logic lives in the CLI layer

## Example Output

### JSON (`bwsk classify resnet18.pt --output json`)
```json
{
  "model_path": "/home/user/resnet18.pt",
  "bwsk_version": "0.1.0",
  "timestamp": "2026-02-28T12:00:00Z",
  "total_layers": 60,
  "s_count": 42,
  "k_count": 15,
  "gray_count": 3,
  "erasure_score": 0.25,
  "layers": [
    { "name": "layer1.0.conv1", "type": "Conv2d",     "classification": "S", "erasure_fraction": 0.0 },
    { "name": "layer1.0.relu",  "type": "ReLU",       "classification": "K", "erasure_fraction": 0.5 },
    { "name": "layer1.0.bn1",   "type": "BatchNorm2d", "classification": "Gray", "erasure_fraction": 0.1 }
  ],
  "warnings": []
}
```

### Text (`bwsk classify resnet18.pt`)
```
BWSK Classifier v0.1.0 — 2026-02-28T12:00:00Z
Model: /home/user/resnet18.pt

Layer Name              Type          Class   Erasure
------------------------------------------------------
layer1.0.conv1          Conv2d        S       0.0000
layer1.0.relu           ReLU          K       0.5000
layer1.0.bn1            BatchNorm2d   Gray    0.1000
...

S: 42  K: 15  Gray: 3  Erasure score: 0.2500
```

## Notes

### Dependencies
- Requires PyTorch (`torch`) to be installed in the environment; `bwsk` does not vendor it
- `torch.fx.symbolic_trace` is used for graph extraction; models must be `fx`-traceable
- Non-traceable models (e.g., those with dynamic control flow) fall back to `nn.Module` tree traversal with a warning

### Assumptions
- Model is saved as a `state_dict` loaded into a known architecture class, or as a full `nn.Module` via `torch.save`
- Classification rules are defined in `src/bwsk/classify.py` and are the single source of truth
- `erasure_fraction` for S-type layers is always `0.0`; for K-type layers it is estimated heuristically and may vary
- Gray classification is stable: the same module type always maps to `"Gray"` for a given bwsk version

## Spec Reference

- [SPEC-001: S/K Operation Classifier](../specs/SPEC_001_sk_classifier.md)
