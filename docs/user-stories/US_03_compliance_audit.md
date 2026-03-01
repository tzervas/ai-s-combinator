# US-03: Audit K-Boundaries for Compliance

**As** a compliance officer,
**I want** to verify that a model's K-boundaries (information erasure points) are documented,
**So that** I can audit the model for regulatory purposes.

## Acceptance Criteria

### K-Boundary Enumeration
- [ ] Report lists every K-boundary (every layer classified as `"K"`) with its fully-qualified `layer_name` and `module_type`
- [ ] Each K-boundary entry includes a human-readable `erasure_description` string that names the operation and explains in one sentence why it erases information (e.g., `"ReLU zeros all negative activations, discarding their magnitude"`)
- [ ] K-boundaries are ordered by their position in the forward pass (depth-first traversal order), with a `position_index` integer field starting at 1
- [ ] Gray-classified layers appear in a separate `"gray_boundaries"` section with a `classification_note` explaining why the layer is ambiguous

### Erasure Fraction Quantification
- [ ] Each K-boundary entry includes `erasure_fraction` as a float in `[0.0, 1.0]`
- [ ] `erasure_fraction` for ReLU is estimated as the fraction of activations that are negative on a calibration forward pass; if no calibration data is provided, the field is `null` with a `"note": "no calibration data; static analysis only"`
- [ ] `erasure_fraction` for Dropout is the configured `p` value (e.g., `0.1` for 10% dropout)
- [ ] `erasure_fraction` for MaxPool2d is `1 - (1 / kernel_area)` (e.g., `0.75` for a 2x2 pool)
- [ ] `erasure_fraction` for masking operations is the fraction of tokens/positions masked, or `null` if statically unknowable
- [ ] The report includes a `cumulative_erasure_score` field equal to the mean `erasure_fraction` across all non-null K-boundaries

### Compliance Document Format
- [ ] `bwsk audit <model_path> --output compliance` produces a self-contained plain-text document (not JSON) formatted for inclusion in a regulatory submission
- [ ] The compliance document is structured with clearly numbered sections:
  1. Document header (model identifier, audit date, auditor, bwsk version)
  2. Executive summary (total K-boundaries, cumulative erasure score, one-sentence risk assessment)
  3. K-boundary inventory table (one row per K-boundary)
  4. Gray-boundary inventory table (one row per Gray layer)
  5. Methodology notes (how classification rules were derived, link to SPEC-001)
  6. Attestation block (placeholder lines for auditor signature and date)
- [ ] `bwsk audit <model_path> --output json` produces a machine-readable audit report that can be ingested by compliance management systems
- [ ] Both output formats include a `document_id` field / header line formatted as `BWSK-AUDIT-<YYYYMMDD>-<8-char model hash>` for unique identification

### Required Fields in the Audit Report (JSON)
- [ ] JSON audit report validates against the following schema (all fields required):
  ```json
  {
    "document_id": "BWSK-AUDIT-20260228-A1B2C3D4",
    "model_path": "<absolute path>",
    "model_sha256": "<64-char hex digest of the model file>",
    "bwsk_version": "<semver>",
    "audit_timestamp": "<ISO 8601 UTC>",
    "total_layers": <integer>,
    "k_boundary_count": <integer>,
    "gray_boundary_count": <integer>,
    "cumulative_erasure_score": <float | null>,
    "risk_assessment": "<one of: LOW | MEDIUM | HIGH>",
    "k_boundaries": [
      {
        "position_index": <integer>,
        "layer_name": "<fully-qualified name>",
        "module_type": "<Python class name>",
        "erasure_description": "<string>",
        "erasure_fraction": <float | null>,
        "erasure_fraction_note": "<string | null>"
      }
    ],
    "gray_boundaries": [
      {
        "position_index": <integer>,
        "layer_name": "<fully-qualified name>",
        "module_type": "<Python class name>",
        "classification_note": "<string>"
      }
    ],
    "methodology": {
      "classification_source": "SPEC-001",
      "graph_extraction": "torch.fx | nn.Module fallback",
      "calibration_data_used": <boolean>
    }
  }
  ```
- [ ] `risk_assessment` is computed as: `LOW` if `cumulative_erasure_score < 0.2`, `MEDIUM` if `< 0.5`, `HIGH` otherwise; if `cumulative_erasure_score` is `null`, `risk_assessment` is `"UNKNOWN"`
- [ ] `model_sha256` is the SHA-256 digest of the raw model file bytes, ensuring the audit is tied to an exact file

### What "Suitable for Compliance Documentation" Means
- [ ] The plain-text compliance document contains no binary data, embedded images, or external references — it is fully self-contained
- [ ] All technical terms (S-type, K-type, erasure fraction) are defined in the Methodology Notes section so a non-ML reader can understand the report without external references
- [ ] The document is reproducible: running `bwsk audit` on the same file at a different time produces an identical report except for `audit_timestamp` and the attestation block
- [ ] Line length in the plain-text document does not exceed 100 characters so the document renders correctly when printed or included in a PDF

### Programmatic API
- [ ] Public function `bwsk.classify.audit_model(model: nn.Module, model_path: str, calibration_data: DataLoader | None = None) -> AuditReport` is importable and callable from Python
- [ ] `AuditReport.to_json() -> str` produces the canonical JSON audit report
- [ ] `AuditReport.to_compliance_text() -> str` produces the plain-text compliance document
- [ ] `AuditReport.document_id` property returns the `BWSK-AUDIT-<date>-<hash>` string

## Example Compliance Document (plain-text excerpt)

```
================================================================================
BWSK INFORMATION ERASURE AUDIT REPORT
Document ID : BWSK-AUDIT-20260228-A1B2C3D4
Model       : /home/user/resnet18.pt
SHA-256     : 3a7f2b... (truncated)
Audit Date  : 2026-02-28T12:00:00Z
BWSK Version: 0.1.0
================================================================================

1. EXECUTIVE SUMMARY

   This report documents 15 K-boundary (information-erasing) operations
   identified in the audited model. The cumulative erasure score is 0.38,
   indicating MEDIUM information erasure risk.

2. K-BOUNDARY INVENTORY

   Pos  Layer Name             Type         Erasure Fraction  Description
   ---  ---------------------  -----------  ----------------  ----------------------------
   1    layer1.0.relu          ReLU         0.4823            ReLU zeros all negative
                                                              activations, discarding their
                                                              magnitude.
   2    layer1.0.maxpool       MaxPool2d    0.7500            MaxPool2d retains 1 of 4
                                                              values per kernel window.
   ...

3. GRAY-BOUNDARY INVENTORY

   Pos  Layer Name             Type         Note
   ---  ---------------------  -----------  -------------------------------------------
   1    layer1.0.bn1           BatchNorm2d  Classification depends on training vs eval
                                            mode and batch statistics availability.

4. METHODOLOGY NOTES

   S-type (information-preserving): operations whose output uniquely determines
   their input given the weights (e.g., linear projection, layer norm).

   K-type (information-erasing): operations that irrecoverably discard input
   information (e.g., ReLU, dropout, max pooling).

   Gray: operations whose classification is context-dependent (e.g., batch norm,
   softmax). See SPEC-001 for the full classification rule table.

   Graph extraction: torch.fx symbolic trace (fallback: nn.Module tree walk).
   Classification source: SPEC-001 v0.1.0.

5. ATTESTATION

   Auditor signature: ______________________________  Date: ______________

   Title: __________________________________

================================================================================
```

## Spec Reference

- [SPEC-001: S/K Operation Classifier](../specs/SPEC_001_sk_classifier.md)
