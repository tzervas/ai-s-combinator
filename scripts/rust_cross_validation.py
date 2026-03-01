"""Cross-validation: Python vs Rust BWSK classification.

Classifies operations from HuggingFace models in both Python and Rust,
then compares results to verify 100% parity. Also benchmarks Rust
classification timing and runs the Burn forward pass benchmark.

Usage:
    uv run python scripts/rust_cross_validation.py

Requires:
    - Rust workspace built: cd rust && cargo build --workspace --examples
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bwsk.classify import OpClass, classify_operation

# Custom rules matching multi_model_benchmark.py
CUSTOM_RULES: dict[str, OpClass] = {
    "Conv1D": OpClass.S,
    "NewGELUActivation": OpClass.K,
    "GELUActivation": OpClass.K,
    "FastGELUActivation": OpClass.K,
    "T5LayerNorm": OpClass.S,
    "OPTLearnedPositionalEmbedding": OpClass.S,
    "RotaryEmbedding": OpClass.S,
    "GPTNeoXRotaryEmbedding": OpClass.S,
}

# Custom rules as string map for Rust
CUSTOM_RULES_STR: dict[str, str] = {name: cls.value for name, cls in CUSTOM_RULES.items()}

RUST_DIR = Path(__file__).resolve().parent.parent / "rust"
CROSS_VALIDATE_BIN = RUST_DIR / "target" / "debug" / "examples" / "cross_validate"
REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "RUST_CROSS_VALIDATION_REPORT.md"

# Models to cross-validate (subset that doesn't require downloading large models)
# We use canonical op names directly rather than loading full HF models.
SYNTHETIC_MODELS = {
    "MLP-5": [
        "nn.Linear",
        "nn.ReLU",
        "nn.Linear",
        "nn.ReLU",
        "nn.Linear",
    ],
    "Transformer-Block": [
        "nn.Linear",
        "nn.Linear",
        "nn.Linear",
        "nn.Linear",
        "nn.Softmax",
        "nn.Linear",
        "nn.Dropout",
        "nn.LayerNorm",
        "nn.Linear",
        "nn.Linear",
        "nn.GELU",
        "nn.Linear",
        "nn.Dropout",
        "nn.LayerNorm",
    ],
    "CNN-Classifier": [
        "nn.Conv2d",
        "nn.ReLU",
        "nn.MaxPool2d",
        "nn.Conv2d",
        "nn.ReLU",
        "nn.MaxPool2d",
        "nn.Flatten",
        "nn.Linear",
        "nn.ReLU",
        "nn.Dropout",
        "nn.Linear",
    ],
    "ResNet-Block": [
        "nn.Conv2d",
        "nn.BatchNorm2d",
        "nn.ReLU",
        "nn.Conv2d",
        "nn.BatchNorm2d",
        "nn.Identity",
    ],
    "All-Ops": [
        "nn.Linear",
        "nn.LazyLinear",
        "nn.Bilinear",
        "nn.Conv1d",
        "nn.Conv2d",
        "nn.Conv3d",
        "nn.ConvTranspose1d",
        "nn.ConvTranspose2d",
        "nn.ConvTranspose3d",
        "nn.LayerNorm",
        "nn.RMSNorm",
        "nn.GroupNorm",
        "nn.BatchNorm1d",
        "nn.BatchNorm2d",
        "nn.BatchNorm3d",
        "nn.ReLU",
        "nn.LeakyReLU",
        "nn.PReLU",
        "nn.GELU",
        "nn.SiLU",
        "nn.Softplus",
        "nn.Sigmoid",
        "nn.Tanh",
        "nn.Softmax",
        "nn.MaxPool1d",
        "nn.MaxPool2d",
        "nn.MaxPool3d",
        "nn.AvgPool1d",
        "nn.AvgPool2d",
        "nn.AvgPool3d",
        "nn.Dropout",
        "nn.Dropout1d",
        "nn.Dropout2d",
        "nn.Dropout3d",
        "nn.Embedding",
        "nn.EmbeddingBag",
        "nn.Identity",
        "nn.Flatten",
        "nn.CrossEntropyLoss",
        "nn.MSELoss",
        "nn.L1Loss",
        "nn.BCELoss",
        "nn.NLLLoss",
        "nn.KLDivLoss",
    ],
    "HF-Custom-Ops": [
        "Conv1D",
        "NewGELUActivation",
        "GELUActivation",
        "FastGELUActivation",
        "T5LayerNorm",
        "OPTLearnedPositionalEmbedding",
        "RotaryEmbedding",
        "GPTNeoXRotaryEmbedding",
    ],
}


def classify_python(ops: list[str]) -> list[dict]:
    """Classify ops using Python classifier.

    Uses classify_operation with a mock module class to classify by
    canonical name, matching how the Rust side works.
    """
    import torch.nn as nn

    # Map canonical names to torch module classes
    op_to_module: dict[str, type] = {
        "nn.Linear": nn.Linear,
        "nn.ReLU": nn.ReLU,
        "nn.GELU": nn.GELU,
        "nn.SiLU": nn.SiLU,
        "nn.Sigmoid": nn.Sigmoid,
        "nn.Tanh": nn.Tanh,
        "nn.Softmax": nn.Softmax,
        "nn.Softplus": nn.Softplus,
        "nn.LeakyReLU": nn.LeakyReLU,
        "nn.PReLU": nn.PReLU,
        "nn.LayerNorm": nn.LayerNorm,
        "nn.GroupNorm": nn.GroupNorm,
        "nn.BatchNorm1d": nn.BatchNorm1d,
        "nn.BatchNorm2d": nn.BatchNorm2d,
        "nn.BatchNorm3d": nn.BatchNorm3d,
        "nn.Conv1d": nn.Conv1d,
        "nn.Conv2d": nn.Conv2d,
        "nn.Conv3d": nn.Conv3d,
        "nn.ConvTranspose1d": nn.ConvTranspose1d,
        "nn.ConvTranspose2d": nn.ConvTranspose2d,
        "nn.ConvTranspose3d": nn.ConvTranspose3d,
        "nn.MaxPool1d": nn.MaxPool1d,
        "nn.MaxPool2d": nn.MaxPool2d,
        "nn.MaxPool3d": nn.MaxPool3d,
        "nn.AvgPool1d": nn.AvgPool1d,
        "nn.AvgPool2d": nn.AvgPool2d,
        "nn.AvgPool3d": nn.AvgPool3d,
        "nn.Dropout": nn.Dropout,
        "nn.Dropout1d": nn.Dropout1d,
        "nn.Dropout2d": nn.Dropout2d,
        "nn.Dropout3d": nn.Dropout3d,
        "nn.Embedding": nn.Embedding,
        "nn.EmbeddingBag": nn.EmbeddingBag,
        "nn.Identity": nn.Identity,
        "nn.Flatten": nn.Flatten,
        "nn.CrossEntropyLoss": nn.CrossEntropyLoss,
        "nn.MSELoss": nn.MSELoss,
        "nn.L1Loss": nn.L1Loss,
        "nn.BCELoss": nn.BCELoss,
        "nn.NLLLoss": nn.NLLLoss,
        "nn.KLDivLoss": nn.KLDivLoss,
        "nn.Bilinear": nn.Bilinear,
        "nn.LazyLinear": nn.LazyLinear,
        "nn.RMSNorm": getattr(nn, "RMSNorm", None),
    }

    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for op_name in ops:
            # For custom rules (HF types), use the custom_rules dict
            if op_name in CUSTOM_RULES:
                cls = CUSTOM_RULES[op_name].value
                results.append(
                    {
                        "op_type": op_name,
                        "classification": cls,
                    }
                )
                continue

            module_cls = op_to_module.get(op_name)
            if module_cls is None:
                results.append(
                    {
                        "op_type": op_name,
                        "classification": "GRAY",
                    }
                )
                continue

            # Instantiate with minimal args
            try:
                if op_name == "nn.Linear":
                    module = module_cls(10, 10)
                elif op_name == "nn.LazyLinear":
                    module = module_cls(10)
                elif op_name == "nn.Bilinear":
                    module = module_cls(10, 10, 10)
                elif op_name in (
                    "nn.LayerNorm",
                    "nn.GroupNorm",
                    "nn.RMSNorm",
                ):
                    if op_name == "nn.GroupNorm":
                        module = module_cls(2, 10)
                    elif op_name == "nn.RMSNorm":
                        if module_cls is None:
                            results.append(
                                {
                                    "op_type": op_name,
                                    "classification": "S",
                                }
                            )
                            continue
                        module = module_cls(10)
                    else:
                        module = module_cls(10)
                elif "BatchNorm" in op_name:
                    module = module_cls(10)
                elif "Conv" in op_name:
                    module = module_cls(3, 3, 3)
                elif "Pool" in op_name:
                    module = module_cls(2)
                elif op_name in (
                    "nn.Embedding",
                    "nn.EmbeddingBag",
                ):
                    module = module_cls(100, 10)
                elif op_name == "nn.Softmax":
                    module = module_cls(dim=-1)
                else:
                    module = module_cls()
            except Exception:
                results.append(
                    {
                        "op_type": op_name,
                        "classification": "GRAY",
                    }
                )
                continue

            result = classify_operation(module, custom_rules=CUSTOM_RULES)
            results.append(
                {
                    "op_type": op_name,
                    "classification": result.classification.value,
                }
            )

    return results


def classify_rust(model_name: str, ops: list[str]) -> dict | None:
    """Classify ops using Rust binary.

    Writes input JSON, runs cross_validate binary, reads output JSON.
    Returns parsed output or None if binary not found.
    """
    if not CROSS_VALIDATE_BIN.exists():
        print(f"  WARNING: Rust binary not found at {CROSS_VALIDATE_BIN}")
        return None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_in:
        json.dump(
            {
                "model_name": model_name,
                "ops": ops,
                "custom_rules": CUSTOM_RULES_STR,
            },
            f_in,
        )
        input_path = f_in.name

    with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as f_out:
        output_path = f_out.name

    try:
        result = subprocess.run(
            [str(CROSS_VALIDATE_BIN), input_path, output_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"  Rust binary failed: {result.stderr}")
            return None
        return json.loads(Path(output_path).read_text())
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"  Error running Rust binary: {e}")
        return None
    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def run_burn_benchmark() -> str | None:
    """Run the Burn forward pass benchmark and capture output."""
    bench_bin = RUST_DIR / "target" / "release" / "examples" / "bench_forward"
    if not bench_bin.exists():
        # Try debug build
        bench_bin = RUST_DIR / "target" / "debug" / "examples" / "bench_forward"
    if not bench_bin.exists():
        print("  WARNING: bench_forward binary not found")
        return None

    try:
        result = subprocess.run(
            [str(bench_bin)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  Burn benchmark failed: {result.stderr}")
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        print("  Burn benchmark timed out")
        return None


def generate_report(
    parity_results: list[dict],
    burn_output: str | None,
) -> str:
    """Generate the cross-validation report."""
    lines = [
        "# Rust Cross-Validation Report",
        "",
        "> Auto-generated by `scripts/rust_cross_validation.py`. Do not edit manually.",
        "",
        "## Classification Parity: Python vs Rust",
        "",
    ]

    total_ops = 0
    total_match = 0
    total_mismatch = 0

    lines.append("| Model | Ops | Matches | Mismatches | Rust Time (µs) | Status |")
    lines.append("|-------|-----|---------|-----------|----------------|--------|")

    for pr in parity_results:
        total_ops += pr["total"]
        total_match += pr["matches"]
        total_mismatch += pr["mismatches"]
        status = "PASS" if pr["mismatches"] == 0 else "FAIL"
        rust_time = pr.get("rust_time_us", "N/A")
        lines.append(
            f"| {pr['model']} | {pr['total']} | "
            f"{pr['matches']} | {pr['mismatches']} | "
            f"{rust_time} | {status} |"
        )

    lines.append("")
    overall = "PASS" if total_mismatch == 0 else "FAIL"
    lines.append(
        f"**Overall**: {total_ops} ops, "
        f"{total_match} matches, "
        f"{total_mismatch} mismatches — **{overall}**"
    )
    lines.append("")

    # Mismatch details
    has_mismatches = any(pr["mismatches"] > 0 for pr in parity_results)
    if has_mismatches:
        lines.append("### Mismatches")
        lines.append("")
        lines.append("| Model | Op | Python | Rust |")
        lines.append("|-------|----|--------|------|")
        for pr in parity_results:
            for mm in pr.get("mismatch_details", []):
                lines.append(f"| {pr['model']} | {mm['op']} | {mm['python']} | {mm['rust']} |")
        lines.append("")

    # Burn benchmark
    if burn_output:
        lines.append("## Burn Forward Pass Benchmark")
        lines.append("")
        lines.append("```")
        lines.append(burn_output.strip())
        lines.append("```")
        lines.append("")

    lines.append(
        "---\n\n*Generated by `scripts/rust_cross_validation.py` — BWSK Combinator AI Framework*\n"
    )
    return "\n".join(lines)


def main() -> int:
    """Run cross-validation between Python and Rust classifiers."""
    print("=" * 60)
    print("BWSK Cross-Validation: Python vs Rust")
    print("=" * 60)

    # Step 1: Build Rust workspace
    print("\nBuilding Rust workspace...")
    build_result = subprocess.run(
        ["cargo", "build", "--workspace", "--examples"],
        cwd=str(RUST_DIR),
        capture_output=True,
        text=True,
    )
    if build_result.returncode != 0:
        print(f"Rust build failed:\n{build_result.stderr}")
        return 1
    print("  Build successful.")

    # Also build release for bench_forward
    print("Building release (for benchmarks)...")
    subprocess.run(
        [
            "cargo",
            "build",
            "--workspace",
            "--examples",
            "--release",
        ],
        cwd=str(RUST_DIR),
        capture_output=True,
        text=True,
    )

    # Step 2: Cross-validate classifications
    print("\nCross-validating classifications...")
    parity_results = []

    for model_name, ops in SYNTHETIC_MODELS.items():
        print(f"\n  Model: {model_name} ({len(ops)} ops)")

        # Python classification
        start = time.perf_counter()
        py_results = classify_python(ops)
        py_time = time.perf_counter() - start
        print(f"    Python: {len(py_results)} ops in {py_time * 1e6:.0f}µs")

        # Rust classification
        rust_output = classify_rust(model_name, ops)

        matches = 0
        mismatches = 0
        mismatch_details = []
        rust_time_us = "N/A"

        if rust_output is not None:
            rust_time_us = rust_output.get("classify_time_us", "N/A")
            rust_per_op = rust_output.get("per_op", [])

            for i, py_r in enumerate(py_results):
                if i < len(rust_per_op):
                    rust_cls = rust_per_op[i]["classification"]
                    py_cls = py_r["classification"]
                    if py_cls == rust_cls:
                        matches += 1
                    else:
                        mismatches += 1
                        mismatch_details.append(
                            {
                                "op": py_r["op_type"],
                                "python": py_cls,
                                "rust": rust_cls,
                            }
                        )
                else:
                    mismatches += 1
        else:
            # No Rust binary — all count as mismatches
            mismatches = len(ops)

        status = "PASS" if mismatches == 0 else "FAIL"
        print(f"    Result: {matches}/{len(ops)} match, {mismatches} mismatch — {status}")
        if mismatch_details:
            for mm in mismatch_details:
                print(f"      MISMATCH: {mm['op']}: Python={mm['python']}, Rust={mm['rust']}")

        parity_results.append(
            {
                "model": model_name,
                "total": len(ops),
                "matches": matches,
                "mismatches": mismatches,
                "mismatch_details": mismatch_details,
                "rust_time_us": rust_time_us,
            }
        )

    # Step 3: Run Burn benchmark
    print("\n\nRunning Burn forward pass benchmark...")
    burn_output = run_burn_benchmark()
    if burn_output:
        print(burn_output)

    # Step 4: Generate report
    print("\nGenerating report...")
    report = generate_report(parity_results, burn_output)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(f"Report written to: {REPORT_PATH}")

    # Summary
    total_mismatches = sum(pr["mismatches"] for pr in parity_results)
    if total_mismatches > 0:
        print(f"\nFAILED: {total_mismatches} classification mismatches")
        return 1

    print("\nPASSED: All classifications match between Python and Rust")
    return 0


if __name__ == "__main__":
    sys.exit(main())
