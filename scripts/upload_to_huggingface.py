"""Upload benchmark results and fine-tuned models to HuggingFace Hub.

Creates public dataset and model repos under the `tzervas` organization:

Dataset repos:
  - tzervas/bwsk-benchmark-scale-sweep
  - tzervas/bwsk-benchmark-architecture-diversity
  - tzervas/bwsk-benchmark-convergence
  - tzervas/bwsk-benchmark-combined

Model repos (fine-tuned checkpoints):
  - tzervas/bwsk-{model}-{mode} for representative models

Usage:
    uv run python scripts/upload_to_huggingface.py
    uv run python scripts/upload_to_huggingface.py --dry-run
    uv run python scripts/upload_to_huggingface.py --datasets-only
    uv run python scripts/upload_to_huggingface.py --models-only
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_NAMESPACE = "tzervas"
SCRIPTS_DIR = Path(__file__).resolve().parent
TAGS = [
    "ai-s-combinator",
    "bwsk",
    "information-preservation",
    "neural-architecture-analysis",
]

DATASET_REPOS = {
    "bwsk-benchmark-scale-sweep": {
        "description": (
            "BWSK S/K classification and training benchmark across 10 transformer models (70M-2.7B)"
        ),
        "source_files": [
            "extended_pythia-70m_results.json",
            "extended_t5-small_results.json",
            "extended_bert-base_results.json",
            "extended_gpt2-small_results.json",
            "extended_pythia-160m_results.json",
            "extended_opt-350m_results.json",
            "extended_gpt2-medium_results.json",
            "extended_pythia-410m_results.json",
            "extended_pythia-1b_results.json",
            "extended_phi-2_results.json",
        ],
    },
    "bwsk-benchmark-architecture-diversity": {
        "description": (
            "BWSK S/K classification benchmark across CNN, ViT, SSM/Mamba, and MoE architectures"
        ),
        "source_files": [
            "extended_resnet50_results.json",
            "extended_efficientnet-b0_results.json",
            "extended_mobilenetv2_results.json",
            "extended_vit-base_results.json",
            "extended_mamba-130m_results.json",
            "extended_mamba-370m_results.json",
            "extended_switch-base-8_results.json",
        ],
    },
    "bwsk-benchmark-convergence": {
        "description": (
            "BWSK convergence experiment: 1500-step training with 3 seeds and statistical analysis"
        ),
        "source_files": [
            "convergence_results.json",
            "convergence_pythia-70m_results.json",
            "convergence_gpt2-medium_results.json",
            "convergence_mamba-130m_results.json",
            "convergence_resnet50_results.json",
        ],
    },
    "bwsk-benchmark-combined": {
        "description": (
            "Combined BWSK benchmark: all 17 models, convergence, and memory/throughput profiling"
        ),
        "source_files": [
            "extended_benchmark_results.json",
            "convergence_results.json",
            "memory_throughput_results.json",
        ],
    },
}


# ---------------------------------------------------------------------------
# README / Card generation
# ---------------------------------------------------------------------------


def generate_dataset_card(repo_name: str, description: str, files: list[str]) -> str:
    """Generate a HuggingFace dataset card (README.md).

    Includes YAML frontmatter for HF Hub discovery and a description
    of the dataset contents.
    """
    tag_yaml = "\n".join(f"  - {t}" for t in TAGS)
    file_list = "\n".join(f"  - `{f}`" for f in files)

    return f"""---
language:
  - en
tags:
{tag_yaml}
task_categories:
  - other
pretty_name: "{repo_name}"
---

# {repo_name}

{description}

## About

This dataset contains benchmark results from the **BWSK Combinator AI Framework**,
which uses combinator logic (B, W, S, K) as a typed architectural description
language for neural networks.

The framework classifies every neural network operation as:
- **S-type** (information-preserving, reversible, coordination-free)
- **K-type** (information-erasing, synchronization point)
- **GRAY** (context-dependent)

## Files

{file_list}

## Usage

```python
import json
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="{HF_NAMESPACE}/{repo_name}",
    filename="<filename>.json",
    repo_type="dataset",
)
with open(path) as f:
    data = json.load(f)
```

## Links

- **GitHub**: [tzervas/ai-s-combinator](https://github.com/tzervas/ai-s-combinator)
- **Whitepaper**: See `docs/WHITEPAPER.md` in the GitHub repo

## License

MIT
"""


def generate_model_card(
    model_slug: str,
    base_model: str,
    mode: str,
    results: dict | None = None,
) -> str:
    """Generate a HuggingFace model card for a fine-tuned checkpoint.

    Includes training configuration, S/K analysis summary, and
    performance metrics extracted from benchmark results.
    """
    tag_yaml = "\n".join(f"  - {t}" for t in TAGS)

    s_ratio = "N/A"
    k_ratio = "N/A"
    final_loss = "N/A"
    memory_mb = "N/A"

    if results:
        cls = results.get("classification")
        if cls:
            s_ratio = f"{cls['s_ratio']:.3f}"
            k_ratio = f"{cls['k_ratio']:.3f}"
        ft = results.get(f"finetune_{mode}")
        if ft:
            final_loss = f"{ft['final_loss']:.4f}"
            memory_mb = f"{ft['peak_memory_mb']:.0f}"

    return f"""---
tags:
{tag_yaml}
base_model: {base_model}
license: mit
---

# BWSK Fine-tuned: {model_slug}

Fine-tuned checkpoint of **{base_model}** using the BWSK framework
in **{mode}** training mode.

## BWSK Analysis

| Metric | Value |
|--------|-------|
| S-type ratio | {s_ratio} |
| K-type ratio | {k_ratio} |
| Training mode | {mode} |
| Final loss | {final_loss} |
| Peak memory (MB) | {memory_mb} |

## Training Configuration

- **Base model**: {base_model}
- **Optimizer**: AdamW
- **Mode**: {mode}

## Links

- **GitHub**: [tzervas/ai-s-combinator](https://github.com/tzervas/ai-s-combinator)
- **Framework**: BWSK Combinator AI Framework

## License

MIT
"""


# ---------------------------------------------------------------------------
# Upload functions
# ---------------------------------------------------------------------------


def upload_dataset_repo(
    api: HfApi,
    repo_name: str,
    config: dict,
    dry_run: bool = False,
) -> None:
    """Create and populate a dataset repo on HuggingFace Hub."""
    full_name = f"{HF_NAMESPACE}/{repo_name}"
    description = config["description"]
    source_files = config["source_files"]

    # Check which files exist
    existing_files = []
    for fname in source_files:
        fpath = SCRIPTS_DIR / fname
        if fpath.exists():
            existing_files.append(fname)

    if not existing_files:
        print(f"  SKIP {repo_name}: no source files found (run benchmarks first)")
        return

    print(f"  {repo_name}: {len(existing_files)}/{len(source_files)} files")

    if dry_run:
        print(f"    [DRY RUN] Would create {full_name}")
        return

    # Create repo
    try:
        create_repo(
            repo_id=full_name,
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )
    except Exception as e:
        print(f"    WARNING: Could not create repo: {e}")

    # Upload files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Generate README
        readme = generate_dataset_card(repo_name, description, existing_files)
        (tmpdir_path / "README.md").write_text(readme)

        # Copy data files
        for fname in existing_files:
            src = SCRIPTS_DIR / fname
            dst = tmpdir_path / fname
            dst.write_text(src.read_text())

        try:
            api.upload_folder(
                folder_path=str(tmpdir_path),
                repo_id=full_name,
                repo_type="dataset",
            )
            print(f"    Uploaded to: https://huggingface.co/datasets/{full_name}")
        except Exception as e:
            print(f"    ERROR uploading: {e}")


def upload_model_repos(
    api: HfApi,
    dry_run: bool = False,
) -> None:
    """Upload fine-tuned model checkpoints to HuggingFace Hub.

    Uploads model cards and metadata for representative models.
    Actual model weights are only uploaded if checkpoints exist locally.
    """
    # Representative models to upload
    model_configs = [
        {
            "slug": "pythia-70m",
            "base": "EleutherAI/pythia-70m",
            "results_file": "extended_pythia-70m_results.json",
        },
        {
            "slug": "gpt2-medium",
            "base": "openai-community/gpt2-medium",
            "results_file": "extended_gpt2-medium_results.json",
        },
        {
            "slug": "resnet50",
            "base": "torchvision/resnet50",
            "results_file": "extended_resnet50_results.json",
        },
    ]

    for mc in model_configs:
        for mode in ["conventional", "bwsk_reversible"]:
            repo_name = f"bwsk-{mc['slug']}-{mode.replace('_', '-')}"
            full_name = f"{HF_NAMESPACE}/{repo_name}"

            # Load results if available
            results = None
            results_path = SCRIPTS_DIR / mc["results_file"]
            if results_path.exists():
                with open(results_path) as f:
                    results = json.load(f)

            print(f"  {repo_name}")

            if dry_run:
                print(f"    [DRY RUN] Would create {full_name}")
                continue

            try:
                create_repo(
                    repo_id=full_name,
                    repo_type="model",
                    private=False,
                    exist_ok=True,
                )
            except Exception as e:
                print(f"    WARNING: Could not create repo: {e}")
                continue

            # Upload model card
            with tempfile.TemporaryDirectory() as tmpdir:
                card = generate_model_card(repo_name, mc["base"], mode, results)
                readme_path = Path(tmpdir) / "README.md"
                readme_path.write_text(card)

                # Also include results JSON if available
                if results:
                    results_dst = Path(tmpdir) / "benchmark_results.json"
                    results_dst.write_text(json.dumps(results, indent=2, default=str))

                try:
                    api.upload_folder(
                        folder_path=str(tmpdir),
                        repo_id=full_name,
                        repo_type="model",
                    )
                    print(f"    Uploaded to: https://huggingface.co/models/{full_name}")
                except Exception as e:
                    print(f"    ERROR uploading: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Upload benchmark data and models to HuggingFace Hub."""
    parser = argparse.ArgumentParser(description="Upload BWSK benchmarks to HuggingFace")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without uploading",
    )
    parser.add_argument(
        "--datasets-only",
        action="store_true",
        help="Upload only dataset repos",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Upload only model repos",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("HUGGINGFACE UPLOAD")
    print(f"Namespace: {HF_NAMESPACE}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)

    api = HfApi()

    # Verify authentication
    try:
        user_info = api.whoami()
        print(f"Authenticated as: {user_info.get('name', 'unknown')}")
    except Exception as e:
        print(f"ERROR: Not authenticated with HuggingFace Hub: {e}")
        print("Run: huggingface-cli login")
        sys.exit(1)

    # Upload datasets
    if not args.models_only:
        print("\n--- Dataset Repos ---")
        for repo_name, config in DATASET_REPOS.items():
            upload_dataset_repo(api, repo_name, config, dry_run=args.dry_run)

    # Upload models
    if not args.datasets_only:
        print("\n--- Model Repos ---")
        upload_model_repos(api, dry_run=args.dry_run)

    print("\n" + "=" * 70)
    print("UPLOAD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
