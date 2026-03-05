"""Generate and upload HuggingFace model cards for BWSK repos.

Supports three card types:
- **consolidated**: One comprehensive card per model for the new single-repo
  layout (tzervas/bwsk-{slug}) — the default for new deployments.
- **model**: Per-variant cards for the old 96-repo layout (legacy).
- **results**: Per-model results cards for the old 16-repo layout (legacy).

Reads existing fulltrain_*_results.json files to create rich README.md model
cards with YAML frontmatter, training results, S/K classification, and
cross-mode comparisons. Uploads via HfApi.upload_file() with rate limiting.

Usage:
    uv run python scripts/generate_model_cards.py --dry-run
    uv run python scripts/generate_model_cards.py --preview pythia-70m
    uv run python scripts/generate_model_cards.py --preview pythia-70m --type consolidated
    uv run python scripts/generate_model_cards.py --slug pythia-70m
    uv run python scripts/generate_model_cards.py --type consolidated
    uv run python scripts/generate_model_cards.py --delay 2.0
    uv run python scripts/generate_model_cards.py  # upload all (consolidated)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_NAMESPACE = "tzervas"
SCRIPTS_DIR = Path(__file__).resolve().parent

# Maps torchvision model names to proper HF model IDs for base_model metadata.
TORCHVISION_TO_HF: dict[str, str] = {
    "resnet50": "microsoft/resnet-50",
    "efficientnet_b0": "google/efficientnet-b0",
    "mobilenet_v2": "google/mobilenet_v2_1.0_224",
}

# Maps arch_type to HuggingFace pipeline_tag values.
ARCH_TYPE_TO_TASK: dict[str, str] = {
    "causal_lm": "text-generation",
    "masked_lm": "fill-mask",
    "seq2seq": "summarization",
    "image_cls": "image-classification",
    "ssm_lm": "text-generation",
}

# Maps arch_type to dataset names for metadata.
ARCH_TYPE_TO_DATASET: dict[str, str] = {
    "causal_lm": "wikitext",
    "masked_lm": "wikitext",
    "seq2seq": "wikitext",
    "image_cls": "cifar10",
    "ssm_lm": "wikitext",
}

# Maps arch_type to the eval metric name HuggingFace expects.
ARCH_TYPE_TO_METRIC: dict[str, str] = {
    "causal_lm": "perplexity",
    "masked_lm": "perplexity",
    "seq2seq": "perplexity",
    "image_cls": "accuracy",
    "ssm_lm": "perplexity",
}

# Human-readable mode names.
MODE_DISPLAY: dict[str, str] = {
    "conventional": "Conventional",
    "bwsk_analyzed": "BWSK Analyzed",
    "bwsk_reversible": "BWSK Reversible",
}

# Human-readable experiment names.
EXPERIMENT_DISPLAY: dict[str, str] = {
    "finetune": "Fine-tune",
    "scratch": "From Scratch",
}

ALL_MODES = ["conventional", "bwsk_analyzed", "bwsk_reversible"]
ALL_EXPERIMENTS = ["finetune", "scratch"]

BWSK_DESCRIPTION = """\
BWSK is a framework that classifies every neural network operation as \
**S-type** (information-preserving, reversible, coordination-free) or \
**K-type** (information-erasing, synchronization point) using combinator \
logic. This classification enables reversible backpropagation through \
S-phases to save memory, and CALM-based parallelism analysis."""

BIBTEX = """\
```bibtex
@software{zervas2026bwsk,
  author = {Zervas, Tyler},
  title = {BWSK: Combinator-Typed Neural Network Analysis},
  year = {2026},
  url = {https://github.com/tzervas/ai-s-combinator},
}
```"""

GITHUB_URL = "https://github.com/tzervas/ai-s-combinator"

# Per-model training config: slug -> (batch_size, seq_len, finetune_lr).
# finetune_lr is from ExtendedModelConfig; scratch LR = finetune_lr * 5.
MODEL_TRAINING_CONFIG: dict[str, tuple[int, int, float]] = {
    "pythia-70m": (8, 512, 5e-5),
    "t5-small": (4, 512, 5e-5),
    "bert-base": (4, 512, 5e-5),
    "gpt2-small": (4, 512, 5e-5),
    "pythia-160m": (4, 512, 3e-5),
    "opt-350m": (2, 512, 2e-5),
    "gpt2-medium": (2, 512, 5e-5),
    "pythia-410m": (2, 512, 2e-5),
    "pythia-1b": (1, 256, 1e-5),
    "resnet50": (32, 224, 1e-3),
    "efficientnet-b0": (32, 224, 1e-3),
    "mobilenetv2": (32, 224, 1e-3),
    "vit-base": (16, 224, 5e-5),
    "mamba-130m": (2, 256, 3e-5),
    "mamba-370m": (1, 256, 2e-5),
    "switch-base-8": (1, 256, 3e-5),
}

# Maps arch_type to the correct AutoModel class for usage examples.
ARCH_TYPE_TO_AUTOMODEL: dict[str, str] = {
    "causal_lm": "AutoModelForCausalLM",
    "masked_lm": "AutoModelForMaskedLM",
    "seq2seq": "AutoModelForSeq2SeqLM",
    "ssm_lm": "AutoModelForCausalLM",
}

# Torchvision model slugs (no transformers integration).
TORCHVISION_SLUGS = {"resnet50", "efficientnet-b0", "mobilenetv2"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_aggregated_results() -> dict[str, dict]:
    """Load all fulltrain_*_results.json files, keyed by slug.

    Returns:
        Dict mapping slug -> aggregated result dict.
    """
    results = {}
    for path in sorted(SCRIPTS_DIR.glob("fulltrain_*_results.json")):
        # Skip per-run files (they have experiment_mode in name)
        name = path.stem  # e.g. "fulltrain_pythia-70m_results"
        parts = name.split("_")
        # Aggregated: fulltrain_{slug}_results  (last part is "results")
        if parts[-1] != "results":
            continue
        with open(path) as f:
            data = json.load(f)
        results[data["slug"]] = data
    return results


def load_per_run(slug: str, experiment: str, mode: str) -> dict | None:
    """Load a single per-run result file.

    Args:
        slug: Model slug (e.g. "pythia-70m").
        experiment: "finetune" or "scratch".
        mode: "conventional", "bwsk_analyzed", or "bwsk_reversible".

    Returns:
        Per-run result dict, or None if file doesn't exist.
    """
    path = SCRIPTS_DIR / f"fulltrain_{slug}_{experiment}_{mode}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_metric(value: float, metric_name: str) -> str:
    """Format a metric value for display.

    Args:
        value: The metric value.
        metric_name: Name of metric ("perplexity" or "accuracy").

    Returns:
        Formatted string, e.g. "29.58" or "97.6%".
    """
    if value == 0.0 and metric_name == "perplexity":
        return "N/A (OOM)"
    if metric_name == "accuracy":
        return f"{value * 100:.1f}%"
    return f"{value:.2f}"


def format_time(seconds: float) -> str:
    """Format wall time as human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like "2m 30s" or "1h 15m".
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def format_memory(mb: float) -> str:
    """Format memory in MB to human-readable string.

    Args:
        mb: Memory in megabytes.

    Returns:
        Formatted string like "4.1 GB" or "512 MB".
    """
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def hf_base_model(hf_id: str) -> str:
    """Map a model's hf_id to a proper HuggingFace base_model identifier.

    Torchvision models don't have HF repos, so we map them to known HF IDs.

    Args:
        hf_id: The hf_id from the results JSON.

    Returns:
        Proper HF model ID string.
    """
    return TORCHVISION_TO_HF.get(hf_id, hf_id)


def memory_savings_pct(conventional_mb: float, reversible_mb: float) -> str:
    """Compute memory savings percentage of reversible vs conventional.

    Args:
        conventional_mb: Peak memory for conventional mode.
        reversible_mb: Peak memory for reversible mode.

    Returns:
        Formatted string like "18.3%" or "0.0%".
    """
    if conventional_mb <= 0:
        return "N/A"
    savings = (conventional_mb - reversible_mb) / conventional_mb * 100
    return f"{max(0, savings):.1f}%"


# ---------------------------------------------------------------------------
# Model repo card generation (96 repos)
# ---------------------------------------------------------------------------


def generate_model_repo_card(
    agg: dict,
    run: dict,
    experiment: str,
    mode: str,
    sibling_runs: dict[str, dict | None],
) -> str:
    """Generate a model card README.md for a single model repo.

    Args:
        agg: Aggregated results for this model.
        run: Per-run results for this specific experiment+mode.
        experiment: "finetune" or "scratch".
        mode: "conventional", "bwsk_analyzed", or "bwsk_reversible".
        sibling_runs: Dict mapping mode -> per-run result for same experiment.

    Returns:
        Complete README.md content with YAML frontmatter.
    """
    slug = agg["slug"]
    model_name = agg["model_name"]
    base_model = hf_base_model(agg["hf_id"])
    arch_type = agg["arch_type"]
    arch_family = agg["arch_family"]
    params_m = agg["params_m"]
    task = ARCH_TYPE_TO_TASK.get(arch_type, "text-generation")
    dataset = ARCH_TYPE_TO_DATASET.get(arch_type, "wikitext")
    metric_name = run.get("metric_name", "perplexity")
    # Ratios are at top level in fulltrain results (not nested).
    s_ratio = agg.get("s_ratio", 0) or 0
    k_ratio = agg.get("k_ratio", 0) or 0
    gray_ratio = 1.0 - s_ratio - k_ratio if s_ratio + k_ratio < 0.999 else 0

    mode_display = MODE_DISPLAY[mode]
    exp_display = EXPERIMENT_DISPLAY[experiment]

    # Best metric for model-index
    best_val = run.get("best_val_metric", 0)
    test_val = run.get("test_metric", 0)
    display_metric = test_val if test_val > 0 else best_val

    verify_token = "false"

    # YAML frontmatter
    lines = ["---"]
    lines.append("license: mit")
    lines.append(f"base_model: {base_model}")
    if slug not in TORCHVISION_SLUGS:
        lines.append("library_name: transformers")
    lines.append(f"pipeline_tag: {task}")
    lines.append("tags:")
    lines.append("  - bwsk")
    lines.append("  - combinator-analysis")
    lines.append(f"  - {arch_family}")
    lines.append(f"  - {mode.replace('_', '-')}")
    lines.append(f"  - {experiment}")
    if mode != "conventional":
        lines.append("  - reversible-backprop")
    lines.append("datasets:")
    lines.append(f"  - {dataset}")
    lines.append("metrics:")
    lines.append(f"  - {metric_name}")

    # model-index for HF leaderboard
    lines.append("model-index:")
    lines.append(f"  - name: bwsk-{slug}-{experiment}-{mode}")
    lines.append("    results:")
    lines.append("      - task:")
    lines.append(f"          type: {task}")
    lines.append("        dataset:")
    lines.append(f"          name: {dataset}")
    lines.append(f"          type: {dataset}")
    lines.append("        metrics:")
    lines.append(f"          - name: {metric_name}")
    lines.append(f"            type: {metric_name}")
    if display_metric > 0:
        lines.append(f"            value: {display_metric:.4f}")
    lines.append(f"            verified: {verify_token}")
    lines.append("---")
    lines.append("")

    # Title
    lines.append(f"# BWSK {model_name} — {exp_display} ({mode_display})")
    lines.append("")

    # One-liner
    if mode == "conventional":
        lines.append(
            f"Standard {experiment} of **{model_name}** "
            f"({params_m}M params) with BWSK S/K classification analysis."
        )
    elif mode == "bwsk_analyzed":
        lines.append(
            f"BWSK-analyzed {experiment} of **{model_name}** "
            f"({params_m}M params) with erasure budget tracking and "
            f"parallelism ratio monitoring."
        )
    else:
        lines.append(
            f"BWSK-reversible {experiment} of **{model_name}** "
            f"({params_m}M params) using S-phase checkpoint-free "
            f"backpropagation for memory savings."
        )
    lines.append("")

    # What is BWSK?
    lines.append("## What is BWSK?")
    lines.append("")
    lines.append(BWSK_DESCRIPTION)
    lines.append("")

    # Model details table
    lines.append("## Model Details")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| **Base Model** | [{base_model}](https://huggingface.co/{base_model}) |")
    lines.append(f"| **Architecture** | {arch_family.title()} ({arch_type}) |")
    lines.append(f"| **Parameters** | {params_m}M |")
    lines.append(f"| **Experiment** | {exp_display} |")
    lines.append(f"| **BWSK Mode** | {mode_display} |")
    dataset_display = "CIFAR-10" if dataset == "cifar10" else "WikiText-2"
    lines.append(f"| **Dataset** | {dataset_display} |")
    lines.append("")

    # S/K Classification
    lines.append("## S/K Classification")
    lines.append("")
    lines.append("| Type | Ratio |")
    lines.append("|------|-------|")
    lines.append(f"| **S-type** (information-preserving) | {s_ratio:.1%} |")
    lines.append(f"| **K-type** (information-erasing) | {k_ratio:.1%} |")
    if gray_ratio > 0:
        lines.append(f"| **Gray** (context-dependent) | {gray_ratio:.1%} |")
    lines.append("")

    # Training results
    lines.append("## Training Results")
    lines.append("")
    final_loss = run.get("final_train_loss", 0)
    best_val_str = format_metric(best_val, metric_name)
    test_val_str = format_metric(test_val, metric_name)
    epochs = run.get("epochs_completed", 0)
    steps = run.get("total_steps", 0)
    wall_time = run.get("wall_time_s", 0)
    peak_mem = run.get("peak_memory_mb", 0)
    early_stopped = run.get("early_stopped", False)

    metric_label = metric_name.title()
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| **Final Train Loss** | {final_loss:.4f} |")
    lines.append(f"| **Best Val {metric_label}** | {best_val_str} |")
    lines.append(f"| **Test {metric_label}** | {test_val_str} |")
    lines.append(f"| **Epochs** | {epochs} |")
    lines.append(f"| **Total Steps** | {steps:,} |")
    lines.append(f"| **Wall Time** | {format_time(wall_time)} |")
    lines.append(f"| **Peak Memory** | {format_memory(peak_mem)} |")
    lines.append(f"| **Early Stopped** | {'Yes' if early_stopped else 'No'} |")
    lines.append("")

    if test_val == 0.0 and metric_name == "perplexity":
        lines.append(
            "> **Note:** Test evaluation returned 0.0 (OOM during "
            "test pass). The model trained successfully — only the "
            "final test evaluation exceeded available VRAM."
        )
        lines.append("")

    # Cross-mode comparison table
    lines.append("## Cross-Mode Comparison")
    lines.append("")
    lines.append(
        f"All three BWSK modes for the **{exp_display.lower()}** experiment on {model_name}:"
    )
    lines.append("")

    header = (
        f"| Mode | Final Loss | Val {metric_label} | Test {metric_label} | Peak Memory | Time |"
    )
    lines.append(header)
    lines.append("|------|------------|" + "|".join(["----------"] * 4) + "|")

    conv_mem = 0.0
    for m in ALL_MODES:
        sib = sibling_runs.get(m)
        if sib is None:
            continue
        m_display = MODE_DISPLAY[m]
        m_loss = sib.get("final_train_loss", 0)
        m_val = format_metric(sib.get("best_val_metric", 0), metric_name)
        m_test = format_metric(sib.get("test_metric", 0), metric_name)
        m_mem = sib.get("peak_memory_mb", 0)
        m_time = format_time(sib.get("wall_time_s", 0))
        if m == "conventional":
            conv_mem = m_mem
        bold = " **←**" if m == mode else ""
        lines.append(
            f"| {m_display}{bold} | {m_loss:.4f} | "
            f"{m_val} | {m_test} | "
            f"{format_memory(m_mem)} | {m_time} |"
        )
    lines.append("")

    # Memory savings note
    rev_run = sibling_runs.get("bwsk_reversible")
    if conv_mem > 0 and rev_run:
        rev_mem = rev_run.get("peak_memory_mb", 0)
        savings = memory_savings_pct(conv_mem, rev_mem)
        lines.append(f"**Memory savings (reversible vs conventional):** {savings}")
        lines.append("")

    # Related models
    lines.append("## Related Models")
    lines.append("")
    for exp in ALL_EXPERIMENTS:
        for m in ALL_MODES:
            repo = f"{HF_NAMESPACE}/bwsk-{slug}-{exp}-{m}"
            marker = " ← this model" if (exp == experiment and m == mode) else ""
            lines.append(f"- [{repo}](https://huggingface.co/{repo}){marker}")
    lines.append(
        f"- [{HF_NAMESPACE}/bwsk-{slug}-full-training-results]"
        f"(https://huggingface.co/{HF_NAMESPACE}/bwsk-{slug}"
        f"-full-training-results) — combined results"
    )
    lines.append("")

    # Training configuration
    cfg = MODEL_TRAINING_CONFIG.get(slug, (8, 512, 5e-5))
    batch, seq, ft_lr = cfg
    lr = ft_lr if experiment == "finetune" else ft_lr * 5

    lines.append("## Training Configuration")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("|---------|-------|")
    lines.append("| **Optimizer** | AdamW |")
    lines.append(f"| **Learning Rate** | {lr:.0e} |")
    lines.append("| **LR Schedule** | Cosine with warmup |")
    lines.append("| **Max Grad Norm** | 1.0 |")
    lines.append("| **Mixed Precision** | AMP (float16) |")
    lines.append("| **Early Stopping** | Patience 3 |")
    lines.append(f"| **Batch Size** | {batch} |")
    if arch_type != "image_cls":
        lines.append(f"| **Sequence Length** | {seq} |")
    lines.append("")

    # Links
    lines.append("## Links")
    lines.append("")
    lines.append(f"- [GitHub Repository]({GITHUB_URL})")
    lines.append(f"- [Whitepaper]({GITHUB_URL}/blob/main/docs/WHITEPAPER.md)")
    lines.append("")

    # Citation
    lines.append("## Citation")
    lines.append("")
    lines.append(BIBTEX)
    lines.append("")

    # License
    lines.append("## License")
    lines.append("")
    lines.append("MIT")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Results repo card generation (16 repos)
# ---------------------------------------------------------------------------


def generate_results_repo_card(agg: dict) -> str:
    """Generate a results repo README.md for a model's combined results.

    Args:
        agg: Aggregated results dict for this model.

    Returns:
        Complete README.md content with YAML frontmatter.
    """
    slug = agg["slug"]
    model_name = agg["model_name"]
    base_model = hf_base_model(agg["hf_id"])
    arch_type = agg["arch_type"]
    arch_family = agg["arch_family"]
    params_m = agg["params_m"]
    # Ratios are at top level in fulltrain results (not nested).
    s_ratio = agg.get("s_ratio", 0) or 0
    k_ratio = agg.get("k_ratio", 0) or 0
    gray_ratio = 1.0 - s_ratio - k_ratio if s_ratio + k_ratio < 0.999 else 0

    # Determine metric name from first available run
    metric_name = "perplexity"
    for exp in ALL_EXPERIMENTS:
        for m in ALL_MODES:
            key = f"{exp}_{m}"
            run_data = agg.get(key)
            if run_data:
                # Aggregated files use different key structure
                break

    # Load per-run files to get detailed metrics
    runs: dict[str, dict[str, dict | None]] = {}
    for exp in ALL_EXPERIMENTS:
        runs[exp] = {}
        for m in ALL_MODES:
            runs[exp][m] = load_per_run(slug, exp, m)
            if runs[exp][m] and runs[exp][m].get("metric_name"):
                metric_name = runs[exp][m]["metric_name"]

    metric_label = metric_name.title()
    dataset = ARCH_TYPE_TO_DATASET.get(arch_type, "wikitext")
    dataset_display = "CIFAR-10" if dataset == "cifar10" else "WikiText-2"

    lines = ["---"]
    lines.append("license: mit")
    lines.append("tags:")
    lines.append("  - bwsk")
    lines.append("  - combinator-analysis")
    lines.append(f"  - {arch_family}")
    lines.append("  - training-results")
    lines.append("  - benchmark")
    lines.append("---")
    lines.append("")

    lines.append(f"# BWSK {model_name} — Full Training Results")
    lines.append("")
    lines.append(
        f"Combined training results for **{model_name}** ({params_m}M "
        f"params) across all 3 BWSK modes and 2 experiments (fine-tune "
        f"+ from scratch) on {dataset_display}."
    )
    lines.append("")

    # What is BWSK?
    lines.append("## What is BWSK?")
    lines.append("")
    lines.append(BWSK_DESCRIPTION)
    lines.append("")

    # Model overview
    lines.append("## Model Overview")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| **Base Model** | [{base_model}](https://huggingface.co/{base_model}) |")
    lines.append(f"| **Architecture** | {arch_family.title()} ({arch_type}) |")
    lines.append(f"| **Parameters** | {params_m}M |")
    lines.append(f"| **Dataset** | {dataset_display} |")
    lines.append(f"| **Eval Metric** | {metric_label} |")
    lines.append("")

    # S/K Classification
    lines.append("## S/K Classification")
    lines.append("")
    lines.append("| Type | Ratio |")
    lines.append("|------|-------|")
    lines.append(f"| **S-type** (information-preserving) | {s_ratio:.1%} |")
    lines.append(f"| **K-type** (information-erasing) | {k_ratio:.1%} |")
    if gray_ratio > 0:
        lines.append(f"| **Gray** (context-dependent) | {gray_ratio:.1%} |")
    lines.append("")

    # Results tables per experiment
    for exp in ALL_EXPERIMENTS:
        exp_display = EXPERIMENT_DISPLAY[exp]
        lines.append(f"## {exp_display} Results")
        lines.append("")

        header = (
            f"| Mode | Final Loss | Val {metric_label} | "
            f"Test {metric_label} | Peak Memory | Time | Epochs |"
        )
        lines.append(header)
        sep = "|------|------------|" + "|".join(["----------"] * 5) + "|"
        lines.append(sep)

        conv_mem = 0.0
        for m in ALL_MODES:
            r = runs[exp].get(m)
            if r is None:
                lines.append(f"| {MODE_DISPLAY[m]} | — | — | — | — | — | — |")
                continue
            loss = r.get("final_train_loss", 0)
            val = format_metric(r.get("best_val_metric", 0), metric_name)
            test = format_metric(r.get("test_metric", 0), metric_name)
            mem = r.get("peak_memory_mb", 0)
            wt = format_time(r.get("wall_time_s", 0))
            ep = r.get("epochs_completed", 0)
            if m == "conventional":
                conv_mem = mem
            lines.append(
                f"| {MODE_DISPLAY[m]} | {loss:.4f} | {val} | "
                f"{test} | {format_memory(mem)} | {wt} | {ep} |"
            )
        lines.append("")

        # Memory savings
        rev = runs[exp].get("bwsk_reversible")
        if conv_mem > 0 and rev:
            rev_mem = rev.get("peak_memory_mb", 0)
            savings = memory_savings_pct(conv_mem, rev_mem)
            lines.append(f"**Memory savings (reversible vs conventional):** {savings}")
            lines.append("")

    # Individual model repos
    lines.append("## Individual Model Repos")
    lines.append("")
    for exp in ALL_EXPERIMENTS:
        for m in ALL_MODES:
            repo = f"{HF_NAMESPACE}/bwsk-{slug}-{exp}-{m}"
            lines.append(f"- [{repo}](https://huggingface.co/{repo})")
    lines.append("")

    # Links
    lines.append("## Links")
    lines.append("")
    lines.append(f"- [GitHub Repository]({GITHUB_URL})")
    lines.append(f"- [Whitepaper]({GITHUB_URL}/blob/main/docs/WHITEPAPER.md)")
    lines.append("")

    # Citation
    lines.append("## Citation")
    lines.append("")
    lines.append(BIBTEX)
    lines.append("")

    # License
    lines.append("## License")
    lines.append("")
    lines.append("MIT")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Consolidated card generation (16 repos — new layout)
# ---------------------------------------------------------------------------

# Subdirectory names inside consolidated repo: {experiment}-{mode} with hyphens.
VARIANT_DIRS: dict[tuple[str, str], str] = {
    (exp, mode): f"{exp}-{mode.replace('_', '-')}" for exp in ALL_EXPERIMENTS for mode in ALL_MODES
}


def generate_consolidated_card(agg: dict) -> str:
    """Generate a comprehensive README.md for a consolidated model repo.

    Includes all 6 variants (2 experiments x 3 modes), S/K classification,
    training results tables, memory savings, and usage instructions.
    This replaces both generate_model_repo_card() and
    generate_results_repo_card() for the new single-repo-per-model layout.

    Args:
        agg: Aggregated results dict (from fulltrain_*_results.json).

    Returns:
        Complete README.md content with YAML frontmatter.
    """
    slug = agg["slug"]
    model_name = agg["model_name"]
    base_model = hf_base_model(agg["hf_id"])
    arch_type = agg["arch_type"]
    arch_family = agg["arch_family"]
    params_m = agg["params_m"]
    task = ARCH_TYPE_TO_TASK.get(arch_type, "text-generation")
    dataset = ARCH_TYPE_TO_DATASET.get(arch_type, "wikitext")
    dataset_display = "CIFAR-10" if dataset == "cifar10" else "WikiText-2"
    s_ratio = agg.get("s_ratio", 0) or 0
    k_ratio = agg.get("k_ratio", 0) or 0
    gray_ratio = max(0, 1.0 - s_ratio - k_ratio) if s_ratio + k_ratio < 0.999 else 0

    # Load all per-run results
    runs: dict[str, dict[str, dict | None]] = {}
    metric_name = "perplexity"
    for exp in ALL_EXPERIMENTS:
        runs[exp] = {}
        for m in ALL_MODES:
            run = load_per_run(slug, exp, m)
            runs[exp][m] = run
            if run and run.get("metric_name"):
                metric_name = run["metric_name"]

    metric_label = metric_name.title()

    # --- YAML frontmatter ---
    lines = ["---"]
    lines.append("license: mit")
    lines.append(f"base_model: {base_model}")
    if slug not in TORCHVISION_SLUGS:
        lines.append("library_name: transformers")
    lines.append(f"pipeline_tag: {task}")
    lines.append("tags:")
    lines.append("  - bwsk")
    lines.append("  - combinator-analysis")
    lines.append(f"  - {arch_family}")
    lines.append("  - reversible-backprop")
    lines.append("  - convergence-training")
    lines.append("datasets:")
    lines.append(f"  - {dataset}")
    lines.append("metrics:")
    lines.append(f"  - {metric_name}")

    # model-index with all available variants
    lines.append("model-index:")
    lines.append(f"  - name: bwsk-{slug}")
    lines.append("    results:")
    for exp in ALL_EXPERIMENTS:
        for m in ALL_MODES:
            run = runs[exp].get(m)
            if run is None:
                continue
            test_val = run.get("test_metric", 0)
            best_val = run.get("best_val_metric", 0)
            display_metric = test_val if test_val > 0 else best_val
            lines.append("      - task:")
            lines.append(f"          type: {task}")
            lines.append(f"          name: {EXPERIMENT_DISPLAY[exp]} ({MODE_DISPLAY[m]})")
            lines.append("        dataset:")
            lines.append(f"          name: {dataset}")
            lines.append(f"          type: {dataset}")
            lines.append("        metrics:")
            lines.append(f"          - name: {metric_name}")
            lines.append(f"            type: {metric_name}")
            if display_metric > 0:
                lines.append(f"            value: {display_metric:.4f}")
            lines.append("            verified: false")
    lines.append("---")
    lines.append("")

    # --- Title and description ---
    lines.append(f"# BWSK {model_name}")
    lines.append("")
    lines.append(
        f"**{model_name}** ({params_m}M params) trained in "
        f"**6 variants** (3 BWSK modes x 2 experiments) on "
        f"{dataset_display} with full convergence training "
        f"and early stopping."
    )
    lines.append("")
    lines.append(
        "This repo contains all model weights, configs, and "
        "training results in a single consolidated repository."
    )
    lines.append("")

    # --- What is BWSK? ---
    lines.append("## What is BWSK?")
    lines.append("")
    lines.append(BWSK_DESCRIPTION)
    lines.append("")

    # --- Model overview ---
    lines.append("## Model Overview")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| **Base Model** | [{base_model}](https://huggingface.co/{base_model}) |")
    lines.append(f"| **Architecture** | {arch_family.title()} ({arch_type}) |")
    lines.append(f"| **Parameters** | {params_m}M |")
    lines.append(f"| **Dataset** | {dataset_display} |")
    lines.append(f"| **Eval Metric** | {metric_label} |")
    lines.append("")

    # --- S/K Classification ---
    lines.append("## S/K Classification")
    lines.append("")
    lines.append("| Type | Ratio |")
    lines.append("|------|-------|")
    lines.append(f"| **S-type** (information-preserving) | {s_ratio:.1%} |")
    lines.append(f"| **K-type** (information-erasing) | {k_ratio:.1%} |")
    if gray_ratio > 0:
        lines.append(f"| **Gray** (context-dependent) | {gray_ratio:.1%} |")
    lines.append("")

    # --- Results tables per experiment ---
    for exp in ALL_EXPERIMENTS:
        exp_display = EXPERIMENT_DISPLAY[exp]
        lines.append(f"## {exp_display} Results")
        lines.append("")

        header = (
            f"| Mode | Final Loss | Val {metric_label} | "
            f"Test {metric_label} | Peak Memory | "
            f"Time | Epochs |"
        )
        lines.append(header)
        sep = "|------|------------|" + "|".join(["----------"] * 5) + "|"
        lines.append(sep)

        conv_mem = 0.0
        for m in ALL_MODES:
            r = runs[exp].get(m)
            if r is None:
                lines.append(f"| {MODE_DISPLAY[m]} | — | — | — | — | — | — |")
                continue
            loss = r.get("final_train_loss", 0)
            val = format_metric(r.get("best_val_metric", 0), metric_name)
            test = format_metric(r.get("test_metric", 0), metric_name)
            mem = r.get("peak_memory_mb", 0)
            wt = format_time(r.get("wall_time_s", 0))
            ep = r.get("epochs_completed", 0)
            if m == "conventional":
                conv_mem = mem
            lines.append(
                f"| {MODE_DISPLAY[m]} | {loss:.4f} | "
                f"{val} | {test} | "
                f"{format_memory(mem)} | {wt} | {ep} |"
            )
        lines.append("")

        # Memory savings
        rev = runs[exp].get("bwsk_reversible")
        if conv_mem > 0 and rev:
            rev_mem = rev.get("peak_memory_mb", 0)
            savings = memory_savings_pct(conv_mem, rev_mem)
            lines.append(f"**Memory savings (reversible vs conventional):** {savings}")
            lines.append("")

    # --- Repo structure ---
    lines.append("## Repository Structure")
    lines.append("")
    lines.append("```")
    lines.append("├── README.md")
    lines.append("├── results.json")
    for exp in ALL_EXPERIMENTS:
        for m in ALL_MODES:
            variant_dir = VARIANT_DIRS[(exp, m)]
            lines.append(f"├── {variant_dir}/")
            lines.append("│   ├── model.safetensors")
            lines.append("│   ├── config.json")
            lines.append("│   └── training_results.json")
    lines.append("```")
    lines.append("")

    # --- Usage ---
    lines.append("## Usage")
    lines.append("")
    lines.append("Load a specific variant:")
    lines.append("")
    if arch_type == "image_cls":
        if slug in TORCHVISION_SLUGS:
            lines.append("```python")
            lines.append("import torch")
            lines.append("# Load fine-tuned conventional variant")
            lines.append("# Weights are in the finetune-conventional/ subdirectory")
            lines.append("```")
        else:
            auto_cls = "AutoModelForImageClassification"
            lines.append("```python")
            lines.append(f"from transformers import {auto_cls}, AutoFeatureExtractor")
            lines.append("")
            lines.append("# Load fine-tuned conventional variant")
            lines.append(f"model = {auto_cls}.from_pretrained(")
            lines.append(f'    "tzervas/bwsk-{slug}", subfolder="finetune-conventional"')
            lines.append(")")
            lines.append("```")
    else:
        auto_cls = ARCH_TYPE_TO_AUTOMODEL.get(arch_type, "AutoModelForCausalLM")
        lines.append("```python")
        lines.append(f"from transformers import {auto_cls}, AutoTokenizer")
        lines.append("")
        lines.append("# Load fine-tuned conventional variant")
        lines.append(f"model = {auto_cls}.from_pretrained(")
        lines.append(f'    "tzervas/bwsk-{slug}", subfolder="finetune-conventional"')
        lines.append(")")
        lines.append("tokenizer = AutoTokenizer.from_pretrained(")
        lines.append(f'    "tzervas/bwsk-{slug}", subfolder="finetune-conventional"')
        lines.append(")")
        lines.append("")
        lines.append("# Load from-scratch BWSK reversible variant")
        lines.append(f"model = {auto_cls}.from_pretrained(")
        lines.append(f'    "tzervas/bwsk-{slug}", subfolder="scratch-bwsk-reversible"')
        lines.append(")")
        lines.append("```")
    lines.append("")

    # --- Training configuration ---
    cfg = MODEL_TRAINING_CONFIG.get(slug, (8, 512, 5e-5))
    batch, seq, ft_lr = cfg
    scratch_lr = ft_lr * 5

    lines.append("## Training Configuration")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("|---------|-------|")
    lines.append("| **Optimizer** | AdamW |")
    lines.append(f"| **LR (fine-tune)** | {ft_lr:.0e} |")
    lines.append(f"| **LR (from-scratch)** | {scratch_lr:.0e} |")
    lines.append("| **LR Schedule** | Cosine with warmup |")
    lines.append("| **Max Grad Norm** | 1.0 |")
    lines.append("| **Mixed Precision** | AMP (float16) |")
    lines.append("| **Early Stopping** | Patience 3 |")
    lines.append(f"| **Batch Size** | {batch} |")
    if arch_type != "image_cls":
        lines.append(f"| **Sequence Length** | {seq} |")
    lines.append("")

    # --- Links ---
    lines.append("## Links")
    lines.append("")
    lines.append(f"- [GitHub Repository]({GITHUB_URL})")
    lines.append(f"- [Whitepaper]({GITHUB_URL}/blob/main/docs/WHITEPAPER.md)")
    lines.append(f"- [Full Training Report]({GITHUB_URL}/blob/main/docs/FULL_TRAINING_REPORT.md)")
    lines.append("")

    # --- Citation ---
    lines.append("## Citation")
    lines.append("")
    lines.append(BIBTEX)
    lines.append("")

    # --- License ---
    lines.append("## License")
    lines.append("")
    lines.append("MIT")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def upload_card(
    api: object,
    repo_id: str,
    content: str,
    dry_run: bool,
) -> bool:
    """Upload a README.md to a HuggingFace repo.

    Args:
        api: HfApi instance.
        repo_id: Full repo ID (e.g. "tzervas/bwsk-pythia-70m-finetune-conventional").
        content: README.md content string.
        dry_run: If True, skip actual upload.

    Returns:
        True if upload succeeded (or dry_run), False on error.
    """
    if dry_run:
        # Validate YAML frontmatter is present
        if not content.startswith("---\n"):
            print(f"  WARNING: {repo_id} missing YAML frontmatter")
            return False
        return True

    try:
        api.upload_file(  # type: ignore[attr-defined]
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message="Add BWSK model card",
        )
        return True
    except Exception as e:
        print(f"  ERROR uploading {repo_id}: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate and optionally upload HuggingFace model cards."""
    parser = argparse.ArgumentParser(description="Generate HuggingFace model cards for BWSK repos")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate all cards and validate, but don't upload",
    )
    parser.add_argument(
        "--preview",
        type=str,
        default=None,
        help="Print card for this slug to stdout (e.g. pythia-70m)",
    )
    parser.add_argument(
        "--slug",
        type=str,
        default=None,
        help="Only process repos for this model slug",
    )
    parser.add_argument(
        "--type",
        choices=[
            "consolidated",
            "model",
            "results",
            "all",
        ],
        default="consolidated",
        help=(
            "Card type: consolidated (new single-repo, default), "
            "model (legacy 96-repo), results (legacy 16-repo), "
            "all (legacy model+results)"
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between uploads (default: 1.0)",
    )
    args = parser.parse_args()

    # Load data
    all_results = load_aggregated_results()
    if not all_results:
        print("ERROR: No fulltrain_*_results.json files found")
        sys.exit(1)

    slugs = sorted(all_results.keys())
    if args.slug:
        if args.slug not in all_results:
            print(f"ERROR: Unknown slug '{args.slug}'")
            print(f"Available: {', '.join(slugs)}")
            sys.exit(1)
        slugs = [args.slug]

    print(f"Found {len(all_results)} models, processing {len(slugs)}")

    # Preview mode: print one card and exit
    if args.preview:
        if args.preview not in all_results:
            print(f"ERROR: Unknown slug '{args.preview}'")
            sys.exit(1)
        agg = all_results[args.preview]
        if args.type == "consolidated":
            print(generate_consolidated_card(agg))
        elif args.type == "results":
            print(generate_results_repo_card(agg))
        else:
            # Show first available model card (legacy)
            for exp in ALL_EXPERIMENTS:
                run = load_per_run(args.preview, exp, "conventional")
                if run:
                    siblings = {}
                    for m in ALL_MODES:
                        siblings[m] = load_per_run(args.preview, exp, m)
                    print(generate_model_repo_card(agg, run, exp, "conventional", siblings))
                    return
            print("ERROR: No per-run results found")
            sys.exit(1)
        return

    # Initialize HF API if not dry-run
    api = None
    if not args.dry_run:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
        except ImportError:
            print("ERROR: huggingface_hub not installed. Use --dry-run or install it.")
            sys.exit(1)

    generated = 0
    uploaded = 0
    errors = 0

    # Generate consolidated cards (16 — new layout)
    if args.type == "consolidated":
        print("\n--- Consolidated Model Cards ---")
        for slug in slugs:
            agg = all_results[slug]
            repo_id = f"{HF_NAMESPACE}/bwsk-{slug}"
            card = generate_consolidated_card(agg)
            generated += 1

            if args.dry_run:
                ok = upload_card(None, repo_id, card, True)
            else:
                print(f"  Uploading {repo_id}...")
                ok = upload_card(api, repo_id, card, False)
                if ok:
                    time.sleep(args.delay)

            if ok:
                uploaded += 1
            else:
                errors += 1

    # Generate model repo cards (96 — legacy)
    if args.type in ("model", "all"):
        print("\n--- Model Repo Cards ---")
        for slug in slugs:
            agg = all_results[slug]
            for exp in ALL_EXPERIMENTS:
                # Load all sibling runs for cross-mode comparison
                siblings: dict[str, dict | None] = {}
                for m in ALL_MODES:
                    siblings[m] = load_per_run(slug, exp, m)

                for mode in ALL_MODES:
                    run = siblings[mode]
                    if run is None:
                        print(f"  SKIP {slug}/{exp}/{mode}: no per-run file")
                        continue

                    repo_id = f"{HF_NAMESPACE}/bwsk-{slug}-{exp}-{mode}"
                    card = generate_model_repo_card(agg, run, exp, mode, siblings)
                    generated += 1

                    if args.dry_run:
                        ok = upload_card(None, repo_id, card, True)
                    else:
                        print(f"  Uploading {repo_id}...")
                        ok = upload_card(api, repo_id, card, False)
                        if ok:
                            time.sleep(args.delay)

                    if ok:
                        uploaded += 1
                    else:
                        errors += 1

    # Generate results repo cards (16)
    if args.type in ("results", "all"):
        print("\n--- Results Repo Cards ---")
        for slug in slugs:
            agg = all_results[slug]
            repo_id = f"{HF_NAMESPACE}/bwsk-{slug}-full-training-results"
            card = generate_results_repo_card(agg)
            generated += 1

            if args.dry_run:
                ok = upload_card(None, repo_id, card, True)
            else:
                print(f"  Uploading {repo_id}...")
                ok = upload_card(api, repo_id, card, False)
                if ok:
                    time.sleep(args.delay)

            if ok:
                uploaded += 1
            else:
                errors += 1

    # Summary
    action = "validated" if args.dry_run else "uploaded"
    print(f"\nDone: {generated} generated, {uploaded} {action}, {errors} errors")

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
