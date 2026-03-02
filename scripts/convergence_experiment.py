"""Convergence experiment: 1500-step training with 3 seeds and statistical analysis.

Tests hypothesis H3: BWSK-reversible training produces statistically equivalent
quality to conventional training.

Runs 4 representative models × 3 modes × 3 seeds = 36 total training runs:
  - Pythia-70M (small transformer)
  - GPT-2 Medium (medium transformer)
  - Mamba-130M (SSM)
  - ResNet-50 (CNN)

Statistical analysis:
  - Mean ± std across 3 seeds per (model, mode)
  - Paired t-test: conventional vs BWSK-reversible final loss
  - Cohen's d effect size
  - 95% confidence intervals
  - Convergence rate: steps to reach 90% of final loss reduction

Usage:
    uv run python scripts/convergence_experiment.py
    uv run python scripts/convergence_experiment.py --dry-run
    uv run python scripts/convergence_experiment.py --models pythia-70m
    uv run python scripts/convergence_experiment.py --steps 500
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bwsk.classify import OpClass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456]
DEFAULT_STEPS = 1500
LOG_EVERY = 100  # Log detailed metrics every N steps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path(__file__).resolve().parent
REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "CONVERGENCE_REPORT.md"

CUSTOM_RULES: dict[str, OpClass] = {
    "Conv1D": OpClass.S,
    "NewGELUActivation": OpClass.K,
    "GELUActivation": OpClass.K,
    "FastGELUActivation": OpClass.K,
    "T5LayerNorm": OpClass.S,
    "OPTLearnedPositionalEmbedding": OpClass.S,
    "RotaryEmbedding": OpClass.S,
    "GPTNeoXRotaryEmbedding": OpClass.S,
    "MambaRMSNorm": OpClass.S,
    "MambaMixer": OpClass.GRAY,
    "MambaCache": OpClass.S,
}

MODES = ["conventional", "bwsk_analyzed", "bwsk_reversible"]


@dataclass
class ConvergenceModelConfig:
    """Configuration for one convergence experiment model."""

    name: str
    slug: str
    source: str  # "huggingface" or "torchvision"
    arch_family: str
    arch_type: str  # "causal_lm", "ssm_lm", "image_cls"
    hf_id: str
    params_m: int
    batch_size: int
    seq_len: int
    dataset: str
    finetune_lr: float


CONVERGENCE_MODELS: list[ConvergenceModelConfig] = [
    ConvergenceModelConfig(
        name="Pythia-70M",
        slug="pythia-70m",
        source="huggingface",
        arch_family="transformer",
        arch_type="causal_lm",
        hf_id="EleutherAI/pythia-70m",
        params_m=70,
        batch_size=8,
        seq_len=512,
        dataset="wikitext",
        finetune_lr=5e-5,
    ),
    ConvergenceModelConfig(
        name="GPT-2 Medium",
        slug="gpt2-medium",
        source="huggingface",
        arch_family="transformer",
        arch_type="causal_lm",
        hf_id="openai-community/gpt2-medium",
        params_m=345,
        batch_size=2,
        seq_len=512,
        dataset="wikitext",
        finetune_lr=5e-5,
    ),
    ConvergenceModelConfig(
        name="Mamba-130M",
        slug="mamba-130m",
        source="huggingface",
        arch_family="ssm",
        arch_type="ssm_lm",
        hf_id="state-spaces/mamba-130m-hf",
        params_m=130,
        batch_size=2,
        seq_len=256,
        dataset="wikitext",
        finetune_lr=3e-5,
    ),
    ConvergenceModelConfig(
        name="ResNet-50",
        slug="resnet50",
        source="torchvision",
        arch_family="cnn",
        arch_type="image_cls",
        hf_id="resnet50",
        params_m=25,
        batch_size=32,
        seq_len=224,
        dataset="cifar10",
        finetune_lr=1e-3,
    ),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StepMetrics:
    """Metrics captured at a specific training step."""

    step: int
    loss: float
    grad_norm: float
    lr: float
    wall_time_s: float
    memory_mb: float


@dataclass
class RunResult:
    """Result of a single training run (model × mode × seed)."""

    model_name: str
    slug: str
    mode: str
    seed: int
    num_steps: int
    loss_curve: list[float]
    step_metrics: list[StepMetrics]
    final_loss: float
    wall_time_s: float
    peak_memory_mb: float
    nan_count: int
    convergence_step_90: int  # Step at which 90% of loss reduction achieved


@dataclass
class StatisticalResult:
    """Statistical analysis comparing two modes for one model."""

    model_name: str
    slug: str
    mode_a: str
    mode_b: str
    mean_loss_a: float
    mean_loss_b: float
    std_loss_a: float
    std_loss_b: float
    t_statistic: float
    p_value: float
    cohens_d: float
    ci_95_lower: float
    ci_95_upper: float
    significant: bool  # p < 0.05
    trending: bool  # 0.01 < p < 0.10


@dataclass
class ModelConvergenceResults:
    """All convergence results for one model."""

    model_name: str
    slug: str
    runs: list[RunResult]
    stats: list[StatisticalResult]
    mean_loss_curves: dict[str, list[float]]  # mode -> mean loss per step
    std_loss_curves: dict[str, list[float]]  # mode -> std loss per step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from bench_utils import peak_memory_mb, reset_memory


def compute_grad_norm(model: nn.Module) -> float:
    """Compute total gradient L2 norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm)


def load_wikitext(split: str = "train") -> str:
    """Load WikiText-2 text for a given split."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(dataset["text"])


def load_cifar10_loader(split: str = "train", batch_size: int = 32):
    """Load CIFAR-10 DataLoader."""
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    is_train = split == "train"
    dataset = datasets.CIFAR10(
        root="/tmp/cifar10",
        train=is_train,
        download=True,
        transform=transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=2)


def prepare_text_batches(
    tokenizer, config: ConvergenceModelConfig, num_steps: int
) -> list[torch.Tensor]:
    """Prepare training batches from WikiText-2."""
    train_text = load_wikitext("train")

    max_tokens = config.seq_len * config.batch_size * num_steps * 2
    # Temporarily override model_max_length to avoid premature truncation.
    saved_max_len = getattr(tokenizer, "model_max_length", max_tokens)
    tokenizer.model_max_length = max(max_tokens, saved_max_len)
    encodings = tokenizer(
        train_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=False,
    )
    tokenizer.model_max_length = saved_max_len
    all_ids = encodings.input_ids[0]

    batches = []
    tokens_per_batch = config.batch_size * config.seq_len

    for i in range(0, len(all_ids) - tokens_per_batch, tokens_per_batch):
        batch = (
            all_ids[i : i + tokens_per_batch]
            .contiguous()
            .reshape(config.batch_size, config.seq_len)
        )
        batches.append(batch)
        if len(batches) >= num_steps:
            break

    return batches


def load_model(config: ConvergenceModelConfig) -> tuple:
    """Load model and tokenizer/loader."""
    if config.arch_type == "image_cls":
        import torchvision.models as models

        model_fn = getattr(models, config.hf_id)
        model = model_fn(weights=None)
        if hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 10)
        return model.to(DEVICE), None
    elif config.arch_type == "ssm_lm":
        # Patch mamba_ssm 2.x for transformers 5.x compatibility
        try:
            import mamba_ssm

            if not hasattr(mamba_ssm, "selective_state_update"):
                from mamba_ssm.ops.triton.selective_state_update import (
                    selective_state_update,
                )

                mamba_ssm.selective_state_update = selective_state_update
        except ImportError:
            pass
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.hf_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(config.hf_id)
        return model.to(DEVICE), tokenizer
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.hf_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(config.hf_id)
        return model.to(DEVICE), tokenizer


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_run(
    config: ConvergenceModelConfig,
    mode: str,
    seed: int,
    num_steps: int,
    text_batches: list[torch.Tensor] | None = None,
    cifar_loader=None,
) -> RunResult:
    """Execute one training run with detailed per-step logging.

    This is the core training loop that captures loss, gradient norm,
    learning rate, wall time, and memory at regular intervals.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model, _ = load_model(config)
    # Enable gradient checkpointing for reversible mode, and always for SSM
    # models (Mamba's slow_forward path creates large intermediate tensors)
    use_checkpointing = mode == "bwsk_reversible" or config.arch_family == "ssm"

    if use_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    use_amp = DEVICE.type == "cuda" and config.params_m >= 300

    # Ensure float32 when not using AMP (some HF models load in bf16)
    if not use_amp:
        model = model.float()

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.finetune_lr)
    warmup_steps = max(1, num_steps // 10)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    reset_memory()
    loss_curve: list[float] = []
    step_metrics: list[StepMetrics] = []
    nan_count = 0
    run_start = time.perf_counter()

    desc = f"  {config.slug}/{mode}/seed={seed}"

    if config.arch_type == "image_cls":
        step = 0
        for images, labels in cifar_loader:
            if step >= num_steps:
                break
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                optimizer.zero_grad()
                step += 1
                continue
            loss.backward()
            grad_norm = compute_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_curve.append(loss.item())

            if step % LOG_EVERY == 0 or step == num_steps - 1:
                step_metrics.append(
                    StepMetrics(
                        step=step,
                        loss=loss.item(),
                        grad_norm=grad_norm,
                        lr=scheduler.get_last_lr()[0],
                        wall_time_s=time.perf_counter() - run_start,
                        memory_mb=peak_memory_mb(),
                    )
                )
            step += 1
    else:
        pad_token_id = 0
        for step, batch in enumerate(
            tqdm(
                text_batches[:num_steps],
                desc=desc,
                leave=False,
            )
        ):
            batch = batch.to(DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                if config.arch_family == "ssm":
                    outputs = model(batch, labels=batch)
                else:
                    attention_mask = batch.ne(pad_token_id).long()
                    outputs = model(
                        batch,
                        attention_mask=attention_mask,
                        labels=batch,
                    )
                loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                optimizer.zero_grad()
                continue
            loss.backward()
            grad_norm = compute_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_curve.append(loss.item())

            if step % LOG_EVERY == 0 or step == num_steps - 1:
                step_metrics.append(
                    StepMetrics(
                        step=step,
                        loss=loss.item(),
                        grad_norm=grad_norm,
                        lr=scheduler.get_last_lr()[0],
                        wall_time_s=time.perf_counter() - run_start,
                        memory_mb=peak_memory_mb(),
                    )
                )

    wall_time = time.perf_counter() - run_start
    mem = peak_memory_mb()

    # Compute convergence step (90% of total loss reduction)
    convergence_step = num_steps
    if len(loss_curve) >= 2:
        initial = loss_curve[0]
        final = loss_curve[-1]
        target = initial - 0.9 * (initial - final)
        for i, loss_val in enumerate(loss_curve):
            if loss_val <= target:
                convergence_step = i
                break

    del model
    reset_memory()

    return RunResult(
        model_name=config.name,
        slug=config.slug,
        mode=mode,
        seed=seed,
        num_steps=num_steps,
        loss_curve=loss_curve,
        step_metrics=[asdict(m) for m in step_metrics],
        final_loss=loss_curve[-1] if loss_curve else float("nan"),
        wall_time_s=wall_time,
        peak_memory_mb=mem,
        nan_count=nan_count,
        convergence_step_90=convergence_step,
    )


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------


def paired_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Compute paired t-test statistic and p-value.

    Uses scipy.stats.ttest_rel for paired samples.
    """
    from scipy import stats

    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return 0.0, 1.0
    t_stat, p_val = stats.ttest_rel(a, b)
    return float(t_stat), float(p_val)


def cohens_d(a: list[float], b: list[float]) -> float:
    """Compute Cohen's d effect size for paired samples."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    import numpy as np

    diffs = np.array(a) - np.array(b)
    return float(np.mean(diffs) / np.std(diffs, ddof=1)) if np.std(diffs, ddof=1) > 0 else 0.0


def confidence_interval_95(values: list[float]) -> tuple[float, float]:
    """Compute 95% CI using t-distribution."""
    import numpy as np
    from scipy import stats

    if len(values) < 2:
        return (float("nan"), float("nan"))
    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)
    ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    return (float(ci[0]), float(ci[1]))


def compute_statistics(
    runs: list[RunResult], model_name: str, slug: str
) -> list[StatisticalResult]:
    """Compute statistical comparisons between modes for one model."""
    results = []

    # Group runs by mode
    mode_losses: dict[str, list[float]] = {}
    for run in runs:
        if run.mode not in mode_losses:
            mode_losses[run.mode] = []
        mode_losses[run.mode].append(run.final_loss)

    # Compare conventional vs bwsk_reversible
    for mode_b in ["bwsk_analyzed", "bwsk_reversible"]:
        mode_a = "conventional"
        if mode_a not in mode_losses or mode_b not in mode_losses:
            continue

        a_vals = mode_losses[mode_a]
        b_vals = mode_losses[mode_b]

        mean_a = sum(a_vals) / len(a_vals)
        mean_b = sum(b_vals) / len(b_vals)
        std_a = (
            (sum((x - mean_a) ** 2 for x in a_vals) / (len(a_vals) - 1)) ** 0.5
            if len(a_vals) > 1
            else 0.0
        )
        std_b = (
            (sum((x - mean_b) ** 2 for x in b_vals) / (len(b_vals) - 1)) ** 0.5
            if len(b_vals) > 1
            else 0.0
        )

        t_stat, p_val = paired_t_test(a_vals, b_vals)
        d = cohens_d(a_vals, b_vals)

        # 95% CI on the difference
        diffs = [a - b for a, b in zip(a_vals, b_vals, strict=True)]
        ci_low, ci_high = confidence_interval_95(diffs)

        results.append(
            StatisticalResult(
                model_name=model_name,
                slug=slug,
                mode_a=mode_a,
                mode_b=mode_b,
                mean_loss_a=mean_a,
                mean_loss_b=mean_b,
                std_loss_a=std_a,
                std_loss_b=std_b,
                t_statistic=t_stat,
                p_value=p_val,
                cohens_d=d,
                ci_95_lower=ci_low,
                ci_95_upper=ci_high,
                significant=p_val < 0.05,
                trending=0.01 < p_val < 0.10,
            )
        )

    return results


def compute_mean_std_curves(
    runs: list[RunResult],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Compute mean and std loss curves per mode across seeds."""
    import numpy as np

    mode_curves: dict[str, list[list[float]]] = {}
    for run in runs:
        if run.mode not in mode_curves:
            mode_curves[run.mode] = []
        mode_curves[run.mode].append(run.loss_curve)

    mean_curves: dict[str, list[float]] = {}
    std_curves: dict[str, list[float]] = {}

    for mode, curves in mode_curves.items():
        min_len = min(len(c) for c in curves)
        trimmed = [c[:min_len] for c in curves]
        arr = np.array(trimmed)
        mean_curves[mode] = arr.mean(axis=0).tolist()
        std_curves[mode] = arr.std(axis=0).tolist()

    return mean_curves, std_curves


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    all_model_results: list[ModelConvergenceResults],
    num_steps: int,
) -> str:
    """Generate the convergence report as markdown."""
    lines: list[str] = []
    lines.append("# Convergence Experiment Report")
    lines.append("")
    lines.append(f"**Steps**: {num_steps} | **Seeds**: {SEEDS} | **Modes**: {', '.join(MODES)}")
    lines.append(f"**Log interval**: every {LOG_EVERY} steps")
    lines.append("")

    for mr in all_model_results:
        lines.append(f"## {mr.model_name}")
        lines.append("")

        # Run summary table
        lines.append("### Training Runs")
        lines.append("")
        lines.append(
            "| Mode | Seed | Final Loss | Wall Time (s) | Peak Mem (MB) | 90% Conv Step | NaN |"
        )
        lines.append(
            "|------|------|-----------|---------------|---------------|---------------|-----|"
        )
        for run in mr.runs:
            lines.append(
                f"| {run.mode} | {run.seed} | "
                f"{run.final_loss:.4f} | {run.wall_time_s:.1f} | "
                f"{run.peak_memory_mb:.0f} | "
                f"{run.convergence_step_90} | {run.nan_count} |"
            )
        lines.append("")

        # Statistical analysis
        if mr.stats:
            lines.append("### Statistical Analysis")
            lines.append("")
            lines.append(
                "| Comparison | Mean A ± Std | Mean B ± Std | t-stat "
                "| p-value | Cohen's d | 95% CI (diff) | Significant |"
            )
            lines.append(
                "|------------|-------------|-------------|--------|"
                "---------|-----------|---------------|-------------|"
            )
            for s in mr.stats:
                sig = "YES" if s.significant else ("trending" if s.trending else "no")
                lines.append(
                    f"| {s.mode_a} vs {s.mode_b} | "
                    f"{s.mean_loss_a:.4f}±{s.std_loss_a:.4f} | "
                    f"{s.mean_loss_b:.4f}±{s.std_loss_b:.4f} | "
                    f"{s.t_statistic:.3f} | {s.p_value:.4f} | "
                    f"{s.cohens_d:.3f} | "
                    f"[{s.ci_95_lower:.4f}, {s.ci_95_upper:.4f}] | "
                    f"{sig} |"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

    # Hypothesis summary
    lines.append("## Hypothesis Testing Summary")
    lines.append("")
    lines.append("### H3: BWSK-reversible produces statistically equivalent quality")
    lines.append("")
    lines.append("| Model | p-value | Cohen's d | Verdict |")
    lines.append("|-------|---------|-----------|---------|")
    for mr in all_model_results:
        for s in mr.stats:
            if s.mode_b == "bwsk_reversible":
                if s.p_value > 0.05:
                    verdict = "Equivalent (fail to reject H0)"
                elif s.p_value > 0.01:
                    verdict = "Trending difference"
                else:
                    verdict = "Significant difference"
                lines.append(
                    f"| {mr.model_name} | {s.p_value:.4f} | {s.cohens_d:.3f} | {verdict} |"
                )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convergence experiment with statistical analysis")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model slugs (default: all 4)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of training steps (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds (default: 42,123,456)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the convergence experiment."""
    args = parse_args()

    seeds = SEEDS
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]

    models = CONVERGENCE_MODELS
    if args.models:
        slugs = {s.strip() for s in args.models.split(",")}
        models = [m for m in CONVERGENCE_MODELS if m.slug in slugs]
        if not models:
            print(f"ERROR: No models found for slugs: {slugs}")
            sys.exit(1)

    num_steps = args.steps
    total_runs = len(models) * len(MODES) * len(seeds)

    print("=" * 70)
    print("CONVERGENCE EXPERIMENT")
    print(f"Models: {len(models)}")
    print(f"Modes: {MODES}")
    print(f"Seeds: {seeds}")
    print(f"Steps: {num_steps}")
    print(f"Total runs: {total_runs}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Configuration validated. Exiting.")
        return

    all_model_results: list[ModelConvergenceResults] = []

    for mi, config in enumerate(models):
        print(f"\n{'=' * 70}\nMODEL {mi + 1}/{len(models)}: {config.name}\n{'=' * 70}")

        # Prepare data once per model
        text_batches = None
        cifar_loader = None

        if config.arch_type == "image_cls":
            cifar_loader = load_cifar10_loader("train", batch_size=config.batch_size)
        else:
            _, tokenizer = load_model(config)
            text_batches = prepare_text_batches(tokenizer, config, num_steps)
            del _
            reset_memory()
            print(f"  Prepared {len(text_batches)} batches")

        runs: list[RunResult] = []

        for mode in MODES:
            for seed in seeds:
                print(f"\n  [{config.slug}] {mode} / seed={seed} ({num_steps} steps)...")
                run = train_one_run(
                    config,
                    mode,
                    seed,
                    num_steps,
                    text_batches=text_batches,
                    cifar_loader=cifar_loader,
                )
                print(
                    f"    Final loss: {run.final_loss:.4f}, "
                    f"Time: {run.wall_time_s:.1f}s, "
                    f"Mem: {run.peak_memory_mb:.0f}MB, "
                    f"90% conv: step {run.convergence_step_90}"
                )
                runs.append(run)

        # Statistical analysis
        stats = compute_statistics(runs, config.name, config.slug)
        mean_curves, std_curves = compute_mean_std_curves(runs)

        model_result = ModelConvergenceResults(
            model_name=config.name,
            slug=config.slug,
            runs=runs,
            stats=stats,
            mean_loss_curves=mean_curves,
            std_loss_curves=std_curves,
        )
        all_model_results.append(model_result)

        # Save per-model results
        json_path = RESULTS_DIR / f"convergence_{config.slug}_results.json"
        with open(json_path, "w") as f:
            json.dump(asdict(model_result), f, indent=2, default=str)
        print(f"  Saved: {json_path}")

    # Save combined results
    combined_path = RESULTS_DIR / "convergence_results.json"
    with open(combined_path, "w") as f:
        json.dump(
            [asdict(r) for r in all_model_results],
            f,
            indent=2,
            default=str,
        )
    print(f"\nCombined results saved to: {combined_path}")

    # Generate report
    report = generate_report(all_model_results, num_steps)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Report saved to: {REPORT_PATH}")

    # Final summary
    print("\n" + "=" * 70)
    print("CONVERGENCE EXPERIMENT COMPLETE")
    for mr in all_model_results:
        print(f"\n  {mr.model_name}:")
        for s in mr.stats:
            if s.mode_b == "bwsk_reversible":
                sig = "EQUIVALENT" if s.p_value > 0.05 else "DIFFERENT"
                print(
                    f"    conv vs reversible: "
                    f"p={s.p_value:.4f}, "
                    f"d={s.cohens_d:.3f} → {sig}"
                )
    print("=" * 70)


if __name__ == "__main__":
    main()
