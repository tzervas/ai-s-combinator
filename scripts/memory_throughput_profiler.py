"""Memory and throughput profiler: detailed breakdown per model per mode.

Measures for each model across 3 training modes:
  - Peak GPU memory (torch.cuda.max_memory_allocated)
  - Memory timeline (sampled each step)
  - Throughput: tokens/sec or images/sec
  - Forward vs backward time (timed separately)
  - Memory breakdown: params + grads + optimizer + activations

Usage:
    uv run python scripts/memory_throughput_profiler.py
    uv run python scripts/memory_throughput_profiler.py --dry-run
    uv run python scripts/memory_throughput_profiler.py --models pythia-70m
    uv run python scripts/memory_throughput_profiler.py --steps 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bwsk.classify import OpClass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_STEPS = 50  # Fewer steps needed for memory profiling
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path(__file__).resolve().parent
REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "MEMORY_THROUGHPUT_REPORT.md"

CUSTOM_RULES: dict[str, OpClass] = {
    "Conv1D": OpClass.S,
    "NewGELUActivation": OpClass.K,
    "GELUActivation": OpClass.K,
    "FastGELUActivation": OpClass.K,
    "T5LayerNorm": OpClass.S,
    "OPTLearnedPositionalEmbedding": OpClass.S,
    "RotaryEmbedding": OpClass.S,
    "GPTNeoXRotaryEmbedding": OpClass.S,
    "PhiRotaryEmbedding": OpClass.S,
    "MambaRMSNorm": OpClass.S,
    "MambaMixer": OpClass.GRAY,
    "MambaCache": OpClass.S,
}


@dataclass
class ProfileModelConfig:
    """Configuration for one profiler model."""

    name: str
    slug: str
    source: str
    arch_family: str
    arch_type: str
    hf_id: str
    params_m: int
    batch_size: int
    seq_len: int
    dataset: str
    finetune_lr: float
    skip_training: bool = False


PROFILER_MODELS: list[ProfileModelConfig] = [
    ProfileModelConfig(
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
    ProfileModelConfig(
        name="Pythia-160M",
        slug="pythia-160m",
        source="huggingface",
        arch_family="transformer",
        arch_type="causal_lm",
        hf_id="EleutherAI/pythia-160m",
        params_m=160,
        batch_size=4,
        seq_len=512,
        dataset="wikitext",
        finetune_lr=3e-5,
    ),
    ProfileModelConfig(
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
    ProfileModelConfig(
        name="Pythia-410M",
        slug="pythia-410m",
        source="huggingface",
        arch_family="transformer",
        arch_type="causal_lm",
        hf_id="EleutherAI/pythia-410m",
        params_m=405,
        batch_size=2,
        seq_len=512,
        dataset="wikitext",
        finetune_lr=2e-5,
    ),
    ProfileModelConfig(
        name="Pythia-1B",
        slug="pythia-1b",
        source="huggingface",
        arch_family="transformer",
        arch_type="causal_lm",
        hf_id="EleutherAI/pythia-1b",
        params_m=1010,
        batch_size=1,
        seq_len=512,
        dataset="wikitext",
        finetune_lr=1e-5,
    ),
    ProfileModelConfig(
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
    ProfileModelConfig(
        name="Mamba-130M",
        slug="mamba-130m",
        source="huggingface",
        arch_family="ssm",
        arch_type="ssm_lm",
        hf_id="state-spaces/mamba-130m-hf",
        params_m=130,
        batch_size=4,
        seq_len=512,
        dataset="wikitext",
        finetune_lr=3e-5,
    ),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MemoryBreakdown:
    """Detailed memory breakdown in MB."""

    params_mb: float
    grads_mb: float
    optimizer_mb: float
    activations_mb: float  # Estimated: peak - params - grads - optimizer
    total_peak_mb: float


@dataclass
class TimingBreakdown:
    """Forward and backward pass timing."""

    forward_time_s: float
    backward_time_s: float
    optimizer_step_s: float
    total_step_s: float


@dataclass
class ModeProfile:
    """Profile results for one model × one mode."""

    mode: str
    memory: MemoryBreakdown
    timing: TimingBreakdown
    memory_timeline: list[float]  # Peak memory at each step
    throughput: float  # tokens/sec or images/sec
    throughput_unit: str  # "tokens/sec" or "images/sec"
    num_steps: int


@dataclass
class ModelProfile:
    """All profile results for one model."""

    model_name: str
    slug: str
    params_m: int
    arch_family: str
    device: str
    timestamp: str
    profiles: list[ModeProfile]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from bench_utils import peak_memory_mb, reset_memory


def current_memory_mb() -> float:
    """Get current GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def param_memory_mb(model: nn.Module) -> float:
    """Compute total parameter memory in MB."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 * 1024)


def grad_memory_mb(model: nn.Module) -> float:
    """Compute total gradient memory in MB."""
    total_bytes = sum(
        p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None
    )
    return total_bytes / (1024 * 1024)


def optimizer_memory_mb(optimizer: torch.optim.Optimizer) -> float:
    """Estimate optimizer state memory in MB.

    AdamW stores 2 state tensors per parameter (m and v), each the same
    size as the parameter.
    """
    total_bytes = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in optimizer.state:
                for v in optimizer.state[p].values():
                    if isinstance(v, torch.Tensor):
                        total_bytes += v.numel() * v.element_size()
    return total_bytes / (1024 * 1024)


def load_wikitext_train() -> str:
    """Load WikiText-2 training text."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return "\n\n".join(dataset["text"])


def prepare_text_batches(
    tokenizer, config: ProfileModelConfig, num_steps: int
) -> list[torch.Tensor]:
    """Prepare training batches from WikiText-2."""
    train_text = load_wikitext_train()
    max_tokens = config.seq_len * config.batch_size * num_steps * 2
    encodings = tokenizer(
        train_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=False,
    )
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


def load_cifar10_loader(batch_size: int = 32):
    """Load CIFAR-10 training DataLoader."""
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
    dataset = datasets.CIFAR10(
        root="/tmp/cifar10",
        train=True,
        download=True,
        transform=transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def load_model(config: ProfileModelConfig) -> tuple:
    """Load model and tokenizer."""
    if config.arch_type == "image_cls":
        import torchvision.models as models

        model_fn = getattr(models, config.hf_id)
        model = model_fn(weights=None)
        if hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, 10)
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
# Profiling loop
# ---------------------------------------------------------------------------


def profile_one_mode(
    config: ProfileModelConfig,
    mode: str,
    num_steps: int,
    text_batches: list[torch.Tensor] | None = None,
    cifar_loader=None,
) -> ModeProfile:
    """Profile one model × one mode with detailed timing and memory breakdown.

    Separates forward pass, backward pass, and optimizer step timing.
    Tracks memory at each step to build a memory timeline.
    """
    model, _ = load_model(config)
    use_checkpointing = mode == "bwsk_reversible"

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

    reset_memory()

    memory_timeline: list[float] = []
    forward_times: list[float] = []
    backward_times: list[float] = []
    optimizer_times: list[float] = []
    total_items = 0  # tokens or images processed

    if config.arch_type == "image_cls":
        step = 0
        for images, labels in cifar_loader:
            if step >= num_steps:
                break
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            forward_times.append(t1 - t0)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                step += 1
                continue

            # Backward
            t2 = time.perf_counter()
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            backward_times.append(t3 - t2)

            # Optimizer
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            t4 = time.perf_counter()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t5 = time.perf_counter()
            optimizer_times.append(t5 - t4)
            optimizer.zero_grad()

            memory_timeline.append(peak_memory_mb())
            total_items += images.size(0)
            step += 1
    else:
        pad_token_id = 0
        for _step, batch in enumerate(text_batches[:num_steps]):
            batch = batch.to(DEVICE)

            # Forward
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
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
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            forward_times.append(t1 - t0)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            # Backward
            t2 = time.perf_counter()
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            backward_times.append(t3 - t2)

            # Optimizer
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            t4 = time.perf_counter()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t5 = time.perf_counter()
            optimizer_times.append(t5 - t4)
            optimizer.zero_grad()

            memory_timeline.append(peak_memory_mb())
            total_items += batch.numel()  # Total tokens

    # Compute memory breakdown
    params_mem = param_memory_mb(model)
    grads_mem = grad_memory_mb(model)
    opt_mem = optimizer_memory_mb(optimizer)
    total_peak = peak_memory_mb()
    activations_mem = max(0.0, total_peak - params_mem - grads_mem - opt_mem)

    # Compute timing averages
    avg_forward = sum(forward_times) / len(forward_times) if forward_times else 0.0
    avg_backward = sum(backward_times) / len(backward_times) if backward_times else 0.0
    avg_opt = sum(optimizer_times) / len(optimizer_times) if optimizer_times else 0.0
    avg_total = avg_forward + avg_backward + avg_opt

    # Throughput
    total_time = sum(forward_times) + sum(backward_times) + sum(optimizer_times)
    throughput = total_items / total_time if total_time > 0 else 0.0
    throughput_unit = "images/sec" if config.arch_type == "image_cls" else "tokens/sec"

    del model, optimizer
    reset_memory()

    return ModeProfile(
        mode=mode,
        memory=MemoryBreakdown(
            params_mb=params_mem,
            grads_mb=grads_mem,
            optimizer_mb=opt_mem,
            activations_mb=activations_mem,
            total_peak_mb=total_peak,
        ),
        timing=TimingBreakdown(
            forward_time_s=avg_forward,
            backward_time_s=avg_backward,
            optimizer_step_s=avg_opt,
            total_step_s=avg_total,
        ),
        memory_timeline=memory_timeline,
        throughput=throughput,
        throughput_unit=throughput_unit,
        num_steps=num_steps,
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(all_profiles: list[ModelProfile]) -> str:
    """Generate markdown report from profiling results."""
    lines: list[str] = []
    lines.append("# Memory & Throughput Profiling Report")
    lines.append("")

    if all_profiles:
        lines.append(f"**Device**: {all_profiles[0].device}")
        lines.append(f"**Date**: {all_profiles[0].timestamp}")
    lines.append("")

    for mp in all_profiles:
        lines.append(f"## {mp.model_name} ({mp.params_m}M params)")
        lines.append("")

        # Memory comparison
        lines.append("### Memory Breakdown (MB)")
        lines.append("")
        lines.append("| Component | " + " | ".join(p.mode for p in mp.profiles) + " |")
        lines.append("|-----------|" + "|".join("---" for _ in mp.profiles) + "|")
        for component in [
            "params_mb",
            "grads_mb",
            "optimizer_mb",
            "activations_mb",
            "total_peak_mb",
        ]:
            label = component.replace("_mb", "").replace("_", " ").title()
            vals = " | ".join(f"{getattr(p.memory, component):.1f}" for p in mp.profiles)
            lines.append(f"| {label} | {vals} |")
        lines.append("")

        # Timing comparison
        lines.append("### Timing Breakdown (seconds per step)")
        lines.append("")
        lines.append("| Phase | " + " | ".join(p.mode for p in mp.profiles) + " |")
        lines.append("|-------|" + "|".join("---" for _ in mp.profiles) + "|")
        for phase in [
            "forward_time_s",
            "backward_time_s",
            "optimizer_step_s",
            "total_step_s",
        ]:
            label = phase.replace("_time_s", "").replace("_s", "").replace("_", " ").title()
            vals = " | ".join(f"{getattr(p.timing, phase):.4f}" for p in mp.profiles)
            lines.append(f"| {label} | {vals} |")
        lines.append("")

        # Throughput
        lines.append("### Throughput")
        lines.append("")
        for p in mp.profiles:
            lines.append(f"- **{p.mode}**: {p.throughput:.0f} {p.throughput_unit}")
        lines.append("")

        # Memory savings
        conv = next((p for p in mp.profiles if p.mode == "conventional"), None)
        rev = next((p for p in mp.profiles if p.mode == "bwsk_reversible"), None)
        if conv and rev and conv.memory.total_peak_mb > 0:
            savings = conv.memory.total_peak_mb - rev.memory.total_peak_mb
            pct = 100 * savings / conv.memory.total_peak_mb
            lines.append(
                f"**Memory savings (reversible vs conventional)**: {savings:.1f} MB ({pct:.1f}%)"
            )
            lines.append("")

        lines.append("---")
        lines.append("")

    # Cross-model scaling tables
    lines.append("## Scaling Analysis")
    lines.append("")
    lines.append("### Peak Memory vs Model Size")
    lines.append("")
    lines.append("| Model | Params (M) | Conv (MB) | Rev (MB) | Savings (%) |")
    lines.append("|-------|-----------|-----------|----------|-------------|")
    for mp in all_profiles:
        conv = next((p for p in mp.profiles if p.mode == "conventional"), None)
        rev = next((p for p in mp.profiles if p.mode == "bwsk_reversible"), None)
        if conv and rev:
            savings_pct = (
                100
                * (conv.memory.total_peak_mb - rev.memory.total_peak_mb)
                / conv.memory.total_peak_mb
                if conv.memory.total_peak_mb > 0
                else 0.0
            )
            lines.append(
                f"| {mp.model_name} | {mp.params_m} | "
                f"{conv.memory.total_peak_mb:.0f} | "
                f"{rev.memory.total_peak_mb:.0f} | "
                f"{savings_pct:.1f}% |"
            )
    lines.append("")

    lines.append("### Throughput vs Model Size")
    lines.append("")
    lines.append("| Model | Params (M) | Conv (items/s) | Rev (items/s) | Overhead (%) |")
    lines.append("|-------|-----------|----------------|---------------|--------------|")
    for mp in all_profiles:
        conv = next((p for p in mp.profiles if p.mode == "conventional"), None)
        rev = next((p for p in mp.profiles if p.mode == "bwsk_reversible"), None)
        if conv and rev:
            overhead = (
                100 * (conv.throughput - rev.throughput) / conv.throughput
                if conv.throughput > 0
                else 0.0
            )
            lines.append(
                f"| {mp.model_name} | {mp.params_m} | "
                f"{conv.throughput:.0f} | {rev.throughput:.0f} | "
                f"{overhead:.1f}% |"
            )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Memory and throughput profiler")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model slugs (default: all)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of profiling steps (default: {DEFAULT_STEPS})",
    )
    return parser.parse_args()


def main() -> None:
    """Run the memory/throughput profiler."""
    args = parse_args()

    models = PROFILER_MODELS
    if args.models:
        slugs = {s.strip() for s in args.models.split(",")}
        models = [m for m in PROFILER_MODELS if m.slug in slugs]
        if not models:
            print(f"ERROR: No models for slugs: {slugs}")
            sys.exit(1)

    num_steps = args.steps

    print("=" * 70)
    print("MEMORY & THROUGHPUT PROFILER")
    print(f"Models: {len(models)}")
    print(f"Steps per mode: {num_steps}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_mem:.1f} GB")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Configuration validated. Exiting.")
        return

    all_profiles: list[ModelProfile] = []

    for mi, config in enumerate(models):
        print(
            f"\n{'=' * 70}\n"
            f"PROFILING {mi + 1}/{len(models)}: {config.name} "
            f"({config.params_m}M)\n"
            f"{'=' * 70}"
        )

        # Prepare data once
        text_batches = None
        cifar_loader = None

        if config.arch_type == "image_cls":
            cifar_loader = load_cifar10_loader(config.batch_size)
        else:
            _, tokenizer = load_model(config)
            text_batches = prepare_text_batches(tokenizer, config, num_steps)
            del _
            reset_memory()

        profiles: list[ModeProfile] = []

        for mode in ["conventional", "bwsk_analyzed", "bwsk_reversible"]:
            print(f"\n  Mode: {mode}...")
            profile = profile_one_mode(
                config,
                mode,
                num_steps,
                text_batches=text_batches,
                cifar_loader=cifar_loader,
            )
            mem = profile.memory
            print(
                f"    Peak: {mem.total_peak_mb:.0f} MB "
                f"(params={mem.params_mb:.0f}, "
                f"grads={mem.grads_mb:.0f}, "
                f"opt={mem.optimizer_mb:.0f}, "
                f"act={mem.activations_mb:.0f})"
            )
            print(
                f"    Timing: fwd={profile.timing.forward_time_s:.4f}s, "
                f"bwd={profile.timing.backward_time_s:.4f}s, "
                f"opt={profile.timing.optimizer_step_s:.4f}s"
            )
            print(f"    Throughput: {profile.throughput:.0f} {profile.throughput_unit}")
            profiles.append(profile)

        model_profile = ModelProfile(
            model_name=config.name,
            slug=config.slug,
            params_m=config.params_m,
            arch_family=config.arch_family,
            device=(torch.cuda.get_device_name() if torch.cuda.is_available() else str(DEVICE)),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            profiles=profiles,
        )
        all_profiles.append(model_profile)

        # Save per-model
        json_path = RESULTS_DIR / f"profile_{config.slug}_results.json"
        with open(json_path, "w") as f:
            json.dump(asdict(model_profile), f, indent=2, default=str)
        print(f"  Saved: {json_path}")

    # Save combined
    combined_path = RESULTS_DIR / "memory_throughput_results.json"
    with open(combined_path, "w") as f:
        json.dump([asdict(p) for p in all_profiles], f, indent=2, default=str)
    print(f"\nCombined results: {combined_path}")

    # Generate report
    report = generate_report(all_profiles)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Report: {REPORT_PATH}")

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    for mp in all_profiles:
        conv = next((p for p in mp.profiles if p.mode == "conventional"), None)
        rev = next((p for p in mp.profiles if p.mode == "bwsk_reversible"), None)
        if conv and rev:
            savings = conv.memory.total_peak_mb - rev.memory.total_peak_mb
            print(
                f"  {mp.model_name}: "
                f"{conv.memory.total_peak_mb:.0f} → "
                f"{rev.memory.total_peak_mb:.0f} MB "
                f"(saves {savings:.0f} MB)"
            )
    print("=" * 70)


if __name__ == "__main__":
    main()
