"""Full training pipeline: train to convergence, compare, upload, cleanup.

Runs epoch-based training with validation and early stopping for each of
16 models across 3 BWSK modes and 2 experiments (fine-tune + from-scratch).
After each model completes, results are compared, uploaded to HuggingFace,
and local artifacts are cleaned up to free disk/VRAM.

Why this exists: the extended_benchmark.py caps training at 300 steps —
sufficient for quick validation but not for genuine quality comparison.
This pipeline trains to convergence to produce publication-quality results.

Usage:
    uv run python scripts/full_training_pipeline.py
    uv run python scripts/full_training_pipeline.py --resume
    uv run python scripts/full_training_pipeline.py --models pythia-70m,resnet50
    uv run python scripts/full_training_pipeline.py --experiment finetune
    uv run python scripts/full_training_pipeline.py --experiment scratch
    uv run python scripts/full_training_pipeline.py --modes conventional,bwsk_reversible
    uv run python scripts/full_training_pipeline.py --max-epochs 3
    uv run python scripts/full_training_pipeline.py --no-upload
    uv run python scripts/full_training_pipeline.py --no-cleanup
    uv run python scripts/full_training_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Reduce CUDA memory fragmentation for large models (Pythia-1B, Switch-Base-8).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bench_utils import peak_memory_mb, reset_memory
from extended_benchmark import (
    ALL_MODELS,
    ARCH_DIVERSITY_MODELS,
    SCALE_SWEEP_MODELS,
    ExtendedModelConfig,
    classify_leaf_modules,
    eval_causal_lm,
    eval_image_cls,
    eval_masked_lm,
    eval_seq2seq,
    eval_ssm_lm,
    get_custom_rules,
    image_forward_step,
    load_ssm_model,
    load_vision_model_torchvision,
    load_vit_model,
    load_wikitext,
    text_forward_step,
)
from training_utils import (
    CheckpointData,
    ChunkedTextDataset,
    CifarDataModule,
    EarlyStopper,
    PipelineState,
    cleanup_checkpoints,
    cleanup_gpu_memory,
    cleanup_hf_cache,
    load_checkpoint,
    save_best_model,
    save_checkpoint,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPTS_DIR
CHECKPOINT_DIR = SCRIPTS_DIR / "checkpoints"
STATE_PATH = SCRIPTS_DIR / "pipeline_state.json"
REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "FULL_TRAINING_REPORT.md"

MODES = ["conventional", "bwsk_analyzed", "bwsk_reversible"]
EXPERIMENTS = ["finetune", "scratch"]


# ---------------------------------------------------------------------------
# Training config per model size tier
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Per-model training hyperparameters.

    Separate from ExtendedModelConfig because these are specific to
    full convergence training (epoch-based with early stopping), not
    the quick 300-step benchmark.

    Args:
        max_epochs: Maximum training epochs before forced stop.
        grad_accum_steps: Gradient accumulation steps for effective
            batch size scaling on large models.
        val_every_steps: Validate every N training steps.
        patience: Early stopping patience (number of val checks).
        lr_finetune: Learning rate for fine-tuning.
        lr_scratch: Learning rate for from-scratch training (typically
            higher to allow faster convergence from random init).
    """

    max_epochs: int = 10
    grad_accum_steps: int = 1
    val_every_steps: int = 500
    patience: int = 3
    lr_finetune: float = 5e-5
    lr_scratch: float = 2.5e-4


def get_training_config(config: ExtendedModelConfig) -> TrainingConfig:
    """Select training hyperparameters based on model size.

    Smaller models can afford more epochs and less gradient accumulation.
    Larger models need fewer epochs with more accumulation to fit in VRAM.

    Args:
        config: The model configuration.

    Returns:
        TrainingConfig tuned for the model's parameter count.
    """
    if config.params_m < 100:
        return TrainingConfig(
            max_epochs=10,
            grad_accum_steps=1,
            val_every_steps=500,
            patience=3,
            lr_finetune=config.finetune_lr,
            lr_scratch=config.finetune_lr * 5,
        )
    if config.params_m <= 500:
        return TrainingConfig(
            max_epochs=5,
            grad_accum_steps=4,
            val_every_steps=500,
            patience=3,
            lr_finetune=config.finetune_lr,
            lr_scratch=config.finetune_lr * 5,
        )
    return TrainingConfig(
        max_epochs=3,
        grad_accum_steps=16,
        val_every_steps=250,
        patience=3,
        lr_finetune=config.finetune_lr,
        lr_scratch=config.finetune_lr * 5,
    )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainRunResult:
    """Result of one training run (one model + one experiment + one mode).

    Args:
        model_slug: URL-safe model identifier.
        experiment: "finetune" or "scratch".
        mode: "conventional", "bwsk_analyzed", or "bwsk_reversible".
        final_train_loss: Last training loss recorded.
        best_val_metric: Best validation metric achieved.
        test_metric: Final test set metric with best model.
        metric_name: Name of the metric (perplexity, accuracy, etc.).
        epochs_completed: Number of epochs trained.
        total_steps: Total training steps completed.
        early_stopped: Whether early stopping triggered.
        wall_time_s: Total wall clock time for this run.
        peak_memory_mb: Peak GPU memory during training.
        loss_curve: Per-step training loss values.
        val_curve: Per-validation-check metric values.
        nan_count: Number of NaN/inf loss steps.
        erasure_budget: K-type fraction (BWSK modes only).
        parallelism_ratio: S-type fraction (BWSK modes only).
    """

    model_slug: str = ""
    experiment: str = ""
    mode: str = ""
    final_train_loss: float = 0.0
    best_val_metric: float = 0.0
    test_metric: float = 0.0
    metric_name: str = ""
    epochs_completed: int = 0
    total_steps: int = 0
    early_stopped: bool = False
    wall_time_s: float = 0.0
    peak_memory_mb: float = 0.0
    loss_curve: list[float] = field(default_factory=list)
    val_curve: list[float] = field(default_factory=list)
    nan_count: int = 0
    erasure_budget: float = 0.0
    parallelism_ratio: float = 0.0


@dataclass
class ModelFullResult:
    """Aggregated results for one model across all experiments and modes.

    Args:
        model_name: Human-readable model name.
        slug: URL-safe model identifier.
        hf_id: HuggingFace model ID.
        arch_family: Architecture family.
        arch_type: Architecture type.
        params_m: Parameter count in millions.
        s_ratio: S-type operation ratio.
        k_ratio: K-type operation ratio.
        runs: All training run results for this model.
        timestamp: When the model was processed.
    """

    model_name: str = ""
    slug: str = ""
    hf_id: str = ""
    arch_family: str = ""
    arch_type: str = ""
    params_m: int = 0
    s_ratio: float = 0.0
    k_ratio: float = 0.0
    runs: list[TrainRunResult] = field(default_factory=list)
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Model loading helpers (extends extended_benchmark loaders)
# ---------------------------------------------------------------------------


def load_model_fresh(
    config: ExtendedModelConfig, from_scratch: bool = False
) -> tuple[nn.Module, object | None]:
    """Load a model, optionally with random weights for from-scratch training.

    For from-scratch mode, creates a fresh model from the HF config
    without loading pretrained weights.

    Args:
        config: Model configuration.
        from_scratch: If True, initialize with random weights.

    Returns:
        Tuple of (model, tokenizer_or_None).
    """
    if config.arch_type == "image_cls":
        if config.source == "torchvision":
            if from_scratch:
                # load_vision_model_torchvision already uses weights=None
                return load_vision_model_torchvision(config), None
            # For fine-tuning, load pretrained weights
            import torchvision.models as models

            model_fn = getattr(models, config.hf_id)
            model = model_fn(weights="DEFAULT")
            # Replace head for CIFAR-10
            if hasattr(model, "fc"):
                model.fc = nn.Linear(model.fc.in_features, 10)
            elif hasattr(model, "classifier"):
                if isinstance(model.classifier, nn.Sequential):
                    last = model.classifier[-1]
                    if isinstance(last, nn.Linear):
                        model.classifier[-1] = nn.Linear(last.in_features, 10)
                elif isinstance(model.classifier, nn.Linear):
                    model.classifier = nn.Linear(model.classifier.in_features, 10)
            return model.to(DEVICE), None
        else:
            # HF ViT
            if from_scratch:
                from transformers import ViTConfig, ViTForImageClassification

                hf_config = ViTConfig.from_pretrained(config.hf_id)
                hf_config.num_labels = 10
                model = ViTForImageClassification(hf_config)
                return model.to(DEVICE), None
            model, _ = load_vit_model(config)
            return model, None

    # Text models
    tokenizer = None
    if config.arch_type == "ssm_lm":
        if from_scratch:
            from extended_benchmark import _patch_mamba_ssm

            _patch_mamba_ssm()
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.hf_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(config.hf_id)
            model = AutoModelForCausalLM.from_config(hf_config)
            return model.to(DEVICE), tokenizer
        model, tokenizer = load_ssm_model(config)
        return model, tokenizer

    # Standard text models (causal_lm, masked_lm, seq2seq)
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.hf_id)

    model_cls_map = {
        "causal_lm": AutoModelForCausalLM,
        "masked_lm": AutoModelForMaskedLM,
        "seq2seq": AutoModelForSeq2SeqLM,
    }
    model_cls = model_cls_map[config.arch_type]

    if from_scratch:
        hf_config = AutoConfig.from_pretrained(config.hf_id)
        model = model_cls.from_config(hf_config)
    else:
        model = model_cls.from_pretrained(config.hf_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model.to(DEVICE), tokenizer


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_model(
    model: nn.Module,
    config: ExtendedModelConfig,
    tokenizer: object | None = None,
    val_text: str = "",
    val_loader=None,
) -> tuple[str, float]:
    """Run validation and return (metric_name, metric_value).

    Dispatches to the correct evaluation function based on model type.

    Args:
        model: The model to evaluate.
        config: Model configuration.
        tokenizer: Tokenizer for text models.
        val_text: Validation text for language models.
        val_loader: Validation DataLoader for vision models.

    Returns:
        Tuple of (metric_name, metric_value).
    """
    model.eval()
    with torch.no_grad():
        if config.arch_type == "image_cls":
            acc, _, _ = eval_image_cls(model, val_loader, use_provenance=False, max_batches=200)
            return "accuracy", acc

        eval_fn_map = {
            "causal_lm": eval_causal_lm,
            "masked_lm": eval_masked_lm,
            "seq2seq": eval_seq2seq,
            "ssm_lm": eval_ssm_lm,
        }
        eval_fn = eval_fn_map[config.arch_type]
        val, _, _ = eval_fn(model, tokenizer, val_text, use_provenance=False)
        metric_name = "pseudo-perplexity" if config.arch_type == "masked_lm" else "perplexity"
        return metric_name, val


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------


def train_one_run(
    config: ExtendedModelConfig,
    train_config: TrainingConfig,
    experiment: str,
    mode: str,
    text_dataset: ChunkedTextDataset | None = None,
    val_text: str = "",
    test_text: str = "",
    cifar_module: CifarDataModule | None = None,
    resume_path: Path | None = None,
) -> TrainRunResult:
    """Train one model in one experiment/mode to convergence.

    Epoch-based training with:
    - Validation every val_every_steps + at epoch end
    - Early stopping on validation metric
    - Gradient accumulation for large models
    - AMP (bf16) for models >= 300M params
    - Gradient clipping (max_norm=1.0)
    - NaN safety: abort if >20% of steps produce NaN

    Args:
        config: Model configuration.
        train_config: Training hyperparameters.
        experiment: "finetune" or "scratch".
        mode: "conventional", "bwsk_analyzed", or "bwsk_reversible".
        text_dataset: Token dataset for text models.
        val_text: Validation text for language models.
        test_text: Test text for language models.
        cifar_module: CIFAR data module for vision models.
        resume_path: Path to checkpoint for resuming.

    Returns:
        TrainRunResult with all training metrics and curves.
    """
    from_scratch = experiment == "scratch"
    lr = train_config.lr_scratch if from_scratch else train_config.lr_finetune

    print(f"    [{experiment}/{mode}] Loading model (from_scratch={from_scratch}, lr={lr:.2e})")

    # Load model
    model, tokenizer = load_model_fresh(config, from_scratch=from_scratch)

    # BWSK analysis
    erasure_budget = 0.0
    parallelism_ratio = 0.0
    if mode in ("bwsk_analyzed", "bwsk_reversible"):
        rules = get_custom_rules(config.arch_family)
        leaves = classify_leaf_modules(model, custom_rules=rules)
        s_count = sum(1 for _, _, c, _ in leaves if c == "S")
        k_count = sum(1 for _, _, c, _ in leaves if c == "K")
        total = len(leaves)
        erasure_budget = k_count / total if total > 0 else 0.0
        parallelism_ratio = s_count / total if total > 0 else 0.0

    # Gradient checkpointing: always for reversible/SSM, large models
    # (>=500M), and MoE (high module count pushes VRAM despite lower param count).
    use_ckpt = (
        mode == "bwsk_reversible"
        or config.arch_family == "ssm"
        or config.arch_family == "moe"
        or config.params_m >= 500
    )
    if use_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    # AMP for large models. MoE excluded: router casts hidden_states to float32
    # internally (for stability) but autocast casts classifier weights to bf16,
    # causing dtype mismatch. Switch-Base-8 also OOMs without AMP on 16GB.
    use_amp = DEVICE.type == "cuda" and config.params_m >= 300 and config.arch_family != "moe"

    # Always ensure fp32 master weights. Some HF models (OPT, Pythia) ship with
    # torch_dtype=float16, and training with fp16 params causes NaN after the
    # first optimizer step due to fp16's narrow representable range. AMP autocast
    # handles bf16 casting dynamically during forward — it needs fp32 master weights.
    model = model.float()

    # Optimizer and scheduler.
    # foreach=False for large models: the default foreach=True pre-allocates
    # temporary buffers for ALL params simultaneously (e.g., sqrt(exp_avg_sq)),
    # causing OOM on models that barely fit in VRAM. foreach=False processes
    # one param group at a time, trading ~10% optimizer step speed for lower
    # peak memory.
    model.train()
    use_foreach = config.params_m < 500
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, foreach=use_foreach)

    # Estimate total steps for cosine schedule
    if config.arch_type == "image_cls" and cifar_module is not None:
        steps_per_epoch = len(cifar_module.train_loader) // train_config.grad_accum_steps
    elif text_dataset is not None:
        loader_len = math.ceil(len(text_dataset) / config.batch_size)
        steps_per_epoch = loader_len // train_config.grad_accum_steps
    else:
        steps_per_epoch = 100  # fallback

    total_steps_est = steps_per_epoch * train_config.max_epochs
    warmup_steps = max(1, total_steps_est // 10)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps_est - warmup_steps),
        eta_min=0,
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, scheduler],
        milestones=[warmup_steps],
    )

    # Early stopping
    es_mode = "max" if config.arch_type == "image_cls" else "min"
    stopper = EarlyStopper(patience=train_config.patience, mode=es_mode)

    # Resume from checkpoint if available
    loss_curve: list[float] = []
    val_curve: list[float] = []
    start_epoch = 0
    global_step = 0
    best_metric = float("inf") if es_mode == "min" else 0.0

    ckpt_dir = CHECKPOINT_DIR / config.slug / experiment / mode
    best_model_path = ckpt_dir / "best_model"
    latest_ckpt_path = ckpt_dir / "latest.pt"

    if resume_path and resume_path.exists():
        print(f"    Resuming from {resume_path}")
        ckpt_data = load_checkpoint(model, optimizer, combined_scheduler, resume_path)
        start_epoch = ckpt_data.epoch
        global_step = ckpt_data.global_step
        best_metric = ckpt_data.best_metric
        loss_curve = ckpt_data.loss_curve
        val_curve = ckpt_data.val_curve

    reset_memory()
    nan_count = 0
    total_nan_threshold = 0.2  # abort if >20% NaN
    start_time = time.perf_counter()
    early_stopped = False

    # --- Training loop ---
    for epoch in range(start_epoch, train_config.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        accum_count = 0

        if config.arch_type == "image_cls" and cifar_module is not None:
            # Vision training
            is_hf_vit = config.source == "huggingface"
            for images, labels in tqdm(
                cifar_module.train_loader,
                desc=f"    Epoch {epoch + 1}",
                leave=False,
            ):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss = image_forward_step(model, images, labels, is_hf_vit=is_hf_vit)
                    loss = loss / train_config.grad_accum_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    accum_count = 0
                    optimizer.zero_grad()
                    continue

                loss.backward()
                accum_count += 1

                if accum_count >= train_config.grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    combined_scheduler.step()
                    optimizer.zero_grad()
                    accum_count = 0
                    global_step += 1

                    step_loss = loss.item() * train_config.grad_accum_steps
                    loss_curve.append(step_loss)
                    epoch_loss += step_loss
                    epoch_steps += 1

                    # Mid-epoch validation
                    if global_step > 0 and global_step % train_config.val_every_steps == 0:
                        metric_name, val_metric = validate_model(
                            model,
                            config,
                            val_loader=cifar_module.val_loader,
                        )
                        val_curve.append(val_metric)
                        print(f"      Step {global_step}: val {metric_name}={val_metric:.4f}")
                        if stopper.step(val_metric):
                            early_stopped = True
                            break
                        if stopper.improved:
                            best_metric = val_metric
                            save_best_model(model, None, best_model_path)
                        model.train()

        elif text_dataset is not None:
            # Text training
            text_loader = torch.utils.data.DataLoader(
                text_dataset,
                batch_size=config.batch_size,
                shuffle=True,
            )
            mask_token_id = getattr(tokenizer, "mask_token_id", 103) or 103
            pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

            for batch in tqdm(
                text_loader,
                desc=f"    Epoch {epoch + 1}",
                leave=False,
            ):
                batch = batch.to(DEVICE)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss = text_forward_step(
                        model,
                        batch,
                        config,
                        mask_token_id,
                        pad_token_id,
                    )
                    loss = loss / train_config.grad_accum_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    accum_count = 0
                    optimizer.zero_grad()
                    continue

                loss.backward()
                accum_count += 1

                if accum_count >= train_config.grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    combined_scheduler.step()
                    optimizer.zero_grad()
                    accum_count = 0
                    global_step += 1

                    step_loss = loss.item() * train_config.grad_accum_steps
                    loss_curve.append(step_loss)
                    epoch_loss += step_loss
                    epoch_steps += 1

                    # Mid-epoch validation
                    if global_step > 0 and global_step % train_config.val_every_steps == 0:
                        metric_name, val_metric = validate_model(
                            model,
                            config,
                            tokenizer=tokenizer,
                            val_text=val_text,
                        )
                        val_curve.append(val_metric)
                        print(f"      Step {global_step}: val {metric_name}={val_metric:.4f}")
                        if stopper.step(val_metric):
                            early_stopped = True
                            break
                        if stopper.improved:
                            best_metric = val_metric
                            save_best_model(model, tokenizer, best_model_path)
                        model.train()

        # NaN safety check — count in consistent units (raw batches).
        # loss_curve counts optimizer steps; multiply by grad_accum to get batches.
        good_batches = len(loss_curve) * train_config.grad_accum_steps
        total_batches = good_batches + nan_count
        if total_batches > 0:
            nan_ratio = nan_count / total_batches
            if nan_ratio > total_nan_threshold:
                print(f"    ABORT: NaN ratio {nan_ratio:.1%} > {total_nan_threshold:.0%} threshold")
                break

        if early_stopped:
            print(f"    Early stopped at epoch {epoch + 1}, best={best_metric:.4f}")
            break

        # End-of-epoch validation
        if config.arch_type == "image_cls" and cifar_module is not None:
            metric_name, val_metric = validate_model(
                model,
                config,
                val_loader=cifar_module.val_loader,
            )
        elif tokenizer is not None:
            metric_name, val_metric = validate_model(
                model,
                config,
                tokenizer=tokenizer,
                val_text=val_text,
            )
        else:
            metric_name = "loss"
            val_metric = epoch_loss / max(1, epoch_steps)

        val_curve.append(val_metric)
        avg_loss = epoch_loss / max(1, epoch_steps)
        print(
            f"    Epoch {epoch + 1}/{train_config.max_epochs}: "
            f"train_loss={avg_loss:.4f}, "
            f"val_{metric_name}={val_metric:.4f}"
        )

        if stopper.step(val_metric):
            early_stopped = True
            print(f"    Early stopped after epoch {epoch + 1}, best={stopper.best:.4f}")
            if stopper.improved:
                best_metric = val_metric
                save_best_model(
                    model,
                    tokenizer if config.arch_type != "image_cls" else None,
                    best_model_path,
                )
            break

        if stopper.improved:
            best_metric = val_metric
            save_best_model(
                model,
                tokenizer if config.arch_type != "image_cls" else None,
                best_model_path,
            )

        # Save checkpoint at end of epoch
        ckpt_data = CheckpointData(
            epoch=epoch + 1,
            global_step=global_step,
            best_metric=best_metric,
            metric_name=metric_name,
            loss_curve=loss_curve,
            val_curve=val_curve,
        )
        save_checkpoint(
            model,
            optimizer,
            combined_scheduler,
            ckpt_data,
            latest_ckpt_path,
        )

    wall_time = time.perf_counter() - start_time
    mem = peak_memory_mb()
    epochs_completed = min(epoch + 1, train_config.max_epochs) if "epoch" in dir() else 0

    # Free training model before test eval to avoid OOM on large models.
    del model
    cleanup_gpu_memory()

    # --- Test evaluation with best model ---
    test_metric = 0.0
    if best_model_path.exists():
        print("    Evaluating best model on test set...")
        try:
            # Load best model for testing
            test_model, test_tok = load_model_fresh(config, from_scratch=from_scratch)
            if not use_amp:
                test_model = test_model.float()
            # Load saved best weights
            if hasattr(test_model, "load_state_dict"):
                if (best_model_path / "model.pt").exists():
                    test_model.load_state_dict(
                        torch.load(
                            best_model_path / "model.pt",
                            map_location=DEVICE,
                            weights_only=True,
                        )
                    )
                elif hasattr(test_model, "from_pretrained"):
                    test_model = type(test_model).from_pretrained(best_model_path).to(DEVICE)

            if config.arch_type == "image_cls" and cifar_module is not None:
                _, test_metric = validate_model(
                    test_model,
                    config,
                    val_loader=cifar_module.test_loader,
                )
            elif test_tok is not None:
                _, test_metric = validate_model(
                    test_model,
                    config,
                    tokenizer=test_tok,
                    val_text=test_text,
                )

            print(f"    Test metric: {test_metric:.4f}")
            del test_model
            cleanup_gpu_memory()
        except Exception as e:
            print(f"    WARNING: Test evaluation failed: {e}")
    else:
        # Use last model's validation metric as proxy
        test_metric = best_metric

    return TrainRunResult(
        model_slug=config.slug,
        experiment=experiment,
        mode=mode,
        final_train_loss=loss_curve[-1] if loss_curve else 0.0,
        best_val_metric=best_metric,
        test_metric=test_metric,
        metric_name=metric_name if "metric_name" in dir() else "",
        epochs_completed=epochs_completed,
        total_steps=global_step,
        early_stopped=early_stopped,
        wall_time_s=wall_time,
        peak_memory_mb=mem,
        loss_curve=loss_curve,
        val_curve=val_curve,
        nan_count=nan_count,
        erasure_budget=erasure_budget,
        parallelism_ratio=parallelism_ratio,
    )


# ---------------------------------------------------------------------------
# Per-model pipeline
# ---------------------------------------------------------------------------


def process_model(
    config: ExtendedModelConfig,
    experiments: list[str],
    modes: list[str],
    state: PipelineState,
    max_epochs_override: int | None = None,
    do_upload: bool = True,
    do_cleanup: bool = True,
    dry_run: bool = False,
) -> ModelFullResult:
    """Run the full pipeline for one model.

    Steps:
    1. S/K classification (once, shared across runs)
    2. Train each experiment × mode combination
    3. Compare results
    4. Upload to HuggingFace (if enabled)
    5. Cleanup local files (if enabled)

    Args:
        config: Model configuration.
        experiments: List of experiments to run.
        modes: List of BWSK modes to train.
        state: Pipeline state for resume support.
        max_epochs_override: Override max_epochs if specified.
        do_upload: Whether to upload to HuggingFace.
        do_cleanup: Whether to clean up local files.
        dry_run: If True, print plan without executing.

    Returns:
        ModelFullResult with all training runs.
    """
    print("\n" + "=" * 70)
    print(f"MODEL: {config.name} ({config.params_m}M, {config.arch_family}/{config.arch_type})")
    print("=" * 70)

    result = ModelFullResult(
        model_name=config.name,
        slug=config.slug,
        hf_id=config.hf_id,
        arch_family=config.arch_family,
        arch_type=config.arch_type,
        params_m=config.params_m,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    if dry_run:
        for exp in experiments:
            for mode in modes:
                print(f"  [DRY RUN] Would train: {exp}/{mode}")
        return result

    # Skip models marked as classification-only
    if config.finetune_only_classification:
        print(f"  SKIPPING: {config.name} is classification-only (OOM risk)")
        return result

    train_config = get_training_config(config)
    if max_epochs_override is not None:
        train_config.max_epochs = max_epochs_override

    # --- S/K Classification (shared) ---
    print("\n  Running S/K classification...")
    try:
        temp_model, _ = load_model_fresh(config, from_scratch=False)
        temp_model = temp_model.float()
        rules = get_custom_rules(config.arch_family)
        leaves = classify_leaf_modules(temp_model, custom_rules=rules)
        s_count = sum(1 for _, _, c, _ in leaves if c == "S")
        k_count = sum(1 for _, _, c, _ in leaves if c == "K")
        total = len(leaves)
        result.s_ratio = s_count / total if total > 0 else 0.0
        result.k_ratio = k_count / total if total > 0 else 0.0
        print(f"  S-ratio: {result.s_ratio:.3f}, K-ratio: {result.k_ratio:.3f}")
        del temp_model
        cleanup_gpu_memory()
    except Exception as e:
        print(f"  WARNING: Classification failed: {e}")

    # --- Prepare data ---
    print("\n  Preparing data...")
    text_dataset = None
    val_text = ""
    test_text = ""
    cifar_module = None

    try:
        if config.arch_type == "image_cls":
            cifar_module = CifarDataModule(batch_size=config.batch_size)
            cifar_module.setup()
            print(
                f"  CIFAR-10: train={len(cifar_module._train_dataset)}, "
                f"val={len(cifar_module._val_dataset)}, "
                f"test={len(cifar_module._test_dataset)}"
            )
        else:
            # Load tokenizer for text dataset preparation
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.hf_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            train_text = load_wikitext("train")
            val_text = load_wikitext("validation")
            test_text = load_wikitext("test")

            # Tokenize training text
            max_tokens = config.seq_len * config.batch_size * 50000
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

            text_dataset = ChunkedTextDataset(all_ids, config.seq_len)
            print(f"  WikiText-2: {len(text_dataset)} chunks (seq_len={config.seq_len})")
    except Exception as e:
        print(f"  ERROR preparing data: {e}")
        traceback.print_exc()
        return result

    # --- Training runs ---
    for exp in experiments:
        for mode in modes:
            if state.is_done(config.slug, exp, mode):
                print(f"\n  [{exp}/{mode}] Already completed, skipping.")
                continue

            state.mark_in_progress(config.slug, exp, mode)
            state.save(STATE_PATH)

            print(f"\n  [{exp}/{mode}] Starting training...")
            try:
                # Check for resume checkpoint
                ckpt_path = CHECKPOINT_DIR / config.slug / exp / mode / "latest.pt"
                resume = ckpt_path if ckpt_path.exists() else None

                run_result = train_one_run(
                    config=config,
                    train_config=train_config,
                    experiment=exp,
                    mode=mode,
                    text_dataset=text_dataset,
                    val_text=val_text,
                    test_text=test_text,
                    cifar_module=cifar_module,
                    resume_path=resume,
                )
                result.runs.append(run_result)

                # Save per-run result
                run_json_path = RESULTS_DIR / f"fulltrain_{config.slug}_{exp}_{mode}.json"
                with open(run_json_path, "w") as f:
                    json.dump(asdict(run_result), f, indent=2, default=str)

                state.mark_complete(config.slug, exp, mode)
                state.save(STATE_PATH)

                print(
                    f"    Done: {run_result.metric_name}="
                    f"{run_result.best_val_metric:.4f}, "
                    f"epochs={run_result.epochs_completed}, "
                    f"steps={run_result.total_steps}, "
                    f"time={run_result.wall_time_s:.0f}s"
                )

            except Exception as e:
                print(f"    ERROR in [{exp}/{mode}]: {e}")
                traceback.print_exc()
                state.mark_failed(config.slug, exp, mode, str(e))
                state.save(STATE_PATH)
                cleanup_gpu_memory()

    # --- Comparison ---
    if result.runs:
        print("\n  --- Comparison ---")
        _print_comparison(result)

    # --- Save model result ---
    model_json_path = RESULTS_DIR / f"fulltrain_{config.slug}_results.json"
    with open(model_json_path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    print(f"\n  Results saved: {model_json_path}")

    # --- Upload ---
    if do_upload and result.runs:
        _upload_model_results(config, result)

    # --- Cleanup ---
    if do_cleanup:
        ckpt_dir = CHECKPOINT_DIR / config.slug
        cleanup_checkpoints(config.slug, ckpt_dir)
        cleanup_hf_cache(config.hf_id)
        cleanup_gpu_memory()

    return result


# ---------------------------------------------------------------------------
# Comparison display
# ---------------------------------------------------------------------------


def _print_comparison(result: ModelFullResult) -> None:
    """Print a side-by-side comparison table for one model's runs."""
    # Group by experiment
    for exp in EXPERIMENTS:
        runs = [r for r in result.runs if r.experiment == exp]
        if not runs:
            continue

        print(f"\n  {exp.upper()} results:")
        header = "  {:20s} {:>12s} {:>12s} {:>12s} {:>8s} {:>8s}".format(
            "Mode",
            "Best Val",
            "Test",
            "Train Loss",
            "Epochs",
            "Time(s)",
        )
        print(header)
        print("  " + "-" * 76)
        for r in runs:
            print(
                f"  {r.mode:20s} {r.best_val_metric:12.4f} "
                f"{r.test_metric:12.4f} {r.final_train_loss:12.4f} "
                f"{r.epochs_completed:8d} {r.wall_time_s:8.0f}"
            )


# ---------------------------------------------------------------------------
# HuggingFace upload
# ---------------------------------------------------------------------------


def _upload_model_results(
    config: ExtendedModelConfig,
    result: ModelFullResult,
) -> None:
    """Upload training results to consolidated HuggingFace repo.

    Uploads all variants into a single repo (tzervas/bwsk-{slug}) with
    subdirectories for each experiment/mode combination. Generates and
    uploads a consolidated README.md model card.

    Args:
        config: Model configuration.
        result: Full model results with all runs.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
    except ImportError:
        print("  WARNING: huggingface_hub not installed, skipping upload")
        return

    # Import here to avoid circular dependency at module level.
    from generate_model_cards import (
        VARIANT_DIRS,
        generate_consolidated_card,
        load_aggregated_results,
    )

    repo_id = f"tzervas/bwsk-{config.slug}"
    print(f"\n  Uploading to HuggingFace: {repo_id}")

    try:
        api.create_repo(repo_id, exist_ok=True)
    except Exception as e:
        print(f"    WARNING: Could not create repo {repo_id}: {e}")
        return

    for run in result.runs:
        variant_dir = VARIANT_DIRS[(run.experiment, run.mode)]
        try:
            # Upload best model into variant subdirectory
            best_path = CHECKPOINT_DIR / config.slug / run.experiment / run.mode / "best_model"
            if best_path.exists():
                api.upload_folder(
                    folder_path=str(best_path),
                    path_in_repo=variant_dir,
                    repo_id=repo_id,
                    commit_message=(
                        f"Add {variant_dir}: {run.metric_name}={run.best_val_metric:.4f}"
                    ),
                )
                print(f"    Uploaded: {variant_dir}/")

            # Upload per-run result JSON into variant subdirectory
            run_json = RESULTS_DIR / f"fulltrain_{config.slug}_{run.experiment}_{run.mode}.json"
            if run_json.exists():
                api.upload_file(
                    path_or_fileobj=str(run_json),
                    path_in_repo=f"{variant_dir}/training_results.json",
                    repo_id=repo_id,
                    commit_message=f"Add {variant_dir} training results",
                )
        except Exception as e:
            print(f"    WARNING: Upload failed for {variant_dir}: {e}")

    # Upload aggregated results.json at root
    try:
        model_json = RESULTS_DIR / f"fulltrain_{config.slug}_results.json"
        if model_json.exists():
            api.upload_file(
                path_or_fileobj=str(model_json),
                path_in_repo="results.json",
                repo_id=repo_id,
                commit_message="Add aggregated training results",
            )
            print("    Uploaded: results.json")
    except Exception as e:
        print(f"    WARNING: Results upload failed: {e}")

    # Generate and upload consolidated README.md
    try:
        all_results = load_aggregated_results()
        agg = all_results.get(config.slug)
        if agg:
            card = generate_consolidated_card(agg)
            api.upload_file(
                path_or_fileobj=card.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message="Add consolidated BWSK model card",
            )
            print("    Uploaded: README.md")
    except Exception as e:
        print(f"    WARNING: README upload failed: {e}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(all_results: list[ModelFullResult]) -> str:
    """Generate markdown report with comparison tables.

    Args:
        all_results: All model results.

    Returns:
        Markdown report string.
    """
    lines = [
        "# Full Training Pipeline Report",
        "",
        "Epoch-based training to convergence with early stopping.",
        "",
    ]

    if all_results:
        r0 = all_results[0]
        lines.append(f"**Date**: {r0.timestamp}")
        device = "GPU" if torch.cuda.is_available() else "CPU"
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name()
        lines.append(f"**Device**: {device}")
        lines.append("")

    for result in all_results:
        lines.append(f"## {result.model_name} ({result.params_m}M, {result.arch_family})")
        lines.append("")
        lines.append(f"S-ratio: {result.s_ratio:.3f}, K-ratio: {result.k_ratio:.3f}")
        lines.append("")

        for exp in EXPERIMENTS:
            runs = [r for r in result.runs if r.experiment == exp]
            if not runs:
                continue

            lines.append(f"### {exp.title()}")
            lines.append("")
            lines.append(
                "| Mode | Best Val | Test | Train Loss | Epochs | Steps | Time (s) | Memory (MB) |"
            )
            lines.append(
                "|------|---------|------|-----------|--------|-------|----------|-------------|"
            )
            for r in runs:
                lines.append(
                    f"| {r.mode} | {r.best_val_metric:.4f} "
                    f"| {r.test_metric:.4f} "
                    f"| {r.final_train_loss:.4f} "
                    f"| {r.epochs_completed} "
                    f"| {r.total_steps} "
                    f"| {r.wall_time_s:.0f} "
                    f"| {r.peak_memory_mb:.0f} |"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

    # Cross-model summary
    lines.append("## Cross-Model Summary")
    lines.append("")
    lines.append(
        "| Model | Params (M) | S-ratio | Best Mode (FT) "
        "| Best Val (FT) | Best Mode (Scratch) | Best Val (Scratch) |"
    )
    lines.append(
        "|-------|-----------|---------|----------------|"
        "---------------|---------------------|-------------------|"
    )
    for result in all_results:
        ft_runs = [r for r in result.runs if r.experiment == "finetune"]
        sc_runs = [r for r in result.runs if r.experiment == "scratch"]

        ft_best = ""
        ft_val = ""
        sc_best = ""
        sc_val = ""

        if ft_runs:
            if ft_runs[0].metric_name == "accuracy":
                best_ft = max(ft_runs, key=lambda r: r.best_val_metric)
            else:
                best_ft = min(ft_runs, key=lambda r: r.best_val_metric)
            ft_best = best_ft.mode
            ft_val = f"{best_ft.best_val_metric:.4f}"

        if sc_runs:
            if sc_runs[0].metric_name == "accuracy":
                best_sc = max(sc_runs, key=lambda r: r.best_val_metric)
            else:
                best_sc = min(sc_runs, key=lambda r: r.best_val_metric)
            sc_best = best_sc.mode
            sc_val = f"{best_sc.best_val_metric:.4f}"

        lines.append(
            f"| {result.model_name} | {result.params_m} "
            f"| {result.s_ratio:.3f} | {ft_best} | {ft_val} "
            f"| {sc_best} | {sc_val} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Full training pipeline: convergence training, comparison, upload, cleanup",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from pipeline_state.json, skipping completed runs",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model slugs (default: all trainable)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="",
        choices=["", "finetune", "scratch"],
        help="Run only one experiment type (default: both)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="",
        help="Comma-separated modes (default: all 3)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max epochs for all models",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip HuggingFace upload",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep local checkpoints and HF cache after runs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without executing",
    )
    parser.add_argument(
        "--scale-only",
        action="store_true",
        help="Run only scale sweep models",
    )
    parser.add_argument(
        "--arch-only",
        action="store_true",
        help="Run only architecture diversity models",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full training pipeline."""
    args = parse_args()

    # Select models
    if args.models:
        slugs = {s.strip() for s in args.models.split(",")}
        models = [m for m in ALL_MODELS if m.slug in slugs]
        if not models:
            print(f"ERROR: No models found for: {slugs}")
            print("Available: " + ", ".join(m.slug for m in ALL_MODELS))
            sys.exit(1)
    elif args.scale_only:
        models = SCALE_SWEEP_MODELS
    elif args.arch_only:
        models = ARCH_DIVERSITY_MODELS
    else:
        models = ALL_MODELS

    # Filter out classification-only models for non-dry-run
    trainable = [m for m in models if not m.finetune_only_classification]

    # Select experiments
    experiments = EXPERIMENTS
    if args.experiment:
        experiments = [args.experiment]

    # Select modes
    modes = MODES
    if args.modes:
        modes = [m.strip() for m in args.modes.split(",")]
        invalid = [m for m in modes if m not in MODES]
        if invalid:
            print(f"ERROR: Invalid modes: {invalid}")
            print(f"Valid modes: {MODES}")
            sys.exit(1)

    # Load or create pipeline state
    state = PipelineState()
    if args.resume:
        state = PipelineState.load(STATE_PATH)
        n_done = len(state.completed)
        print(f"Resuming: {n_done} runs already completed")

    # Summary
    total_runs = len(trainable) * len(experiments) * len(modes)
    already_done = sum(
        1
        for m in trainable
        for e in experiments
        for mode in modes
        if state.is_done(m.slug, e, mode)
    )

    print("=" * 70)
    print("FULL TRAINING PIPELINE")
    print(f"Models: {len(trainable)} (of {len(models)} selected)")
    print(f"Experiments: {experiments}")
    print(f"Modes: {modes}")
    print(f"Total runs: {total_runs} ({already_done} already done)")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {mem_gb:.1f} GB")
    print(f"Upload: {not args.no_upload}")
    print(f"Cleanup: {not args.no_cleanup}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)

    if args.dry_run:
        for m in trainable:
            tc = get_training_config(m)
            if args.max_epochs:
                tc.max_epochs = args.max_epochs
            print(
                f"\n  {m.name} ({m.slug}): "
                f"epochs={tc.max_epochs}, "
                f"accum={tc.grad_accum_steps}, "
                f"val_every={tc.val_every_steps}"
            )
            for e in experiments:
                for mode in modes:
                    done = state.is_done(m.slug, e, mode)
                    status = "DONE" if done else "PENDING"
                    print(f"    {e}/{mode}: {status}")
        return

    # --- Main loop ---
    all_results: list[ModelFullResult] = []

    for i, config in enumerate(trainable):
        print(f"\n[{i + 1}/{len(trainable)}] {config.name} ({config.slug})")
        try:
            result = process_model(
                config=config,
                experiments=experiments,
                modes=modes,
                state=state,
                max_epochs_override=args.max_epochs,
                do_upload=not args.no_upload,
                do_cleanup=not args.no_cleanup,
                dry_run=args.dry_run,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  FATAL ERROR for {config.name}: {e}")
            traceback.print_exc()
            cleanup_gpu_memory()

    # --- Generate combined report ---
    if all_results:
        report = generate_report(all_results)
        with open(REPORT_PATH, "w") as f:
            f.write(report)
        print(f"\nReport saved: {REPORT_PATH}")

        # Save combined JSON
        combined_path = RESULTS_DIR / "full_training_results.json"
        with open(combined_path, "w") as f:
            json.dump(
                [asdict(r) for r in all_results],
                f,
                indent=2,
                default=str,
            )
        print(f"Combined results: {combined_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"Models processed: {len(all_results)}")
    print(f"Total completed runs: {len(state.completed)}")
    print(f"Total failed runs: {len(state.failed)}")
    if state.failed:
        print("\nFailed runs:")
        for f in state.failed:
            print(f"  {f[0]}/{f[1]}/{f[2]}: {f[3]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
