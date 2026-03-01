"""Multi-model benchmark: BWSK analysis vs conventional PyTorch.

Benchmarks 6 models across 3 architecture types on WikiText-2:
  - BERT-base (110M, encoder-only, masked LM)
  - GPT-2 Medium (345M, decoder-only, causal LM)
  - T5-small (60M, encoder-decoder, seq2seq)
  - OPT-350M (331M, decoder-only, causal LM)
  - Pythia-410M (405M, decoder-only, causal LM)
  - Pythia-1B (1010M, decoder-only, causal LM)

Each model gets the same 6-section analysis as the GPT-2 benchmark:
  A. Architecture Analysis — S/K classification
  B. Evaluation — perplexity or pseudo-perplexity (model-type-specific)
  C. Fine-tuning Comparison — 3 modes (conventional, BWSK-analyzed, BWSK-reversible)
  D. Memory Profiling — peak GPU memory per training mode
  E. CALM Analysis — per-block parallelism and distribution partitioning
  F. Quality Summary — side-by-side comparison table

Plus a cross-model comparison report at the end.

Usage:
    uv run python scripts/multi_model_benchmark.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bwsk.calm import analyze_calm, partition_for_distribution
from bwsk.classify import OpClass, classify_operation
from bwsk.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
FINETUNE_STEPS = 300
FINETUNE_LR = 5e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path(__file__).resolve().parent
REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "MULTI_MODEL_BENCHMARK_REPORT.md"

# Custom classification rules for HuggingFace-specific types not in BWSK's DB.
# These cover custom module types across all models we benchmark.
CUSTOM_RULES: dict[str, OpClass] = {
    "Conv1D": OpClass.S,  # HF GPT-2 uses custom Conv1D (linear projection)
    "NewGELUActivation": OpClass.K,  # HF GELU variant
    "GELUActivation": OpClass.K,
    "FastGELUActivation": OpClass.K,
    "T5LayerNorm": OpClass.S,  # RMSNorm variant, invertible given scale
    "OPTLearnedPositionalEmbedding": OpClass.S,  # Learned embed, injective
    "RotaryEmbedding": OpClass.S,  # Pure rotation, no info loss
    "GPTNeoXRotaryEmbedding": OpClass.S,  # Pythia rotary embeddings
}


@dataclass
class ModelConfig:
    """Configuration for one benchmark model.

    Each model specifies its own fine-tuning learning rate because larger
    models (300M+) are more sensitive to LR — too high causes gradient
    explosions that produce NaN loss values.
    """

    name: str  # Display name
    hf_id: str  # HuggingFace model ID
    arch_type: str  # "causal_lm", "masked_lm", "seq2seq"
    params_m: int  # Approximate parameter count in millions
    batch_size: int  # Training batch size (tuned to fit 16GB VRAM)
    seq_len: int  # Sequence length for training
    block_paths: list[tuple[str, str]]  # [(section, dot.path.to.blocks), ...]
    finetune_lr: float = 5e-5  # Per-model fine-tuning LR


MODELS: list[ModelConfig] = [
    ModelConfig(
        name="BERT-base",
        hf_id="google-bert/bert-base-uncased",
        arch_type="masked_lm",
        params_m=110,
        batch_size=4,
        seq_len=512,
        block_paths=[("encoder", "bert.encoder.layer")],
    ),
    ModelConfig(
        name="GPT-2 Medium",
        hf_id="openai-community/gpt2-medium",
        arch_type="causal_lm",
        params_m=345,
        batch_size=2,
        seq_len=512,
        block_paths=[("decoder", "transformer.h")],
    ),
    ModelConfig(
        name="T5-small",
        hf_id="google-t5/t5-small",
        arch_type="seq2seq",
        params_m=60,
        batch_size=4,
        seq_len=512,
        block_paths=[
            ("encoder", "encoder.block"),
            ("decoder", "decoder.block"),
        ],
    ),
    ModelConfig(
        name="OPT-350M",
        hf_id="facebook/opt-350m",
        arch_type="causal_lm",
        params_m=331,
        batch_size=2,
        seq_len=512,
        block_paths=[("decoder", "model.decoder.layers")],
        finetune_lr=2e-5,
    ),
    ModelConfig(
        name="Pythia-410M",
        hf_id="EleutherAI/pythia-410m",
        arch_type="causal_lm",
        params_m=405,
        batch_size=2,
        seq_len=512,
        block_paths=[("decoder", "gpt_neox.layers")],
        finetune_lr=2e-5,
    ),
    ModelConfig(
        name="Pythia-1B",
        hf_id="EleutherAI/pythia-1b",
        arch_type="causal_lm",
        params_m=1010,
        batch_size=1,
        seq_len=512,
        block_paths=[("decoder", "gpt_neox.layers")],
        finetune_lr=1e-5,
    ),
]


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------


@dataclass
class BlockClassification:
    """Classification results for a single transformer block."""

    block_idx: int
    section: str
    s_count: int
    k_count: int
    gray_count: int
    total: int
    s_ratio: float


@dataclass
class ClassificationResults:
    """Full model S/K classification results."""

    total_modules: int
    s_count: int
    k_count: int
    gray_count: int
    s_ratio: float
    k_ratio: float
    gray_ratio: float
    per_block: list[BlockClassification]


@dataclass
class EvalResult:
    """Evaluation result (perplexity or pseudo-perplexity)."""

    metric_name: str  # "perplexity" or "pseudo-perplexity"
    value: float
    value_with_provenance: float
    baseline_time_s: float
    provenance_time_s: float
    provenance_overhead_s: float
    provenance_s_phases: int
    provenance_k_boundaries: int
    provenance_erasure_budget: float


@dataclass
class FinetuneResult:
    """Fine-tuning result for one mode."""

    mode: str
    final_loss: float
    eval_metric: float  # perplexity or pseudo-perplexity after fine-tuning
    wall_time_s: float
    peak_memory_mb: float
    loss_curve: list[float]
    nan_count: int = 0
    erasure_budget: float = 0.0
    parallelism_ratio: float = 0.0


@dataclass
class CALMBlockResult:
    """CALM analysis for one transformer block."""

    block_idx: int
    section: str
    total_children: int
    monotone_count: int
    sync_count: int
    parallelism_ratio: float
    num_sync_barriers: int


@dataclass
class ModelResults:
    """All benchmark results for one model."""

    model_name: str = ""
    hf_id: str = ""
    arch_type: str = ""
    params_m: int = 0
    batch_size: int = 0
    seq_len: int = 0
    device: str = ""
    timestamp: str = ""
    classification: ClassificationResults | None = None
    evaluation: EvalResult | None = None
    finetune_conventional: FinetuneResult | None = None
    finetune_bwsk_analyzed: FinetuneResult | None = None
    finetune_bwsk_reversible: FinetuneResult | None = None
    calm_blocks: list[CALMBlockResult] = field(default_factory=list)
    calm_partition_2: list[list[str]] = field(default_factory=list)
    calm_partition_4: list[list[str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def reset_memory() -> None:
    """Reset GPU memory tracking."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb() -> float:
    """Get peak GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def classify_leaf_modules(
    model: nn.Module,
) -> list[tuple[str, nn.Module, str, str]]:
    """Classify all leaf modules in a model.

    Returns list of (name, module, classification, op_type) tuples.
    Uses classify_operation on each leaf since torch.fx can't trace HF models.
    """
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, module in model.named_modules():
            children = list(module.children())
            if len(children) > 0:
                continue
            result = classify_operation(module, custom_rules=CUSTOM_RULES)
            results.append(
                (
                    name,
                    module,
                    result.classification.value,
                    result.op_type,
                )
            )
    return results


def get_blocks(model: nn.Module, dot_path: str) -> nn.ModuleList:
    """Navigate a dot-separated path to get a module list."""
    obj = model
    for attr in dot_path.split("."):
        obj = getattr(obj, attr)
    return obj


def load_model(config: ModelConfig) -> tuple[nn.Module, object]:
    """Load model and tokenizer for a given config."""
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.hf_id)

    if config.arch_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(config.hf_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    elif config.arch_type == "masked_lm":
        model = AutoModelForMaskedLM.from_pretrained(config.hf_id)
    elif config.arch_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(config.hf_id)
    else:
        raise ValueError(f"Unknown arch_type: {config.arch_type}")

    return model.to(DEVICE), tokenizer


def load_wikitext(split: str = "test") -> str:
    """Load WikiText-2 text for a given split."""
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    return "\n\n".join(dataset["text"])


# ---------------------------------------------------------------------------
# Section A: Architecture Analysis (generic for all models)
# ---------------------------------------------------------------------------


def run_classification(
    model: nn.Module,
    config: ModelConfig,
) -> ClassificationResults:
    """Classify all leaf modules as S/K/GRAY."""
    print("\n  SECTION A: Architecture Analysis — S/K Classification")

    leaves = classify_leaf_modules(model)

    s_count = sum(1 for _, _, c, _ in leaves if c == "S")
    k_count = sum(1 for _, _, c, _ in leaves if c == "K")
    gray_count = sum(1 for _, _, c, _ in leaves if c == "GRAY")
    total = len(leaves)

    print(f"    Total leaf modules: {total}")
    print(f"    S-type: {s_count} ({100 * s_count / total:.1f}%)")
    print(f"    K-type: {k_count} ({100 * k_count / total:.1f}%)")
    print(f"    GRAY:   {gray_count} ({100 * gray_count / total:.1f}%)")

    # Per-block analysis
    per_block: list[BlockClassification] = []

    for section_name, dot_path in config.block_paths:
        blocks = get_blocks(model, dot_path)

        for block_idx, _block in enumerate(blocks):
            # Count S/K/GRAY among this block's leaves
            block_prefix = f"{dot_path}.{block_idx}."
            block_leaves = [(n, m, c, t) for n, m, c, t in leaves if n.startswith(block_prefix)]
            if not block_leaves:
                continue

            bs = sum(1 for _, _, c, _ in block_leaves if c == "S")
            bk = sum(1 for _, _, c, _ in block_leaves if c == "K")
            bg = sum(1 for _, _, c, _ in block_leaves if c == "GRAY")
            bt = len(block_leaves)

            per_block.append(
                BlockClassification(
                    block_idx=block_idx,
                    section=section_name,
                    s_count=bs,
                    k_count=bk,
                    gray_count=bg,
                    total=bt,
                    s_ratio=bs / bt if bt > 0 else 0.0,
                )
            )

    if per_block:
        print("    Per-block (first 3): ", end="")
        for b in per_block[:3]:
            print(
                f"[{b.section} {b.block_idx}: S={b.s_count} K={b.k_count} "
                f"G={b.gray_count} ratio={b.s_ratio:.2f}] ",
                end="",
            )
        print()

    return ClassificationResults(
        total_modules=total,
        s_count=s_count,
        k_count=k_count,
        gray_count=gray_count,
        s_ratio=s_count / total if total > 0 else 0.0,
        k_ratio=k_count / total if total > 0 else 0.0,
        gray_ratio=gray_count / total if total > 0 else 0.0,
        per_block=per_block,
    )


# ---------------------------------------------------------------------------
# Section B: Evaluation (model-type-specific)
# ---------------------------------------------------------------------------


def eval_causal_lm(
    model: nn.Module,
    tokenizer: object,
    text: str,
    use_provenance: bool = False,
    stride: int = 512,
    max_length: int = 1024,
) -> tuple[float, float, ProvenanceTracker | None]:
    """Compute perplexity for causal LM models with sliding window.

    Passes attention_mask to model forward to avoid NaN from positional
    encoding in models like OPT that derive position IDs from attention_mask.

    Returns (perplexity, wall_time, tracker_or_None).
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(DEVICE)
    seq_len = input_ids.size(1)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0

    tracker = None
    if use_provenance:
        tracker = ProvenanceTracker()
        tracker.attach(model)

    nlls = []
    start = time.perf_counter()

    with torch.no_grad():
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            if end - begin <= 1:
                continue
            chunk = input_ids[:, begin:end]
            attention_mask = chunk.ne(pad_token_id).long()
            target = chunk.clone()
            if begin > 0:
                target[:, : max_length - stride - 1] = -100
            outputs = model(chunk, attention_mask=attention_mask, labels=target)
            loss_val = outputs.loss.item()
            if not (torch.isnan(outputs.loss) or torch.isinf(outputs.loss)):
                nlls.append(loss_val)
            if end >= seq_len:
                break

    wall = time.perf_counter() - start
    ppl = torch.exp(torch.tensor(nlls).mean()).item() if nlls else float("nan")
    return ppl, wall, tracker


def eval_masked_lm(
    model: nn.Module,
    tokenizer: object,
    text: str,
    use_provenance: bool = False,
    window_size: int = 512,
    stride: int = 256,
    mask_prob: float = 0.15,
) -> tuple[float, float, ProvenanceTracker | None]:
    """Compute pseudo-perplexity for masked LM models.

    Masks 15% of tokens per window, computes cross-entropy loss on masked
    positions, averages across windows, and returns exp(avg_loss).

    Returns (pseudo_perplexity, wall_time, tracker_or_None).
    """
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    all_ids = encodings.input_ids[0]

    mask_token_id = tokenizer.mask_token_id

    tracker = None
    if use_provenance:
        tracker = ProvenanceTracker()
        tracker.attach(model)

    nlls = []
    start = time.perf_counter()

    with torch.no_grad():
        for begin in range(0, len(all_ids) - window_size, stride):
            chunk = all_ids[begin : begin + window_size].unsqueeze(0).to(DEVICE)

            # Create mask (15% of tokens)
            mask = torch.rand(chunk.shape, device=DEVICE) < mask_prob

            labels = chunk.clone()
            labels[~mask] = -100  # Only compute loss on masked positions

            masked_input = chunk.clone()
            masked_input[mask] = mask_token_id

            outputs = model(masked_input, labels=labels)
            loss_val = outputs.loss.item()
            if not (torch.isnan(outputs.loss) or torch.isinf(outputs.loss)):
                nlls.append(loss_val)

    wall = time.perf_counter() - start
    ppl = torch.exp(torch.tensor(nlls).mean()).item() if nlls else float("nan")
    return ppl, wall, tracker


def eval_seq2seq(
    model: nn.Module,
    tokenizer: object,
    text: str,
    use_provenance: bool = False,
    seq_len: int = 256,
    stride: int = 128,
) -> tuple[float, float, ProvenanceTracker | None]:
    """Compute perplexity for seq2seq models.

    Splits text into (encoder_input, decoder_target) pairs and computes
    cross-entropy loss on the decoder output.

    Returns (perplexity, wall_time, tracker_or_None).
    """
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    all_ids = encodings.input_ids[0]

    tracker = None
    if use_provenance:
        tracker = ProvenanceTracker()
        tracker.attach(model)

    nlls = []
    start = time.perf_counter()
    chunk_size = seq_len * 2  # Half for encoder, half for decoder

    with torch.no_grad():
        for begin in range(0, len(all_ids) - chunk_size, stride):
            chunk = all_ids[begin : begin + chunk_size]
            enc_input = chunk[:seq_len].contiguous().unsqueeze(0).to(DEVICE)
            dec_target = chunk[seq_len:].contiguous().unsqueeze(0).to(DEVICE)

            outputs = model(input_ids=enc_input, labels=dec_target)
            loss_val = outputs.loss.item()
            if not (torch.isnan(outputs.loss) or torch.isinf(outputs.loss)):
                nlls.append(loss_val)

    wall = time.perf_counter() - start
    ppl = torch.exp(torch.tensor(nlls).mean()).item() if nlls else float("nan")
    return ppl, wall, tracker


def run_evaluation(
    model: nn.Module,
    tokenizer: object,
    text: str,
    config: ModelConfig,
) -> EvalResult:
    """Run evaluation: baseline and with provenance hooks."""
    print("\n  SECTION B: Evaluation (pre-trained model on WikiText-2)")

    model.eval()

    eval_fn = {
        "causal_lm": eval_causal_lm,
        "masked_lm": eval_masked_lm,
        "seq2seq": eval_seq2seq,
    }[config.arch_type]

    metric_name = "pseudo-perplexity" if config.arch_type == "masked_lm" else "perplexity"

    # Baseline
    val_base, time_base, _ = eval_fn(model, tokenizer, text, use_provenance=False)
    print(f"    Baseline {metric_name}: {val_base:.2f} ({time_base:.1f}s)")

    # With provenance
    val_prov, time_prov, tracker = eval_fn(model, tokenizer, text, use_provenance=True)
    overhead = time_prov - time_base
    print(f"    With provenance: {val_prov:.2f} ({time_prov:.1f}s, overhead: {overhead:.1f}s)")

    s_phases = 0
    k_boundaries = 0
    erasure_budget = 0.0
    if tracker is not None:
        graph = tracker.finalize()
        s_phases = len(graph.s_phases)
        k_boundaries = len(graph.k_boundaries)
        erasure_budget = graph.erasure_budget
        tracker.detach()
        print(
            f"    Provenance: S-phases={s_phases}, "
            f"K-boundaries={k_boundaries}, "
            f"erasure_budget={erasure_budget:.3f}"
        )

    diff = abs(val_base - val_prov)
    if diff < 0.5:
        print(f"    PASS: Provenance hooks do not affect computation (diff={diff:.4f})")
    else:
        print(f"    NOTE: {metric_name} diff={diff:.4f} (numerical noise expected)")

    return EvalResult(
        metric_name=metric_name,
        value=val_base,
        value_with_provenance=val_prov,
        baseline_time_s=time_base,
        provenance_time_s=time_prov,
        provenance_overhead_s=overhead,
        provenance_s_phases=s_phases,
        provenance_k_boundaries=k_boundaries,
        provenance_erasure_budget=erasure_budget,
    )


# ---------------------------------------------------------------------------
# Section C: Fine-tuning (model-type-specific forward step)
# ---------------------------------------------------------------------------


def prepare_batches(
    tokenizer: object,
    config: ModelConfig,
) -> list[torch.Tensor]:
    """Prepare fine-tuning batches from WikiText-2 train split."""
    train_text = load_wikitext("train")

    max_tokens = config.seq_len * config.batch_size * FINETUNE_STEPS * 2
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
        if len(batches) >= FINETUNE_STEPS:
            break

    return batches


def forward_step(
    model: nn.Module,
    batch: torch.Tensor,
    config: ModelConfig,
    mask_token_id: int = 103,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Single forward step returning loss, handling all model types.

    Passes attention_mask for causal LM models to prevent NaN from
    positional encoding in OPT (which derives position IDs via cumsum
    on attention_mask) and Pythia (rotary embeddings).

    For causal LM: model(batch, attention_mask=..., labels=batch)
    For masked LM: mask 15% of tokens, compute loss on masked positions
    For seq2seq: split batch into encoder input / decoder target halves
    """
    if config.arch_type == "causal_lm":
        attention_mask = batch.ne(pad_token_id).long()
        outputs = model(batch, attention_mask=attention_mask, labels=batch)
        return outputs.loss

    if config.arch_type == "masked_lm":
        mask = torch.rand(batch.shape, device=batch.device) < 0.15
        labels = batch.clone()
        labels[~mask] = -100
        masked_input = batch.clone()
        masked_input[mask] = mask_token_id
        outputs = model(masked_input, labels=labels)
        return outputs.loss

    if config.arch_type == "seq2seq":
        mid = batch.size(1) // 2
        enc_input = batch[:, :mid].contiguous()
        dec_labels = batch[:, mid:].contiguous()
        outputs = model(input_ids=enc_input, labels=dec_labels)
        return outputs.loss

    raise ValueError(f"Unknown arch_type: {config.arch_type}")


def eval_after_finetune(
    model: nn.Module,
    tokenizer: object,
    text: str,
    config: ModelConfig,
) -> float:
    """Quick evaluation after fine-tuning. Returns the metric value."""
    model.eval()
    eval_fn = {
        "causal_lm": eval_causal_lm,
        "masked_lm": eval_masked_lm,
        "seq2seq": eval_seq2seq,
    }[config.arch_type]
    val, _, _ = eval_fn(model, tokenizer, text, use_provenance=False)
    return val


def finetune_one_mode(
    config: ModelConfig,
    tokenizer: object,
    batches: list[torch.Tensor],
    test_text: str,
    mode: str,
    use_checkpointing: bool = False,
) -> FinetuneResult:
    """Run fine-tuning for one mode, returning results.

    Uses per-model learning rate, bf16 mixed precision for 300M+ param
    models, linear warmup (10% of steps), and attention_mask to prevent
    NaN in OPT/Pythia. Gradient checkpointing uses use_reentrant=False
    to avoid silent NaN gradients.

    Args:
        config: Model configuration.
        tokenizer: The tokenizer.
        batches: Pre-tokenized training batches.
        test_text: Text for evaluation after fine-tuning.
        mode: "conventional", "bwsk_analyzed", or "bwsk_reversible".
        use_checkpointing: Whether to enable gradient checkpointing.
    """
    model, _ = load_model(config)

    erasure_budget = 0.0
    parallelism_ratio = 0.0

    if mode in ("bwsk_analyzed", "bwsk_reversible"):
        leaves = classify_leaf_modules(model)
        s_count = sum(1 for _, _, c, _ in leaves if c == "S")
        k_count = sum(1 for _, _, c, _ in leaves if c == "K")
        total = len(leaves)
        erasure_budget = k_count / total if total > 0 else 0.0
        parallelism_ratio = s_count / total if total > 0 else 0.0

    if use_checkpointing:
        # use_reentrant=False prevents silent NaN gradients with
        # gradient checkpointing on models that use non-standard
        # activation patterns (OPT, Pythia).
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Disable KV cache — incompatible with gradient checkpointing
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    mask_token_id = getattr(tokenizer, "mask_token_id", 103) or 103
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

    # bf16 mixed precision for 300M+ models to reduce numerical errors
    # and memory usage while maintaining fp32-equivalent dynamic range.
    use_amp = DEVICE.type == "cuda" and config.params_m >= 300

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.finetune_lr)

    # Linear warmup: 10% of steps (30 out of 300) prevents gradient
    # explosions in the first steps that cause NaN loss values.
    warmup_steps = max(1, len(batches) // 10)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    reset_memory()
    loss_curve = []
    nan_count = 0
    start = time.perf_counter()

    for batch in tqdm(batches, desc=f"  {mode}", leave=False):
        batch = batch.to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = forward_step(model, batch, config, mask_token_id, pad_token_id)
        if torch.isnan(loss) or torch.isinf(loss):
            # Skip steps with NaN/inf loss (numerical instability)
            nan_count += 1
            optimizer.zero_grad()
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        loss_curve.append(loss.item())

    wall = time.perf_counter() - start
    mem = peak_memory_mb()

    total_steps = len(batches)
    if nan_count > 0:
        pct = 100 * nan_count / total_steps
        level = "WARNING" if pct > 50 else "INFO"
        print(f"      [{level}] NaN/inf steps: {nan_count}/{total_steps} ({pct:.0f}%)")

    if use_checkpointing:
        model.gradient_checkpointing_disable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True

    eval_val = eval_after_finetune(model, tokenizer, test_text, config)

    del model
    reset_memory()

    return FinetuneResult(
        mode=mode,
        final_loss=loss_curve[-1] if loss_curve else 0.0,
        eval_metric=eval_val,
        wall_time_s=wall,
        peak_memory_mb=mem,
        loss_curve=loss_curve,
        nan_count=nan_count,
        erasure_budget=erasure_budget,
        parallelism_ratio=parallelism_ratio,
    )


def run_finetuning(
    config: ModelConfig,
    tokenizer: object,
    batches: list[torch.Tensor],
    test_text: str,
) -> tuple[FinetuneResult, FinetuneResult, FinetuneResult]:
    """Run all three fine-tuning modes."""
    print(f"\n  SECTION C: Fine-tuning Comparison ({FINETUNE_STEPS} steps)")
    print(f"    LR: {config.finetune_lr}, AMP: {config.params_m >= 300}")

    mode_labels = {
        "conventional": "Mode 1: Conventional",
        "bwsk_analyzed": "Mode 2: BWSK-analyzed",
        "bwsk_reversible": ("Mode 3: BWSK-reversible (gradient checkpointing)"),
    }

    results = []
    for mode, ckpt in [
        ("conventional", False),
        ("bwsk_analyzed", False),
        ("bwsk_reversible", True),
    ]:
        print(f"    {mode_labels[mode]}...")
        r = finetune_one_mode(config, tokenizer, batches, test_text, mode, ckpt)
        metric_name = "pseudo-PPL" if config.arch_type == "masked_lm" else "PPL"
        print(
            f"      Loss: {r.final_loss:.4f}, "
            f"{metric_name}: {r.eval_metric:.2f}, "
            f"Mem: {r.peak_memory_mb:.0f}MB, "
            f"Time: {r.wall_time_s:.1f}s"
        )
        if mode != "conventional":
            print(
                f"      Erasure budget: {r.erasure_budget:.3f}, "
                f"Parallelism ratio: {r.parallelism_ratio:.3f}"
            )
        results.append(r)

    r1, r2, r3 = results
    if r1.peak_memory_mb > 0 and r3.peak_memory_mb > 0:
        savings = r1.peak_memory_mb - r3.peak_memory_mb
        pct = 100 * savings / r1.peak_memory_mb if r1.peak_memory_mb > 0 else 0.0
        print(f"    Memory savings (reversible vs conventional): {savings:.0f}MB ({pct:.1f}%)")

    return r1, r2, r3


# ---------------------------------------------------------------------------
# Section E: CALM Analysis
# ---------------------------------------------------------------------------


def run_calm_analysis(
    model: nn.Module,
    config: ModelConfig,
) -> tuple[list[CALMBlockResult], list, list]:
    """Run CALM analysis on transformer blocks."""
    print("\n  SECTION E: CALM Analysis")

    block_results: list[CALMBlockResult] = []

    for section_name, dot_path in config.block_paths:
        blocks = get_blocks(model, dot_path)
        for block_idx, block in enumerate(blocks):
            report = analyze_calm(block)
            block_results.append(
                CALMBlockResult(
                    block_idx=block_idx,
                    section=section_name,
                    total_children=report.total_modules,
                    monotone_count=report.monotone_count,
                    sync_count=report.sync_count,
                    parallelism_ratio=report.parallelism_ratio,
                    num_sync_barriers=report.num_sync_barriers,
                )
            )

    if block_results:
        # Print first few
        print(f"    Blocks analyzed: {len(block_results)}")
        for br in block_results[:3]:
            print(
                f"    [{br.section} {br.block_idx}] "
                f"children={br.total_children}, "
                f"par_ratio={br.parallelism_ratio:.2f}, "
                f"barriers={br.num_sync_barriers}"
            )
        if len(block_results) > 3:
            print(f"    ... and {len(block_results) - 3} more")

    # Distribution partitioning on first block
    first_path = config.block_paths[0][1]
    first_blocks = get_blocks(model, first_path)
    p2 = partition_for_distribution(first_blocks[0], num_devices=2)
    p4 = partition_for_distribution(first_blocks[0], num_devices=4)
    print(f"    Partition (block 0, 2 devices): {p2}")
    print(f"    Partition (block 0, 4 devices): {p4}")

    return block_results, p2, p4


# ---------------------------------------------------------------------------
# Per-model benchmark runner
# ---------------------------------------------------------------------------


def benchmark_model(config: ModelConfig, test_text: str) -> ModelResults:
    """Run the full 6-section benchmark for one model."""
    print("\n" + "=" * 70)
    print(f"BENCHMARKING: {config.name} ({config.params_m}M params, {config.arch_type})")
    print(f"  HF ID: {config.hf_id}")
    print(f"  Batch size: {config.batch_size}, Seq len: {config.seq_len}")
    print("=" * 70)

    results = ModelResults(
        model_name=config.name,
        hf_id=config.hf_id,
        arch_type=config.arch_type,
        params_m=config.params_m,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
    )
    results.device = torch.cuda.get_device_name() if torch.cuda.is_available() else str(DEVICE)
    results.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Load model and tokenizer
    print("\n  Loading model...")
    model, tokenizer = load_model(config)

    # Section A: Classification
    results.classification = run_classification(model, config)

    # Section B: Evaluation
    results.evaluation = run_evaluation(model, tokenizer, test_text, config)

    # Section E: CALM (do before freeing model)
    calm_blocks, p2, p4 = run_calm_analysis(model, config)
    results.calm_blocks = calm_blocks
    results.calm_partition_2 = p2
    results.calm_partition_4 = p4

    # Free model for fine-tuning (each mode loads fresh)
    del model
    reset_memory()

    # Prepare fine-tuning data
    print("\n  Preparing fine-tuning batches...")
    batches = prepare_batches(tokenizer, config)
    print(f"    Prepared {len(batches)} batches")

    # Section C: Fine-tuning (also covers Section D: Memory)
    r1, r2, r3 = run_finetuning(config, tokenizer, batches, test_text)
    results.finetune_conventional = r1
    results.finetune_bwsk_analyzed = r2
    results.finetune_bwsk_reversible = r3

    # Section D: Memory summary
    print("\n  SECTION D: Memory Profiling")
    print(f"    Conventional:    {r1.peak_memory_mb:.0f} MB")
    print(f"    BWSK-Analyzed:   {r2.peak_memory_mb:.0f} MB")
    print(f"    BWSK-Reversible: {r3.peak_memory_mb:.0f} MB")

    del batches
    reset_memory()

    print(f"\n  {config.name} benchmark complete.")
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_model_section(r: ModelResults) -> list[str]:
    """Generate a report section for one model."""
    lines: list[str] = []
    lines.append(f"## {r.model_name} ({r.params_m}M, {r.arch_type})")
    lines.append("")
    lines.append(f"**HuggingFace ID**: `{r.hf_id}`")
    lines.append(f"**Batch size**: {r.batch_size}, **Seq len**: {r.seq_len}")
    lines.append("")

    # Classification
    if r.classification:
        c = r.classification
        lines.append("### A. S/K Classification")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total leaf modules | {c.total_modules} |")
        lines.append(f"| S-type | {c.s_count} ({100 * c.s_ratio:.1f}%) |")
        lines.append(f"| K-type | {c.k_count} ({100 * c.k_ratio:.1f}%) |")
        lines.append(f"| GRAY | {c.gray_count} ({100 * c.gray_ratio:.1f}%) |")
        lines.append("")
        if c.per_block:
            lines.append("| Section | Block | S | K | GRAY | S-ratio |")
            lines.append("|---------|-------|---|---|------|---------|")
            for b in c.per_block:
                lines.append(
                    f"| {b.section} | {b.block_idx} | "
                    f"{b.s_count} | "
                    f"{b.k_count} | {b.gray_count} | "
                    f"{b.s_ratio:.2f} |"
                )
            lines.append("")

    # Evaluation
    if r.evaluation:
        e = r.evaluation
        lines.append(f"### B. Evaluation ({e.metric_name})")
        lines.append("")
        lines.append("| Metric | Baseline | With Provenance |")
        lines.append("|--------|----------|-----------------|")
        lines.append(f"| {e.metric_name} | {e.value:.2f} | {e.value_with_provenance:.2f} |")
        lines.append(f"| Wall time (s) | {e.baseline_time_s:.1f} | {e.provenance_time_s:.1f} |")
        lines.append("")
        lines.append(
            f"Provenance overhead: {e.provenance_overhead_s:.1f}s | "
            f"S-phases: {e.provenance_s_phases} | "
            f"K-boundaries: {e.provenance_k_boundaries} | "
            f"Erasure budget: {e.provenance_erasure_budget:.3f}"
        )
        lines.append("")

    # Fine-tuning
    r1 = r.finetune_conventional
    r2 = r.finetune_bwsk_analyzed
    r3 = r.finetune_bwsk_reversible
    if r1 and r2 and r3:
        metric = r.evaluation.metric_name if r.evaluation else "perplexity"
        lines.append(f"### C/D. Fine-tuning & Memory ({FINETUNE_STEPS} steps)")
        lines.append("")
        lines.append("| Metric | Conventional | BWSK-Analyzed | BWSK-Reversible |")
        lines.append("|--------|-------------|---------------|-----------------|")
        lines.append(
            f"| Final loss | {r1.final_loss:.4f} | {r2.final_loss:.4f} | {r3.final_loss:.4f} |"
        )
        lines.append(
            f"| {metric} | {r1.eval_metric:.2f} | {r2.eval_metric:.2f} | {r3.eval_metric:.2f} |"
        )
        lines.append(
            f"| Wall time (s) | {r1.wall_time_s:.1f} | "
            f"{r2.wall_time_s:.1f} "
            f"| {r3.wall_time_s:.1f} |"
        )
        lines.append(
            f"| Peak memory (MB) | {r1.peak_memory_mb:.0f} | "
            f"{r2.peak_memory_mb:.0f} | {r3.peak_memory_mb:.0f} |"
        )
        lines.append(f"| Erasure budget | — | {r2.erasure_budget:.3f} | {r3.erasure_budget:.3f} |")
        lines.append(
            f"| Parallelism ratio | — | {r2.parallelism_ratio:.3f} | {r3.parallelism_ratio:.3f} |"
        )
        lines.append(f"| NaN steps | {r1.nan_count} | {r2.nan_count} | {r3.nan_count} |")
        lines.append("")
        if r1.peak_memory_mb > 0:
            savings = r1.peak_memory_mb - r3.peak_memory_mb
            pct = 100 * savings / r1.peak_memory_mb
            lines.append(f"**Memory savings**: {savings:.0f} MB ({pct:.1f}% reduction)")
            lines.append("")

    # CALM
    if r.calm_blocks:
        lines.append("### E. CALM Analysis")
        lines.append("")
        lines.append("| Section | Block | Children | Monotone | Sync | Par. Ratio | Barriers |")
        lines.append("|---------|-------|----------|----------|------|------------|----------|")
        for br in r.calm_blocks:
            lines.append(
                f"| {br.section} | {br.block_idx} | "
                f"{br.total_children} | "
                f"{br.monotone_count} | {br.sync_count} | "
                f"{br.parallelism_ratio:.2f} | "
                f"{br.num_sync_barriers} |"
            )
        lines.append("")
        if r.calm_partition_2:
            lines.append(f"Partition (2 devices): {r.calm_partition_2}")
        if r.calm_partition_4:
            lines.append(f"Partition (4 devices): {r.calm_partition_4}")
        lines.append("")

    return lines


def generate_cross_model_comparison(
    all_results: list[ModelResults],
) -> list[str]:
    """Generate cross-model comparison tables."""
    lines: list[str] = []
    lines.append("## Cross-Model Comparison")
    lines.append("")

    # S/K Classification comparison
    lines.append("### S/K Classification Across Architectures")
    lines.append("")
    lines.append("| Model | Params | Type | Leaves | S% | K% | GRAY% |")
    lines.append("|-------|--------|------|--------|----|----|-------|")
    for r in all_results:
        if r.classification:
            c = r.classification
            lines.append(
                f"| {r.model_name} | {r.params_m}M | "
                f"{r.arch_type} | "
                f"{c.total_modules} | {100 * c.s_ratio:.1f} | "
                f"{100 * c.k_ratio:.1f} | "
                f"{100 * c.gray_ratio:.1f} |"
            )
    lines.append("")

    # Memory comparison
    lines.append("### Memory Savings from BWSK-Justified Gradient Checkpointing")
    lines.append("")
    lines.append(
        "| Model | Conventional (MB) | BWSK-Reversible (MB) | Savings (MB) | Savings (%) |"
    )
    lines.append("|-------|-------------------|---------------------|-------------|------------|")
    for r in all_results:
        r1 = r.finetune_conventional
        r3 = r.finetune_bwsk_reversible
        if r1 and r3 and r1.peak_memory_mb > 0:
            savings = r1.peak_memory_mb - r3.peak_memory_mb
            pct = 100 * savings / r1.peak_memory_mb
            lines.append(
                f"| {r.model_name} | {r1.peak_memory_mb:.0f} | "
                f"{r3.peak_memory_mb:.0f} | "
                f"{savings:.0f} | {pct:.1f} |"
            )
    lines.append("")

    # Training quality comparison
    lines.append("### Training Quality (Conventional vs BWSK-Reversible)")
    lines.append("")
    lines.append("| Model | Conv. Loss | Rev. Loss | Conv. Eval | Rev. Eval | Metric |")
    lines.append("|-------|-----------|-----------|-----------|----------|--------|")
    for r in all_results:
        r1 = r.finetune_conventional
        r3 = r.finetune_bwsk_reversible
        metric = r.evaluation.metric_name if r.evaluation else "perplexity"
        if r1 and r3:
            lines.append(
                f"| {r.model_name} | {r1.final_loss:.4f} | "
                f"{r3.final_loss:.4f} | "
                f"{r1.eval_metric:.2f} | {r3.eval_metric:.2f} "
                f"| {metric} |"
            )
    lines.append("")

    # CALM comparison
    lines.append("### CALM Parallelism Across Architectures")
    lines.append("")
    lines.append("| Model | Avg Par. Ratio | Avg Barriers/Block | Total Blocks |")
    lines.append("|-------|---------------|--------------------|--------------| ")
    for r in all_results:
        if r.calm_blocks:
            avg_par = sum(b.parallelism_ratio for b in r.calm_blocks) / len(r.calm_blocks)
            avg_bar = sum(b.num_sync_barriers for b in r.calm_blocks) / len(r.calm_blocks)
            lines.append(
                f"| {r.model_name} | {avg_par:.2f} | {avg_bar:.1f} | {len(r.calm_blocks)} |"
            )
    lines.append("")

    # Provenance overhead comparison
    lines.append("### Provenance Hook Overhead")
    lines.append("")
    lines.append("| Model | Baseline (s) | With Provenance (s) | Overhead (s) | Eval Match |")
    lines.append("|-------|-------------|--------------------|--------------|-----------| ")
    for r in all_results:
        if r.evaluation:
            e = r.evaluation
            diff = abs(e.value - e.value_with_provenance)
            match = "Yes" if diff < 0.5 else f"diff={diff:.2f}"
            lines.append(
                f"| {r.model_name} | {e.baseline_time_s:.1f} | "
                f"{e.provenance_time_s:.1f} | "
                f"{e.provenance_overhead_s:.1f} | {match} |"
            )
    lines.append("")

    return lines


def generate_full_report(all_results: list[ModelResults]) -> str:
    """Generate the complete multi-model benchmark report."""
    lines: list[str] = []
    lines.append("# Multi-Model Benchmark Report: BWSK vs Conventional PyTorch")
    lines.append("")

    device = all_results[0].device if all_results else "unknown"
    timestamp = all_results[0].timestamp if all_results else ""
    lines.append(f"**Device**: {device}")
    lines.append(f"**Generated**: {timestamp}")
    lines.append(f"**Models benchmarked**: {len(all_results)}")
    lines.append(f"**Fine-tuning steps**: {FINETUNE_STEPS}")
    lines.append("")
    lines.append(
        "> This report is auto-generated by "
        "`scripts/multi_model_benchmark.py`. "
        "Do not edit manually."
    )
    lines.append("")

    # Cross-model comparison first (the main value)
    lines.append("---")
    lines.append("")
    lines.extend(generate_cross_model_comparison(all_results))

    # Per-model details
    lines.append("---")
    lines.append("")
    lines.append("# Per-Model Details")
    lines.append("")

    for r in all_results:
        lines.append("---")
        lines.append("")
        lines.extend(generate_model_section(r))

    lines.append("---")
    lines.append("")
    lines.append("*Generated by `scripts/multi_model_benchmark.py` — BWSK Combinator AI Framework*")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full multi-model benchmark."""
    print("=" * 70)
    print("Multi-Model Benchmark: BWSK vs Conventional PyTorch")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Models: {len(MODELS)}")
    for cfg in MODELS:
        print(f"  - {cfg.name} ({cfg.params_m}M, {cfg.arch_type})")

    # Load dataset once
    print("\nLoading WikiText-2 test set...")
    test_text = load_wikitext("test")

    all_results: list[ModelResults] = []

    for i, config in enumerate(MODELS):
        print(f"\n{'#' * 70}")
        print(f"# MODEL {i + 1}/{len(MODELS)}: {config.name}")
        print(f"{'#' * 70}")

        try:
            result = benchmark_model(config, test_text)
            all_results.append(result)

            # Save per-model JSON
            json_path = (
                RESULTS_DIR / f"benchmark_"
                f"{config.name.lower().replace(' ', '_').replace('-', '_')}"
                f".json"
            )
            json_path.write_text(json.dumps(asdict(result), indent=2, default=str))
            print(f"  Saved: {json_path}")

        except Exception as e:
            print(f"\n  ERROR benchmarking {config.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Generate combined report
    if all_results:
        print("\n" + "=" * 70)
        print("Generating combined report...")
        print("=" * 70)

        report = generate_full_report(all_results)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report)
        print(f"Report written to: {REPORT_PATH}")

        # Combined JSON
        combined_path = RESULTS_DIR / "multi_model_benchmark_results.json"
        combined_data = {
            "models": [asdict(r) for r in all_results],
            "device": all_results[0].device,
            "timestamp": all_results[0].timestamp,
            "finetune_steps": FINETUNE_STEPS,
        }
        combined_path.write_text(json.dumps(combined_data, indent=2, default=str))
        print(f"Combined data written to: {combined_path}")

    print(f"\nBenchmark complete! {len(all_results)}/{len(MODELS)} models succeeded.")


if __name__ == "__main__":
    main()
