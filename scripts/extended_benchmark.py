"""Extended benchmark: 17 models across scale sweep + architecture diversity.

Extends the multi-model benchmark to cover:
  - Scale sweep: 10 transformer models (70M→2.7B) on WikiText-2
  - Architecture diversity: 7 non-transformer models (CNN, ViT, SSM, MoE)
    on CIFAR-10 or WikiText-2

Each model gets the same 6-section analysis:
  A. Architecture Analysis — S/K classification
  B. Evaluation — perplexity / accuracy (model-type-specific)
  C. Fine-tuning Comparison — 3 modes (conventional, BWSK-analyzed, BWSK-reversible)
  D. Memory Profiling — peak GPU memory per training mode
  E. CALM Analysis — per-block parallelism and distribution partitioning
  F. Quality Summary — side-by-side comparison table

Usage:
    uv run python scripts/extended_benchmark.py
    uv run python scripts/extended_benchmark.py --dry-run
    uv run python scripts/extended_benchmark.py --models pythia-70m,resnet50
"""

from __future__ import annotations

import argparse
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

FINETUNE_STEPS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path(__file__).resolve().parent
REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "EXTENDED_BENCHMARK_REPORT.md"

# Custom classification rules for HuggingFace-specific module types.
# Grouped by architecture family for clarity.
TRANSFORMER_RULES: dict[str, OpClass] = {
    "Conv1D": OpClass.S,  # HF GPT-2 custom Conv1D (linear projection)
    "NewGELUActivation": OpClass.K,
    "GELUActivation": OpClass.K,
    "FastGELUActivation": OpClass.K,
    "T5LayerNorm": OpClass.S,  # RMSNorm variant, invertible
    "OPTLearnedPositionalEmbedding": OpClass.S,
    "RotaryEmbedding": OpClass.S,
    "GPTNeoXRotaryEmbedding": OpClass.S,
    "PhiRotaryEmbedding": OpClass.S,  # Phi-2 rotary
}

VIT_RULES: dict[str, OpClass] = {
    "ViTSelfAttention": OpClass.GRAY,
    "ViTAttention": OpClass.GRAY,
}

MAMBA_RULES: dict[str, OpClass] = {
    "MambaRMSNorm": OpClass.S,  # RMSNorm variant, invertible given scale
    "MambaMixer": OpClass.GRAY,  # Selective scan: gating (K) + linear recurrence (S)
    "MambaCache": OpClass.S,  # State cache, no info loss
}

MOE_RULES: dict[str, OpClass] = {
    "SwitchTransformersTop1Router": OpClass.K,  # argmax → maximal erasure
    "SwitchTransformersSparseMLP": OpClass.GRAY,
    "T5DenseActDense": OpClass.GRAY,
    "T5LayerNorm": OpClass.S,
}


def get_custom_rules(arch_family: str) -> dict[str, OpClass]:
    """Return merged custom classification rules for a given architecture family.

    Each family starts with the base transformer rules and adds family-specific
    rules on top. CNN models use no custom rules because Conv2d, MaxPool2d,
    ReLU, and BatchNorm2d are all in the default BWSK database.
    """
    base = dict(TRANSFORMER_RULES)
    if arch_family == "vit":
        base.update(VIT_RULES)
    elif arch_family == "ssm":
        base.update(MAMBA_RULES)
    elif arch_family == "moe":
        base.update(MOE_RULES)
    elif arch_family == "cnn":
        return {}  # All CNN ops are in the default DB
    return base


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


@dataclass
class ExtendedModelConfig:
    """Configuration for one benchmark model.

    Extends ModelConfig with source (huggingface/torchvision), architecture
    family, dataset, and custom classification rules. Per-model learning rates
    prevent gradient explosions in larger models.
    """

    name: str
    slug: str  # URL-safe identifier
    source: str  # "huggingface" or "torchvision"
    arch_family: str  # "transformer", "cnn", "vit", "ssm", "moe"
    arch_type: str  # "causal_lm", "masked_lm", "seq2seq", "image_cls", "ssm_lm"
    hf_id: str  # HuggingFace model ID or torchvision model name
    params_m: int  # Approximate parameter count in millions
    batch_size: int
    seq_len: int  # Sequence length for text or image size for vision
    dataset: str  # "wikitext" or "cifar10"
    block_paths: list[tuple[str, str]]
    finetune_lr: float = 5e-5
    finetune_only_classification: bool = False  # Skip finetune for very large models


# --- Scale Sweep: 10 transformer models (70M → 2.7B) ---

SCALE_SWEEP_MODELS: list[ExtendedModelConfig] = [
    ExtendedModelConfig(
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
        block_paths=[("decoder", "gpt_neox.layers")],
        finetune_lr=5e-5,
    ),
    ExtendedModelConfig(
        name="T5-small",
        slug="t5-small",
        source="huggingface",
        arch_family="transformer",
        arch_type="seq2seq",
        hf_id="google-t5/t5-small",
        params_m=60,
        batch_size=4,
        seq_len=512,
        dataset="wikitext",
        block_paths=[
            ("encoder", "encoder.block"),
            ("decoder", "decoder.block"),
        ],
        finetune_lr=5e-5,
    ),
    ExtendedModelConfig(
        name="BERT-base",
        slug="bert-base",
        source="huggingface",
        arch_family="transformer",
        arch_type="masked_lm",
        hf_id="google-bert/bert-base-uncased",
        params_m=110,
        batch_size=4,
        seq_len=512,
        dataset="wikitext",
        block_paths=[("encoder", "bert.encoder.layer")],
        finetune_lr=5e-5,
    ),
    ExtendedModelConfig(
        name="GPT-2 Small",
        slug="gpt2-small",
        source="huggingface",
        arch_family="transformer",
        arch_type="causal_lm",
        hf_id="openai-community/gpt2",
        params_m=124,
        batch_size=4,
        seq_len=512,
        dataset="wikitext",
        block_paths=[("decoder", "transformer.h")],
        finetune_lr=5e-5,
    ),
    ExtendedModelConfig(
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
        block_paths=[("decoder", "gpt_neox.layers")],
        finetune_lr=3e-5,
    ),
    ExtendedModelConfig(
        name="OPT-350M",
        slug="opt-350m",
        source="huggingface",
        arch_family="transformer",
        arch_type="causal_lm",
        hf_id="facebook/opt-350m",
        params_m=331,
        batch_size=2,
        seq_len=512,
        dataset="wikitext",
        block_paths=[("decoder", "model.decoder.layers")],
        finetune_lr=2e-5,
    ),
    ExtendedModelConfig(
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
        block_paths=[("decoder", "transformer.h")],
        finetune_lr=5e-5,
    ),
    ExtendedModelConfig(
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
        block_paths=[("decoder", "gpt_neox.layers")],
        finetune_lr=2e-5,
    ),
    ExtendedModelConfig(
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
        block_paths=[("decoder", "gpt_neox.layers")],
        finetune_lr=1e-5,
    ),
    ExtendedModelConfig(
        name="Phi-2",
        slug="phi-2",
        source="huggingface",
        arch_family="transformer",
        arch_type="causal_lm",
        hf_id="microsoft/phi-2",
        params_m=2700,
        batch_size=1,
        seq_len=256,
        dataset="wikitext",
        block_paths=[("decoder", "model.layers")],
        finetune_lr=5e-6,
        finetune_only_classification=True,  # OOM risk on 16GB
    ),
]

# --- Architecture Diversity: 7 non-transformer models ---

ARCH_DIVERSITY_MODELS: list[ExtendedModelConfig] = [
    ExtendedModelConfig(
        name="ResNet-50",
        slug="resnet50",
        source="torchvision",
        arch_family="cnn",
        arch_type="image_cls",
        hf_id="resnet50",
        params_m=25,
        batch_size=32,
        seq_len=224,  # Image size
        dataset="cifar10",
        block_paths=[
            ("layer1", "layer1"),
            ("layer2", "layer2"),
            ("layer3", "layer3"),
            ("layer4", "layer4"),
        ],
        finetune_lr=1e-3,
    ),
    ExtendedModelConfig(
        name="EfficientNet-B0",
        slug="efficientnet-b0",
        source="torchvision",
        arch_family="cnn",
        arch_type="image_cls",
        hf_id="efficientnet_b0",
        params_m=5,
        batch_size=32,
        seq_len=224,
        dataset="cifar10",
        block_paths=[("features", "features")],
        finetune_lr=1e-3,
    ),
    ExtendedModelConfig(
        name="MobileNetV2",
        slug="mobilenetv2",
        source="torchvision",
        arch_family="cnn",
        arch_type="image_cls",
        hf_id="mobilenet_v2",
        params_m=3,
        batch_size=32,
        seq_len=224,
        dataset="cifar10",
        block_paths=[("features", "features")],
        finetune_lr=1e-3,
    ),
    ExtendedModelConfig(
        name="ViT-base",
        slug="vit-base",
        source="huggingface",
        arch_family="vit",
        arch_type="image_cls",
        hf_id="google/vit-base-patch16-224",
        params_m=86,
        batch_size=16,
        seq_len=224,
        dataset="cifar10",
        block_paths=[("encoder", "vit.encoder.layer")],
        finetune_lr=5e-5,
    ),
    ExtendedModelConfig(
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
        block_paths=[("backbone", "backbone.layers")],
        finetune_lr=3e-5,
    ),
    ExtendedModelConfig(
        name="Mamba-370M",
        slug="mamba-370m",
        source="huggingface",
        arch_family="ssm",
        arch_type="ssm_lm",
        hf_id="state-spaces/mamba-370m-hf",
        params_m=370,
        batch_size=1,
        seq_len=256,
        dataset="wikitext",
        block_paths=[("backbone", "backbone.layers")],
        finetune_lr=2e-5,
    ),
    ExtendedModelConfig(
        name="Switch-Base-8",
        slug="switch-base-8",
        source="huggingface",
        arch_family="moe",
        arch_type="seq2seq",
        hf_id="google/switch-base-8",
        params_m=220,
        batch_size=1,
        seq_len=256,
        dataset="wikitext",
        block_paths=[
            ("encoder", "encoder.block"),
            ("decoder", "decoder.block"),
        ],
        finetune_lr=3e-5,
    ),
]

ALL_MODELS = SCALE_SWEEP_MODELS + ARCH_DIVERSITY_MODELS


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------


@dataclass
class BlockClassification:
    """Classification results for a single block."""

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
    """Evaluation result."""

    metric_name: str
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
    eval_metric: float
    wall_time_s: float
    peak_memory_mb: float
    loss_curve: list[float]
    nan_count: int = 0
    erasure_budget: float = 0.0
    parallelism_ratio: float = 0.0


@dataclass
class CALMBlockResult:
    """CALM analysis for one block."""

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
    slug: str = ""
    hf_id: str = ""
    arch_family: str = ""
    arch_type: str = ""
    params_m: int = 0
    batch_size: int = 0
    seq_len: int = 0
    dataset: str = ""
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
    skipped_finetune: bool = False
    skipped_reason: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from bench_utils import peak_memory_mb, reset_memory


def classify_leaf_modules(
    model: nn.Module,
    custom_rules: dict[str, OpClass] | None = None,
) -> list[tuple[str, nn.Module, str, str]]:
    """Classify all leaf modules in a model.

    Returns list of (name, module, classification, op_type) tuples.
    Uses classify_operation on each leaf since torch.fx can't trace HF models.
    """
    results = []
    rules = custom_rules or {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, module in model.named_modules():
            children = list(module.children())
            if len(children) > 0:
                continue
            result = classify_operation(module, custom_rules=rules)
            results.append((name, module, result.classification.value, result.op_type))
    return results


def get_blocks(model: nn.Module, dot_path: str) -> nn.ModuleList:
    """Navigate a dot-separated path to get a module list."""
    obj = model
    for attr in dot_path.split("."):
        obj = getattr(obj, attr)
    return obj


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_text_model(
    config: ExtendedModelConfig,
) -> tuple[nn.Module, object]:
    """Load a HuggingFace text model and tokenizer."""
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


def _patch_mamba_ssm() -> None:
    """Patch mamba_ssm to re-export functions at paths transformers expects.

    mamba_ssm 2.x moved selective_state_update to ops.triton submodule,
    but transformers 5.x expects it at the top-level mamba_ssm namespace.
    """
    try:
        import mamba_ssm

        if not hasattr(mamba_ssm, "selective_state_update"):
            from mamba_ssm.ops.triton.selective_state_update import (
                selective_state_update,
            )

            mamba_ssm.selective_state_update = selective_state_update
    except ImportError:
        pass


def load_ssm_model(
    config: ExtendedModelConfig,
) -> tuple[nn.Module, object]:
    """Load a Mamba SSM model using the HuggingFace -hf variant.

    Uses AutoModelForCausalLM which handles MambaForCausalLM automatically
    when the model config specifies model_type='mamba'. Applies a
    compatibility patch for mamba_ssm 2.x before loading.
    """
    _patch_mamba_ssm()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.hf_id)
    return model.to(DEVICE), tokenizer


def load_vision_model_torchvision(
    config: ExtendedModelConfig,
) -> nn.Module:
    """Load a torchvision model and replace the classification head for CIFAR-10.

    Replaces the final FC layer with a 10-class head for CIFAR-10 fine-tuning.
    """
    import torchvision.models as models

    model_fn = getattr(models, config.hf_id)
    model = model_fn(weights=None)

    # Replace classification head for 10-class CIFAR-10
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 10)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            last = model.classifier[-1]
            if isinstance(last, nn.Linear):
                in_features = last.in_features
                model.classifier[-1] = nn.Linear(in_features, 10)
        elif isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, 10)

    return model.to(DEVICE)


def load_vit_model(
    config: ExtendedModelConfig,
) -> tuple[nn.Module, object]:
    """Load a ViT model for image classification from HuggingFace.

    Configures for 10-class CIFAR-10. Uses ViTForImageClassification with
    ignore_mismatched_sizes to handle the head replacement.
    """
    from transformers import ViTForImageClassification, ViTImageProcessor

    feature_extractor = ViTImageProcessor.from_pretrained(config.hf_id)
    model = ViTForImageClassification.from_pretrained(
        config.hf_id,
        num_labels=10,
        ignore_mismatched_sizes=True,
    )
    return model.to(DEVICE), feature_extractor


def load_wikitext(split: str = "test") -> str:
    """Load WikiText-2 text for a given split."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(dataset["text"])


def load_cifar10(split: str = "test"):
    """Load CIFAR-10 as a torchvision dataset with transforms.

    Returns a DataLoader with 224x224 resized images normalized to ImageNet stats.
    """
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

    ds_split = split == "test"
    dataset = datasets.CIFAR10(
        root="/tmp/cifar10",
        train=not ds_split,
        download=True,
        transform=transform,
    )
    return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)


# ---------------------------------------------------------------------------
# Section A: Classification
# ---------------------------------------------------------------------------


def run_classification(
    model: nn.Module,
    config: ExtendedModelConfig,
) -> ClassificationResults:
    """Classify all leaf modules as S/K/GRAY."""
    print("\n  SECTION A: Architecture Analysis — S/K Classification")

    rules = get_custom_rules(config.arch_family)
    leaves = classify_leaf_modules(model, custom_rules=rules)

    s_count = sum(1 for _, _, c, _ in leaves if c == "S")
    k_count = sum(1 for _, _, c, _ in leaves if c == "K")
    gray_count = sum(1 for _, _, c, _ in leaves if c == "GRAY")
    total = len(leaves)

    print(f"    Total leaf modules: {total}")
    print(f"    S-type: {s_count} ({100 * s_count / total:.1f}%)")
    print(f"    K-type: {k_count} ({100 * k_count / total:.1f}%)")
    print(f"    GRAY:   {gray_count} ({100 * gray_count / total:.1f}%)")

    per_block: list[BlockClassification] = []

    for section_name, dot_path in config.block_paths:
        try:
            blocks = get_blocks(model, dot_path)
        except AttributeError:
            print(f"    WARNING: Could not find {dot_path}")
            continue

        for block_idx, _block in enumerate(blocks):
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
    """Compute perplexity for causal LM models with sliding window."""
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
    mean_nll = torch.tensor(nlls).mean() if nlls else torch.tensor(float("nan"))
    # Clamp to avoid overflow in exp() — perplexity > exp(20) ≈ 485M is meaningless
    ppl = torch.exp(mean_nll.clamp(max=20.0)).item()
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
    """Compute pseudo-perplexity for masked LM models."""
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
            mask = torch.rand(chunk.shape, device=DEVICE) < mask_prob
            labels = chunk.clone()
            labels[~mask] = -100
            masked_input = chunk.clone()
            masked_input[mask] = mask_token_id
            outputs = model(masked_input, labels=labels)
            loss_val = outputs.loss.item()
            if not (torch.isnan(outputs.loss) or torch.isinf(outputs.loss)):
                nlls.append(loss_val)

    wall = time.perf_counter() - start
    mean_nll = torch.tensor(nlls).mean() if nlls else torch.tensor(float("nan"))
    # Clamp to avoid overflow in exp() — perplexity > exp(20) ≈ 485M is meaningless
    ppl = torch.exp(mean_nll.clamp(max=20.0)).item()
    return ppl, wall, tracker


def eval_seq2seq(
    model: nn.Module,
    tokenizer: object,
    text: str,
    use_provenance: bool = False,
    seq_len: int = 256,
    stride: int = 128,
) -> tuple[float, float, ProvenanceTracker | None]:
    """Compute perplexity for seq2seq models."""
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    all_ids = encodings.input_ids[0]

    tracker = None
    if use_provenance:
        tracker = ProvenanceTracker()
        tracker.attach(model)

    nlls = []
    start = time.perf_counter()
    chunk_size = seq_len * 2

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
    mean_nll = torch.tensor(nlls).mean() if nlls else torch.tensor(float("nan"))
    # Clamp to avoid overflow in exp() — perplexity > exp(20) ≈ 485M is meaningless
    ppl = torch.exp(mean_nll.clamp(max=20.0)).item()
    return ppl, wall, tracker


def eval_image_cls(
    model: nn.Module,
    dataloader,
    use_provenance: bool = False,
    max_batches: int = 50,
) -> tuple[float, float, ProvenanceTracker | None]:
    """Compute accuracy for image classification models.

    Returns (accuracy, wall_time, tracker_or_None).
    """
    tracker = None
    if use_provenance:
        tracker = ProvenanceTracker()
        tracker.attach(model)

    correct = 0
    total = 0
    start = time.perf_counter()

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            # Handle HuggingFace ViT output format
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    wall = time.perf_counter() - start
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, wall, tracker


def eval_ssm_lm(
    model: nn.Module,
    tokenizer: object,
    text: str,
    use_provenance: bool = False,
    stride: int = 512,
    max_length: int = 1024,
) -> tuple[float, float, ProvenanceTracker | None]:
    """Compute perplexity for SSM/Mamba models (same as causal LM)."""
    return eval_causal_lm(model, tokenizer, text, use_provenance, stride, max_length)


def run_evaluation(
    model: nn.Module,
    config: ExtendedModelConfig,
    tokenizer_or_loader=None,
    text: str = "",
) -> EvalResult:
    """Run evaluation section, dispatching by model type."""
    print("\n  SECTION B: Evaluation")

    model.eval()

    if config.arch_type == "image_cls":
        metric_name = "accuracy"
        val_base, time_base, _ = eval_image_cls(model, tokenizer_or_loader, use_provenance=False)
        val_prov, time_prov, tracker = eval_image_cls(
            model, tokenizer_or_loader, use_provenance=True
        )
    else:
        eval_fn = {
            "causal_lm": eval_causal_lm,
            "masked_lm": eval_masked_lm,
            "seq2seq": eval_seq2seq,
            "ssm_lm": eval_ssm_lm,
        }[config.arch_type]
        metric_name = "pseudo-perplexity" if config.arch_type == "masked_lm" else "perplexity"
        val_base, time_base, _ = eval_fn(model, tokenizer_or_loader, text, use_provenance=False)
        val_prov, time_prov, tracker = eval_fn(
            model, tokenizer_or_loader, text, use_provenance=True
        )

    overhead = time_prov - time_base
    print(f"    Baseline {metric_name}: {val_base:.4f} ({time_base:.1f}s)")
    print(f"    With provenance: {val_prov:.4f} ({time_prov:.1f}s, overhead: {overhead:.1f}s)")

    s_phases = 0
    k_boundaries = 0
    erasure_budget = 0.0
    if tracker is not None:
        graph = tracker.finalize()
        s_phases = len(graph.s_phases)
        k_boundaries = len(graph.k_boundaries)
        erasure_budget = graph.erasure_budget
        tracker.detach()

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
# Section C: Fine-tuning
# ---------------------------------------------------------------------------


def prepare_text_batches(
    tokenizer: object,
    config: ExtendedModelConfig,
) -> list[torch.Tensor]:
    """Prepare fine-tuning batches from WikiText-2 train split."""
    train_text = load_wikitext("train")

    max_tokens = config.seq_len * config.batch_size * FINETUNE_STEPS * 2
    # Temporarily override model_max_length to avoid premature truncation.
    # Some tokenizers (OPT, Pythia) have model_max_length=2048 which would
    # limit us to 1-2 batches. We handle chunking ourselves below.
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
        if len(batches) >= FINETUNE_STEPS:
            break

    return batches


def text_forward_step(
    model: nn.Module,
    batch: torch.Tensor,
    config: ExtendedModelConfig,
    mask_token_id: int = 103,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Single forward step for text models, returning loss."""
    if config.arch_type in ("causal_lm", "ssm_lm"):
        attention_mask = batch.ne(pad_token_id).long()
        # Mamba models don't accept attention_mask
        if config.arch_family == "ssm":
            outputs = model(batch, labels=batch)
        else:
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


def image_forward_step(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    is_hf_vit: bool = False,
) -> torch.Tensor:
    """Single forward step for image classification models."""
    if is_hf_vit:
        outputs = model(pixel_values=images, labels=labels)
        return outputs.loss

    # torchvision models return raw logits
    logits = model(images)
    return nn.functional.cross_entropy(logits, labels)


def finetune_one_mode(
    config: ExtendedModelConfig,
    mode: str,
    text_tokenizer=None,
    text_batches: list[torch.Tensor] | None = None,
    test_text: str = "",
    cifar_loader=None,
    test_loader=None,
    use_checkpointing: bool = False,
) -> FinetuneResult:
    """Run fine-tuning for one mode, returning results.

    Handles both text and vision models. Uses per-model LR, bf16 mixed
    precision for 300M+ param models, linear warmup (10% of steps), and
    gradient clipping.
    """
    # Load fresh model
    if config.arch_type == "image_cls":
        if config.source == "torchvision":
            model = load_vision_model_torchvision(config)
        else:
            model, _ = load_vit_model(config)
    elif config.arch_type == "ssm_lm":
        model, _ = load_ssm_model(config)
    else:
        model, _ = load_text_model(config)

    rules = get_custom_rules(config.arch_family)
    erasure_budget = 0.0
    parallelism_ratio = 0.0

    if mode in ("bwsk_analyzed", "bwsk_reversible"):
        leaves = classify_leaf_modules(model, custom_rules=rules)
        s_count = sum(1 for _, _, c, _ in leaves if c == "S")
        k_count = sum(1 for _, _, c, _ in leaves if c == "K")
        total = len(leaves)
        erasure_budget = k_count / total if total > 0 else 0.0
        parallelism_ratio = s_count / total if total > 0 else 0.0

    if use_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    use_amp = DEVICE.type == "cuda" and config.params_m >= 300

    # Always ensure fp32 master weights. Some HF models (OPT, Pythia) ship with
    # torch_dtype=float16, and training with fp16 params causes NaN after the
    # first optimizer step due to fp16's narrow representable range. AMP autocast
    # handles bf16 casting dynamically during forward — it needs fp32 master weights.
    model = model.float()

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.finetune_lr)

    num_steps = FINETUNE_STEPS
    warmup_steps = max(1, num_steps // 10)
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

    if config.arch_type == "image_cls":
        # Vision fine-tuning loop
        is_hf_vit = config.source == "huggingface"
        step = 0
        for images, labels in cifar_loader:
            if step >= num_steps:
                break
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = image_forward_step(model, images, labels, is_hf_vit=is_hf_vit)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                optimizer.zero_grad()
                step += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_curve.append(loss.item())
            step += 1
    else:
        # Text fine-tuning loop
        mask_token_id = getattr(text_tokenizer, "mask_token_id", 103) or 103
        pad_token_id = getattr(text_tokenizer, "pad_token_id", 0) or 0

        for batch in tqdm(text_batches[:num_steps], desc=f"  {mode}", leave=False):
            batch = batch.to(DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = text_forward_step(model, batch, config, mask_token_id, pad_token_id)
            if torch.isnan(loss) or torch.isinf(loss):
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

    if nan_count > 0:
        total_steps = num_steps
        pct = 100 * nan_count / total_steps
        level = "WARNING" if pct > 50 else "INFO"
        print(f"      [{level}] NaN/inf steps: {nan_count}/{total_steps} ({pct:.0f}%)")

    if use_checkpointing and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = True

    # Quick eval after fine-tuning
    eval_val = 0.0
    if config.arch_type == "image_cls" and test_loader is not None:
        eval_val, _, _ = eval_image_cls(model, test_loader)
    elif config.arch_type != "image_cls" and text_tokenizer is not None:
        model.eval()
        eval_fn = {
            "causal_lm": eval_causal_lm,
            "masked_lm": eval_masked_lm,
            "seq2seq": eval_seq2seq,
            "ssm_lm": eval_ssm_lm,
        }[config.arch_type]
        eval_val, _, _ = eval_fn(model, text_tokenizer, test_text, use_provenance=False)

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
    config: ExtendedModelConfig,
    text_tokenizer=None,
    text_batches: list[torch.Tensor] | None = None,
    test_text: str = "",
    cifar_loader=None,
    test_loader=None,
) -> tuple[FinetuneResult, FinetuneResult, FinetuneResult]:
    """Run all three fine-tuning modes."""
    print(f"\n  SECTION C: Fine-tuning ({FINETUNE_STEPS} steps)")
    print(f"    LR: {config.finetune_lr}, AMP: {config.params_m >= 300}")

    # Always use gradient checkpointing for SSM models (Mamba's slow_forward
    # path creates large intermediate tensors that cause OOM without it)
    always_ckpt = config.arch_family == "ssm"

    results = []
    for mode, ckpt in [
        ("conventional", always_ckpt),
        ("bwsk_analyzed", always_ckpt),
        ("bwsk_reversible", True),
    ]:
        print(f"    {mode}...")
        r = finetune_one_mode(
            config,
            mode,
            text_tokenizer=text_tokenizer,
            text_batches=text_batches,
            test_text=test_text,
            cifar_loader=cifar_loader,
            test_loader=test_loader,
            use_checkpointing=ckpt,
        )
        print(
            f"      Loss: {r.final_loss:.4f}, "
            f"Eval: {r.eval_metric:.4f}, "
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
    config: ExtendedModelConfig,
) -> tuple[list[CALMBlockResult], list, list]:
    """Run CALM analysis on model blocks."""
    print("\n  SECTION E: CALM Analysis")

    block_results: list[CALMBlockResult] = []

    for section_name, dot_path in config.block_paths:
        try:
            blocks = get_blocks(model, dot_path)
        except AttributeError:
            print(f"    WARNING: Could not find {dot_path}")
            continue

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
    p2: list = []
    p4: list = []
    if config.block_paths:
        try:
            first_path = config.block_paths[0][1]
            first_blocks = get_blocks(model, first_path)
            if len(first_blocks) > 0:
                p2 = partition_for_distribution(first_blocks[0], num_devices=2)
                p4 = partition_for_distribution(first_blocks[0], num_devices=4)
                print(f"    Partition (block 0, 2 devices): {p2}")
                print(f"    Partition (block 0, 4 devices): {p4}")
        except (AttributeError, IndexError) as e:
            print(f"    WARNING: Could not partition: {e}")

    return block_results, p2, p4


# ---------------------------------------------------------------------------
# Per-model benchmark runner
# ---------------------------------------------------------------------------


def benchmark_model(config: ExtendedModelConfig, dry_run: bool = False) -> ModelResults:
    """Run the full benchmark for one model."""
    print("\n" + "=" * 70)
    print(
        f"BENCHMARKING: {config.name} "
        f"({config.params_m}M params, {config.arch_family}/{config.arch_type})"
    )
    print(f"  Source: {config.source}, HF ID: {config.hf_id}")
    print(
        f"  Batch size: {config.batch_size}, "
        f"Seq/Img len: {config.seq_len}, Dataset: {config.dataset}"
    )
    print("=" * 70)

    results = ModelResults(
        model_name=config.name,
        slug=config.slug,
        hf_id=config.hf_id,
        arch_family=config.arch_family,
        arch_type=config.arch_type,
        params_m=config.params_m,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        dataset=config.dataset,
    )
    results.device = torch.cuda.get_device_name() if torch.cuda.is_available() else str(DEVICE)
    results.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    if dry_run:
        print("  [DRY RUN] Skipping actual computation")
        return results

    # --- Load model ---
    print("\n  Loading model...")
    text_tokenizer = None
    test_text = ""
    cifar_train_loader = None
    cifar_test_loader = None

    try:
        if config.arch_type == "image_cls":
            if config.source == "torchvision":
                model = load_vision_model_torchvision(config)
            else:
                model, text_tokenizer = load_vit_model(config)
            cifar_test_loader = load_cifar10("test")
            cifar_train_loader = load_cifar10("train")
        elif config.arch_type == "ssm_lm":
            model, text_tokenizer = load_ssm_model(config)
            test_text = load_wikitext("test")
        else:
            model, text_tokenizer = load_text_model(config)
            test_text = load_wikitext("test")
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        results.skipped_finetune = True
        results.skipped_reason = f"Load error: {e}"
        return results

    # Ensure model is in float32 — some HF models (e.g., Switch Transformer)
    # store weights in bfloat16 natively, causing dtype mismatches with float32 inputs.
    model = model.float()

    # --- Section A: Classification ---
    results.classification = run_classification(model, config)

    # --- Section B: Evaluation ---
    try:
        if config.arch_type == "image_cls":
            results.evaluation = run_evaluation(model, config, cifar_test_loader)
        else:
            results.evaluation = run_evaluation(model, config, text_tokenizer, test_text)
    except Exception as e:
        print(f"  ERROR in evaluation: {e}")

    # --- Section E: CALM ---
    try:
        calm_blocks, p2, p4 = run_calm_analysis(model, config)
        results.calm_blocks = calm_blocks
        results.calm_partition_2 = p2
        results.calm_partition_4 = p4
    except Exception as e:
        print(f"  ERROR in CALM analysis: {e}")

    # Free model for fine-tuning
    del model
    reset_memory()

    # --- Section C: Fine-tuning ---
    if config.finetune_only_classification:
        print(f"\n  SKIPPING fine-tuning for {config.name} (finetune_only_classification=True)")
        results.skipped_finetune = True
        results.skipped_reason = "Classification-only mode (OOM risk)"
    else:
        try:
            text_batches = None
            if config.arch_type != "image_cls" and text_tokenizer:
                print("\n  Preparing fine-tuning batches...")
                text_batches = prepare_text_batches(text_tokenizer, config)
                print(f"    Prepared {len(text_batches)} batches")

            r1, r2, r3 = run_finetuning(
                config,
                text_tokenizer=text_tokenizer,
                text_batches=text_batches,
                test_text=test_text,
                cifar_loader=cifar_train_loader,
                test_loader=cifar_test_loader,
            )
            results.finetune_conventional = r1
            results.finetune_bwsk_analyzed = r2
            results.finetune_bwsk_reversible = r3

            # Memory summary
            print("\n  SECTION D: Memory Profiling")
            print(f"    Conventional:    {r1.peak_memory_mb:.0f} MB")
            print(f"    BWSK-Analyzed:   {r2.peak_memory_mb:.0f} MB")
            print(f"    BWSK-Reversible: {r3.peak_memory_mb:.0f} MB")
        except Exception as e:
            print(f"  ERROR in fine-tuning: {e}")
            results.skipped_finetune = True
            results.skipped_reason = str(e)

    reset_memory()
    print(f"\n  {config.name} benchmark complete.")
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_model_section(r: ModelResults) -> list[str]:
    """Generate a markdown report section for one model."""
    lines: list[str] = []
    lines.append(f"## {r.model_name} ({r.params_m}M, {r.arch_family}/{r.arch_type})")
    lines.append("")
    lines.append(f"**Source**: {r.hf_id} | **Dataset**: {r.dataset}")
    lines.append(f"**Batch size**: {r.batch_size}, **Seq/Img len**: {r.seq_len}")
    lines.append("")

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

    if r.evaluation:
        e = r.evaluation
        lines.append(f"### B. Evaluation ({e.metric_name})")
        lines.append("")
        lines.append("| Baseline | With Provenance |")
        lines.append("|----------|-----------------|")
        lines.append(f"| {e.value:.4f} | {e.value_with_provenance:.4f} |")
        lines.append(f"\nProvenance overhead: {e.provenance_overhead_s:.1f}s")
        lines.append("")

    r1 = r.finetune_conventional
    r2 = r.finetune_bwsk_analyzed
    r3 = r.finetune_bwsk_reversible
    if r1 and r2 and r3:
        lines.append(f"### C/D. Fine-tuning & Memory ({FINETUNE_STEPS} steps)")
        lines.append("")
        lines.append("| Metric | Conventional | BWSK-Analyzed | BWSK-Reversible |")
        lines.append("|--------|-------------|---------------|-----------------|")
        lines.append(
            f"| Final loss | {r1.final_loss:.4f} | {r2.final_loss:.4f} | {r3.final_loss:.4f} |"
        )
        lines.append(
            f"| Eval metric | {r1.eval_metric:.4f} | {r2.eval_metric:.4f} | {r3.eval_metric:.4f} |"
        )
        lines.append(
            f"| Peak memory (MB) | {r1.peak_memory_mb:.0f} | "
            f"{r2.peak_memory_mb:.0f} | {r3.peak_memory_mb:.0f} |"
        )
        lines.append(
            f"| Wall time (s) | {r1.wall_time_s:.1f} | "
            f"{r2.wall_time_s:.1f} | {r3.wall_time_s:.1f} |"
        )
        lines.append(f"| NaN steps | {r1.nan_count} | {r2.nan_count} | {r3.nan_count} |")
        lines.append("")

        if r1.peak_memory_mb > 0 and r3.peak_memory_mb > 0:
            savings = r1.peak_memory_mb - r3.peak_memory_mb
            pct = 100 * savings / r1.peak_memory_mb
            lines.append(f"**Memory savings**: {savings:.0f}MB ({pct:.1f}%)")
            lines.append("")
    elif r.skipped_finetune:
        lines.append("### C/D. Fine-tuning: SKIPPED")
        lines.append(f"Reason: {r.skipped_reason}")
        lines.append("")

    if r.calm_blocks:
        avg_par = sum(b.parallelism_ratio for b in r.calm_blocks) / len(r.calm_blocks)
        lines.append("### E. CALM Analysis")
        lines.append(f"Average parallelism ratio: {avg_par:.3f}")
        lines.append("")

    lines.append("---")
    lines.append("")
    return lines


def generate_cross_model_comparison(
    all_results: list[ModelResults],
) -> list[str]:
    """Generate cross-model comparison tables."""
    lines: list[str] = []
    lines.append("## Cross-Model Comparison")
    lines.append("")

    # S/K Ratio table
    lines.append("### S/K Ratios by Model")
    lines.append("")
    lines.append("| Model | Family | Params (M) | S-ratio | K-ratio | GRAY-ratio |")
    lines.append("|-------|--------|-----------|---------|---------|------------|")
    for r in all_results:
        if r.classification:
            c = r.classification
            lines.append(
                f"| {r.model_name} | {r.arch_family} | {r.params_m} "
                f"| {c.s_ratio:.3f} | {c.k_ratio:.3f} "
                f"| {c.gray_ratio:.3f} |"
            )
    lines.append("")

    # Memory comparison
    lines.append("### Memory Comparison")
    lines.append("")
    lines.append("| Model | Conventional (MB) | Reversible (MB) | Savings (%) |")
    lines.append("|-------|-------------------|-----------------|-------------|")
    for r in all_results:
        r1 = r.finetune_conventional
        r3 = r.finetune_bwsk_reversible
        if r1 and r3 and r1.peak_memory_mb > 0:
            savings_pct = 100 * (r1.peak_memory_mb - r3.peak_memory_mb) / r1.peak_memory_mb
            lines.append(
                f"| {r.model_name} | {r1.peak_memory_mb:.0f} "
                f"| {r3.peak_memory_mb:.0f} | {savings_pct:.1f}% |"
            )
    lines.append("")

    # CALM comparison
    lines.append("### CALM Parallelism by Architecture")
    lines.append("")
    lines.append("| Model | Family | Avg Parallelism Ratio |")
    lines.append("|-------|--------|-----------------------|")
    for r in all_results:
        if r.calm_blocks:
            avg = sum(b.parallelism_ratio for b in r.calm_blocks) / len(r.calm_blocks)
            lines.append(f"| {r.model_name} | {r.arch_family} | {avg:.3f} |")
    lines.append("")

    return lines


def generate_report(all_results: list[ModelResults]) -> str:
    """Generate the full markdown report."""
    lines: list[str] = []
    lines.append("# Extended Benchmark Report: BWSK Analysis Across 17 Models")
    lines.append("")
    lines.append(
        "This report covers the extended BWSK benchmark across "
        "10 transformer models (scale sweep) and "
        "7 non-transformer models (architecture diversity)."
    )
    lines.append("")

    if all_results:
        r0 = all_results[0]
        lines.append(f"**Device**: {r0.device}")
        lines.append(f"**Date**: {r0.timestamp}")
        lines.append(f"**Fine-tuning steps**: {FINETUNE_STEPS} per mode")
    lines.append("")

    # Scale sweep section
    scale_results = [r for r in all_results if r.arch_family == "transformer"]
    if scale_results:
        lines.append("# Part 1: Scale Sweep (Transformers)")
        lines.append("")
        for r in scale_results:
            lines.extend(generate_model_section(r))

    # Architecture diversity section
    arch_results = [r for r in all_results if r.arch_family != "transformer"]
    if arch_results:
        lines.append("# Part 2: Architecture Diversity")
        lines.append("")
        for r in arch_results:
            lines.extend(generate_model_section(r))

    # Cross-model comparison
    lines.extend(generate_cross_model_comparison(all_results))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extended BWSK benchmark: 17 models")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running benchmarks",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model slugs to run (default: all)",
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
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate report from existing per-model JSON files without running benchmarks",
    )
    return parser.parse_args()


def regenerate_report_from_json() -> None:
    """Regenerate the report from existing per-model JSON files."""
    json_files = sorted(RESULTS_DIR.glob("extended_*_results.json"))
    # Exclude the combined file
    json_files = [f for f in json_files if "benchmark_results" not in f.name]

    if not json_files:
        print("ERROR: No per-model result JSON files found.")
        sys.exit(1)

    all_results: list[ModelResults] = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        # Reconstruct ClassificationResult
        cls_data = data.get("classification")
        cls_result = None
        if cls_data:
            per_block = [
                BlockClassification(**b) for b in cls_data.get("per_block", [])
            ]
            cls_result = ClassificationResults(
                total_modules=cls_data["total_modules"],
                s_count=cls_data["s_count"],
                k_count=cls_data["k_count"],
                gray_count=cls_data["gray_count"],
                s_ratio=cls_data["s_ratio"],
                k_ratio=cls_data["k_ratio"],
                gray_ratio=cls_data["gray_ratio"],
                per_block=per_block,
            )

        # Reconstruct EvalResult
        eval_data = data.get("evaluation")
        eval_result = None
        if eval_data:
            eval_result = EvalResult(**eval_data)

        # Reconstruct FinetuneResults
        def parse_ft(ft_data):
            if ft_data is None:
                return None
            return FinetuneResult(**ft_data)

        # Reconstruct CALMBlock list
        calm_blocks = None
        if data.get("calm_blocks"):
            calm_blocks = [CALMBlockResult(**b) for b in data["calm_blocks"]]

        mr = ModelResults(
            model_name=data["model_name"],
            slug=data["slug"],
            hf_id=data["hf_id"],
            arch_family=data["arch_family"],
            arch_type=data["arch_type"],
            params_m=data["params_m"],
            batch_size=data["batch_size"],
            seq_len=data["seq_len"],
            dataset=data["dataset"],
            device=data.get("device", ""),
            timestamp=data.get("timestamp", ""),
            classification=cls_result,
            evaluation=eval_result,
            finetune_conventional=parse_ft(data.get("finetune_conventional")),
            finetune_bwsk_analyzed=parse_ft(data.get("finetune_bwsk_analyzed")),
            finetune_bwsk_reversible=parse_ft(
                data.get("finetune_bwsk_reversible"),
            ),
            calm_blocks=calm_blocks,
            calm_partition_2=data.get("calm_partition_2"),
            calm_partition_4=data.get("calm_partition_4"),
            skipped_finetune=data.get("skipped_finetune", False),
            skipped_reason=data.get("skipped_reason", ""),
        )
        all_results.append(mr)
        print(f"  Loaded: {jf.name} ({mr.model_name})")

    # Save combined JSON
    combined_path = RESULTS_DIR / "extended_benchmark_results.json"
    with open(combined_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)
    print(f"\nCombined results saved to: {combined_path}")

    # Generate report
    report = generate_report(all_results)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Models in report: {len(all_results)}")


def main() -> None:
    """Run the extended benchmark suite."""
    args = parse_args()

    if args.report_only:
        print("Regenerating report from existing JSON files...")
        regenerate_report_from_json()
        return

    # Select models
    if args.models:
        slugs = {s.strip() for s in args.models.split(",")}
        models = [m for m in ALL_MODELS if m.slug in slugs]
        if not models:
            print(f"ERROR: No models found for slugs: {slugs}")
            print("Available slugs: " + ", ".join(m.slug for m in ALL_MODELS))
            sys.exit(1)
    elif args.scale_only:
        models = SCALE_SWEEP_MODELS
    elif args.arch_only:
        models = ARCH_DIVERSITY_MODELS
    else:
        models = ALL_MODELS

    print("=" * 70)
    print("EXTENDED BWSK BENCHMARK")
    print(f"Models: {len(models)}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Fine-tuning steps: {FINETUNE_STEPS}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)

    all_results: list[ModelResults] = []

    for i, config in enumerate(models):
        print(f"\n[{i + 1}/{len(models)}] {config.name} ({config.slug})")
        try:
            result = benchmark_model(config, dry_run=args.dry_run)
            all_results.append(result)

            # Save per-model JSON
            json_path = RESULTS_DIR / f"extended_{config.slug}_results.json"
            with open(json_path, "w") as f:
                json.dump(asdict(result), f, indent=2, default=str)
            print(f"  Saved: {json_path}")
        except Exception as e:
            print(f"  FATAL ERROR for {config.name}: {e}")
            import traceback

            traceback.print_exc()

    # Regenerate combined results from ALL per-model JSON files
    # (not just this run's results, to avoid clobbering when using --models)
    json_files = sorted(RESULTS_DIR.glob("extended_*_results.json"))
    json_files = [f for f in json_files if "benchmark_results" not in f.name]
    all_json = []
    for jf in json_files:
        with open(jf) as fh:
            all_json.append(json.load(fh))
    combined_path = RESULTS_DIR / "extended_benchmark_results.json"
    with open(combined_path, "w") as fh:
        json.dump(all_json, fh, indent=2, default=str)
    print(f"\nCombined results saved to: {combined_path}")

    # Generate report
    report = generate_report(all_results)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Report saved to: {REPORT_PATH}")

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print(f"Models benchmarked: {len(all_results)}")
    for r in all_results:
        s_ratio = f"S={r.classification.s_ratio:.3f}" if r.classification else "N/A"
        status = "DONE" if not r.skipped_finetune else "CLASSIFICATION ONLY"
        print(f"  {r.model_name}: {s_ratio} [{status}]")
    print("=" * 70)


if __name__ == "__main__":
    main()
