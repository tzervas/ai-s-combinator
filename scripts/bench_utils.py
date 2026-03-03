"""Shared GPU memory utilities for BWSK benchmark scripts.

Extracts the duplicated reset_memory() and peak_memory_mb() helpers that
appear in all 5 benchmark scripts, and adds VRAM estimation functions
needed by the GPU task scheduler.

Why this module exists: every benchmark script independently defined
identical memory helpers. Centralizing them here eliminates duplication
and provides a single place to add VRAM-aware scheduling utilities.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path


def reset_memory() -> None:
    """Reset GPU memory tracking.

    Triggers garbage collection and clears PyTorch's CUDA cache so that
    subsequent peak_memory_mb() calls reflect only the current workload.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def peak_memory_mb() -> float:
    """Get peak GPU memory allocated in MB since the last reset.

    Returns 0.0 if CUDA is unavailable (CPU-only mode).
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def free_vram_mb() -> float:
    """Current free VRAM in MB via torch.cuda.mem_get_info.

    Returns 0.0 if CUDA is unavailable.
    """
    try:
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def total_vram_mb() -> float:
    """Total GPU VRAM in MB.

    Returns 0.0 if CUDA is unavailable.
    """
    try:
        import torch

        if torch.cuda.is_available():
            _free, total = torch.cuda.mem_get_info()
            return total / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def estimate_vram_mb(
    params_m: float,
    batch_size: int = 4,
    seq_len: int = 512,
    mode: str = "training",
) -> float:
    """Estimate peak VRAM usage in MB for a model workload.

    Uses a two-term model calibrated against real RTX 5080 measurements:

    1. **Model state**: params_m * per_param_factor (tiered by size).
    2. **Batch/activation scaling**: proportional to batch * seq_len.

    Plus ~400 MB CUDA overhead (context, kernels, cuDNN workspace).

    Estimates are deliberately conservative (tend to under-estimate) since
    the scheduler applies a 90% safety margin. The calibration file
    (vram_calibration.json) provides exact data after the first run.

    Args:
        params_m: Model parameter count in millions.
        batch_size: Training/inference batch size.
        seq_len: Sequence length (tokens for NLP, image_size for vision).
        mode: "training" or "inference".

    Returns:
        Estimated peak VRAM in MB.

    Raises:
        ValueError: If params_m is negative.
    """
    if params_m < 0:
        raise ValueError(f"params_m must be non-negative, got {params_m}")

    # Simple two-term model calibrated against RTX 5080 measurements.
    #
    # Real VRAM depends heavily on architecture (attention pattern, number
    # of layers, intermediate sizes) so no simple formula is perfect.
    # This provides reasonable scheduling estimates; the calibration file
    # (vram_calibration.json) provides exact data after the first run.
    #
    # Term 1: Model state — weights + grads + optimizer.
    #   Scales as params_m * per_param_factor.
    #   Factor decreases with model size (larger models are more param-efficient
    #   in practice due to shared embeddings, tied weights, etc.).
    per_param_mb = 20.0 if params_m < 100 else (15.0 if params_m < 500 else 10.0)

    if mode == "training":
        model_state_mb = params_m * per_param_mb
    else:
        # Inference: ~3x less than training (no grads/optimizer)
        model_state_mb = params_m * per_param_mb / 3.0

    # Term 2: Activation/batch scaling.
    #   Larger batches and sequences increase peak memory.
    #   Normalized to batch=4, seq=512 baseline.
    batch_factor = (batch_size / 4.0) * (seq_len / 512.0)
    activation_mb = 200.0 * batch_factor

    # CUDA overhead: context init, kernel cache, cuDNN workspace
    cuda_overhead_mb = 400.0

    return model_state_mb + activation_mb + cuda_overhead_mb


# --- Calibration file support ---

CALIBRATION_PATH = Path(__file__).parent / "vram_calibration.json"


def load_calibration() -> dict[str, float]:
    """Load VRAM calibration data from disk.

    Returns a dict mapping model slugs to measured peak VRAM in MB.
    Returns empty dict if no calibration file exists.
    """
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH) as f:
            return json.load(f)
    return {}


def save_calibration(data: dict[str, float]) -> None:
    """Save VRAM calibration data to disk.

    Args:
        data: Dict mapping model slugs to measured peak VRAM in MB.
    """
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def calibrated_vram_mb(
    model_slug: str,
    params_m: float,
    batch_size: int = 4,
    seq_len: int = 512,
    mode: str = "training",
    buffer_factor: float = 1.05,
) -> float:
    """Get VRAM estimate, preferring calibrated measurement over heuristic.

    If the model has a calibrated measurement in vram_calibration.json,
    returns that value plus a small buffer (default 5%). Otherwise falls
    back to the heuristic estimate_vram_mb().

    Args:
        model_slug: URL-safe model identifier (e.g. "pythia-70m").
        params_m: Parameter count in millions (fallback).
        batch_size: Batch size (fallback).
        seq_len: Sequence length (fallback).
        mode: "training" or "inference" (fallback).
        buffer_factor: Multiplier on calibrated value (default 1.05 = 5%).

    Returns:
        Estimated peak VRAM in MB.
    """
    cal = load_calibration()
    if model_slug in cal:
        return cal[model_slug] * buffer_factor
    return estimate_vram_mb(params_m, batch_size, seq_len, mode)


def cleanup_hf_cache(hf_model_id: str) -> None:
    """Remove a model from the HuggingFace cache directory.

    Frees disk space after the model has been fine-tuned and uploaded.
    The cache path follows HF's convention: models--{org}--{name}.

    Args:
        hf_model_id: HuggingFace model ID (e.g., "EleutherAI/pythia-70m").
    """
    import shutil

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    cache_name = "models--" + hf_model_id.replace("/", "--")
    model_cache = cache_dir / cache_name
    if model_cache.exists():
        shutil.rmtree(model_cache)
        print(f"  Cleaned HF cache: {model_cache}")
