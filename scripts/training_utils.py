"""Shared training infrastructure for the full training pipeline.

Provides reusable components for epoch-based training to convergence:
- ChunkedTextDataset: flat token tensor to non-overlapping windows
- CifarDataModule: CIFAR-10 with train/val/test splits and augmentation
- EarlyStopper: patience-based early stopping on validation metric
- Checkpoint helpers: save/load/cleanup for training state
- HF cache cleanup: remove downloaded model files after upload

Why this module exists: the extended_benchmark.py training loop is
step-capped (300 steps) and lacks validation, early stopping, and
checkpointing. Full convergence training needs these facilities, and
they should be reusable across experiments.
"""

from __future__ import annotations

import gc
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

# ---------------------------------------------------------------------------
# ChunkedTextDataset
# ---------------------------------------------------------------------------


class ChunkedTextDataset(Dataset):
    """Non-overlapping fixed-length windows from a flat token tensor.

    Converts a 1-D token tensor into (seq_len,) chunks suitable for
    language model training. DataLoader handles batching and per-epoch
    shuffling.

    Why non-overlapping: overlapping windows bias the loss toward
    tokens that appear in multiple windows. Non-overlapping gives each
    token equal weight and avoids redundant computation.

    Args:
        token_ids: 1-D tensor of token IDs (e.g., from tokenizer).
        seq_len: Length of each chunk window.
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int) -> None:
        if token_ids.dim() != 1:
            raise ValueError(f"token_ids must be 1-D, got {token_ids.dim()}-D")
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.num_chunks = len(token_ids) // seq_len

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.seq_len
        return self.token_ids[start : start + self.seq_len]


# ---------------------------------------------------------------------------
# CifarDataModule
# ---------------------------------------------------------------------------


class CifarDataModule:
    """CIFAR-10 data module with train/val/test splits and augmentation.

    Train: 45,000 images (90% of training set) with random crop + flip.
    Val: 5,000 images (10% of training set) without augmentation.
    Test: 10,000 images (standard test split) without augmentation.

    All images are resized to 224x224 and normalized to ImageNet stats,
    matching pretrained model expectations.

    Why 90/10 split: standard practice for validation-based early stopping
    while keeping enough training data for convergence.

    Args:
        batch_size: Batch size for all dataloaders.
        num_workers: Number of dataloader workers.
        data_root: Directory to download/cache CIFAR-10.
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 2,
        data_root: str = "/tmp/cifar10",
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None
        self._test_loader: DataLoader | None = None

    def _get_transforms(self, augment: bool = False):
        """Build image transforms pipeline.

        Augmented transforms add random crop and horizontal flip for training.
        Both pipelines resize to 224 and normalize to ImageNet stats.
        """
        from torchvision import transforms

        if augment:
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def setup(self) -> None:
        """Download and prepare datasets with splits."""
        from torchvision import datasets

        train_full = datasets.CIFAR10(
            root=self.data_root,
            train=True,
            download=True,
            transform=self._get_transforms(augment=True),
        )
        # 90/10 train/val split with fixed seed for reproducibility
        n_train = int(0.9 * len(train_full))
        n_val = len(train_full) - n_train
        train_sub, val_sub = random_split(
            train_full,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        # Val subset needs non-augmented transforms. We wrap it with a
        # dataset that applies the eval transform instead.
        val_full = datasets.CIFAR10(
            root=self.data_root,
            train=True,
            download=False,
            transform=self._get_transforms(augment=False),
        )
        val_indices = val_sub.indices
        val_dataset = Subset(val_full, val_indices)

        self._train_dataset = train_sub
        self._val_dataset = val_dataset
        self._test_dataset = datasets.CIFAR10(
            root=self.data_root,
            train=False,
            download=True,
            transform=self._get_transforms(augment=False),
        )

    @property
    def train_loader(self) -> DataLoader:
        """Training dataloader with shuffling."""
        if self._train_loader is None:
            self._train_loader = DataLoader(
                self._train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
        return self._train_loader

    @property
    def val_loader(self) -> DataLoader:
        """Validation dataloader without shuffling."""
        if self._val_loader is None:
            self._val_loader = DataLoader(
                self._val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return self._val_loader

    @property
    def test_loader(self) -> DataLoader:
        """Test dataloader without shuffling."""
        if self._test_loader is None:
            self._test_loader = DataLoader(
                self._test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return self._test_loader


# ---------------------------------------------------------------------------
# EarlyStopper
# ---------------------------------------------------------------------------


class EarlyStopper:
    """Patience-based early stopping on a validation metric.

    Tracks the best metric value seen so far and counts consecutive
    epochs without improvement. Signals to stop when patience is
    exhausted.

    Why patience-based: simple, well-understood, and sufficient for
    most training scenarios. Avoids complex schedules while preventing
    overfitting.

    Args:
        patience: Number of checks without improvement before stopping.
        min_delta: Minimum improvement to count as progress.
        mode: "min" for metrics to minimize (loss, perplexity),
              "max" for metrics to maximize (accuracy).
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float | None = None
        self.counter = 0
        self._improved = False

    def step(self, metric: float) -> bool:
        """Record a new metric value and return True if training should stop.

        Args:
            metric: The validation metric value.

        Returns:
            True if patience is exhausted and training should stop.
        """
        self._improved = False

        if self.best is None:
            self.best = metric
            self._improved = True
            return False

        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
            self._improved = True
        else:
            self.counter += 1

        return self.counter >= self.patience

    @property
    def improved(self) -> bool:
        """True if the last step() call was a new best."""
        return self._improved


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


@dataclass
class CheckpointData:
    """Metadata stored alongside a training checkpoint.

    Args:
        epoch: Current epoch number (0-indexed).
        global_step: Total training steps completed.
        best_metric: Best validation metric seen so far.
        metric_name: Name of the tracked metric (e.g., "perplexity").
        loss_curve: Training loss values per step.
        val_curve: Validation metric values per check.
    """

    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("inf")
    metric_name: str = ""
    loss_curve: list[float] = field(default_factory=list)
    val_curve: list[float] = field(default_factory=list)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    ckpt_data: CheckpointData,
    path: Path,
) -> None:
    """Save a full training checkpoint to disk.

    Saves model state_dict, optimizer state_dict, scheduler state_dict,
    and metadata (epoch, step, metrics, curves) in a single file.

    Args:
        model: The model being trained.
        optimizer: The optimizer.
        scheduler: The LR scheduler (must have state_dict()).
        ckpt_data: Metadata about training progress.
        path: File path to save the checkpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": asdict(ckpt_data),
    }
    if hasattr(scheduler, "state_dict"):
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    path: Path,
) -> CheckpointData:
    """Load a training checkpoint from disk.

    Restores model, optimizer, and scheduler state, and returns the
    metadata (epoch, step, metrics).

    Args:
        model: The model to restore weights into.
        optimizer: The optimizer to restore state into.
        scheduler: The LR scheduler to restore state into.
        path: File path of the checkpoint.

    Returns:
        CheckpointData with the saved metadata.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if hasattr(scheduler, "load_state_dict") and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    meta = state.get("metadata", {})
    return CheckpointData(
        epoch=meta.get("epoch", 0),
        global_step=meta.get("global_step", 0),
        best_metric=meta.get("best_metric", float("inf")),
        metric_name=meta.get("metric_name", ""),
        loss_curve=meta.get("loss_curve", []),
        val_curve=meta.get("val_curve", []),
    )


def save_best_model(
    model: torch.nn.Module,
    tokenizer: object | None,
    path: Path,
) -> None:
    """Save model in HuggingFace save_pretrained format.

    Falls back to torch.save for models without save_pretrained
    (e.g., torchvision models).

    Args:
        model: The trained model.
        tokenizer: Optional tokenizer to save alongside the model.
        path: Directory to save into.
    """
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(path)
    else:
        torch.save(model.state_dict(), path / "model.pt")


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------


def cleanup_checkpoints(model_slug: str, checkpoint_dir: Path) -> None:
    """Remove checkpoint directory for a model after successful upload.

    Args:
        model_slug: Model identifier (for logging).
        checkpoint_dir: Path to the checkpoint directory to remove.
    """
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print(f"  Cleaned up checkpoints: {checkpoint_dir}")


def cleanup_hf_cache(hf_model_id: str) -> None:
    """Remove a model from the HuggingFace cache directory.

    Frees disk space after the model has been fine-tuned and uploaded.
    The cache path follows HF's convention: models--{org}--{name}.

    Args:
        hf_model_id: HuggingFace model ID (e.g., "EleutherAI/pythia-70m").
    """
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    # HF cache uses -- as separator for org/model
    cache_name = "models--" + hf_model_id.replace("/", "--")
    model_cache = cache_dir / cache_name
    if model_cache.exists():
        shutil.rmtree(model_cache)
        print(f"  Cleaned HF cache: {model_cache}")


def cleanup_gpu_memory() -> None:
    """Release GPU memory via garbage collection and CUDA cache clearing."""
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Pipeline state (resume support)
# ---------------------------------------------------------------------------


@dataclass
class PipelineState:
    """Tracks pipeline progress for resume support.

    Args:
        completed: List of (model_slug, experiment, mode) tuples that
            have finished successfully.
        in_progress: Currently running (model_slug, experiment, mode)
            or None.
        failed: List of (model_slug, experiment, mode, error) tuples.
    """

    completed: list[list[str]] = field(default_factory=list)
    in_progress: list[str] | None = None
    failed: list[list[str]] = field(default_factory=list)

    def is_done(self, slug: str, experiment: str, mode: str) -> bool:
        """Check if a specific run has already completed."""
        return [slug, experiment, mode] in self.completed

    def mark_complete(self, slug: str, experiment: str, mode: str) -> None:
        """Mark a run as completed."""
        key = [slug, experiment, mode]
        if key not in self.completed:
            self.completed.append(key)
        self.in_progress = None

    def mark_in_progress(self, slug: str, experiment: str, mode: str) -> None:
        """Mark a run as in progress."""
        self.in_progress = [slug, experiment, mode]

    def mark_failed(self, slug: str, experiment: str, mode: str, error: str) -> None:
        """Mark a run as failed."""
        self.failed.append([slug, experiment, mode, error])
        self.in_progress = None

    def save(self, path: Path) -> None:
        """Save state to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> PipelineState:
        """Load state from JSON file, or return fresh state if not found."""
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        return cls(
            completed=data.get("completed", []),
            in_progress=data.get("in_progress"),
            failed=data.get("failed", []),
        )
