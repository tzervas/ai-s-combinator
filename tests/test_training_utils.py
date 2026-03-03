"""Tests for training_utils: datasets, early stopping, checkpoints, cleanup."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from training_utils import (
    CheckpointData,
    ChunkedTextDataset,
    EarlyStopper,
    PipelineState,
    cleanup_checkpoints,
    cleanup_hf_cache,
    load_checkpoint,
    save_best_model,
    save_checkpoint,
)

# ---------------------------------------------------------------------------
# ChunkedTextDataset
# ---------------------------------------------------------------------------


class TestChunkedTextDataset:
    """Tests for ChunkedTextDataset."""

    def test_basic_chunking(self):
        """Flat tensor of 100 tokens with seq_len=10 yields 10 chunks."""
        tokens = torch.arange(100)
        ds = ChunkedTextDataset(tokens, seq_len=10)
        assert len(ds) == 10
        assert ds[0].shape == (10,)
        assert torch.equal(ds[0], torch.arange(10))
        assert torch.equal(ds[9], torch.arange(90, 100))

    def test_truncates_remainder(self):
        """Tokens that don't fill a full chunk are dropped."""
        tokens = torch.arange(25)
        ds = ChunkedTextDataset(tokens, seq_len=10)
        assert len(ds) == 2  # 25 // 10 = 2, remainder 5 dropped

    def test_single_chunk(self):
        """Exact fit: one chunk."""
        tokens = torch.arange(8)
        ds = ChunkedTextDataset(tokens, seq_len=8)
        assert len(ds) == 1
        assert torch.equal(ds[0], torch.arange(8))

    def test_empty_when_too_short(self):
        """Fewer tokens than seq_len yields empty dataset."""
        tokens = torch.arange(3)
        ds = ChunkedTextDataset(tokens, seq_len=10)
        assert len(ds) == 0

    def test_rejects_2d_input(self):
        """Must be 1-D tensor."""
        with pytest.raises(ValueError, match="1-D"):
            ChunkedTextDataset(torch.zeros(2, 3), seq_len=3)

    def test_rejects_zero_seq_len(self):
        """seq_len must be positive."""
        with pytest.raises(ValueError, match="seq_len"):
            ChunkedTextDataset(torch.arange(10), seq_len=0)

    def test_chunks_are_non_overlapping(self):
        """Chunks should not share any token positions."""
        tokens = torch.arange(30)
        ds = ChunkedTextDataset(tokens, seq_len=10)
        all_tokens = torch.cat([ds[i] for i in range(len(ds))])
        assert torch.equal(all_tokens, torch.arange(30))

    def test_works_with_dataloader(self):
        """Integration: DataLoader batches chunks correctly."""
        tokens = torch.arange(40)
        ds = ChunkedTextDataset(tokens, seq_len=10)
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0].shape == (2, 10)


# ---------------------------------------------------------------------------
# EarlyStopper
# ---------------------------------------------------------------------------


class TestEarlyStopper:
    """Tests for EarlyStopper."""

    def test_min_mode_stops_on_plateau(self):
        """Stops after patience checks without improvement in min mode."""
        stopper = EarlyStopper(patience=2, mode="min")
        assert not stopper.step(1.0)  # first value, always best
        assert stopper.improved
        assert not stopper.step(0.9)  # improved
        assert stopper.improved
        assert not stopper.step(0.95)  # worse, counter=1
        assert not stopper.improved
        assert stopper.step(0.95)  # worse again, counter=2 >= patience
        assert not stopper.improved

    def test_max_mode_stops_on_plateau(self):
        """Stops after patience checks without improvement in max mode."""
        stopper = EarlyStopper(patience=2, mode="max")
        assert not stopper.step(0.5)
        assert not stopper.step(0.6)  # improved
        assert not stopper.step(0.55)  # worse, counter=1
        assert stopper.step(0.55)  # worse, counter=2

    def test_min_delta_threshold(self):
        """Small improvements below min_delta don't count."""
        stopper = EarlyStopper(patience=2, mode="min", min_delta=0.1)
        assert not stopper.step(1.0)
        assert not stopper.step(0.95)  # improved by 0.05 < min_delta
        assert not stopper.improved  # doesn't count
        assert stopper.step(0.92)  # still no real improvement

    def test_resets_counter_on_improvement(self):
        """Counter resets when a genuinely better metric appears."""
        stopper = EarlyStopper(patience=3, mode="min")
        stopper.step(1.0)
        stopper.step(1.1)  # worse, counter=1
        stopper.step(1.2)  # worse, counter=2
        stopper.step(0.5)  # improved! counter=0
        assert stopper.counter == 0
        assert stopper.best == pytest.approx(0.5)

    def test_invalid_mode_raises(self):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode"):
            EarlyStopper(mode="invalid")

    def test_best_starts_none(self):
        """Best is None before any step."""
        stopper = EarlyStopper()
        assert stopper.best is None
        assert not stopper.improved


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


class TestCheckpoints:
    """Tests for checkpoint save/load."""

    def test_round_trip(self, tmp_path):
        """Save and load produces identical model + metadata."""
        model = nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        ckpt_data = CheckpointData(
            epoch=3,
            global_step=150,
            best_metric=2.5,
            metric_name="perplexity",
            loss_curve=[3.0, 2.8, 2.6],
            val_curve=[3.1, 2.9],
        )

        path = tmp_path / "ckpt.pt"
        save_checkpoint(model, optimizer, scheduler, ckpt_data, path)
        assert path.exists()

        # Load into fresh model
        model2 = nn.Linear(4, 2)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1)
        loaded = load_checkpoint(model2, optimizer2, scheduler2, path)

        assert loaded.epoch == 3
        assert loaded.global_step == 150
        assert loaded.best_metric == pytest.approx(2.5)
        assert loaded.metric_name == "perplexity"
        assert loaded.loss_curve == [3.0, 2.8, 2.6]
        assert loaded.val_curve == [3.1, 2.9]

        # Model weights should match
        for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
            assert torch.equal(p1, p2)

    def test_save_creates_parent_dirs(self, tmp_path):
        """save_checkpoint creates intermediate directories."""
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        path = tmp_path / "deep" / "nested" / "ckpt.pt"
        save_checkpoint(model, optimizer, scheduler, CheckpointData(), path)
        assert path.exists()


class TestSaveBestModel:
    """Tests for save_best_model."""

    def test_saves_state_dict_for_plain_module(self, tmp_path):
        """Non-HF models get saved as model.pt."""
        model = nn.Linear(4, 2)
        save_best_model(model, tokenizer=None, path=tmp_path / "best")
        assert (tmp_path / "best" / "model.pt").exists()


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------


class TestCleanup:
    """Tests for cleanup functions."""

    def test_cleanup_checkpoints_removes_dir(self, tmp_path):
        """cleanup_checkpoints removes the checkpoint directory."""
        ckpt_dir = tmp_path / "checkpoints" / "pythia-70m"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "ckpt.pt").write_text("fake")
        cleanup_checkpoints("pythia-70m", ckpt_dir)
        assert not ckpt_dir.exists()

    def test_cleanup_checkpoints_noop_if_missing(self, tmp_path):
        """cleanup_checkpoints is a no-op if directory doesn't exist."""
        cleanup_checkpoints("missing", tmp_path / "nonexistent")

    def test_cleanup_hf_cache(self, tmp_path):
        """cleanup_hf_cache removes the correct cache directory."""
        cache_dir = tmp_path / ".cache" / "huggingface" / "hub"
        model_dir = cache_dir / "models--EleutherAI--pythia-70m"
        model_dir.mkdir(parents=True)
        (model_dir / "data.bin").write_text("fake")

        with patch("training_utils.Path.home", return_value=tmp_path):
            cleanup_hf_cache("EleutherAI/pythia-70m")

        assert not model_dir.exists()


# ---------------------------------------------------------------------------
# PipelineState
# ---------------------------------------------------------------------------


class TestPipelineState:
    """Tests for pipeline state persistence."""

    def test_fresh_state(self):
        """New state has empty lists."""
        state = PipelineState()
        assert not state.is_done("m", "finetune", "conventional")
        assert state.completed == []

    def test_mark_complete(self):
        """Marking complete makes is_done return True."""
        state = PipelineState()
        state.mark_complete("pythia-70m", "finetune", "conventional")
        assert state.is_done("pythia-70m", "finetune", "conventional")
        assert not state.is_done("pythia-70m", "finetune", "bwsk_analyzed")

    def test_mark_complete_idempotent(self):
        """Double mark doesn't duplicate."""
        state = PipelineState()
        state.mark_complete("m", "e", "mode")
        state.mark_complete("m", "e", "mode")
        assert len(state.completed) == 1

    def test_save_load_round_trip(self, tmp_path):
        """State survives JSON serialization."""
        state = PipelineState()
        state.mark_complete("pythia-70m", "finetune", "conventional")
        state.mark_failed("bert", "scratch", "bwsk_analyzed", "OOM")

        path = tmp_path / "state.json"
        state.save(path)
        loaded = PipelineState.load(path)

        assert loaded.is_done("pythia-70m", "finetune", "conventional")
        assert len(loaded.failed) == 1
        assert loaded.failed[0][3] == "OOM"

    def test_load_missing_file_returns_fresh(self, tmp_path):
        """Loading from non-existent file returns empty state."""
        state = PipelineState.load(tmp_path / "nonexistent.json")
        assert state.completed == []

    def test_mark_in_progress(self):
        """In-progress tracking."""
        state = PipelineState()
        state.mark_in_progress("m", "finetune", "conv")
        assert state.in_progress == ["m", "finetune", "conv"]
        state.mark_complete("m", "finetune", "conv")
        assert state.in_progress is None
