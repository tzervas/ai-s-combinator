"""Tests for VRAM estimation and GPU task scheduler.

Tests the bench_utils VRAM estimation functions and the gpu_scheduler
bin-packing logic without requiring a GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts/ to path so we can import bench_utils and gpu_scheduler
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from bench_utils import (
    calibrated_vram_mb,
    estimate_vram_mb,
    load_calibration,
    save_calibration,
)
from gpu_scheduler import GpuTask, VRAMScheduler

# ---------------------------------------------------------------------------
# VRAM estimation tests
# ---------------------------------------------------------------------------


class TestEstimateVram:
    """Test the heuristic VRAM estimation model."""

    def test_small_model_reasonable_range(self):
        """T5-small (60M) should estimate between 1000-3000 MB."""
        est = estimate_vram_mb(60, batch_size=4, seq_len=512, mode="training")
        assert 1000 < est < 3000, f"T5-small estimate {est:.0f} MB out of range"

    def test_medium_model_reasonable_range(self):
        """BERT-base (110M) should estimate between 1500-5000 MB."""
        est = estimate_vram_mb(110, batch_size=4, seq_len=512, mode="training")
        assert 1500 < est < 5000, f"BERT-base estimate {est:.0f} MB out of range"

    def test_large_model_reasonable_range(self):
        """Pythia-1B (1010M) should estimate between 8000-14000 MB."""
        est = estimate_vram_mb(1010, batch_size=2, seq_len=512, mode="training")
        assert 8000 < est < 14000, f"Pythia-1B estimate {est:.0f} MB out of range"

    def test_inference_less_than_training(self):
        """Inference should use less VRAM than training (no grads/optimizer)."""
        train = estimate_vram_mb(110, batch_size=4, seq_len=512, mode="training")
        infer = estimate_vram_mb(110, batch_size=4, seq_len=512, mode="inference")
        assert infer < train

    def test_larger_batch_uses_more_vram(self):
        """Doubling batch size should increase VRAM estimate."""
        small_batch = estimate_vram_mb(110, batch_size=2, seq_len=512)
        large_batch = estimate_vram_mb(110, batch_size=8, seq_len=512)
        assert large_batch > small_batch

    def test_larger_model_uses_more_vram(self):
        """More parameters should mean more VRAM."""
        small = estimate_vram_mb(60, batch_size=4, seq_len=512)
        large = estimate_vram_mb(345, batch_size=4, seq_len=512)
        assert large > small

    def test_zero_params_returns_overhead_only(self):
        """Zero-param model should return just CUDA overhead + activations."""
        est = estimate_vram_mb(0, batch_size=1, seq_len=1)
        # Should be close to the 400 MB CUDA overhead + small activation term
        assert 350 < est < 600

    def test_monotonic_with_params(self):
        """Estimate should increase monotonically with params."""
        prev = 0.0
        for params_m in [10, 50, 100, 200, 500, 1000]:
            est = estimate_vram_mb(params_m)
            assert est > prev, f"Not monotonic at {params_m}M"
            prev = est


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------


class TestCalibration:
    """Test calibration file read/write."""

    def test_load_missing_file_returns_empty(self, tmp_path):
        """load_calibration returns {} when no file exists."""
        with patch("bench_utils.CALIBRATION_PATH", tmp_path / "missing.json"):
            assert load_calibration() == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        """save_calibration + load_calibration preserves data."""
        cal_path = tmp_path / "cal.json"
        data = {"pythia-70m": 3967.0, "bert-base": 3214.0}
        with patch("bench_utils.CALIBRATION_PATH", cal_path):
            save_calibration(data)
            loaded = load_calibration()
        assert loaded == data

    def test_calibrated_vram_prefers_measurement(self, tmp_path):
        """calibrated_vram_mb uses measured value when available."""
        cal_path = tmp_path / "cal.json"
        cal_path.write_text(json.dumps({"pythia-70m": 3967.0}))
        with patch("bench_utils.CALIBRATION_PATH", cal_path):
            result = calibrated_vram_mb("pythia-70m", params_m=70)
        # Should be 3967 * 1.05 = 4165.35
        assert result == pytest.approx(3967.0 * 1.05, rel=1e-3)

    def test_calibrated_vram_falls_back_to_heuristic(self, tmp_path):
        """calibrated_vram_mb falls back to heuristic for unknown models."""
        cal_path = tmp_path / "cal.json"
        cal_path.write_text(json.dumps({}))
        with patch("bench_utils.CALIBRATION_PATH", cal_path):
            result = calibrated_vram_mb("unknown-model", params_m=110)
        expected = estimate_vram_mb(110)
        assert result == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# GpuTask tests
# ---------------------------------------------------------------------------


class TestGpuTask:
    """Test GpuTask dataclass."""

    def test_creation(self):
        task = GpuTask(
            task_id="test-1",
            script="extended_benchmark",
            model_slug="pythia-70m",
            params_m=70,
            batch_size=4,
            seq_len=512,
            estimated_vram_mb=3900.0,
        )
        assert task.status == "pending"
        assert task.actual_peak_mb is None
        assert task.task_id == "test-1"

    def test_default_status_is_pending(self):
        task = GpuTask(
            task_id="t",
            script="s",
            model_slug="m",
            params_m=1,
            batch_size=1,
            seq_len=1,
            estimated_vram_mb=100.0,
        )
        assert task.status == "pending"


# ---------------------------------------------------------------------------
# VRAMScheduler tests
# ---------------------------------------------------------------------------


class TestVRAMScheduler:
    """Test VRAM-aware scheduling logic."""

    def _make_task(self, slug: str, vram_mb: float, script: str = "extended_benchmark") -> GpuTask:
        return GpuTask(
            task_id=f"{script}:{slug}",
            script=script,
            model_slug=slug,
            params_m=100,
            batch_size=4,
            seq_len=512,
            estimated_vram_mb=vram_mb,
        )

    def test_single_large_model_gets_solo_round(self):
        """A model using >50% of budget gets its own round."""
        scheduler = VRAMScheduler(total_vram_mb=16000.0, safety_factor=0.9)
        scheduler.tasks = [self._make_task("big", 10000.0)]
        rounds = scheduler.plan_schedule()
        assert len(rounds) == 1
        assert len(rounds[0]) == 1
        assert rounds[0][0].model_slug == "big"

    def test_small_models_packed_together(self):
        """Two small models that fit in budget are packed into one round."""
        scheduler = VRAMScheduler(total_vram_mb=16000.0, safety_factor=0.9)
        # Budget = 14400 MB. Two 5000 MB tasks = 10000 < 14400
        scheduler.tasks = [
            self._make_task("a", 5000.0),
            self._make_task("b", 5000.0),
        ]
        rounds = scheduler.plan_schedule()
        assert len(rounds) == 1
        assert len(rounds[0]) == 2

    def test_overflow_creates_new_round(self):
        """When adding a task would exceed budget, start a new round."""
        scheduler = VRAMScheduler(total_vram_mb=16000.0, safety_factor=0.9)
        # Budget = 14400. 8000 + 8000 = 16000 > 14400
        scheduler.tasks = [
            self._make_task("a", 8000.0),
            self._make_task("b", 8000.0),
        ]
        rounds = scheduler.plan_schedule()
        assert len(rounds) == 2

    def test_first_fit_decreasing_order(self):
        """Tasks are sorted largest-first (FFD bin packing)."""
        scheduler = VRAMScheduler(total_vram_mb=20000.0, safety_factor=0.9)
        # Budget = 18000
        scheduler.tasks = [
            self._make_task("small", 2000.0),
            self._make_task("big", 10000.0),
            self._make_task("medium", 4000.0),
        ]
        rounds = scheduler.plan_schedule()
        # Big (10000) goes first. Medium (4000) fits with big (14000 < 18000).
        # Small (2000) also fits (16000 < 18000).
        # All 3 should fit in one round with 20GB budget.
        assert len(rounds) == 1
        # Big should be placed first (sorted descending)
        assert rounds[0][0].model_slug == "big"
        slugs = {t.model_slug for t in rounds[0]}
        assert slugs == {"big", "medium", "small"}

    def test_many_small_models_pack_efficiently(self):
        """Many small models should pack into fewer rounds than models."""
        scheduler = VRAMScheduler(total_vram_mb=16000.0, safety_factor=0.9)
        # Budget = 14400. 7 tasks at 2000 each.
        # Can fit 7 in round 1 (14000 < 14400)
        scheduler.tasks = [self._make_task(f"m{i}", 2000.0) for i in range(7)]
        rounds = scheduler.plan_schedule()
        assert len(rounds) == 1
        assert len(rounds[0]) == 7

    def test_empty_schedule(self):
        """No tasks produces no rounds."""
        scheduler = VRAMScheduler(total_vram_mb=16000.0, safety_factor=0.9)
        scheduler.tasks = []
        rounds = scheduler.plan_schedule()
        assert len(rounds) == 0

    def test_realistic_17_model_schedule(self):
        """Simulate scheduling all 17 extended benchmark models."""
        scheduler = VRAMScheduler(total_vram_mb=16990.0, safety_factor=0.9)
        # Approximate VRAM estimates for 17 models
        models = [
            ("pythia-1b", 10200),
            ("gpt2-medium", 6800),
            ("opt-350m", 5500),
            ("pythia-410m", 5200),
            ("phi-2", 12000),
            ("bert-base", 3100),
            ("pythia-160m", 3500),
            ("pythia-70m", 3900),
            ("t5-small", 1850),
            ("mobilenetv2", 2500),
            ("efficientnet-b0", 2800),
            ("resnet50", 2200),
            ("vit-base", 3200),
            ("switch-base-8", 4500),
            ("mamba-130m", 3000),
            ("mamba-370m", 5000),
            ("gpt2-small", 3500),
        ]
        scheduler.tasks = [self._make_task(slug, vram) for slug, vram in models]
        rounds = scheduler.plan_schedule()

        # Should have fewer rounds than models (packing works)
        assert len(rounds) < len(models)
        # Every task should appear exactly once
        all_slugs = [t.model_slug for r in rounds for t in r]
        assert sorted(all_slugs) == sorted(slug for slug, _ in models)
        # No round should exceed budget
        budget = 16990.0 * 0.9
        for r in rounds:
            total = sum(t.estimated_vram_mb for t in r)
            assert total <= budget, f"Round exceeds budget: {total:.0f} > {budget:.0f}"

    def test_add_task(self):
        """add_task creates a GpuTask with correct estimates."""
        scheduler = VRAMScheduler(total_vram_mb=16000.0)
        scheduler.add_task(
            script="extended_benchmark",
            model_slug="pythia-70m",
            params_m=70,
            batch_size=4,
            seq_len=512,
        )
        assert len(scheduler.tasks) == 1
        task = scheduler.tasks[0]
        assert task.model_slug == "pythia-70m"
        assert task.estimated_vram_mb > 0

    def test_format_schedule_output(self):
        """format_schedule returns a non-empty string."""
        scheduler = VRAMScheduler(total_vram_mb=16000.0, safety_factor=0.9)
        scheduler.tasks = [
            self._make_task("a", 5000.0),
            self._make_task("b", 3000.0),
        ]
        output = scheduler.format_schedule()
        assert "Round 1" in output
        assert "5000" in output or "5,000" in output or "5.0" in output

    def test_task_exceeds_budget_still_scheduled(self):
        """Task larger than budget gets its own round (with warning)."""
        scheduler = VRAMScheduler(total_vram_mb=4000.0, safety_factor=0.9)
        # Budget = 3600, task needs 10000 — exceeds budget
        scheduler.tasks = [self._make_task("huge", 10000.0)]
        rounds = scheduler.plan_schedule()
        # Should still schedule it (solo round), user was warned
        assert len(rounds) == 1
        assert rounds[0][0].model_slug == "huge"

    def test_auto_detect_fallback(self):
        """VRAMScheduler uses fallback when no GPU detected."""
        with patch("gpu_scheduler.total_vram_mb_fn", return_value=0.0):
            scheduler = VRAMScheduler()
        # Should fall back to 16384 MB
        assert scheduler.total_vram_mb == 16384.0

    def test_explicit_vram_overrides_detection(self):
        """Explicit total_vram_mb is used even if GPU is available."""
        scheduler = VRAMScheduler(total_vram_mb=8000.0)
        assert scheduler.total_vram_mb == 8000.0


# ---------------------------------------------------------------------------
# Edge case tests for estimate_vram_mb
# ---------------------------------------------------------------------------


class TestEstimateVramEdgeCases:
    """Test edge cases in VRAM estimation."""

    def test_negative_params_raises(self):
        """Negative params_m should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            estimate_vram_mb(-10)

    def test_very_large_params(self):
        """Very large models produce large but finite estimates."""
        est = estimate_vram_mb(100_000)  # 100B params
        assert est > 0
        assert est < 10_000_000  # Less than 10 TB
