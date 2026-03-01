"""Tests for CALM monotone analysis."""

import pytest
import torch.nn as nn

from bwsk.calm import analyze_calm, partition_for_distribution


class TestAnalyzeCalm:
    def test_all_s_model(self):
        """Model with all S-type ops is fully parallelizable."""
        model = nn.Sequential(nn.Linear(10, 10), nn.LayerNorm(10), nn.Linear(10, 5))
        report = analyze_calm(model)
        assert report.parallelism_ratio == pytest.approx(1.0)
        assert report.num_sync_barriers == 0
        assert report.monotone_count == 3

    def test_mixed_model(self):
        """Model with mixed S/K ops identifies sync points."""
        model = nn.Sequential(
            nn.Linear(10, 10),  # S
            nn.LayerNorm(10),  # S
            nn.ReLU(),  # K
            nn.Linear(10, 10),  # S
            nn.ReLU(),  # K
            nn.Linear(10, 5),  # S
        )
        report = analyze_calm(model)

        assert report.total_modules == 6
        assert report.monotone_count == 4  # 4 S-type
        assert report.sync_count == 2  # 2 K-type
        assert report.parallelism_ratio == pytest.approx(4 / 6)
        assert report.num_sync_barriers == 2

    def test_segments_structure(self):
        """Segments are correctly partitioned."""
        model = nn.Sequential(
            nn.Linear(10, 10),  # S
            nn.LayerNorm(10),  # S
            nn.ReLU(),  # K
            nn.Linear(10, 5),  # S
        )
        report = analyze_calm(model)

        # [S,S], [K], [S]
        assert len(report.segments) == 3
        assert report.segments[0].is_monotone is True
        assert report.segments[0].size == 2
        assert report.segments[1].is_monotone is False
        assert report.segments[1].size == 1
        assert report.segments[2].is_monotone is True
        assert report.segments[2].size == 1

    def test_all_k_model(self):
        """Model with all K-type ops: no parallelism."""
        model = nn.Sequential(nn.ReLU(), nn.Dropout(0.5))
        report = analyze_calm(model)
        assert report.parallelism_ratio == pytest.approx(0.0)
        assert report.num_sync_barriers == 2

    def test_to_dict(self):
        """Report serializes correctly."""
        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        report = analyze_calm(model)
        d = report.to_dict()
        assert "parallelism_ratio" in d
        assert "num_sync_barriers" in d
        assert "segments" in d


class TestPartitionForDistribution:
    def test_single_device(self):
        """Single device: everything on one device."""
        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        devices = partition_for_distribution(model, num_devices=1)
        assert len(devices) == 1
        assert len(devices[0]) == 2

    def test_two_devices(self):
        """Two devices: split at K-boundaries."""
        model = nn.Sequential(
            nn.Linear(10, 10),  # S -> device 0
            nn.LayerNorm(10),  # S -> device 0
            nn.ReLU(),  # K -> device 0 (sync point)
            nn.Linear(10, 10),  # S -> device 1
            nn.ReLU(),  # K -> device 1 (sync point)
            nn.Linear(10, 5),  # S -> device 0 (wraps)
        )
        devices = partition_for_distribution(model, num_devices=2)
        assert len(devices) == 2
        # Each device should have some modules
        assert len(devices[0]) > 0
        assert len(devices[1]) > 0
        # Total should be 6
        assert sum(len(d) for d in devices) == 6

    def test_all_s_no_split(self):
        """All S-type model stays on one device."""
        model = nn.Sequential(nn.Linear(10, 10), nn.LayerNorm(10), nn.Linear(10, 5))
        devices = partition_for_distribution(model, num_devices=2)
        # No K-boundaries, so everything stays on device 0
        assert len(devices[0]) == 3
        assert len(devices[1]) == 0
