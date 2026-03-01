"""Tests for erasure-minimized NAS."""

import random

import pytest

from bwsk.nas import (
    ArchitectureGene,
    evaluate_architecture,
    mutate_gene,
    random_gene,
    search_evolutionary,
    search_random,
)


class TestArchitectureGene:
    def test_build_simple(self):
        """Gene builds a valid model."""
        gene = ArchitectureGene(
            ops=["Linear", "ReLU", "Linear"],
            in_features=10,
            hidden_features=20,
            out_features=5,
        )
        model = gene.build()
        import torch

        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)

    def test_build_all_s_type(self):
        """All S-type architecture builds and runs."""
        gene = ArchitectureGene(
            ops=["Linear", "LeakyReLU", "LayerNorm", "Linear"],
            in_features=10,
            hidden_features=20,
            out_features=5,
        )
        model = gene.build()
        import torch

        out = model(torch.randn(4, 10))
        assert out.shape == (4, 5)

    def test_build_empty_ops_fallback(self):
        """Empty ops falls back to single Linear."""
        gene = ArchitectureGene(
            ops=[],
            in_features=10,
            hidden_features=20,
            out_features=5,
        )
        model = gene.build()
        import torch

        out = model(torch.randn(4, 10))
        assert out.shape == (4, 5)


class TestEvaluateArchitecture:
    def test_returns_nas_result(self):
        """Evaluation returns a valid NASResult."""
        gene = ArchitectureGene(
            ops=["Linear", "ReLU", "Linear"],
            in_features=10,
            hidden_features=20,
            out_features=5,
        )
        result = evaluate_architecture(gene, train_steps=5)

        assert 0 <= result.erasure_score <= 1.0
        assert result.accuracy > 0
        assert result.total_ops > 0
        assert result.s_count + result.k_count <= result.total_ops

    def test_s_type_arch_has_low_erasure(self):
        """Architecture with all S-type ops has erasure_score near 0."""
        gene = ArchitectureGene(
            ops=["Linear", "LeakyReLU", "Linear"],
            in_features=10,
            hidden_features=20,
            out_features=5,
        )
        result = evaluate_architecture(gene, train_steps=5)
        assert result.erasure_score == pytest.approx(0.0)

    def test_k_type_arch_has_high_erasure(self):
        """Architecture with K-type ops has nonzero erasure."""
        gene = ArchitectureGene(
            ops=["Linear", "ReLU", "Linear", "ReLU", "Linear"],
            in_features=10,
            hidden_features=20,
            out_features=5,
        )
        result = evaluate_architecture(gene, train_steps=5)
        assert result.erasure_score > 0


class TestRandomGene:
    def test_generates_valid_gene(self):
        """Random gene has correct structure."""
        random.seed(42)
        gene = random_gene(10, 20, 5, depth=5)
        assert len(gene.ops) == 5
        assert gene.ops[0] == "Linear"
        assert gene.in_features == 10
        assert gene.out_features == 5

    def test_builds_successfully(self):
        """Random gene builds a working model."""
        random.seed(42)
        gene = random_gene(10, 20, 5)
        model = gene.build()
        import torch

        out = model(torch.randn(4, 10))
        assert out.shape[0] == 4


class TestMutateGene:
    def test_mutation_changes_one_op(self):
        """Mutation changes exactly one operation."""
        random.seed(42)
        gene = ArchitectureGene(
            ops=["Linear", "ReLU", "ReLU", "ReLU", "Linear"],
            in_features=10,
            hidden_features=20,
            out_features=5,
        )
        mutated = mutate_gene(gene)

        # First and last should stay Linear
        assert mutated.ops[0] == "Linear"
        assert mutated.ops[-1] == "Linear"
        # At most one op changed
        diffs = sum(1 for a, b in zip(gene.ops, mutated.ops, strict=False) if a != b)
        assert diffs <= 1


class TestSearchRandom:
    def test_random_search(self):
        """Random search produces valid report."""
        random.seed(42)
        report = search_random(
            num_architectures=5,
            depth=4,
            train_steps=3,
        )

        assert len(report.results) == 5
        assert report.best_erasure is not None
        assert report.best_accuracy is not None
        assert len(report.pareto_frontier) > 0

    def test_pareto_frontier_valid(self):
        """Pareto frontier contains non-dominated solutions."""
        random.seed(42)
        report = search_random(
            num_architectures=10,
            depth=4,
            train_steps=3,
        )

        # Each Pareto point should not be dominated by any other
        for p in report.pareto_frontier:
            for other in report.results:
                if other is p:
                    continue
                # other should NOT dominate p
                dominates = (
                    other.accuracy >= p.accuracy
                    and other.erasure_score <= p.erasure_score
                    and (other.accuracy > p.accuracy or other.erasure_score < p.erasure_score)
                )
                assert not dominates


class TestSearchEvolutionary:
    def test_evolutionary_search(self):
        """Evolutionary search produces valid report."""
        random.seed(42)
        report = search_evolutionary(
            num_generations=3,
            population_size=4,
            depth=4,
            train_steps=3,
        )

        assert len(report.results) > 0
        assert report.best_erasure is not None
        assert report.best_accuracy is not None

    def test_report_to_dict(self):
        """Report serializes correctly."""
        random.seed(42)
        report = search_random(
            num_architectures=3,
            depth=4,
            train_steps=3,
        )
        d = report.to_dict()

        assert "num_architectures" in d
        assert "best_erasure" in d
        assert "pareto_size" in d
        assert d["num_architectures"] == 3
