"""Erasure-minimized Neural Architecture Search (NAS).

Searches for architectures that minimize K-type (information-erasing)
operations while maintaining task performance. The key insight: since
~75% of transformer computation is S-type, we can find architectures
that push this even higher by replacing K-type operations with S-type
alternatives where possible.

Search space:
- S-type operations: Linear, LeakyReLU, Softplus, LayerNorm, Identity
- K-type operations: ReLU, GELU, Sigmoid, MaxPool2d, Dropout
- Architecture: variable-depth sequential models with operator choices

Optimization objective:
- Primary: minimize erasure score (fraction of K-type ops)
- Secondary: maintain task accuracy above a threshold

Search algorithms:
- Random search (baseline)
- Evolutionary search (mutation + selection on Pareto frontier)

Why erasure-minimized NAS? Standard NAS optimizes for accuracy/latency.
Our NAS additionally optimizes for information preservation, yielding
architectures with better provenance tracking, more reversible backprop
potential, and higher CALM parallelism ratios.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from bwsk.classify import classify_model

# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

# Available operations for each position in the architecture
S_TYPE_OPS = [
    ("Linear", lambda in_d, out_d: nn.Linear(in_d, out_d)),
    ("LeakyReLU", lambda in_d, out_d: nn.LeakyReLU()),
    ("Softplus", lambda in_d, out_d: nn.Softplus()),
    ("LayerNorm", lambda in_d, out_d: nn.LayerNorm(in_d)),
    ("Identity", lambda in_d, out_d: nn.Identity()),
]

K_TYPE_OPS = [
    ("ReLU", lambda in_d, out_d: nn.ReLU()),
    ("GELU", lambda in_d, out_d: nn.GELU()),
    ("Sigmoid", lambda in_d, out_d: nn.Sigmoid()),
    ("Dropout", lambda in_d, out_d: nn.Dropout(0.1)),
]

ALL_OPS = S_TYPE_OPS + K_TYPE_OPS


@dataclass
class ArchitectureGene:
    """Encoding of a single architecture in the search space.

    An architecture is a sequence of (operation_name, changes_dim) pairs.
    Linear operations can change dimensions; activations preserve them.
    """

    ops: list[str]
    in_features: int
    hidden_features: int
    out_features: int

    def build(self) -> nn.Module:
        """Construct the nn.Module from this gene encoding.

        Returns:
            A Sequential model matching this architecture.
        """
        op_map = {name: factory for name, factory in ALL_OPS}
        layers: list[nn.Module] = []

        current_dim = self.in_features
        for i, op_name in enumerate(self.ops):
            if op_name not in op_map:
                continue

            if op_name == "Linear":
                # Last Linear outputs to out_features
                if i == len(self.ops) - 1 or all(o != "Linear" for o in self.ops[i + 1 :]):
                    out_dim = self.out_features
                else:
                    out_dim = self.hidden_features
                layers.append(op_map[op_name](current_dim, out_dim))
                current_dim = out_dim
            else:
                layers.append(op_map[op_name](current_dim, current_dim))

        if not layers:
            layers.append(nn.Linear(self.in_features, self.out_features))

        return nn.Sequential(*layers)


@dataclass
class NASResult:
    """Result of evaluating a single architecture."""

    gene: ArchitectureGene
    erasure_score: float
    accuracy: float  # Proxy metric (lower loss = higher accuracy)
    s_count: int
    k_count: int
    total_ops: int
    parallelism_ratio: float = 0.0


@dataclass
class NASReport:
    """Complete NAS search report."""

    results: list[NASResult] = field(default_factory=list)
    best_erasure: NASResult | None = None
    best_accuracy: NASResult | None = None
    pareto_frontier: list[NASResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "num_architectures": len(self.results),
            "best_erasure": {
                "ops": self.best_erasure.gene.ops,
                "erasure_score": self.best_erasure.erasure_score,
                "accuracy": self.best_erasure.accuracy,
            }
            if self.best_erasure
            else None,
            "best_accuracy": {
                "ops": self.best_accuracy.gene.ops,
                "erasure_score": self.best_accuracy.erasure_score,
                "accuracy": self.best_accuracy.accuracy,
            }
            if self.best_accuracy
            else None,
            "pareto_size": len(self.pareto_frontier),
        }


# ---------------------------------------------------------------------------
# Architecture evaluation
# ---------------------------------------------------------------------------


def evaluate_architecture(
    gene: ArchitectureGene,
    train_steps: int = 10,
) -> NASResult:
    """Evaluate an architecture on a synthetic task.

    Builds the model, trains briefly on random data, and measures
    both erasure score and task performance (loss as accuracy proxy).

    Args:
        gene: Architecture to evaluate.
        train_steps: Number of training steps.

    Returns:
        NASResult with erasure and accuracy metrics.
    """
    model = gene.build()

    # Classify the model
    try:
        report = classify_model(model)
        erasure_score = report.erasure_score
        s_count = report.s_count
        k_count = report.k_count
        total_ops = report.total_ops
    except Exception:
        # If tracing fails, penalize with max erasure
        erasure_score = 1.0
        s_count = 0
        k_count = len(gene.ops)
        total_ops = len(gene.ops)

    # Quick training on synthetic data to measure task performance
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    x = torch.randn(32, gene.in_features)
    y = torch.randn(32, gene.out_features)

    final_loss = float("inf")
    for _ in range(train_steps):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    # Accuracy proxy: lower loss = better
    accuracy = 1.0 / (1.0 + final_loss)

    return NASResult(
        gene=gene,
        erasure_score=erasure_score,
        accuracy=accuracy,
        s_count=s_count,
        k_count=k_count,
        total_ops=total_ops,
        parallelism_ratio=(s_count / total_ops if total_ops > 0 else 0.0),
    )


# ---------------------------------------------------------------------------
# Search algorithms
# ---------------------------------------------------------------------------


def random_gene(
    in_features: int = 10,
    hidden: int = 20,
    out_features: int = 5,
    depth: int = 5,
    s_bias: float = 0.5,
) -> ArchitectureGene:
    """Generate a random architecture gene.

    Args:
        in_features: Input dimension.
        hidden: Hidden dimension.
        out_features: Output dimension.
        depth: Number of operations.
        s_bias: Probability of choosing S-type ops (0-1).

    Returns:
        A random ArchitectureGene.
    """
    ops = []
    has_linear = False

    for i in range(depth):
        if i == 0 or (i == depth - 1 and not has_linear):
            # First and last must be Linear to ensure correct dimensions
            ops.append("Linear")
            has_linear = True
        elif random.random() < s_bias:
            name, _ = random.choice(S_TYPE_OPS)
            ops.append(name)
            if name == "Linear":
                has_linear = True
        else:
            name, _ = random.choice(K_TYPE_OPS)
            ops.append(name)

    return ArchitectureGene(
        ops=ops,
        in_features=in_features,
        hidden_features=hidden,
        out_features=out_features,
    )


def mutate_gene(gene: ArchitectureGene) -> ArchitectureGene:
    """Mutate an architecture gene by changing one operation.

    Args:
        gene: Gene to mutate.

    Returns:
        A new gene with one operation changed.
    """
    new_ops = list(gene.ops)
    if len(new_ops) <= 2:
        return gene  # Too short to mutate non-Linear positions

    # Mutate a random non-first, non-last position
    idx = random.randint(1, len(new_ops) - 2)
    name, _ = random.choice(ALL_OPS)
    new_ops[idx] = name

    return ArchitectureGene(
        ops=new_ops,
        in_features=gene.in_features,
        hidden_features=gene.hidden_features,
        out_features=gene.out_features,
    )


def _compute_pareto(
    results: list[NASResult],
) -> list[NASResult]:
    """Compute the Pareto frontier of accuracy vs. erasure.

    A result is Pareto-optimal if no other result dominates it
    (i.e., is better in both accuracy and erasure score).
    """
    pareto: list[NASResult] = []
    for r in results:
        dominated = False
        for other in results:
            if other is r:
                continue
            # other dominates r if it has both better accuracy
            # and lower erasure
            if (
                other.accuracy >= r.accuracy
                and other.erasure_score <= r.erasure_score
                and (other.accuracy > r.accuracy or other.erasure_score < r.erasure_score)
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    return pareto


def search_random(
    num_architectures: int = 20,
    in_features: int = 10,
    hidden: int = 20,
    out_features: int = 5,
    depth: int = 5,
    train_steps: int = 10,
) -> NASReport:
    """Random search over the architecture space.

    Args:
        num_architectures: Number of random architectures to evaluate.
        in_features: Input dimension.
        hidden: Hidden dimension.
        out_features: Output dimension.
        depth: Number of operations per architecture.
        train_steps: Training steps per evaluation.

    Returns:
        NASReport with all results and Pareto frontier.
    """
    results = []
    for _ in range(num_architectures):
        gene = random_gene(in_features, hidden, out_features, depth)
        result = evaluate_architecture(gene, train_steps)
        results.append(result)

    return _build_report(results)


def search_evolutionary(
    num_generations: int = 5,
    population_size: int = 10,
    in_features: int = 10,
    hidden: int = 20,
    out_features: int = 5,
    depth: int = 5,
    train_steps: int = 10,
) -> NASReport:
    """Evolutionary search optimizing for low erasure + high accuracy.

    Uses tournament selection and mutation to evolve architectures
    toward the Pareto frontier of accuracy vs. erasure score.

    Args:
        num_generations: Number of evolution generations.
        population_size: Population size per generation.
        in_features: Input dimension.
        hidden: Hidden dimension.
        out_features: Output dimension.
        depth: Number of operations per architecture.
        train_steps: Training steps per evaluation.

    Returns:
        NASReport with all results and Pareto frontier.
    """
    # Initialize population
    population = [
        random_gene(in_features, hidden, out_features, depth) for _ in range(population_size)
    ]

    all_results: list[NASResult] = []

    for _gen in range(num_generations):
        # Evaluate
        gen_results = [evaluate_architecture(g, train_steps) for g in population]
        all_results.extend(gen_results)

        # Select: keep top half by combined score
        # Score: accuracy - erasure_score (maximize)
        gen_results.sort(key=lambda r: r.accuracy - r.erasure_score, reverse=True)
        survivors = gen_results[: population_size // 2]

        # Breed next generation via mutation
        next_gen: list[ArchitectureGene] = [r.gene for r in survivors]
        while len(next_gen) < population_size:
            parent = random.choice(survivors)
            child = mutate_gene(parent.gene)
            next_gen.append(child)

        population = next_gen

    return _build_report(all_results)


def _build_report(results: list[NASResult]) -> NASReport:
    """Build a NAS report from a list of results."""
    if not results:
        return NASReport()

    best_erasure = min(results, key=lambda r: r.erasure_score)
    best_accuracy = max(results, key=lambda r: r.accuracy)
    pareto = _compute_pareto(results)

    return NASReport(
        results=results,
        best_erasure=best_erasure,
        best_accuracy=best_accuracy,
        pareto_frontier=pareto,
    )
