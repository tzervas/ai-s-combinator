"""Generate publication-quality figures from experiment results.

Reads JSON outputs from extended_benchmark, convergence_experiment, and
memory_throughput_profiler, and produces figures for the BWSK whitepaper.

Figures generated:
  1. S/K ratio vs model size (scatter + trend line)
  2. S/K ratio by architecture family (grouped bar chart)
  3. Loss convergence curves (mean ± std, 4 models × 3 modes)
  4. Memory savings vs model size (line plot)
  5. Throughput overhead vs model size (line plot)
  6. CALM parallelism ratio by architecture (bar chart)
  7. Memory breakdown stacked bar chart
  8. Architecture S/K profile comparison (radar/polar chart)

Usage:
    uv run python scripts/generate_whitepaper_figures.py
    uv run python scripts/generate_whitepaper_figures.py --output-dir docs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Use non-interactive backend for server/CI environments
matplotlib.use("Agg")

# Publication-quality defaults
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPTS_DIR.parent / "docs" / "figures"

# Color palette for training modes
MODE_COLORS = {
    "conventional": "#1f77b4",
    "bwsk_analyzed": "#ff7f0e",
    "bwsk_reversible": "#2ca02c",
}

# Color palette for architecture families
FAMILY_COLORS = {
    "transformer": "#1f77b4",
    "cnn": "#ff7f0e",
    "vit": "#2ca02c",
    "ssm": "#d62728",
    "moe": "#9467bd",
}


def load_json(path: Path) -> dict | list | None:
    """Load JSON file, returning None if not found."""
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: S/K ratio vs model size
# ---------------------------------------------------------------------------


def fig_sk_ratio_vs_size(data: list[dict], output_dir: Path) -> None:
    """Scatter plot of S-ratio vs parameter count with trend line."""
    params = []
    s_ratios = []
    families = []

    for m in data:
        if m.get("classification"):
            c = m["classification"]
            params.append(m["params_m"])
            s_ratios.append(c["s_ratio"])
            families.append(m.get("arch_family", "unknown"))

    if not params:
        print("  No classification data for fig 1")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for family in sorted(set(families)):
        mask = [f == family for f in families]
        x = [p for p, m in zip(params, mask, strict=True) if m]
        y = [s for s, m in zip(s_ratios, mask, strict=True) if m]
        color = FAMILY_COLORS.get(family, "#999999")
        ax.scatter(x, y, c=color, label=family, s=80, zorder=3)

    # Trend line (all data)
    if len(params) > 2:
        z = np.polyfit(np.log10(params), s_ratios, 1)
        p = np.poly1d(z)
        x_line = np.logspace(np.log10(min(params)), np.log10(max(params)), 50)
        ax.plot(
            x_line,
            p(np.log10(x_line)),
            "--",
            color="gray",
            alpha=0.7,
            label="Trend",
        )

    ax.axhline(y=0.75, color="red", linestyle=":", alpha=0.5, label="75% hypothesis")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("S-type ratio")
    ax.set_title("S/K Ratio vs Model Size")
    ax.legend()
    ax.set_ylim(0, 1)

    path = output_dir / "fig1_sk_ratio_vs_size.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2: S/K ratio by architecture family
# ---------------------------------------------------------------------------


def fig_sk_ratio_by_family(data: list[dict], output_dir: Path) -> None:
    """Grouped bar chart of S/K/GRAY ratios by architecture family."""
    family_data: dict[str, list[dict]] = {}
    for m in data:
        if m.get("classification"):
            family = m.get("arch_family", "unknown")
            if family not in family_data:
                family_data[family] = []
            family_data[family].append(m["classification"])

    if not family_data:
        print("  No data for fig 2")
        return

    families = sorted(family_data.keys())
    s_means = []
    k_means = []
    g_means = []

    for f in families:
        cs = family_data[f]
        s_means.append(np.mean([c["s_ratio"] for c in cs]))
        k_means.append(np.mean([c["k_ratio"] for c in cs]))
        g_means.append(np.mean([c["gray_ratio"] for c in cs]))

    x = np.arange(len(families))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, s_means, width, label="S-type", color="#2ca02c")
    ax.bar(x, k_means, width, label="K-type", color="#d62728")
    ax.bar(x + width, g_means, width, label="GRAY", color="#7f7f7f")

    ax.set_xlabel("Architecture Family")
    ax.set_ylabel("Ratio")
    ax.set_title("S/K/GRAY Ratios by Architecture Family")
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.legend()
    ax.set_ylim(0, 1)

    path = output_dir / "fig2_sk_ratio_by_family.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3: Convergence curves
# ---------------------------------------------------------------------------


def fig_convergence_curves(convergence_data: list[dict], output_dir: Path) -> None:
    """Loss convergence curves with mean ± std shading."""
    if not convergence_data:
        print("  No convergence data for fig 3")
        return

    n_models = len(convergence_data)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, model_data in zip(axes, convergence_data, strict=True):
        model_name = model_data["model_name"]
        mean_curves = model_data.get("mean_loss_curves", {})
        std_curves = model_data.get("std_loss_curves", {})

        for mode in ["conventional", "bwsk_analyzed", "bwsk_reversible"]:
            if mode in mean_curves:
                mean = np.array(mean_curves[mode])
                std = np.array(std_curves.get(mode, [0] * len(mean)))
                steps = np.arange(len(mean))
                color = MODE_COLORS[mode]

                ax.plot(steps, mean, color=color, label=mode, linewidth=1.5)
                ax.fill_between(
                    steps,
                    mean - std,
                    mean + std,
                    alpha=0.2,
                    color=color,
                )

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(model_name)
        ax.legend(fontsize=8)

    fig.suptitle("Training Convergence (mean ± 1 std, 3 seeds)", fontsize=14)
    fig.tight_layout()

    path = output_dir / "fig3_convergence_curves.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4: Memory savings vs model size
# ---------------------------------------------------------------------------


def fig_memory_savings(profile_data: list[dict], output_dir: Path) -> None:
    """Line plot of memory savings (%) vs model size."""
    params = []
    savings_pct = []

    for mp in profile_data:
        profiles = mp.get("profiles", [])
        conv = next((p for p in profiles if p["mode"] == "conventional"), None)
        rev = next((p for p in profiles if p["mode"] == "bwsk_reversible"), None)
        if conv and rev:
            conv_mem = conv["memory"]["total_peak_mb"]
            rev_mem = rev["memory"]["total_peak_mb"]
            if conv_mem > 0:
                params.append(mp["params_m"])
                savings_pct.append(100 * (conv_mem - rev_mem) / conv_mem)

    if not params:
        print("  No profile data for fig 4")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(params, savings_pct, "o-", color="#2ca02c", markersize=8)

    ax.axhline(y=15, color="red", linestyle=":", alpha=0.5, label="15% target")
    ax.axhline(y=40, color="red", linestyle=":", alpha=0.5, label="40% target")
    ax.fill_between(
        [min(params) * 0.8, max(params) * 1.2],
        15,
        40,
        alpha=0.1,
        color="red",
        label="H4 target range",
    )

    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Memory Savings (%)")
    ax.set_title("Memory Savings (BWSK-Reversible vs Conventional)")
    ax.legend()

    path = output_dir / "fig4_memory_savings.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 5: Throughput overhead vs model size
# ---------------------------------------------------------------------------


def fig_throughput_overhead(profile_data: list[dict], output_dir: Path) -> None:
    """Line plot of throughput overhead (%) vs model size."""
    params = []
    overhead_pct = []

    for mp in profile_data:
        profiles = mp.get("profiles", [])
        conv = next((p for p in profiles if p["mode"] == "conventional"), None)
        rev = next((p for p in profiles if p["mode"] == "bwsk_reversible"), None)
        if conv and rev:
            conv_tp = conv["throughput"]
            rev_tp = rev["throughput"]
            if conv_tp > 0:
                params.append(mp["params_m"])
                overhead_pct.append(100 * (conv_tp - rev_tp) / conv_tp)

    if not params:
        print("  No profile data for fig 5")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(params, overhead_pct, "o-", color="#d62728", markersize=8)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Throughput Overhead (%)")
    ax.set_title("Throughput Overhead (BWSK-Reversible vs Conventional)")

    path = output_dir / "fig5_throughput_overhead.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 6: CALM parallelism by architecture
# ---------------------------------------------------------------------------


def fig_calm_parallelism(benchmark_data: list[dict], output_dir: Path) -> None:
    """Bar chart of average CALM parallelism ratio by architecture."""
    model_names = []
    par_ratios = []
    colors = []

    for m in benchmark_data:
        calm = m.get("calm_blocks", [])
        if calm:
            avg_par = np.mean([b["parallelism_ratio"] for b in calm])
            model_names.append(m["model_name"])
            par_ratios.append(avg_par)
            family = m.get("arch_family", "unknown")
            colors.append(FAMILY_COLORS.get(family, "#999999"))

    if not model_names:
        print("  No CALM data for fig 6")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_names))
    ax.bar(x, par_ratios, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel("Average Parallelism Ratio")
    ax.set_title("CALM Parallelism Ratio by Architecture")
    ax.set_ylim(0, 1)

    # Legend for families
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=legend_elements)

    path = output_dir / "fig6_calm_parallelism.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 7: Memory breakdown stacked bar
# ---------------------------------------------------------------------------


def fig_memory_breakdown(profile_data: list[dict], output_dir: Path) -> None:
    """Stacked bar chart of memory breakdown per model (conventional mode)."""
    model_names = []
    params_mem = []
    grads_mem = []
    opt_mem = []
    act_mem = []

    for mp in profile_data:
        conv = next(
            (p for p in mp.get("profiles", []) if p["mode"] == "conventional"),
            None,
        )
        if conv:
            model_names.append(mp["model_name"])
            mem = conv["memory"]
            params_mem.append(mem["params_mb"])
            grads_mem.append(mem["grads_mb"])
            opt_mem.append(mem["optimizer_mb"])
            act_mem.append(mem["activations_mb"])

    if not model_names:
        print("  No profile data for fig 7")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.6

    ax.bar(x, params_mem, width, label="Parameters", color="#1f77b4")
    ax.bar(
        x,
        grads_mem,
        width,
        bottom=params_mem,
        label="Gradients",
        color="#ff7f0e",
    )
    bottom2 = [p + g for p, g in zip(params_mem, grads_mem, strict=True)]
    ax.bar(
        x,
        opt_mem,
        width,
        bottom=bottom2,
        label="Optimizer State",
        color="#2ca02c",
    )
    bottom3 = [b + o for b, o in zip(bottom2, opt_mem, strict=True)]
    ax.bar(
        x,
        act_mem,
        width,
        bottom=bottom3,
        label="Activations",
        color="#d62728",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Breakdown by Component (Conventional Training)")
    ax.legend()

    path = output_dir / "fig7_memory_breakdown.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 8: S/K profile comparison (grouped bar by model)
# ---------------------------------------------------------------------------


def fig_sk_profile_comparison(data: list[dict], output_dir: Path) -> None:
    """Grouped bar chart showing S/K/GRAY for each model, ordered by S-ratio."""
    models_with_cls = [m for m in data if m.get("classification")]
    if not models_with_cls:
        print("  No data for fig 8")
        return

    # Sort by S-ratio
    models_with_cls.sort(key=lambda m: m["classification"]["s_ratio"], reverse=True)

    names = [m["model_name"] for m in models_with_cls]
    s_vals = [m["classification"]["s_ratio"] for m in models_with_cls]
    k_vals = [m["classification"]["k_ratio"] for m in models_with_cls]
    g_vals = [m["classification"]["gray_ratio"] for m in models_with_cls]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(names))
    width = 0.25

    ax.bar(x - width, s_vals, width, label="S-type", color="#2ca02c")
    ax.bar(x, k_vals, width, label="K-type", color="#d62728")
    ax.bar(x + width, g_vals, width, label="GRAY", color="#7f7f7f")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Ratio")
    ax.set_title("S/K/GRAY Profile by Model (sorted by S-ratio)")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(y=0.75, color="red", linestyle=":", alpha=0.5)

    path = output_dir / "fig8_sk_profile_comparison.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate all whitepaper figures."""
    parser = argparse.ArgumentParser(description="Generate whitepaper figures")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for figures",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING WHITEPAPER FIGURES")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load data
    benchmark_data = load_json(SCRIPTS_DIR / "extended_benchmark_results.json")
    convergence_data = load_json(SCRIPTS_DIR / "convergence_results.json")
    profile_data = load_json(SCRIPTS_DIR / "memory_throughput_results.json")

    # Also try loading from multi_model_benchmark as fallback
    if benchmark_data is None:
        benchmark_data = load_json(SCRIPTS_DIR / "multi_model_benchmark_results.json")
        if benchmark_data is not None and isinstance(benchmark_data, dict):
            benchmark_data = [benchmark_data]

    if benchmark_data:
        print("\nFigure 1: S/K ratio vs model size")
        fig_sk_ratio_vs_size(benchmark_data, output_dir)

        print("Figure 2: S/K ratio by architecture family")
        fig_sk_ratio_by_family(benchmark_data, output_dir)

        print("Figure 6: CALM parallelism by architecture")
        fig_calm_parallelism(benchmark_data, output_dir)

        print("Figure 8: S/K profile comparison")
        fig_sk_profile_comparison(benchmark_data, output_dir)
    else:
        print("\nNo benchmark data found. Run extended_benchmark.py first.")

    if convergence_data:
        print("\nFigure 3: Convergence curves")
        fig_convergence_curves(convergence_data, output_dir)
    else:
        print("\nNo convergence data found. Run convergence_experiment.py first.")

    if profile_data:
        print("\nFigure 4: Memory savings vs model size")
        fig_memory_savings(profile_data, output_dir)

        print("Figure 5: Throughput overhead vs model size")
        fig_throughput_overhead(profile_data, output_dir)

        print("Figure 7: Memory breakdown")
        fig_memory_breakdown(profile_data, output_dir)
    else:
        print("\nNo profile data found. Run memory_throughput_profiler.py first.")

    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print(f"Figures saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
