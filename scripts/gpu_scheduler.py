"""VRAM-aware GPU task scheduler for BWSK benchmark suite.

Bin-packs benchmark tasks by estimated VRAM usage so that small models
run concurrently while large models get solo rounds. Each task runs as
a subprocess so that PyTorch's CUDA caching allocator releases all VRAM
on process exit — guaranteeing a clean slate between rounds.

Why subprocesses: PyTorch's CUDA memory allocator does not return freed
GPU memory to the OS within a single process. Only process termination
guarantees full VRAM reclamation. Running each task as a subprocess
means each round starts with all VRAM available.

Usage:
    # Show schedule without running
    uv run python scripts/gpu_scheduler.py --schedule-only --script extended_benchmark

    # Run extended benchmark with scheduling
    uv run python scripts/gpu_scheduler.py --script extended_benchmark

    # Schedule all benchmark scripts
    uv run python scripts/gpu_scheduler.py --all

    # Populate calibration from existing result JSONs
    uv run python scripts/gpu_scheduler.py --calibrate
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from bench_utils import calibrated_vram_mb, load_calibration, save_calibration
from bench_utils import total_vram_mb as total_vram_mb_fn

# ---------------------------------------------------------------------------
# Known model configs for each script
# ---------------------------------------------------------------------------

# Maps script name -> list of (slug, params_m, batch_size, seq_len) tuples.
# These mirror the configs defined in each benchmark script so the scheduler
# can estimate VRAM without importing heavy PyTorch dependencies.

SCRIPT_MODELS: dict[str, list[tuple[str, int, int, int]]] = {
    "extended_benchmark": [
        ("pythia-70m", 70, 4, 512),
        ("pythia-160m", 160, 4, 512),
        ("pythia-410m", 410, 4, 512),
        ("pythia-1b", 1010, 2, 512),
        ("gpt2-small", 124, 4, 512),
        ("gpt2-medium", 345, 4, 512),
        ("phi-2", 2700, 1, 512),
        ("opt-350m", 350, 4, 512),
        ("bert-base", 110, 8, 512),
        ("t5-small", 60, 8, 512),
        ("resnet50", 25, 32, 224),
        ("mobilenetv2", 3, 32, 224),
        ("efficientnet-b0", 5, 32, 224),
        ("vit-base", 86, 16, 224),
        ("mamba-130m", 130, 4, 512),
        ("mamba-370m", 370, 4, 512),
        ("switch-base-8", 262, 4, 512),
    ],
    "convergence_experiment": [
        ("pythia-70m", 70, 4, 512),
        ("gpt2-medium", 345, 4, 512),
        ("mamba-130m", 130, 4, 512),
        ("resnet50", 25, 32, 224),
    ],
    "memory_throughput_profiler": [
        ("pythia-70m", 70, 4, 512),
        ("pythia-410m", 410, 4, 512),
        ("bert-base", 110, 8, 512),
        ("gpt2-medium", 345, 4, 512),
        ("t5-small", 60, 8, 512),
        ("resnet50", 25, 32, 224),
        ("mamba-130m", 130, 4, 512),
    ],
    "multi_model_benchmark": [
        ("bert-base", 110, 8, 512),
        ("gpt2-medium", 345, 4, 512),
        ("t5-small", 60, 8, 512),
        ("opt-350m", 350, 4, 512),
        ("pythia-70m", 70, 4, 512),
        ("pythia-1b", 1010, 2, 512),
    ],
    "gpt2_benchmark": [
        ("gpt2", 124, 4, 512),
    ],
}


# ---------------------------------------------------------------------------
# GpuTask dataclass
# ---------------------------------------------------------------------------


@dataclass
class GpuTask:
    """A single GPU benchmark task with VRAM metadata.

    Represents one model run within one benchmark script. The scheduler
    uses estimated_vram_mb for bin-packing and fills actual_peak_mb
    after execution for calibration.

    Args:
        task_id: Unique identifier (script:slug format).
        script: Benchmark script name (without .py).
        model_slug: URL-safe model identifier.
        params_m: Parameter count in millions.
        batch_size: Training batch size.
        seq_len: Sequence length or image size.
        estimated_vram_mb: Predicted peak VRAM from estimate or calibration.
        status: Task lifecycle state.
        actual_peak_mb: Measured peak VRAM after execution.
    """

    task_id: str
    script: str
    model_slug: str
    params_m: float
    batch_size: int
    seq_len: int
    estimated_vram_mb: float
    status: str = "pending"
    actual_peak_mb: float | None = None


# ---------------------------------------------------------------------------
# VRAMScheduler
# ---------------------------------------------------------------------------


class VRAMScheduler:
    """VRAM-aware scheduler that bin-packs GPU tasks into concurrent rounds.

    Uses first-fit-decreasing (FFD) bin packing: sort tasks by estimated
    VRAM descending, then greedily assign each to the first round where
    it fits within the budget. This is a well-known approximation algorithm
    for bin packing that works well in practice.

    Why FFD over optimal: bin packing is NP-hard, and FFD is guaranteed
    to use at most 11/9 * OPT + 6/9 bins. With only ~17 tasks, FFD
    produces near-optimal schedules instantly.

    Args:
        total_vram_mb: Total GPU VRAM in MB. Auto-detected if 0.
        safety_factor: Fraction of total VRAM to use as budget (default 0.9).
    """

    def __init__(
        self,
        total_vram_mb: float = 0.0,
        safety_factor: float = 0.9,
    ) -> None:
        if total_vram_mb > 0:
            self.total_vram_mb = total_vram_mb
        else:
            detected = total_vram_mb_fn()
            if detected > 0:
                self.total_vram_mb = detected
            else:
                # Fallback for CPU-only environments
                self.total_vram_mb = 16384.0
        self.safety_factor = safety_factor
        self.tasks: list[GpuTask] = []

    @property
    def budget_mb(self) -> float:
        """Available VRAM budget after safety margin."""
        return self.total_vram_mb * self.safety_factor

    def add_task(
        self,
        script: str,
        model_slug: str,
        params_m: float,
        batch_size: int = 4,
        seq_len: int = 512,
        mode: str = "training",
    ) -> None:
        """Add a single task with auto-estimated VRAM.

        Args:
            script: Benchmark script name.
            model_slug: Model identifier.
            params_m: Parameter count in millions.
            batch_size: Training batch size.
            seq_len: Sequence length or image size.
            mode: "training" or "inference".
        """
        vram = calibrated_vram_mb(model_slug, params_m, batch_size, seq_len, mode)
        self.tasks.append(
            GpuTask(
                task_id=f"{script}:{model_slug}",
                script=script,
                model_slug=model_slug,
                params_m=params_m,
                batch_size=batch_size,
                seq_len=seq_len,
                estimated_vram_mb=vram,
            )
        )

    def add_tasks_for_script(self, script_name: str, model_filter: set[str] | None = None) -> None:
        """Add all known tasks for a benchmark script.

        Args:
            script_name: Key in SCRIPT_MODELS (e.g. "extended_benchmark").
            model_filter: If set, only include models whose slug is in this set.
        """
        configs = SCRIPT_MODELS.get(script_name, [])
        added = 0
        for slug, params_m, batch_size, seq_len in configs:
            if model_filter and slug not in model_filter:
                continue
            self.add_task(script_name, slug, params_m, batch_size, seq_len)
            added += 1
        if model_filter and added == 0:
            available = [slug for slug, *_ in configs]
            print(
                f"WARNING: No models matched filter for {script_name}. "
                f"Available: {', '.join(available)}"
            )

    def plan_schedule(self) -> list[list[GpuTask]]:
        """Plan execution rounds using first-fit-decreasing bin packing.

        Sorts tasks by estimated VRAM (largest first) and greedily packs
        them into rounds that fit within the VRAM budget.

        Returns:
            List of rounds, where each round is a list of tasks to run
            concurrently.
        """
        if not self.tasks:
            return []

        budget = self.budget_mb
        # Sort descending by estimated VRAM (FFD)
        sorted_tasks = sorted(self.tasks, key=lambda t: t.estimated_vram_mb, reverse=True)

        # Warn about tasks that individually exceed the budget
        for task in sorted_tasks:
            if task.estimated_vram_mb > budget:
                print(
                    f"WARNING: {task.model_slug} estimated VRAM "
                    f"({task.estimated_vram_mb:.0f} MB) exceeds budget "
                    f"({budget:.0f} MB). May cause OOM."
                )

        rounds: list[list[GpuTask]] = []
        round_used: list[float] = []  # Current VRAM used per round

        for task in sorted_tasks:
            placed = False
            for i, used in enumerate(round_used):
                if used + task.estimated_vram_mb <= budget:
                    rounds[i].append(task)
                    round_used[i] += task.estimated_vram_mb
                    placed = True
                    break
            if not placed:
                rounds.append([task])
                round_used.append(task.estimated_vram_mb)

        return rounds

    def format_schedule(self) -> str:
        """Format the planned schedule as a human-readable string."""
        rounds = self.plan_schedule()
        if not rounds:
            return "No tasks to schedule."

        lines = []
        gpu_name = "GPU"
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name()
        except ImportError:
            pass

        lines.append(
            f"=== VRAM Schedule ({gpu_name}, "
            f"{self.total_vram_mb / 1024:.1f} GB, "
            f"safety={self.safety_factor:.0%}) ==="
        )
        lines.append("")

        for i, round_tasks in enumerate(rounds, 1):
            total_vram = sum(t.estimated_vram_mb for t in round_tasks)
            if len(round_tasks) == 1:
                t = round_tasks[0]
                lines.append(f"Round {i}: [{t.model_slug}] — {t.estimated_vram_mb:.0f} MB (solo)")
            else:
                lines.append(
                    f"Round {i}: [{', '.join(t.model_slug for t in round_tasks)}] "
                    f"— {' + '.join(str(int(t.estimated_vram_mb)) for t in round_tasks)} "
                    f"= {total_vram:.0f} MB"
                )
            lines.append("")

        total_tasks = sum(len(r) for r in rounds)
        lines.append(
            f"Total: {total_tasks} tasks in {len(rounds)} rounds (budget: {self.budget_mb:.0f} MB)"
        )
        return "\n".join(lines)

    def execute(self, dry_run: bool = False) -> dict[str, float]:
        """Execute the planned schedule, running rounds sequentially.

        Within each round, tasks launch as concurrent subprocesses.
        Process exit guarantees full VRAM reclamation between rounds.

        Args:
            dry_run: If True, print schedule without executing.

        Returns:
            Dict mapping task_id to actual peak VRAM MB (from result JSONs).
        """
        rounds = self.plan_schedule()
        print(self.format_schedule())

        if dry_run:
            return {}

        results: dict[str, float] = {}

        for i, round_tasks in enumerate(rounds, 1):
            print(f"\n--- Executing Round {i}/{len(rounds)} ---")
            processes: list[tuple[GpuTask, subprocess.Popen]] = []

            for task in round_tasks:
                cmd = self._build_command(task)
                print(f"  Starting: {task.model_slug} ({task.script})")
                print(f"    Command: {' '.join(cmd)}")
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                processes.append((task, proc))

            # Wait for all tasks in this round
            for task, proc in processes:
                stdout, _ = proc.communicate()
                task.status = "completed" if proc.returncode == 0 else "failed"
                if proc.returncode != 0:
                    print(f"  FAILED: {task.model_slug} (exit {proc.returncode})")
                    if stdout:
                        # Print last 20 lines of output for debugging
                        tail = "\n".join(stdout.strip().split("\n")[-20:])
                        print(f"    Output tail:\n{tail}")
                else:
                    print(f"  Completed: {task.model_slug}")

                # Try to read actual peak VRAM from result JSON
                peak = self._read_peak_from_results(task)
                if peak is not None:
                    task.actual_peak_mb = peak
                    results[task.task_id] = peak

            print(f"--- Round {i} complete ---")

        # Update calibration with actual measurements
        if results:
            cal = load_calibration()
            for task_id, peak in results.items():
                slug = task_id.split(":", 1)[1] if ":" in task_id else task_id
                cal[slug] = peak
            save_calibration(cal)
            print(f"\nCalibration updated with {len(results)} measurements.")

        # Regenerate combined JSONs from per-model results
        scripts_with_tasks = {t.script for t in self.tasks}
        if "extended_benchmark" in scripts_with_tasks:
            self._regenerate_combined("extended_benchmark")
        if "convergence_experiment" in scripts_with_tasks:
            self._regenerate_combined("convergence_experiment")
        if "memory_throughput_profiler" in scripts_with_tasks:
            self._regenerate_combined("memory_throughput_profiler")

        return results

    def _regenerate_combined(self, script_name: str) -> None:
        """Regenerate combined JSON from per-model results after scheduled run.

        When models run as separate subprocesses, each overwrites the combined
        JSON. This method merges per-model JSONs back into a single combined
        file so downstream scripts (figures, upload) see all results.
        """
        results_dir = Path(__file__).parent

        if script_name == "extended_benchmark":
            # extended_benchmark has --report-only flag
            print("\nRegenerating extended benchmark combined report...")
            cmd = [
                "uv",
                "run",
                "python",
                str(results_dir / "extended_benchmark.py"),
                "--report-only",
            ]
            subprocess.run(cmd, check=False)
            return

        # For convergence and profiler: merge per-model JSONs manually
        prefix_map = {
            "convergence_experiment": ("convergence_", "convergence_results.json"),
            "memory_throughput_profiler": ("profiler_", "memory_throughput_results.json"),
        }
        prefix, combined_name = prefix_map.get(script_name, (None, None))
        if not prefix:
            return

        pattern = f"{prefix}*_results.json"
        per_model_files = sorted(results_dir.glob(pattern))
        if not per_model_files and script_name == "convergence_experiment":
            per_model_files = sorted(results_dir.glob("convergence_*_results.json"))
            per_model_files = [
                f for f in per_model_files if "convergence_results.json" not in f.name
            ]

        if not per_model_files:
            print(f"  No per-model JSONs found for {script_name}")
            return

        combined = []
        for jf in per_model_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
                combined.append(data)
            except (json.JSONDecodeError, OSError):
                continue

        if combined:
            combined_path = results_dir / combined_name
            with open(combined_path, "w") as f:
                json.dump(combined, f, indent=2, default=str)
            print(f"  Regenerated: {combined_path} ({len(combined)} models)")

    def _build_command(self, task: GpuTask) -> list[str]:
        """Build the subprocess command for a task."""
        script_path = Path(__file__).parent / f"{task.script}.py"
        cmd = ["uv", "run", "python", str(script_path)]

        # Most scripts support --models flag
        if task.script != "gpt2_benchmark":
            cmd.extend(["--models", task.model_slug])

        return cmd

    def _read_peak_from_results(self, task: GpuTask) -> float | None:
        """Try to read actual peak VRAM from the script's result JSON."""
        results_dir = Path(__file__).parent

        # Try common result file patterns
        patterns = [
            f"extended_{task.model_slug}_results.json",
            f"convergence_{task.model_slug}_results.json",
            f"benchmark_{task.model_slug.replace('-', '_')}.json",
        ]

        for pattern in patterns:
            path = results_dir / pattern
            if path.exists():
                try:
                    with open(path) as f:
                        data = json.load(f)
                    peak = _find_peak_memory(data)
                    if peak is not None:
                        return peak
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

        return None


# ---------------------------------------------------------------------------
# Calibration from existing results
# ---------------------------------------------------------------------------


def _find_peak_memory(data: dict | list) -> float | None:
    """Search a result dict for peak memory measurement.

    Checks common key patterns at the top level and one level deep.
    Returns the maximum peak_memory_mb found (representing actual usage).
    """
    if not isinstance(data, dict):
        return None

    candidates: list[float] = []
    memory_keys = ["peak_memory_mb", "memory_peak_mb", "bwsk_peak_memory_mb"]

    # Top-level keys
    for key in memory_keys:
        val = data.get(key)
        if val is not None:
            with contextlib.suppress(ValueError, TypeError):
                candidates.append(float(val))

    # One level deep (e.g. finetune_conventional.peak_memory_mb)
    for _key, section_val in data.items():
        if isinstance(section_val, dict):
            for mkey in memory_keys:
                val = section_val.get(mkey)
                if val is not None:
                    with contextlib.suppress(ValueError, TypeError):
                        candidates.append(float(val))

    return max(candidates) if candidates else None


def calibrate_from_results() -> dict[str, float]:
    """Scan existing result JSONs and populate calibration file.

    Reads all *_results.json files in scripts/ and extracts peak VRAM
    measurements. This lets the scheduler use real data without running
    benchmarks first.

    Returns:
        Dict of model slug -> peak VRAM MB.
    """
    results_dir = Path(__file__).parent
    cal = load_calibration()
    found = 0

    for json_path in sorted(results_dir.glob("*_results.json")):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Extract slug from filename pattern: extended_pythia-70m_results.json
        name = json_path.stem  # e.g. "extended_pythia-70m_results"
        parts = name.rsplit("_results", 1)[0]  # "extended_pythia-70m"

        # Find peak memory in data — check nested structures too.
        # Extended benchmark stores it in finetune_conventional.peak_memory_mb
        peak = _find_peak_memory(data)

        if peak is not None and peak > 0:
            # Extract model slug: remove script prefix
            for prefix in [
                "extended_",
                "convergence_",
                "benchmark_",
                "profiler_",
            ]:
                if parts.startswith(prefix):
                    slug = parts[len(prefix) :]
                    cal[slug] = peak
                    found += 1
                    break

    if found:
        save_calibration(cal)
        print(f"Calibrated {found} models from existing result files.")
    else:
        print("No peak memory data found in result files.")

    return cal


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VRAM-aware GPU task scheduler for BWSK benchmarks",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="",
        help="Schedule tasks for one script (e.g. extended_benchmark)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model slugs to include",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Schedule tasks from all benchmark scripts",
    )
    parser.add_argument(
        "--schedule-only",
        action="store_true",
        help="Show schedule without executing",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Populate calibration from existing result JSONs",
    )
    parser.add_argument(
        "--safety",
        type=float,
        default=0.9,
        help="VRAM safety factor (default: 0.9 = use 90%% of VRAM)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the GPU scheduler CLI."""
    args = parse_args()

    if args.calibrate:
        calibrate_from_results()
        return

    # VRAMScheduler auto-detects GPU VRAM when total_vram_mb=0
    scheduler = VRAMScheduler(safety_factor=args.safety)

    model_filter = None
    if args.models:
        model_filter = {s.strip() for s in args.models.split(",")}

    if args.all:
        for script_name in SCRIPT_MODELS:
            scheduler.add_tasks_for_script(script_name, model_filter)
    elif args.script:
        if args.script not in SCRIPT_MODELS:
            print(f"ERROR: Unknown script '{args.script}'")
            print(f"Available: {', '.join(SCRIPT_MODELS)}")
            sys.exit(1)
        scheduler.add_tasks_for_script(args.script, model_filter)
    else:
        print("ERROR: Specify --script <name>, --all, or --calibrate")
        sys.exit(1)

    if args.schedule_only:
        print(scheduler.format_schedule())
    else:
        scheduler.execute()


if __name__ == "__main__":
    main()
