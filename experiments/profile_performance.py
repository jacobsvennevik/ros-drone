#!/usr/bin/env python3
"""Performance profiling for topological mapping system.

This script profiles key components to identify bottlenecks:
- Coactivity tracking (spike registration, matrix updates)
- Graph construction (edge building, integration window checking)
- Betti number computation (clique extraction, persistent homology)

Usage:
    python3 experiments/profile_performance.py \
        --duration 60 \
        --num-cells 100 \
        --output results/profile_report.txt
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)
from hippocampus_core.env import Agent, Environment


def profile_coactivity_tracking(
    num_cells: int,
    duration: float,
    dt: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Profile coactivity tracking performance.

    Parameters
    ----------
    num_cells:
        Number of place cells.
    duration:
        Simulation duration in seconds.
    dt:
        Time step.
    seed:
        Random seed.

    Returns
    -------
    dict
        Performance metrics.
    """
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(seed)

    config = PlaceCellControllerConfig(
        num_place_cells=num_cells,
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.2,
        coactivity_threshold=5.0,
        max_edge_distance=0.3,
        integration_window=None,  # No integration window for pure coactivity profiling
    )

    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent = Agent(environment=env, random_state=rng)

    num_steps = int(duration / dt)

    # Profile the step function
    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.perf_counter()
    for _ in range(num_steps):
        position = agent.step(dt)
        controller.step(position, dt)
    end_time = time.perf_counter()

    profiler.disable()

    # Extract timing
    total_time = end_time - start_time
    time_per_step = total_time / num_steps

    # Get coactivity matrix size
    coactivity_matrix = controller.get_coactivity_matrix()
    matrix_size = coactivity_matrix.size

    # Analyze profiler stats
    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream)
    stats.sort_stats("cumulative")
    stats.print_stats(20)  # Top 20 functions
    stats_output = stats_stream.getvalue()

    # Extract key function timings
    coactivity_time = 0.0
    spike_time = 0.0
    rate_time = 0.0

    for line in stats_output.split("\n"):
        if "coactivity" in line.lower() or "register_spikes" in line.lower():
            parts = line.split()
            if len(parts) > 4:
                try:
                    coactivity_time += float(parts[2])  # cumulative time
                except (ValueError, IndexError):
                    pass
        if "sample_spikes" in line.lower():
            parts = line.split()
            if len(parts) > 4:
                try:
                    spike_time += float(parts[2])
                except (ValueError, IndexError):
                    pass
        if "get_rates" in line.lower():
            parts = line.split()
            if len(parts) > 4:
                try:
                    rate_time += float(parts[2])
                except (ValueError, IndexError):
                    pass

    return {
        "total_time": total_time,
        "time_per_step": time_per_step,
        "steps_per_second": num_steps / total_time,
        "coactivity_time": coactivity_time,
        "spike_time": spike_time,
        "rate_time": rate_time,
        "matrix_size": matrix_size,
        "profiler_stats": stats_output,
    }


def profile_graph_construction(
    num_cells: int,
    duration: float,
    integration_window: float | None = None,
    dt: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Profile graph construction performance.

    Parameters
    ----------
    num_cells:
        Number of place cells.
    duration:
        Simulation duration in seconds.
    integration_window:
        Integration window in seconds (None for no window).
    dt:
        Time step.
    seed:
        Random seed.

    Returns
    -------
    dict
        Performance metrics.
    """
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(seed)

    config = PlaceCellControllerConfig(
        num_place_cells=num_cells,
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.2,
        coactivity_threshold=5.0,
        max_edge_distance=0.3,
        integration_window=integration_window,
    )

    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent = Agent(environment=env, random_state=rng)

    num_steps = int(duration / dt)
    sample_interval = max(1, num_steps // 20)  # Sample 20 times

    # Run simulation
    for _ in range(num_steps):
        position = agent.step(dt)
        controller.step(position, dt)

    # Profile graph construction
    graph_times: List[float] = []
    edge_counts: List[int] = []

    profiler = cProfile.Profile()
    for _ in range(10):  # Profile multiple graph builds
        profiler.enable()
        start = time.perf_counter()
        graph = controller.get_graph()
        end = time.perf_counter()
        profiler.disable()

        graph_times.append(end - start)
        edge_counts.append(graph.num_edges())

    avg_graph_time = np.mean(graph_times)
    std_graph_time = np.std(graph_times)

    # Analyze profiler stats
    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    stats_output = stats_stream.getvalue()

    # Extract key function timings
    build_time = 0.0
    distance_time = 0.0

    for line in stats_output.split("\n"):
        if "build_from_coactivity" in line.lower():
            parts = line.split()
            if len(parts) > 4:
                try:
                    build_time += float(parts[2])
                except (ValueError, IndexError):
                    pass
        if "distance" in line.lower() or "norm" in line.lower():
            parts = line.split()
            if len(parts) > 4:
                try:
                    distance_time += float(parts[2])
                except (ValueError, IndexError):
                    pass

    return {
        "avg_graph_time": avg_graph_time,
        "std_graph_time": std_graph_time,
        "avg_edges": np.mean(edge_counts),
        "build_time": build_time,
        "distance_time": distance_time,
        "profiler_stats": stats_output,
    }


def profile_betti_computation(
    num_cells: int,
    duration: float,
    dt: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Profile Betti number computation performance.

    Parameters
    ----------
    num_cells:
        Number of place cells.
    duration:
        Simulation duration in seconds.
    dt:
        Time step.
    seed:
        Random seed.

    Returns
    -------
    dict
        Performance metrics.
    """
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(seed)

    config = PlaceCellControllerConfig(
        num_place_cells=num_cells,
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.2,
        coactivity_threshold=5.0,
        max_edge_distance=0.3,
        integration_window=None,
    )

    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent = Agent(environment=env, random_state=rng)

    num_steps = int(duration / dt)

    # Run simulation
    for _ in range(num_steps):
        position = agent.step(dt)
        controller.step(position, dt)

    graph = controller.get_graph()

    # Profile Betti computation
    betti_times: List[float] = []
    clique_times: List[float] = []

    profiler = cProfile.Profile()

    for _ in range(5):  # Profile multiple Betti computations
        # Profile clique extraction
        profiler.enable()
        start = time.perf_counter()
        cliques = graph.get_maximal_cliques()
        clique_time = time.perf_counter() - start
        profiler.disable()
        clique_times.append(clique_time)

        # Profile Betti computation
        try:
            profiler.enable()
            start = time.perf_counter()
            betti = graph.compute_betti_numbers(max_dim=2)
            betti_time = time.perf_counter() - start
            profiler.disable()
            betti_times.append(betti_time)
        except ImportError:
            betti_times.append(np.nan)
            break

    avg_betti_time = np.mean([t for t in betti_times if not np.isnan(t)])
    avg_clique_time = np.mean(clique_times)

    # Analyze profiler stats
    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    stats_output = stats_stream.getvalue()

    # Extract key function timings
    ripser_time = 0.0
    gudhi_time = 0.0

    for line in stats_output.split("\n"):
        if "ripser" in line.lower():
            parts = line.split()
            if len(parts) > 4:
                try:
                    ripser_time += float(parts[2])
                except (ValueError, IndexError):
                    pass
        if "gudhi" in line.lower():
            parts = line.split()
            if len(parts) > 4:
                try:
                    gudhi_time += float(parts[2])
                except (ValueError, IndexError):
                    pass

    return {
        "avg_betti_time": avg_betti_time,
        "avg_clique_time": avg_clique_time,
        "num_cliques": len(cliques),
        "ripser_time": ripser_time,
        "gudhi_time": gudhi_time,
        "profiler_stats": stats_output,
    }


def generate_report(
    coactivity_results: Dict[str, float],
    graph_results: Dict[str, float],
    betti_results: Dict[str, float],
    output_path: Path | None = None,
) -> None:
    """Generate performance profiling report.

    Parameters
    ----------
    coactivity_results:
        Coactivity tracking profiling results.
    graph_results:
        Graph construction profiling results.
    betti_results:
        Betti computation profiling results.
    output_path:
        Optional path to save report.
    """
    report_lines: List[str] = []

    report_lines.append("=" * 70)
    report_lines.append("Performance Profiling Report")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Coactivity Tracking
    report_lines.append("1. COACTIVITY TRACKING PERFORMANCE")
    report_lines.append("-" * 70)
    report_lines.append(f"Total time: {coactivity_results['total_time']:.3f} s")
    report_lines.append(f"Time per step: {coactivity_results['time_per_step']*1000:.3f} ms")
    report_lines.append(f"Steps per second: {coactivity_results['steps_per_second']:.1f}")
    report_lines.append(f"Coactivity matrix size: {coactivity_results['matrix_size']:,}")
    report_lines.append("")
    report_lines.append("Key Function Timings (cumulative):")
    report_lines.append(f"  - Coactivity registration: {coactivity_results['coactivity_time']:.3f} s")
    report_lines.append(f"  - Spike sampling: {coactivity_results['spike_time']:.3f} s")
    report_lines.append(f"  - Rate computation: {coactivity_results['rate_time']:.3f} s")
    report_lines.append("")

    # Graph Construction
    report_lines.append("2. GRAPH CONSTRUCTION PERFORMANCE")
    report_lines.append("-" * 70)
    report_lines.append(f"Average graph build time: {graph_results['avg_graph_time']*1000:.3f} ms")
    report_lines.append(f"Std dev: {graph_results['std_graph_time']*1000:.3f} ms")
    report_lines.append(f"Average edges: {graph_results['avg_edges']:.0f}")
    report_lines.append("")
    report_lines.append("Key Function Timings (cumulative):")
    report_lines.append(f"  - Build from coactivity: {graph_results['build_time']:.3f} s")
    report_lines.append(f"  - Distance calculations: {graph_results['distance_time']:.3f} s")
    report_lines.append("")

    # Betti Computation
    report_lines.append("3. BETTI NUMBER COMPUTATION PERFORMANCE")
    report_lines.append("-" * 70)
    if not np.isnan(betti_results["avg_betti_time"]):
        report_lines.append(f"Average Betti computation time: {betti_results['avg_betti_time']*1000:.3f} ms")
        report_lines.append(f"Average clique extraction time: {betti_results['avg_clique_time']*1000:.3f} ms")
        report_lines.append(f"Number of cliques: {betti_results['num_cliques']}")
        report_lines.append("")
        report_lines.append("Key Function Timings (cumulative):")
        report_lines.append(f"  - Ripser time: {betti_results['ripser_time']:.3f} s")
        report_lines.append(f"  - Gudhi time: {betti_results['gudhi_time']:.3f} s")
    else:
        report_lines.append("Betti computation not available (ripser/gudhi not installed)")
    report_lines.append("")

    # Recommendations
    report_lines.append("4. RECOMMENDATIONS")
    report_lines.append("-" * 70)

    # Coactivity recommendations
    if coactivity_results["time_per_step"] > 0.01:  # > 10 ms per step
        report_lines.append("⚠ Coactivity tracking is slow (>10 ms/step)")
        report_lines.append("  - Consider optimizing spike registration")
        report_lines.append("  - Check if matrix updates can be batched")
    else:
        report_lines.append("✓ Coactivity tracking is efficient")

    # Graph recommendations
    if graph_results["avg_graph_time"] > 0.1:  # > 100 ms
        report_lines.append("⚠ Graph construction is slow (>100 ms)")
        report_lines.append("  - Consider caching distance calculations")
        report_lines.append("  - Optimize integration window checking")
    else:
        report_lines.append("✓ Graph construction is efficient")

    # Betti recommendations
    if not np.isnan(betti_results["avg_betti_time"]):
        if betti_results["avg_betti_time"] > 1.0:  # > 1 second
            report_lines.append("⚠ Betti computation is slow (>1 s)")
            report_lines.append("  - Consider reducing number of cliques")
            report_lines.append("  - Use faster backend (ripser vs gudhi)")
        else:
            report_lines.append("✓ Betti computation is efficient")

    report_lines.append("")
    report_lines.append("=" * 70)

    # Detailed profiler stats
    report_lines.append("")
    report_lines.append("DETAILED PROFILER STATISTICS")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("Coactivity Tracking (Top 20 functions):")
    report_lines.append(coactivity_results["profiler_stats"])
    report_lines.append("")
    report_lines.append("Graph Construction (Top 20 functions):")
    report_lines.append(graph_results["profiler_stats"])
    report_lines.append("")
    if not np.isnan(betti_results["avg_betti_time"]):
        report_lines.append("Betti Computation (Top 20 functions):")
        report_lines.append(betti_results["profiler_stats"])

    report_text = "\n".join(report_lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report_text)
        print(f"\nSaved profiling report to: {output_path}")
    else:
        print(report_text)


def main() -> None:
    """Run performance profiling."""
    parser = argparse.ArgumentParser(description="Profile topological mapping performance")
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Simulation duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--num-cells",
        type=int,
        default=100,
        help="Number of place cells (default: 100)",
    )
    parser.add_argument(
        "--integration-window",
        type=float,
        default=None,
        help="Integration window in seconds (default: None)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/profile_report.txt"),
        help="Output path for report (default: results/profile_report.txt)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Performance Profiling")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Duration: {args.duration} s")
    print(f"  Place cells: {args.num_cells}")
    print(f"  Integration window: {args.integration_window or 'None'}")
    print(f"  Seed: {args.seed}")
    print()

    # Profile coactivity tracking
    print("Profiling coactivity tracking...")
    coactivity_results = profile_coactivity_tracking(
        num_cells=args.num_cells,
        duration=args.duration,
        seed=args.seed,
    )
    print(f"  ✓ Completed in {coactivity_results['total_time']:.2f} s")
    print()

    # Profile graph construction
    print("Profiling graph construction...")
    graph_results = profile_graph_construction(
        num_cells=args.num_cells,
        duration=args.duration,
        integration_window=args.integration_window,
        seed=args.seed,
    )
    print(f"  ✓ Completed (avg: {graph_results['avg_graph_time']*1000:.2f} ms)")
    print()

    # Profile Betti computation
    print("Profiling Betti number computation...")
    try:
        betti_results = profile_betti_computation(
            num_cells=args.num_cells,
            duration=args.duration,
            seed=args.seed,
        )
        if not np.isnan(betti_results["avg_betti_time"]):
            print(f"  ✓ Completed (avg: {betti_results['avg_betti_time']*1000:.2f} ms)")
        else:
            print("  ⚠ Skipped (ripser/gudhi not available)")
    except ImportError:
        betti_results = {
            "avg_betti_time": np.nan,
            "avg_clique_time": 0.0,
            "num_cliques": 0,
            "ripser_time": 0.0,
            "gudhi_time": 0.0,
            "profiler_stats": "Betti computation not available",
        }
        print("  ⚠ Skipped (ripser/gudhi not available)")
    print()

    # Generate report
    print("Generating report...")
    generate_report(coactivity_results, graph_results, betti_results, args.output)

    print()
    print("=" * 70)
    print("Profiling complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

