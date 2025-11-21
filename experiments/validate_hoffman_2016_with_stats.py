#!/usr/bin/env python3
"""Statistical validation experiment for Hoffman et al. (2016) topological mapping.

This script extends validate_hoffman_2016.py with multi-trial statistical aggregation:
- Runs multiple trials with different seeds
- Computes mean, std, confidence intervals
- Generates plots with error bars
- Outputs statistical reports (CSV/JSON)

Usage:
    python experiments/validate_hoffman_2016_with_stats.py \
        --num-trials 10 \
        --integration-windows 0 120 240 480 \
        --duration 900
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Import from existing validation script
# Add experiments to path for imports
EXPERIMENTS_PATH = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_PATH) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_PATH))

# Import functions from validate_hoffman_2016
import importlib.util
spec = importlib.util.spec_from_file_location("validate_hoffman_2016", EXPERIMENTS_PATH / "validate_hoffman_2016.py")
validate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_module)

run_learning_experiment = validate_module.run_learning_experiment
estimate_learning_time = validate_module.estimate_learning_time
_sample_uniform_positions = validate_module._sample_uniform_positions
_generate_obstacle_ring_positions = validate_module._generate_obstacle_ring_positions
_generate_ring_spoke_positions = validate_module._generate_ring_spoke_positions

from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellControllerConfig,
)
from hippocampus_core.env import Agent, CircularObstacle, Environment
from hippocampus_core.stats import (
    aggregate_trials,
    aggregate_time_series,
    compute_confidence_interval,
    compute_success_rate,
    format_statistic,
)


def run_multiple_trials(
    env: Environment,
    integration_window: Optional[float],
    num_trials: int = 10,
    base_seed: int = 42,
    duration_seconds: float = 300.0,
    dt: float = 0.05,
    num_place_cells: int = 100,
    sigma: float = 0.15,
    coactivity_threshold: float = 5.0,
    max_edge_distance: Optional[float] = None,
    place_cell_positions: Optional[np.ndarray] = None,
    trajectory_mode: str = "random",
    trajectory_params: Optional[dict] = None,
    expected_b1: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run multiple trials with different seeds.

    Parameters
    ----------
    env:
        Environment to navigate in.
    integration_window:
        Integration window in seconds.
    num_trials:
        Number of trials to run.
    base_seed:
        Base random seed (each trial uses base_seed + trial_index).
    duration_seconds:
        Total simulation duration.
    dt:
        Simulation time step.
    num_place_cells:
        Number of place cells.
    sigma:
        Place field size parameter.
    coactivity_threshold:
        Minimum coactivity threshold.
    max_edge_distance:
        Maximum edge distance.
    place_cell_positions:
        Optional explicit place cell positions.
    trajectory_mode:
        Trajectory generation mode.
    trajectory_params:
        Optional trajectory parameters.
    expected_b1:
        Expected Betti number b₁.

    Returns
    -------
    List[Dict[str, Any]]
        List of result dictionaries, one per trial.
    """
    results_list = []

    print(f"Running {num_trials} trials with integration window ϖ={integration_window}s...")

    for trial_idx in range(num_trials):
        seed = base_seed + trial_idx
        print(f"  Trial {trial_idx + 1}/{num_trials} (seed={seed})...", end="", flush=True)

        try:
            result = run_learning_experiment(
                env=env,
                integration_window=integration_window,
                duration_seconds=duration_seconds,
                dt=dt,
                num_place_cells=num_place_cells,
                sigma=sigma,
                seed=seed,
                expected_b1=expected_b1,
                coactivity_threshold=coactivity_threshold,
                max_edge_distance=max_edge_distance,
                place_cell_positions=place_cell_positions,
                trajectory_mode=trajectory_mode,
                trajectory_params=trajectory_params,
            )

            # Add trial metadata
            result["trial_index"] = trial_idx
            result["seed"] = seed

            # Compute final values
            if result["edges"]:
                result["final_edges"] = result["edges"][-1]
            if result["components"]:
                result["final_components"] = result["components"][-1]
            if result["betti_0"]:
                result["final_betti_0"] = result["betti_0"][-1]
            if result["betti_1"]:
                result["final_betti_1"] = result["betti_1"][-1]

            # Estimate learning time
            result["t_min"] = estimate_learning_time(result, target_betti_0=1, target_betti_1=expected_b1 or 0)

            results_list.append(result)
            print(" ✓")

        except Exception as e:
            print(f" ✗ Error: {e}")
            continue

    print(f"Completed {len(results_list)}/{num_trials} trials successfully.")
    return results_list


def plot_with_error_bars(
    times: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    ax: plt.Axes,
    label: str,
    color: Optional[str] = None,
    alpha: float = 0.7,
) -> None:
    """Plot time series with error bars.

    Parameters
    ----------
    times:
        Time points.
    means:
        Mean values.
    stds:
        Standard deviations.
    ax:
        Matplotlib axes.
    label:
        Label for the plot.
    color:
        Optional color.
    alpha:
        Transparency.
    """
    ax.plot(times, means, label=label, color=color, alpha=alpha, linewidth=2)
    ax.fill_between(
        times,
        means - stds,
        means + stds,
        alpha=alpha * 0.3,
        color=color,
        label=f"{label} (±1 std)",
    )


def create_statistical_plots(
    aggregated_time_series: Dict[str, tuple],
    aggregated_final: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None,
) -> None:
    """Create plots with statistical information.

    Parameters
    ----------
    aggregated_time_series:
        Dictionary of time series data (times, means, stds).
    aggregated_final:
        Dictionary of final value statistics.
    output_path:
        Optional path to save figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Statistical Aggregation of Topological Learning Experiments", fontsize=14, fontweight="bold")

    # Plot 1: Edges over time
    ax1 = axes[0, 0]
    if "edges" in aggregated_time_series:
        times, means, stds = aggregated_time_series["edges"]
        plot_with_error_bars(times, means, stds, ax1, "Edges", color="blue")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Number of Edges")
    ax1.set_title("Edge Formation Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Betti numbers over time
    ax2 = axes[0, 1]
    if "betti_0" in aggregated_time_series:
        times, means, stds = aggregated_time_series["betti_0"]
        plot_with_error_bars(times, means, stds, ax2, "b₀ (Components)", color="green")
    if "betti_1" in aggregated_time_series:
        times, means, stds = aggregated_time_series["betti_1"]
        plot_with_error_bars(times, means, stds, ax2, "b₁ (Holes)", color="red")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Betti Number")
    ax2.set_title("Topology Evolution Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Box plot of final values
    ax3 = axes[1, 0]
    if aggregated_final:
        metrics = []
        data_list = []
        labels = []

        for key in ["final_edges", "final_components", "final_betti_0", "final_betti_1"]:
            if key in aggregated_final:
                stats = aggregated_final[key]
                # Create synthetic data from statistics for box plot
                # (In practice, you'd use raw data, but this works for visualization)
                mean = stats["mean"]
                std = stats["std"]
                q25 = stats["q25"]
                q75 = stats["q75"]
                median = stats["median"]
                min_val = stats["min"]
                max_val = stats["max"]

                # Approximate distribution for box plot
                # Use quartiles and median
                data_list.append([min_val, q25, median, q75, max_val])
                labels.append(key.replace("final_", "").replace("_", " ").title())

        if data_list:
            bp = ax3.boxplot(data_list, labels=labels, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightblue")
                patch.set_alpha(0.7)
            ax3.set_ylabel("Value")
            ax3.set_title("Final Statistics Distribution")
            ax3.grid(True, alpha=0.3, axis="y")
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis("off")

    if aggregated_final:
        table_data = []
        table_data.append(["Metric", "Mean ± Std", "Median", "95% CI"])

        for key in ["final_edges", "final_components", "final_betti_0", "final_betti_1", "t_min"]:
            if key in aggregated_final:
                stats = aggregated_final[key]
                metric_name = key.replace("final_", "").replace("_", " ").title()
                mean_std = format_statistic(stats["mean"], stats["std"])
                median = f"{stats['median']:.2f}"
                ci = f"[{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}]"
                table_data.append([metric_name, mean_std, median, ci])

        table = ax4.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            cellLoc="left",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax4.set_title("Statistical Summary", fontweight="bold", pad=20)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved statistical plots to: {output_path}")
    else:
        plt.show()


def save_statistical_report(
    aggregated_final: Dict[str, Dict[str, float]],
    aggregated_time_series: Dict[str, tuple],
    output_path: Path,
    format: str = "json",
) -> None:
    """Save statistical report to file.

    Parameters
    ----------
    aggregated_final:
        Dictionary of final value statistics.
    aggregated_time_series:
        Dictionary of time series data.
    output_path:
        Path to save report.
    format:
        Output format: "json" or "csv".
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        report = {
            "final_statistics": aggregated_final,
            "time_series_summary": {
                key: {
                    "times": times.tolist(),
                    "means": means.tolist(),
                    "stds": stds.tolist(),
                }
                for key, (times, means, stds) in aggregated_time_series.items()
            },
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Saved JSON report to: {output_path}")

    elif format == "csv":
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Mean", "Std", "Median", "Q25", "Q75", "Min", "Max", "CI_Lower", "CI_Upper"])

            for key, stats in aggregated_final.items():
                writer.writerow([
                    key,
                    stats["mean"],
                    stats["std"],
                    stats["median"],
                    stats["q25"],
                    stats["q75"],
                    stats["min"],
                    stats["max"],
                    stats["ci_lower"],
                    stats["ci_upper"],
                ])

        print(f"Saved CSV report to: {output_path}")


def main() -> None:
    """Run statistical validation experiment."""
    parser = argparse.ArgumentParser(
        description="Statistical validation of Hoffman et al. (2016) with multiple trials"
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=10,
        help="Number of trials to run (default: 10)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--integration-windows",
        type=float,
        nargs="+",
        default=[0, 120, 240, 480],
        help="Integration window values in seconds (default: 0 120 240 480)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="Simulation duration in seconds (default: 300)",
    )
    parser.add_argument(
        "--num-cells",
        type=int,
        default=100,
        help="Number of place cells (default: 100)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.15,
        help="Place field size parameter (default: 0.15)",
    )
    parser.add_argument(
        "--coactivity-threshold",
        type=float,
        default=5.0,
        help="Coactivity threshold (default: 5.0)",
    )
    parser.add_argument(
        "--obstacle",
        action="store_true",
        help="Use obstacle environment",
    )
    parser.add_argument(
        "--obstacle-radius",
        type=float,
        default=0.15,
        help="Obstacle radius (default: 0.15)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for figure (default: results/validate_stats.png)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Output path for statistical report (JSON/CSV)",
    )
    parser.add_argument(
        "--report-format",
        type=str,
        choices=["json", "csv"],
        default="json",
        help="Report format (default: json)",
    )

    args = parser.parse_args()

    # Create environment
    if args.obstacle:
        obstacle = CircularObstacle(center_x=0.5, center_y=0.5, radius=args.obstacle_radius)
        env = Environment(width=1.0, height=1.0, obstacles=[obstacle])
        expected_b1 = 1
    else:
        env = Environment(width=1.0, height=1.0)
        expected_b1 = 0

    max_edge_distance = args.sigma * 2.0  # Default multiplier

    print("=" * 70)
    print("Statistical Validation: Hoffman et al. (2016)")
    print("=" * 70)
    print(f"Number of trials: {args.num_trials}")
    print(f"Integration windows: {args.integration_windows} seconds")
    print(f"Simulation duration: {args.duration} seconds ({args.duration/60:.1f} minutes)")
    print(f"Number of place cells: {args.num_cells}")
    print(f"Place-field sigma: {args.sigma}")
    print(f"Coactivity threshold: {args.coactivity_threshold}")
    print(f"Obstacle environment: {'Yes' if args.obstacle else 'No'}")
    if args.obstacle:
        print(f"  Obstacle radius: {args.obstacle_radius}")
    print()

    # Run trials for each integration window
    all_results: Dict[Optional[float], List[Dict[str, Any]]] = {}

    for integration_window_sec in args.integration_windows:
        integration_window = None if integration_window_sec == 0 else integration_window_sec

        results_list = run_multiple_trials(
            env=env,
            integration_window=integration_window,
            num_trials=args.num_trials,
            base_seed=args.base_seed,
            duration_seconds=args.duration,
            num_place_cells=args.num_cells,
            sigma=args.sigma,
            coactivity_threshold=args.coactivity_threshold,
            max_edge_distance=max_edge_distance,
            expected_b1=expected_b1,
        )

        all_results[integration_window] = results_list

    # Aggregate statistics
    print("\n" + "=" * 70)
    print("Statistical Aggregation")
    print("=" * 70)

    # Aggregate for each integration window
    aggregated_by_window: Dict[Optional[float], Dict[str, Any]] = {}

    for integration_window, results_list in all_results.items():
        if not results_list:
            continue

        # Aggregate final values
        aggregated_final = aggregate_trials(results_list)

        # Aggregate time series
        aggregated_time_series = aggregate_time_series(results_list)

        # Compute success rates
        def reached_target_b0(result: Dict[str, Any]) -> bool:
            return result.get("final_betti_0", -1) == 1

        def reached_target_b1(result: Dict[str, Any]) -> bool:
            return result.get("final_betti_1", -1) == expected_b1

        success_b0, num_success_b0, total = compute_success_rate(results_list, reached_target_b0)
        success_b1, num_success_b1, total = compute_success_rate(results_list, reached_target_b1)

        aggregated_by_window[integration_window] = {
            "final_stats": aggregated_final,
            "time_series": aggregated_time_series,
            "success_rates": {
                "b0_target": {"rate": success_b0, "successful": num_success_b0, "total": total},
                "b1_target": {"rate": success_b1, "successful": num_success_b1, "total": total},
            },
        }

        # Print summary
        window_str = f"ϖ={integration_window}s" if integration_window else "ϖ=None"
        print(f"\n{window_str}:")
        if "final_edges" in aggregated_final:
            stats = aggregated_final["final_edges"]
            print(f"  Final edges: {format_statistic(stats['mean'], stats['std'])}")
        if "final_betti_0" in aggregated_final:
            stats = aggregated_final["final_betti_0"]
            print(f"  Final b₀: {format_statistic(stats['mean'], stats['std'])}")
        if "final_betti_1" in aggregated_final:
            stats = aggregated_final["final_betti_1"]
            print(f"  Final b₁: {format_statistic(stats['mean'], stats['std'])}")
        if "t_min" in aggregated_final:
            stats = aggregated_final["t_min"]
            if stats["mean"] > 0:
                print(f"  T_min: {format_statistic(stats['mean'], stats['std'])} seconds")
        print(f"  Success rate (b₀=1): {success_b0*100:.1f}% ({num_success_b0}/{total})")
        print(f"  Success rate (b₁={expected_b1}): {success_b1*100:.1f}% ({num_success_b1}/{total})")

    # Create plots (use first integration window for detailed plots)
    if aggregated_by_window:
        first_window = sorted(aggregated_by_window.keys(), key=lambda x: x or 0)[0]
        first_agg = aggregated_by_window[first_window]

        output_path = Path(args.output) if args.output else Path("results/validate_stats.png")
        create_statistical_plots(
            first_agg["time_series"],
            first_agg["final_stats"],
            output_path,
        )

    # Save report
    if args.report:
        report_path = Path(args.report)
        if aggregated_by_window:
            first_window = sorted(aggregated_by_window.keys(), key=lambda x: x or 0)[0]
            first_agg = aggregated_by_window[first_window]
            save_statistical_report(
                first_agg["final_stats"],
                first_agg["time_series"],
                report_path,
                format=args.report_format,
            )

    print("\n" + "=" * 70)
    print("Statistical validation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

