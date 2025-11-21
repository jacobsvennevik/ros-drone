#!/usr/bin/env python3
"""Parameter sweep for Yartsev et al. (2011) grid validation.

This script runs the grid stability analysis across different parameter sets:
- Different calibration intervals
- Different grid sizes
- Different grid tau values
- Different simulation durations

Results are saved to results/sweeps/ with summary plots.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hippocampus_core.controllers.bat_navigation_controller import (
    BatNavigationController,
    BatNavigationControllerConfig,
)
from hippocampus_core.env import Agent, Environment


def theta_index(signal, dt: float) -> float:
    """Compute theta-band power index."""
    signal = np.asarray(signal)
    if len(signal) == 0:
        return 0.0
    
    fft = np.fft.rfft(signal - np.mean(signal))
    freqs = np.fft.rfftfreq(signal.size, d=dt)
    theta_band = (freqs >= 4.0) & (freqs <= 10.0)
    
    total_power = np.sum(np.abs(fft) ** 2)
    if total_power == 0:
        return 0.0
    
    return np.sum(np.abs(fft[theta_band]) ** 2) / total_power


def simulate(
    duration_seconds: float = 600.0,
    dt: float = 0.05,
    grid_size: tuple[int, int] = (20, 20),
    calibration_interval: int = 400,
    grid_tau: float = 0.05,
    seed: int = 11,
):
    """Run Yartsev grid validation simulation."""
    env = Environment(width=1.0, height=1.0)
    config = BatNavigationControllerConfig(
        num_place_cells=80,
        hd_num_neurons=60,
        grid_size=grid_size,
        calibration_interval=calibration_interval,
        integration_window=480.0,
        grid_tau=grid_tau,
    )
    controller = BatNavigationController(env, config=config, rng=np.random.default_rng(seed))
    agent = Agent(env, random_state=np.random.default_rng(seed + 1), track_heading=True)

    num_steps = int(duration_seconds / dt)
    theta_power = []
    grid_norm = []
    velocity_history = []
    
    for step in range(num_steps):
        obs = agent.step(dt, include_theta=True)
        controller.step(obs, dt)
        velocity_history.append(controller.grid_attractor.estimate_position().copy())
        grid_norm.append(controller.grid_attractor.drift_metric())

        if (step + 1) % 200 == 0:
            recent = np.array(velocity_history[-200:])
            if len(recent) > 0:
                theta_power.append(theta_index(recent[:, 0], dt))

    return {
        "theta_power": np.array(theta_power) if theta_power else np.array([]),
        "grid_norm": np.array(grid_norm),
        "mean_grid_drift": np.mean(grid_norm),
        "std_grid_drift": np.std(grid_norm),
        "final_grid_drift": grid_norm[-1] if grid_norm else 0.0,
        "mean_theta_power": np.mean(theta_power) if theta_power else 0.0,
        "max_theta_power": np.max(theta_power) if theta_power else 0.0,
    }


def sweep_calibration_interval(
    intervals: list[int],
    duration_seconds: float = 600.0,
    num_trials: int = 5,
    base_seed: int = 11,
) -> dict:
    """Sweep calibration interval parameter."""
    results = {}
    for calib_int in intervals:
        trial_results = []
        for trial in range(num_trials):
            seed = base_seed + trial
            result = simulate(
                duration_seconds=duration_seconds,
                calibration_interval=calib_int,
                seed=seed,
            )
            trial_results.append(result)
        
        if not trial_results:
            # Guard against empty trial results
            results[calib_int] = {
                "mean_grid_drift": 0.0,
                "std_grid_drift": 0.0,
                "mean_theta_power": 0.0,
                "std_theta_power": 0.0,
                "trials": [],
            }
        else:
            results[calib_int] = {
                "mean_grid_drift": np.mean([r["mean_grid_drift"] for r in trial_results]),
                "std_grid_drift": np.std([r["mean_grid_drift"] for r in trial_results]),
                "mean_theta_power": np.mean([r["mean_theta_power"] for r in trial_results]),
                "std_theta_power": np.std([r["mean_theta_power"] for r in trial_results]),
                "trials": trial_results,
            }
    return results


def sweep_grid_size(
    grid_sizes: list[tuple[int, int]],
    duration_seconds: float = 600.0,
    num_trials: int = 5,
    base_seed: int = 11,
) -> dict:
    """Sweep grid size parameter."""
    results = {}
    for grid_size in grid_sizes:
        trial_results = []
        for trial in range(num_trials):
            seed = base_seed + trial
            result = simulate(
                duration_seconds=duration_seconds,
                grid_size=grid_size,
                seed=seed,
            )
            trial_results.append(result)
        
        grid_str = f"{grid_size[0]}x{grid_size[1]}"
        if not trial_results:
            # Guard against empty trial results
            results[grid_str] = {
                "mean_grid_drift": 0.0,
                "std_grid_drift": 0.0,
                "mean_theta_power": 0.0,
                "std_theta_power": 0.0,
                "trials": [],
            }
        else:
            results[grid_str] = {
                "mean_grid_drift": np.mean([r["mean_grid_drift"] for r in trial_results]),
                "std_grid_drift": np.std([r["mean_grid_drift"] for r in trial_results]),
                "mean_theta_power": np.mean([r["mean_theta_power"] for r in trial_results]),
                "std_theta_power": np.std([r["mean_theta_power"] for r in trial_results]),
                "trials": trial_results,
            }
    return results


def plot_sweep_results(
    sweep_results: dict,
    parameter_name: str,
    output_path: Path,
    metric: str = "grid_drift",
    title_prefix: str = "Yartsev Grid Validation",
):
    """Plot parameter sweep results."""
    if not sweep_results:
        raise ValueError(f"No sweep results to plot for {parameter_name}")
    
    # Sort parameter values intelligently
    def sort_key(key):
        """Sort key that handles both numeric and string grid sizes."""
        if isinstance(key, (int, float)):
            return (0, key)  # Numeric values first
        # Try to parse grid size strings like "12x12"
        if isinstance(key, str) and "x" in key:
            try:
                size = int(key.split("x")[0])
                return (1, size)  # Sort by first dimension
            except ValueError:
                pass
        # Default string sort
        return (2, str(key))
    
    param_values = sorted(sweep_results.keys(), key=sort_key)
    
    if metric == "grid_drift":
        means = [sweep_results[p]["mean_grid_drift"] for p in param_values]
        stds = [sweep_results[p]["std_grid_drift"] for p in param_values]
        ylabel = "Grid Drift Metric"
    else:
        means = [sweep_results[p]["mean_theta_power"] for p in param_values]
        stds = [sweep_results[p]["std_theta_power"] for p in param_values]
        ylabel = "Theta-Band Power"
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Convert param_values to numeric if possible
    try:
        x_values = [float(p) for p in param_values]
    except (ValueError, TypeError):
        x_values = list(range(len(param_values)))
        ax.set_xticks(x_values)
        ax.set_xticklabels(param_values, rotation=45, ha="right")
    
    ax.errorbar(
        x_values,
        means,
        yerr=stds,
        marker="o",
        linewidth=2,
        capsize=5,
    )
    
    ax.set_xlabel(f"{parameter_name}")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix}: {ylabel} vs {parameter_name}")
    ax.grid(True, alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved sweep plot to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter sweep for Yartsev grid validation"
    )
    parser.add_argument(
        "--sweep",
        type=str,
        choices=["calibration", "grid_size", "both"],
        default="both",
        help="Which parameter to sweep",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sweeps"),
        help="Output directory for results (default: results/sweeps)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=600.0,
        help="Simulation duration in seconds (default: 600)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials per parameter value (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Base random seed (default: 11)",
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Yartsev Grid Validation Parameter Sweep")
    print("=" * 70)
    print()
    
    if args.sweep in ("calibration", "both"):
        print("Sweeping calibration interval...")
        calib_intervals = [200, 300, 400, 500, 600]
        calib_results = sweep_calibration_interval(
            calib_intervals,
            duration_seconds=args.duration,
            num_trials=args.trials,
            base_seed=args.seed,
        )
        
        # Print summary
        print("\nCalibration Interval Results:")
        print(f"{'Interval':>12} | {'Grid Drift':>15} | {'Theta Power':>15}")
        print("-" * 50)
        for calib_int in sorted(calib_results.keys()):
            r = calib_results[calib_int]
            print(
                f"{calib_int:>12} | "
                f"{r['mean_grid_drift']:>7.4f} ± {r['std_grid_drift']:>5.4f} | "
                f"{r['mean_theta_power']:>7.4f} ± {r['std_theta_power']:>5.4f}"
            )
        
        # Plot grid drift
        plot_sweep_results(
            calib_results,
            "Calibration Interval",
            output_dir / "yartsev_sweep_calibration_drift.png",
            metric="grid_drift",
        )
        # Plot theta power
        plot_sweep_results(
            calib_results,
            "Calibration Interval",
            output_dir / "yartsev_sweep_calibration_theta.png",
            metric="theta_power",
        )
        print()
    
    if args.sweep in ("grid_size", "both"):
        print("Sweeping grid size...")
        grid_sizes = [(12, 12), (16, 16), (20, 20), (24, 24)]
        grid_results = sweep_grid_size(
            grid_sizes,
            duration_seconds=args.duration,
            num_trials=args.trials,
            base_seed=args.seed,
        )
        
        # Print summary
        print("\nGrid Size Results:")
        print(f"{'Grid Size':>12} | {'Grid Drift':>15} | {'Theta Power':>15}")
        print("-" * 50)
        for grid_str in sorted(grid_results.keys()):
            r = grid_results[grid_str]
            print(
                f"{grid_str:>12} | "
                f"{r['mean_grid_drift']:>7.4f} ± {r['std_grid_drift']:>5.4f} | "
                f"{r['mean_theta_power']:>7.4f} ± {r['std_theta_power']:>5.4f}"
            )
        
        # Plot grid drift
        plot_sweep_results(
            grid_results,
            "Grid Size",
            output_dir / "yartsev_sweep_grid_size_drift.png",
            metric="grid_drift",
        )
        # Plot theta power
        plot_sweep_results(
            grid_results,
            "Grid Size",
            output_dir / "yartsev_sweep_grid_size_theta.png",
            metric="theta_power",
        )
        print()
    
    print("Sweep complete!")


if __name__ == "__main__":
    main()

