#!/usr/bin/env python3
"""Parameter sweep for Rubin et al. (2014) HD validation.

This script runs the HD tuning analysis across different parameter sets:
- Different calibration intervals
- Different HD neuron counts
- Different sigma values
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


def rayleigh_vector(theta: np.ndarray, weights: np.ndarray) -> float:
    """Compute Rayleigh vector length for directional tuning."""
    theta = np.asarray(theta)
    weights = np.asarray(weights)
    
    # Guard against empty arrays
    if len(theta) == 0 or len(weights) == 0:
        return 0.0
    
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        return 0.0
    
    vector = np.sum(weights * np.exp(1j * theta))
    return np.abs(vector) / weight_sum


def compute_tuning(obs_history, rate_history, place_cell_centers, sigma, cell_index=None):
    """Compute HD tuning inside/outside place field."""
    if cell_index is None:
        cell_index = int(np.argmax(rate_history.mean(axis=0)))

    positions = obs_history[:, :2]
    headings = obs_history[:, 2]
    rates = rate_history[:, cell_index]
    center = place_cell_centers[cell_index, :2]

    distance = np.linalg.norm(positions - center, axis=1)
    in_mask = distance <= sigma
    out_mask = ~in_mask

    heading_bins = np.linspace(-np.pi, np.pi, 25)
    def tuning_curve(mask):
        counts, _ = np.histogram(headings[mask], bins=heading_bins, weights=rates[mask])
        sample_counts, _ = np.histogram(headings[mask], bins=heading_bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            curve = np.where(sample_counts > 0, counts / sample_counts, 0.0)
        centers = 0.5 * (heading_bins[1:] + heading_bins[:-1])
        return centers, curve

    centers, in_curve = tuning_curve(in_mask)
    _, out_curve = tuning_curve(out_mask)

    return {
        "cell_index": cell_index,
        "center": center,
        "rayleigh_in": rayleigh_vector(headings[in_mask], rates[in_mask] + 1e-6),
        "rayleigh_out": rayleigh_vector(headings[out_mask], rates[out_mask] + 1e-6),
        "curve_centers": centers,
        "in_curve": in_curve,
        "out_curve": out_curve,
    }


def run_simulation(
    duration_seconds: float = 300.0,
    dt: float = 0.05,
    num_place_cells: int = 60,
    hd_num_neurons: int = 72,
    calibration_interval: int = 250,
    sigma: float = 0.1,
    seed: int = 7,
):
    """Run Rubin HD validation simulation."""
    env = Environment(width=1.0, height=1.0)
    config = BatNavigationControllerConfig(
        num_place_cells=num_place_cells,
        hd_num_neurons=hd_num_neurons,
        grid_size=(16, 16),
        calibration_interval=calibration_interval,
        integration_window=480.0,
        sigma=sigma,
    )
    controller = BatNavigationController(env, config=config, rng=np.random.default_rng(seed))
    agent = Agent(env, random_state=np.random.default_rng(seed + 1), track_heading=True)

    num_steps = int(duration_seconds / dt)
    obs_history = np.zeros((num_steps, 3), dtype=float)
    rate_history = np.zeros((num_steps, num_place_cells), dtype=float)

    for idx in range(num_steps):
        obs = agent.step(dt, include_theta=True)
        controller.step(obs, dt)
        obs_history[idx] = obs
        rate_history[idx] = controller.last_rates

    analysis = compute_tuning(
        obs_history,
        rate_history,
        controller.place_cells.get_positions(),
        sigma=config.sigma,
    )

    return {
        "rayleigh_in": analysis["rayleigh_in"],
        "rayleigh_out": analysis["rayleigh_out"],
        "obs_history": obs_history,
        "rate_history": rate_history,
        "analysis": analysis,
    }


def sweep_calibration_interval(
    intervals: list[int],
    duration_seconds: float = 300.0,
    num_trials: int = 5,
    base_seed: int = 7,
) -> dict:
    """Sweep calibration interval parameter."""
    results = {}
    for calib_int in intervals:
        trial_results = []
        for trial in range(num_trials):
            seed = base_seed + trial
            result = run_simulation(
                duration_seconds=duration_seconds,
                calibration_interval=calib_int,
                seed=seed,
            )
            trial_results.append(result)
        
        if not trial_results:
            # Guard against empty trial results
            results[calib_int] = {
                "mean_rayleigh_in": 0.0,
                "mean_rayleigh_out": 0.0,
                "std_rayleigh_in": 0.0,
                "std_rayleigh_out": 0.0,
                "trials": [],
            }
        else:
            results[calib_int] = {
                "mean_rayleigh_in": np.mean([r["rayleigh_in"] for r in trial_results]),
                "mean_rayleigh_out": np.mean([r["rayleigh_out"] for r in trial_results]),
                "std_rayleigh_in": np.std([r["rayleigh_in"] for r in trial_results]),
                "std_rayleigh_out": np.std([r["rayleigh_out"] for r in trial_results]),
                "trials": trial_results,
            }
    return results


def sweep_hd_neurons(
    neuron_counts: list[int],
    duration_seconds: float = 300.0,
    num_trials: int = 5,
    base_seed: int = 7,
) -> dict:
    """Sweep HD neuron count parameter."""
    results = {}
    for hd_neurons in neuron_counts:
        trial_results = []
        for trial in range(num_trials):
            seed = base_seed + trial
            result = run_simulation(
                duration_seconds=duration_seconds,
                hd_num_neurons=hd_neurons,
                seed=seed,
            )
            trial_results.append(result)
        
        if not trial_results:
            # Guard against empty trial results
            results[hd_neurons] = {
                "mean_rayleigh_in": 0.0,
                "mean_rayleigh_out": 0.0,
                "std_rayleigh_in": 0.0,
                "std_rayleigh_out": 0.0,
                "trials": [],
            }
        else:
            results[hd_neurons] = {
                "mean_rayleigh_in": np.mean([r["rayleigh_in"] for r in trial_results]),
                "mean_rayleigh_out": np.mean([r["rayleigh_out"] for r in trial_results]),
                "std_rayleigh_in": np.std([r["rayleigh_in"] for r in trial_results]),
                "std_rayleigh_out": np.std([r["rayleigh_out"] for r in trial_results]),
                "trials": trial_results,
            }
    return results


def plot_sweep_results(
    sweep_results: dict,
    parameter_name: str,
    output_path: Path,
    title_prefix: str = "Rubin HD Validation",
):
    """Plot parameter sweep results."""
    if not sweep_results:
        raise ValueError(f"No sweep results to plot for {parameter_name}")
    
    param_values = sorted(sweep_results.keys())
    mean_in = [sweep_results[p]["mean_rayleigh_in"] for p in param_values]
    mean_out = [sweep_results[p]["mean_rayleigh_out"] for p in param_values]
    std_in = [sweep_results[p]["std_rayleigh_in"] for p in param_values]
    std_out = [sweep_results[p]["std_rayleigh_out"] for p in param_values]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.errorbar(
        param_values,
        mean_in,
        yerr=std_in,
        marker="o",
        label="Inside field",
        linewidth=2,
        capsize=5,
    )
    ax.errorbar(
        param_values,
        mean_out,
        yerr=std_out,
        marker="s",
        label="Outside field",
        linewidth=2,
        capsize=5,
        linestyle="--",
    )
    
    ax.set_xlabel(f"{parameter_name}")
    ax.set_ylabel("Rayleigh Vector Length")
    ax.set_title(f"{title_prefix}: HD Tuning vs {parameter_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved sweep plot to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter sweep for Rubin HD validation"
    )
    parser.add_argument(
        "--sweep",
        type=str,
        choices=["calibration", "hd_neurons", "both"],
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
        default=300.0,
        help="Simulation duration in seconds (default: 300)",
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
        default=7,
        help="Base random seed (default: 7)",
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Rubin HD Validation Parameter Sweep")
    print("=" * 70)
    print()
    
    if args.sweep in ("calibration", "both"):
        print("Sweeping calibration interval...")
        calib_intervals = [100, 200, 250, 300, 400, 500]
        calib_results = sweep_calibration_interval(
            calib_intervals,
            duration_seconds=args.duration,
            num_trials=args.trials,
            base_seed=args.seed,
        )
        
        # Print summary
        print("\nCalibration Interval Results:")
        print(f"{'Interval':>12} | {'Rayleigh (in)':>15} | {'Rayleigh (out)':>15}")
        print("-" * 50)
        for calib_int in sorted(calib_results.keys()):
            r = calib_results[calib_int]
            print(
                f"{calib_int:>12} | "
                f"{r['mean_rayleigh_in']:>7.3f} ± {r['std_rayleigh_in']:>5.3f} | "
                f"{r['mean_rayleigh_out']:>7.3f} ± {r['std_rayleigh_out']:>5.3f}"
            )
        
        # Plot
        plot_sweep_results(
            calib_results,
            "Calibration Interval",
            output_dir / "rubin_sweep_calibration.png",
        )
        print()
    
    if args.sweep in ("hd_neurons", "both"):
        print("Sweeping HD neuron count...")
        hd_counts = [36, 48, 60, 72, 90, 108]
        hd_results = sweep_hd_neurons(
            hd_counts,
            duration_seconds=args.duration,
            num_trials=args.trials,
            base_seed=args.seed,
        )
        
        # Print summary
        print("\nHD Neuron Count Results:")
        print(f"{'HD Neurons':>12} | {'Rayleigh (in)':>15} | {'Rayleigh (out)':>15}")
        print("-" * 50)
        for hd_count in sorted(hd_results.keys()):
            r = hd_results[hd_count]
            print(
                f"{hd_count:>12} | "
                f"{r['mean_rayleigh_in']:>7.3f} ± {r['std_rayleigh_in']:>5.3f} | "
                f"{r['mean_rayleigh_out']:>7.3f} ± {r['std_rayleigh_out']:>5.3f}"
            )
        
        # Plot
        plot_sweep_results(
            hd_results,
            "HD Neuron Count",
            output_dir / "rubin_sweep_hd_neurons.png",
        )
        print()
    
    print("Sweep complete!")


if __name__ == "__main__":
    main()

