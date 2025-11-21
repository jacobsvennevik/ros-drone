"""Automated parameter sweep for grid/HD attractor and controller parameters.

This script performs YAML-driven parameter sweeps to test sensitivity of
normalize_mode, adaptive_calibration, reward_timescale, and other parameters.
Results are saved for parameter-sensitivity analysis.

Usage:
    python experiments/parameter_sweep.py --config experiments/parameter_sweep_config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hippocampus_core.env import Agent, Environment
from hippocampus_core.grid_cells import GridAttractorConfig
from hippocampus_core.head_direction import HeadDirectionConfig
from hippocampus_core.controllers.bat_navigation_controller import (
    BatNavigationController,
    BatNavigationControllerConfig,
)


def run_single_simulation(
    params: Dict[str, Any],
    duration_seconds: float = 60.0,
    dt: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a single simulation with given parameters and return metrics.
    
    Parameters
    ----------
    params:
        Dictionary of parameter values (e.g., {'normalize_mode': 'subtractive', ...})
    duration_seconds:
        Simulation duration in seconds
    dt:
        Time step in seconds
    seed:
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Dictionary of outcome metrics (drift, stability, etc.)
    """
    env = Environment(width=1.0, height=1.0)
    
    # Build controller config with parameters
    controller_config = BatNavigationControllerConfig(
        num_place_cells=params.get("num_place_cells", 80),
        hd_num_neurons=params.get("hd_num_neurons", 60),
        grid_size=params.get("grid_size", (15, 15)),
        grid_tau=params.get("grid_tau", 0.05),
        hd_tau=params.get("hd_tau", 0.05),
        calibration_interval=params.get("calibration_interval", 200),
        adaptive_calibration=params.get("adaptive_calibration", False),
        calibration_drift_threshold=params.get("calibration_drift_threshold", 0.1),
        normalize_mode=params.get("normalize_mode", "subtractive"),
    )
    
    controller = BatNavigationController(
        env, config=controller_config, rng=np.random.default_rng(seed)
    )
    agent = Agent(env, random_state=np.random.default_rng(seed + 1), track_heading=True)
    
    num_steps = int(duration_seconds / dt)
    grid_drifts = []
    hd_errors = []
    grid_means = []
    hd_means = []
    
    for step in range(num_steps):
        obs = agent.step(dt, include_theta=True)
        controller.step(obs, dt)
        
        # Collect metrics
        if step % 10 == 0:  # Sample every 10 steps
            grid_drift = controller.grid_attractor.drift_metric()
            grid_drifts.append(grid_drift)
            
            grid_state = controller.grid_attractor.state
            grid_means.append(np.mean(grid_state))
            
            hd_state = controller.hd_attractor.state
            hd_means.append(np.mean(hd_state))
            
            # HD error (wrapped angle difference)
            true_heading = obs[2]
            estimated_heading = controller.hd_attractor.estimate_heading()
            error = abs(((estimated_heading - true_heading + np.pi) % (2 * np.pi)) - np.pi)
            hd_errors.append(error)
    
    # Compute summary statistics
    return {
        "params": params.copy(),
        "mean_grid_drift": float(np.mean(grid_drifts)),
        "std_grid_drift": float(np.std(grid_drifts)),
        "max_grid_drift": float(np.max(grid_drifts)),
        "mean_hd_error": float(np.mean(hd_errors)),
        "std_hd_error": float(np.std(hd_errors)),
        "max_hd_error": float(np.max(hd_errors)),
        "mean_grid_state_mean": float(np.mean(grid_means)),
        "mean_hd_state_mean": float(np.mean(hd_means)),
        "grid_normalization_check": abs(np.mean(grid_means)) < 1e-5,
        "hd_normalization_check": abs(np.mean(hd_means)) < 1e-5,
    }


def load_sweep_config(config_path: Path) -> Dict[str, Any]:
    """Load parameter sweep configuration from YAML file.
    
    Expected format:
        parameters:
          normalize_mode: ["subtractive", "divisive"]
          adaptive_calibration: [true, false]
          reward_timescale: [0.5, 1.0, 2.0]
        simulation:
          duration_seconds: 60.0
          dt: 0.05
          num_trials: 3
        output:
          results_file: "results/parameter_sweep_results.json"
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def generate_parameter_combinations(
    parameter_grid: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters from grid.
    
    Parameters
    ----------
    parameter_grid:
        Dictionary mapping parameter names to lists of values
    
    Returns
    -------
    list
        List of parameter dictionaries, one for each combination
    """
    import itertools
    
    keys = list(parameter_grid.keys())
    values = list(parameter_grid.values())
    
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


def main():
    parser = argparse.ArgumentParser(description="Run parameter sweep")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "parameter_sweep_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file path (overrides config)",
    )
    args = parser.parse_args()
    
    # Load configuration
    if not args.config.exists():
        print(f"Configuration file not found: {args.config}")
        print("Creating default configuration file...")
        create_default_config(args.config)
        print(f"Default config created at {args.config}")
        print("Please edit it and run again.")
        return
    
    config = load_sweep_config(args.config)
    
    # Extract parameters
    parameter_grid = config.get("parameters", {})
    sim_config = config.get("simulation", {})
    output_config = config.get("output", {})
    
    duration_seconds = sim_config.get("duration_seconds", 60.0)
    dt = sim_config.get("dt", 0.05)
    num_trials = sim_config.get("num_trials", 1)
    base_seed = sim_config.get("seed", 42)
    
    output_file = args.output or Path(output_config.get("results_file", "parameter_sweep_results.json"))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate parameter combinations
    combinations = generate_parameter_combinations(parameter_grid)
    print(f"Testing {len(combinations)} parameter combinations...")
    print(f"Each with {num_trials} trial(s)")
    print(f"Total simulations: {len(combinations) * num_trials}")
    
    results = []
    
    for i, params in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
        
        trial_results = []
        for trial in range(num_trials):
            seed = base_seed + i * num_trials + trial
            print(f"  Trial {trial+1}/{num_trials} (seed={seed})...", end=" ", flush=True)
            
            try:
                result = run_single_simulation(
                    params,
                    duration_seconds=duration_seconds,
                    dt=dt,
                    seed=seed,
                )
                trial_results.append(result)
                print("✓")
            except Exception as e:
                print(f"✗ Error: {e}")
                # Continue with next trial
        
        if trial_results:
            # Aggregate across trials
            aggregated = {
                "params": params,
                "num_trials": len(trial_results),
                "metrics": {
                    "mean_grid_drift": np.mean([r["mean_grid_drift"] for r in trial_results]),
                    "std_grid_drift": np.std([r["mean_grid_drift"] for r in trial_results]),
                    "mean_hd_error": np.mean([r["mean_hd_error"] for r in trial_results]),
                    "std_hd_error": np.std([r["mean_hd_error"] for r in trial_results]),
                },
                "trials": trial_results,
            }
            results.append(aggregated)
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Completed! Results saved to {output_file}")
    print(f"Total combinations tested: {len(results)}")


def create_default_config(config_path: Path) -> None:
    """Create default parameter sweep configuration file."""
    default_config = {
        "parameters": {
            "normalize_mode": ["subtractive", "divisive"],
            "adaptive_calibration": [False, True],
            "calibration_drift_threshold": [0.05, 0.1, 0.2],
        },
        "simulation": {
            "duration_seconds": 60.0,
            "dt": 0.05,
            "num_trials": 3,
            "seed": 42,
        },
        "output": {
            "results_file": "results/parameter_sweep_results.json",
        },
    }
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()

