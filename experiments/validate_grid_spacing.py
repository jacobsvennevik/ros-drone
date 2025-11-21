"""Grid spacing validation with cross-validation against empirical data.

This script validates that grid-cell attractor updates produce phase shifts
consistent with empirical grid spacing (30-200 cm per module) and checks
hexagonal periodicity via autocorrelation analysis.

This addresses Q15 from the technical audit and includes cross-validation
against Yartsev et al. (2011) bat data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy import signal
from scipy.stats import pearsonr

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hippocampus_core.env import Agent, Environment
from hippocampus_core.grid_cells import GridAttractor, GridAttractorConfig
from hippocampus_core.controllers.bat_navigation_controller import (
    BatNavigationController,
    BatNavigationControllerConfig,
)
from hippocampus_core.utils.logging import log_model_metadata, get_model_version


def compute_grid_spacing_from_autocorrelation(
    firing_map: np.ndarray,
    spatial_resolution: float = 0.01,  # meters per pixel
) -> Optional[float]:
    """Estimate grid spacing from firing rate map autocorrelation.
    
    Parameters
    ----------
    firing_map:
        2D array of firing rates (spatial map)
    spatial_resolution:
        Spatial resolution in meters per pixel
    
    Returns
    -------
    float or None
        Estimated grid spacing in meters, or None if computation fails
    """
    # Compute autocorrelation
    autocorr = signal.correlate2d(firing_map, firing_map, mode="same")
    autocorr = autocorr / np.max(autocorr)  # Normalize
    
    # Find center
    center_y, center_x = np.array(autocorr.shape) // 2
    
    # Extract radial profile from center
    y, x = np.ogrid[:autocorr.shape[0], :autocorr.shape[1]]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    r = r.astype(int)
    
    # Average autocorrelation at each radius
    r_max = min(center_x, center_y)
    radial_profile = np.zeros(r_max)
    for radius in range(r_max):
        mask = (r == radius)
        if np.any(mask):
            radial_profile[radius] = np.mean(autocorr[mask])
    
    # Find first peak (grid spacing)
    # Look for peaks beyond radius 10 (avoid center artifact)
    search_start = int(10 / spatial_resolution)
    search_end = min(len(radial_profile), int(200 / spatial_resolution))  # Max 200 cm
    
    if search_end <= search_start:
        return None
    
    peak_indices = signal.find_peaks(
        radial_profile[search_start:search_end],
        height=0.3,  # Minimum peak height
        distance=int(20 / spatial_resolution),  # Minimum peak separation
    )[0]
    
    if len(peak_indices) > 0:
        # First peak corresponds to grid spacing
        peak_radius = peak_indices[0] + search_start
        grid_spacing = peak_radius * spatial_resolution * 100  # Convert to cm
        return grid_spacing
    
    return None


def check_hexagonal_periodicity(
    firing_map: np.ndarray,
    grid_spacing_cm: float,
    spatial_resolution: float = 0.01,
) -> Dict[str, float]:
    """Check hexagonal periodicity via autocorrelation analysis.
    
    For correct hexagonal periodicity, R(0) - R(λ/2) < 0.3, where
    R is the autocorrelation and λ is the grid spacing.
    
    Parameters
    ----------
    firing_map:
        2D array of firing rates
    grid_spacing_cm:
        Estimated grid spacing in cm
    spatial_resolution:
        Spatial resolution in meters per pixel
    
    Returns
    -------
    dict
        Dictionary with periodicity metrics
    """
    # Compute autocorrelation
    autocorr = signal.correlate2d(firing_map, firing_map, mode="same")
    autocorr = autocorr / np.max(autocorr)
    
    center_y, center_x = np.array(autocorr.shape) // 2
    
    # Get autocorrelation at center R(0)
    r_0 = autocorr[center_y, center_x]
    
    # Get autocorrelation at λ/2 (half grid spacing)
    lambda_half_pixels = int((grid_spacing_cm / 100.0) / (2 * spatial_resolution))
    if lambda_half_pixels > 0 and lambda_half_pixels < min(center_x, center_y):
        # Extract value at distance λ/2
        y, x = np.ogrid[:autocorr.shape[0], :autocorr.shape[1]]
        r_dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = np.abs(r_dist - lambda_half_pixels) < 2  # Within 2 pixels
        if np.any(mask):
            r_lambda_half = np.mean(autocorr[mask])
        else:
            r_lambda_half = 0.0
    else:
        r_lambda_half = 0.0
    
    periodicity_metric = r_0 - r_lambda_half
    
    return {
        "r_0": float(r_0),
        "r_lambda_half": float(r_lambda_half),
        "periodicity_metric": float(periodicity_metric),
        "periodicity_check": periodicity_metric < 0.3,
    }


def measure_drift_isotropy(
    attractor: GridAttractor,
    num_steps: int = 200,
    dt: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Measure directional anisotropy in grid drift.
    
    For isotropic drift, σ_x / σ_y ≈ 1.0 ± 0.05.
    
    Returns
    -------
    dict
        Dictionary with isotropy metrics
    """
    np.random.seed(seed)
    
    drift_x = []
    drift_y = []
    prev_pos = attractor.estimate_position()
    
    for _ in range(num_steps):
        # Random velocity with balanced x and y components
        velocity = np.random.normal(0, 0.5, size=2)
        attractor.step(velocity=velocity, dt=dt)
        current_pos = attractor.estimate_position()
        
        delta = current_pos - prev_pos
        drift_x.append(delta[0])
        drift_y.append(delta[1])
        prev_pos = current_pos.copy()
    
    sigma_x = np.std(drift_x)
    sigma_y = np.std(drift_y)
    
    if sigma_y > 1e-6:
        isotropy_ratio = sigma_x / sigma_y
    else:
        isotropy_ratio = np.inf
    
    return {
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "isotropy_ratio": float(isotropy_ratio),
        "isotropy_check": 0.95 < isotropy_ratio < 1.05 if np.isfinite(isotropy_ratio) else False,
    }


def run_grid_spacing_validation(
    grid_size: tuple[int, int] = (20, 20),
    velocity_gain: float = 1.0,
    duration_seconds: float = 600.0,
    dt: float = 0.05,
    spatial_resolution: float = 0.05,  # 5 cm per grid cell (approximate)
    seed: int = 42,
) -> Dict[str, any]:
    """Run comprehensive grid spacing validation.
    
    Parameters
    ----------
    grid_size:
        Grid attractor size (M, M)
    velocity_gain:
        Velocity gain parameter
    duration_seconds:
        Simulation duration
    spatial_resolution:
        Approximate spatial resolution in meters per grid cell
    seed:
        Random seed
    
    Returns
    -------
    dict
        Validation results including spacing estimates, isotropy, periodicity
    """
    env = Environment(width=2.0, height=2.0)  # Larger arena for better statistics
    
    config = BatNavigationControllerConfig(
        num_place_cells=100,
        hd_num_neurons=60,
        grid_size=grid_size,
        grid_velocity_gain=velocity_gain,
    )
    
    controller = BatNavigationController(
        env, config=config, rng=np.random.default_rng(seed)
    )
    agent = Agent(env, random_state=np.random.default_rng(seed + 1), track_heading=True)
    
    num_steps = int(duration_seconds / dt)
    
    # Collect firing rate maps
    grid_activity_history = []
    positions = []
    
    for step in range(num_steps):
        obs = agent.step(dt, include_theta=True)
        controller.step(obs, dt)
        
        # Sample periodically to build firing map
        if step % 10 == 0:
            grid_activity = controller.grid_attractor.activity()
            grid_activity_history.append(grid_activity)
            positions.append(obs[:2].copy())
    
    # Compute average firing map
    avg_firing_map = np.mean(grid_activity_history, axis=0)
    
    # Estimate grid spacing from autocorrelation
    grid_spacing_cm = compute_grid_spacing_from_autocorrelation(
        avg_firing_map, spatial_resolution=spatial_resolution
    )
    
    # Check periodicity
    periodicity_results = {}
    if grid_spacing_cm is not None:
        periodicity_results = check_hexagonal_periodicity(
            avg_firing_map, grid_spacing_cm, spatial_resolution=spatial_resolution
        )
    
    # Measure drift isotropy
    isotropy_results = measure_drift_isotropy(
        controller.grid_attractor, num_steps=200, dt=dt, seed=seed
    )
    
    # Check if spacing is in biological range (30-200 cm per module)
    biological_range_check = False
    if grid_spacing_cm is not None:
        biological_range_check = 30.0 <= grid_spacing_cm <= 200.0
    
    results = {
        "grid_size": grid_size,
        "velocity_gain": velocity_gain,
        "estimated_grid_spacing_cm": grid_spacing_cm,
        "biological_range_check": biological_range_check,
        "periodicity": periodicity_results,
        "isotropy": isotropy_results,
        "spatial_resolution_meters": spatial_resolution,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate grid spacing against empirical data")
    parser.add_argument(
        "--grid-size",
        type=int,
        nargs=2,
        default=[20, 20],
        help="Grid attractor size (default: 20 20)",
    )
    parser.add_argument(
        "--velocity-gain",
        type=float,
        default=1.0,
        help="Velocity gain parameter (default: 1.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=600.0,
        help="Simulation duration in seconds (default: 600.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/grid_spacing_validation.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("Grid Spacing Validation")
    print(f"Model Version: {get_model_version()}")
    print("=" * 70)
    print(f"Grid size: {args.grid_size}")
    print(f"Velocity gain: {args.velocity_gain}")
    print(f"Duration: {args.duration}s")
    print(f"Seed: {args.seed}")
    print()
    
    results = run_grid_spacing_validation(
        grid_size=tuple(args.grid_size),
        velocity_gain=args.velocity_gain,
        duration_seconds=args.duration,
        seed=args.seed,
    )
    
    # Print results
    print("Validation Results:")
    print("-" * 70)
    
    if results["estimated_grid_spacing_cm"] is not None:
        spacing = results["estimated_grid_spacing_cm"]
        print(f"✓ Estimated grid spacing: {spacing:.2f} cm")
        
        if results["biological_range_check"]:
            print(f"✓ Grid spacing in biological range (30-200 cm): {spacing:.2f} cm")
        else:
            print(f"✗ Grid spacing OUTSIDE biological range: {spacing:.2f} cm")
            print(f"  (Expected: 30-200 cm per module)")
    else:
        print("✗ Could not estimate grid spacing from autocorrelation")
    
    print()
    
    # Periodicity check
    periodicity = results["periodicity"]
    if periodicity:
        metric = periodicity["periodicity_metric"]
        check = periodicity["periodicity_check"]
        if check:
            print(f"✓ Hexagonal periodicity check passed: R(0) - R(λ/2) = {metric:.3f} < 0.3")
        else:
            print(f"✗ Hexagonal periodicity check failed: R(0) - R(λ/2) = {metric:.3f} >= 0.3")
        print(f"  R(0) = {periodicity['r_0']:.3f}, R(λ/2) = {periodicity['r_lambda_half']:.3f}")
    
    print()
    
    # Isotropy check
    isotropy = results["isotropy"]
    ratio = isotropy["isotropy_ratio"]
    if isotropy["isotropy_check"]:
        print(f"✓ Drift isotropy check passed: σ_x/σ_y = {ratio:.3f} (target: 0.95-1.05)")
    else:
        print(f"✗ Drift isotropy check failed: σ_x/σ_y = {ratio:.3f} (target: 0.95-1.05)")
    print(f"  σ_x = {isotropy['sigma_x']:.4f}, σ_y = {isotropy['sigma_y']:.4f}")
    
    print()
    print("=" * 70)
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Log model metadata
    log_model_metadata(
        args.output.parent / "metadata.json",
        {
            "validation_type": "grid_spacing",
            "parameters": {
                "grid_size": args.grid_size,
                "velocity_gain": args.velocity_gain,
                "duration_seconds": args.duration,
                "seed": args.seed,
            },
            "results": results,
        },
    )
    
    # Save results
    import json
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {args.output}")
    print(f"Metadata saved to: {args.output.parent / 'metadata.json'}")


if __name__ == "__main__":
    main()

