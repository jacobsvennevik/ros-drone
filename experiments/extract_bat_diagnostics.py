"""Helper functions for extracting HD/grid diagnostics from BatNavigationController.

This module provides utilities to extract and aggregate HD/grid statistics
when using BatNavigationController in validation experiments.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Optional import - will be None if bat controller not available
try:
    from hippocampus_core.controllers.bat_navigation_controller import (
        BatNavigationController,
    )
except ImportError:
    BatNavigationController = None


def extract_hd_grid_diagnostics(
    controller: object,
) -> Optional[dict[str, float | np.ndarray]]:
    """Extract HD and grid diagnostics from a controller.

    Parameters
    ----------
    controller:
        Controller instance (must be BatNavigationController).

    Returns
    -------
    dict or None
        Dictionary with HD/grid diagnostics if available, None otherwise.
        Keys:
        - 'hd_estimate': Estimated heading (radians)
        - 'hd_error': HD estimation error (if ground truth available)
        - 'hd_activity_norm': L2 norm of HD activity vector
        - 'grid_estimate': Estimated position [x, y]
        - 'grid_drift': Grid drift metric
        - 'grid_activity_norm': L2 norm of grid activity matrix
        - 'mean_place_rate': Mean place cell firing rate
        - 'max_place_rate': Max place cell firing rate
    """
    if BatNavigationController is None:
        return None

    if not isinstance(controller, BatNavigationController):
        return None

    diagnostics = {}

    # HD diagnostics
    try:
        hd_activity = controller.hd_attractor.activity()
        hd_estimate = controller.hd_attractor.estimate_heading()
        diagnostics["hd_estimate"] = float(hd_estimate)
        diagnostics["hd_activity_norm"] = float(np.linalg.norm(hd_activity))
    except AttributeError:
        pass

    # Grid diagnostics
    try:
        grid_activity = controller.grid_attractor.activity()
        grid_estimate = controller.grid_attractor.estimate_position()
        grid_drift = controller.grid_attractor.drift_metric()
        diagnostics["grid_estimate"] = grid_estimate.copy()
        diagnostics["grid_drift"] = float(grid_drift)
        diagnostics["grid_activity_norm"] = float(np.linalg.norm(grid_activity))
    except AttributeError:
        pass

    # Place cell diagnostics
    try:
        if controller.last_rates is not None:
            rates = controller.last_rates
            diagnostics["mean_place_rate"] = float(np.mean(rates))
            diagnostics["max_place_rate"] = float(np.max(rates))
    except AttributeError:
        pass

    # HD error (if ground truth available)
    # This would need to be computed externally based on ground truth heading

    return diagnostics if diagnostics else None


def aggregate_hd_grid_stats(
    diagnostics_list: list[dict[str, float | np.ndarray]],
) -> dict[str, dict[str, float]]:
    """Aggregate HD/grid diagnostics across multiple samples.

    Parameters
    ----------
    diagnostics_list:
        List of diagnostic dictionaries from extract_hd_grid_diagnostics().

    Returns
    -------
    dict
        Dictionary of aggregated statistics.
        Format: {'metric_name': {'mean': float, 'std': float, ...}}
    """
    if not diagnostics_list:
        return {}

    # Collect all values for each metric
    metric_values: dict[str, list[float]] = {}

    for diag in diagnostics_list:
        for key, value in diag.items():
            if key == "grid_estimate":
                # Skip grid_estimate (position) - would need special handling
                continue
            if isinstance(value, (int, float, np.number)):
                if key not in metric_values:
                    metric_values[key] = []
                metric_values[key].append(float(value))

    # Compute statistics
    aggregated = {}

    for key, values in metric_values.items():
        if not values:
            continue
        values_arr = np.array(values)
        aggregated[key] = {
            "mean": float(np.mean(values_arr)),
            "std": float(np.std(values_arr)),
            "min": float(np.min(values_arr)),
            "max": float(np.max(values_arr)),
            "median": float(np.median(values_arr)),
        }

    return aggregated


def print_hd_grid_summary(aggregated: dict[str, dict[str, float]]) -> None:
    """Print a formatted summary of HD/grid statistics.

    Parameters
    ----------
    aggregated:
        Aggregated statistics from aggregate_hd_grid_stats().
    """
    if not aggregated:
        print("No HD/grid diagnostics available.")
        return

    print("\n" + "=" * 70)
    print("HD/Grid Diagnostics Summary")
    print("=" * 70)

    for key, stats in sorted(aggregated.items()):
        mean = stats["mean"]
        std = stats["std"]
        print(f"{key:20s}: {mean:8.4f} ± {std:8.4f}  [min={stats['min']:.4f}, max={stats['max']:.4f}]")

    print("=" * 70)


# Example usage in a validation experiment:
"""
from experiments.extract_bat_diagnostics import (
    extract_hd_grid_diagnostics,
    aggregate_hd_grid_stats,
    print_hd_grid_summary,
)

# Inside your simulation loop:
diagnostics_history = []
for step in range(num_steps):
    # ... run simulation ...
    diag = extract_hd_grid_diagnostics(controller)
    if diag:
        diagnostics_history.append(diag)

# After simulation:
if diagnostics_history:
    aggregated = aggregate_hd_grid_stats(diagnostics_history)
    print_hd_grid_summary(aggregated)
"""

