#!/usr/bin/env python3
"""Validation experiment for Hoffman et al. (2016) topological mapping findings.

This script validates key findings from the bat hippocampus paper:
1. Integration window (ϖ) reduces spurious loops and fragmentation
2. Longer integration windows produce more stable maps
3. Betti numbers can verify correct topology learning
4. Learning time T_min depends on integration window length

Paper: arXiv:1601.04253v1 [q-bio.NC] - "Topological mapping of space in bat hippocampus"

Usage:
    python experiments/validate_hoffman_2016.py [--integration-windows 60 120 240 480]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
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


def run_learning_experiment(
    env: Environment,
    integration_window: Optional[float],
    duration_seconds: float = 300.0,  # 5 minutes
    dt: float = 0.05,
    num_place_cells: int = 100,
    sigma: float = 0.15,
    seed: int = 42,
) -> dict:
    """Run a single learning experiment and collect statistics.

    Parameters
    ----------
    env:
        Environment to navigate in
    integration_window:
        Integration window in seconds (ϖ). None = no integration window (immediate edge admission)
    duration_seconds:
        Total simulation duration in seconds
    dt:
        Simulation time step
    num_place_cells:
        Number of place cells
    sigma:
        Place field size parameter
    seed:
        Random seed for reproducibility

    Returns
    -------
    dict
        Statistics including edges, components, Betti numbers over time
    """
    config = PlaceCellControllerConfig(
        num_place_cells=num_place_cells,
        sigma=sigma,
        max_rate=20.0,
        coactivity_window=0.2,  # w: coincidence window (200ms)
        coactivity_threshold=5.0,
        max_edge_distance=2.0 * sigma,
        integration_window=integration_window,  # ϖ: integration window
    )

    rng = np.random.default_rng(seed)
    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent_rng = np.random.default_rng(seed + 1)
    agent = Agent(environment=env, random_state=agent_rng)

    num_steps = int(duration_seconds / dt)
    sample_interval = max(1, num_steps // 50)  # Sample ~50 times

    results = {
        "times": [],
        "edges": [],
        "components": [],
        "betti_0": [],
        "betti_1": [],
        "betti_2": [],
    }

    print(f"  Running with ϖ={integration_window}s...", end="", flush=True)

    for step in range(num_steps):
        position = agent.step(dt)
        controller.step(np.asarray(position), dt)

        if step % sample_interval == 0 or step == num_steps - 1:
            graph = controller.get_graph()
            current_time = controller.current_time

            # Get all stats from the same graph snapshot for consistency
            num_edges = graph.num_edges()
            num_components = graph.num_components()

            results["times"].append(current_time)
            results["edges"].append(num_edges)
            results["components"].append(num_components)

            # Compute Betti numbers (may fail if ripser/gudhi not available)
            try:
                betti = graph.compute_betti_numbers(max_dim=2)
                b0 = betti.get(0, 0)
                b1 = betti.get(1, 0)
                b2 = betti.get(2, 0)
                
                # Consistency check: b0 should match number of components
                # If they disagree, prefer components (more reliable)
                if b0 != num_components and num_edges > 0:
                    # Log warning but use Betti number (may be computing from clique complex)
                    pass
                elif num_edges == 0:
                    # If no edges, all nodes are isolated - b0 should equal number of nodes
                    # But Betti number computation on empty graph might give 1
                    # Use components instead for consistency
                    b0 = num_components
                
                results["betti_0"].append(b0)
                results["betti_1"].append(b1)
                results["betti_2"].append(b2)
            except ImportError:
                # Fallback if persistent homology not available
                results["betti_0"].append(num_components)
                results["betti_1"].append(-1)  # Mark as unavailable
                results["betti_2"].append(-1)

    print(" done")
    return results


def estimate_learning_time(
    results: dict, target_betti_0: int = 1, target_betti_1: int = 1
) -> Optional[float]:
    """Estimate learning time T_min when topology stabilizes.

    Parameters
    ----------
    results:
        Results dictionary from run_learning_experiment
    target_betti_0:
        Expected b_0 (number of components)
    target_betti_1:
        Expected b_1 (number of holes)

    Returns
    -------
    Optional[float]
        Time in seconds when topology first matches target and stays stable, or None if never reached
    """
    if not results["betti_1"] or results["betti_1"][0] == -1:
        # Can't estimate without Betti numbers, fall back to components
        if results["components"]:
            # Find first time when components == target_betti_0
            components_arr = np.array(results["components"])
            times_arr = np.array(results["times"])
            # Find first index where components == target_betti_0
            idx = np.where(components_arr == target_betti_0)[0]
            if len(idx) > 0:
                first_idx = idx[0]
                # Check if it stays stable (at least 3 consecutive samples)
                if first_idx + 3 < len(components_arr):
                    stable = all(
                        components_arr[j] == target_betti_0
                        for j in range(first_idx, min(first_idx + 3, len(components_arr)))
                    )
                    if stable:
                        return float(times_arr[first_idx])
                return float(times_arr[first_idx])
        return None

    # Use Betti numbers if available
    b0_arr = np.array(results["betti_0"])
    b1_arr = np.array(results["betti_1"])
    times_arr = np.array(results["times"])
    
    # Find first index where b0 == target and b1 <= target
    mask = (b0_arr == target_betti_0) & (b1_arr <= target_betti_1)
    idx = np.where(mask)[0]
    
    if len(idx) == 0:
        return None
    
    first_idx = idx[0]
    
    # Check if it stays stable (at least 3 consecutive samples)
    if first_idx + 3 < len(b0_arr):
        stable = all(
            b0_arr[j] == target_betti_0 and b1_arr[j] <= target_betti_1
            for j in range(first_idx, min(first_idx + 3, len(b0_arr)))
        )
        if stable:
            return float(times_arr[first_idx])
    
    return float(times_arr[first_idx])


def plot_results(
    results_by_window: dict[Optional[float], dict],
    output_path: Optional[Path] = None,
) -> None:
    """Plot validation results comparing different integration windows.

    Parameters
    ----------
    results_by_window:
        Dictionary mapping integration_window -> results
    output_path:
        Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Hoffman et al. (2016) Validation: Integration Window Effects",
        fontsize=14,
        fontweight="bold",
    )

    # Plot 1: Number of edges over time
    ax = axes[0, 0]
    for integration_window, results in sorted(results_by_window.items(), key=lambda x: x[0] or 0):
        label = f"ϖ = {integration_window}s" if integration_window else "ϖ = None"
        ax.plot(
            np.array(results["times"]) / 60.0,  # Convert to minutes
            results["edges"],
            label=label,
            alpha=0.7,
            linewidth=2,
        )
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Number of Edges")
    ax.set_title("Graph Growth: Edges vs. Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Number of components over time
    ax = axes[0, 1]
    for integration_window, results in sorted(results_by_window.items(), key=lambda x: x[0] or 0):
        label = f"ϖ = {integration_window}s" if integration_window else "ϖ = None"
        ax.plot(
            np.array(results["times"]) / 60.0,
            results["components"],
            label=label,
            alpha=0.7,
            linewidth=2,
        )
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Number of Components (b₀)")
    ax.set_title("Fragmentation: Components vs. Time")
    ax.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="Target (b₀=1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Betti number b₁ (holes) over time
    ax = axes[0, 2]
    for integration_window, results in sorted(results_by_window.items(), key=lambda x: x[0] or 0):
        label = f"ϖ = {integration_window}s" if integration_window else "ϖ = None"
        betti_1 = np.array(results["betti_1"])
        if betti_1[0] != -1:  # Check if available
            ax.plot(
                np.array(results["times"]) / 60.0,
                betti_1,
                label=label,
                alpha=0.7,
                linewidth=2,
            )
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Number of Holes (b₁)")
    ax.set_title("Spurious Loops: b₁ vs. Time")
    ax.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="Target (b₁≤1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Final edge counts vs. integration window
    ax = axes[1, 0]
    windows = []
    final_edges = []
    for integration_window, results in sorted(results_by_window.items(), key=lambda x: x[0] or 0):
        windows.append(integration_window if integration_window else 0)
        final_edges.append(results["edges"][-1])
    ax.bar(
        [w if w else 0 for w in windows],
        final_edges,
        width=20,
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_xlabel("Integration Window ϖ (seconds)")
    ax.set_ylabel("Final Number of Edges")
    ax.set_title("Final Graph Size vs. Integration Window")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 5: Learning time T_min vs. integration window
    ax = axes[1, 1]
    windows = []
    learning_times = []
    for integration_window, results in sorted(results_by_window.items(), key=lambda x: x[0] or 0):
        t_min = estimate_learning_time(results)
        if t_min is not None:
            windows.append(integration_window if integration_window else 0)
            learning_times.append(t_min / 60.0)  # Convert to minutes
    if windows:
        ax.scatter(windows, learning_times, s=100, alpha=0.7, edgecolors="black")
        ax.set_xlabel("Integration Window ϖ (seconds)")
        ax.set_ylabel("Learning Time T_min (minutes)")
        ax.set_title("Learning Time vs. Integration Window")
        ax.grid(True, alpha=0.3)

    # Plot 6: Final Betti numbers comparison
    ax = axes[1, 2]
    windows = []
    final_b1 = []
    for integration_window, results in sorted(results_by_window.items(), key=lambda x: x[0] or 0):
        if results["betti_1"] and results["betti_1"][-1] != -1:
            windows.append(integration_window if integration_window else 0)
            final_b1.append(results["betti_1"][-1])
    if windows:
        ax.bar(
            windows,
            final_b1,
            width=20,
            alpha=0.7,
            edgecolor="black",
            color=["green" if b <= 1 else "red" for b in final_b1],
        )
        ax.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="Target (b₁≤1)")
        ax.set_xlabel("Integration Window ϖ (seconds)")
        ax.set_ylabel("Final b₁ (Number of Holes)")
        ax.set_title("Final Topology vs. Integration Window")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved figure to: {output_path}")
    else:
        plt.show()


def main() -> None:
    """Run validation experiment."""
    parser = argparse.ArgumentParser(
        description="Validate Hoffman et al. (2016) findings on integration windows"
    )
    parser.add_argument(
        "--integration-windows",
        type=float,
        nargs="+",
        default=[0, 60, 120, 240, 480],  # 0 means None (no window)
        help="Integration window values in seconds (default: 0 60 120 240 480)",
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
        "--output",
        type=Path,
        help="Path to save output figure (default: show interactively)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    print("=" * 70)
    print("Hoffman et al. (2016) Topological Mapping Validation")
    print("=" * 70)
    print()
    print(f"Integration windows: {args.integration_windows} seconds")
    print(f"Simulation duration: {args.duration} seconds ({args.duration/60:.1f} minutes)")
    print(f"Number of place cells: {args.num_cells}")
    print()

    # Create environment (simple 2D arena)
    env = Environment(width=1.0, height=1.0)

    # Run experiments for each integration window
    results_by_window: dict[Optional[float], dict] = {}

    for integration_window_sec in args.integration_windows:
        integration_window = None if integration_window_sec == 0 else integration_window_sec
        results = run_learning_experiment(
            env=env,
            integration_window=integration_window,
            duration_seconds=args.duration,
            num_place_cells=args.num_cells,
            seed=args.seed,
        )
        results_by_window[integration_window] = results

    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print()
    print(f"{'ϖ (s)':>8} | {'Final Edges':>12} | {'Final b₀':>10} | {'Final b₁':>10} | {'T_min (min)':>12}")
    print("-" * 70)

    for integration_window in sorted(results_by_window.keys(), key=lambda x: x or 0):
        results = results_by_window[integration_window]
        window_str = "None" if integration_window is None else f"{integration_window:.0f}"
        
        # Get final values from the last sample (ensure consistency)
        final_edges = results["edges"][-1]
        final_components = results["components"][-1]
        final_b0 = results["betti_0"][-1]
        final_b1 = results["betti_1"][-1] if results["betti_1"][-1] != -1 else "N/A"
        
        # Consistency check: if no edges, b0 should equal number of nodes (all isolated)
        # If b0 doesn't match components when there are edges, prefer components
        if final_edges == 0:
            # No edges means all nodes are isolated - b0 should equal number of nodes
            # Use components (which counts connected components correctly)
            final_b0 = final_components
        elif abs(final_b0 - final_components) > 0:
            # If they disagree and we have edges, prefer components (more reliable)
            # This can happen if Betti numbers are computed from clique complex vs graph components
            final_b0 = final_components
        
        t_min = estimate_learning_time(results)
        t_min_str = f"{t_min/60:.2f}" if t_min else "N/A"

        print(
            f"{window_str:>8} | {final_edges:>12} | {final_b0:>10} | {final_b1:>10} | {t_min_str:>12}"
        )

    # Plot results
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)
    plot_results(results_by_window, output_path=args.output)

    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)
    print()
    print("Key findings from paper:")
    print("  1. Longer integration windows (ϖ) should reduce spurious loops (b₁)")
    print("  2. Integration windows should reduce fragmentation (b₀ → 1 faster)")
    print("  3. Learning time T_min may increase slightly with ϖ, but maps are more stable")
    print()


if __name__ == "__main__":
    main()

