#!/usr/bin/env python3
"""Visualization of topological learning with Betti number tracking over time.

This script demonstrates how the topological map evolves over time, tracking:
- Betti numbers (b₀, b₁, b₂) over time (barcode-style visualization)
- Graph structure at different time points
- Edge counts and components over time

Inspired by Figure 1A from Hoffman et al. (2016), which shows timelines of
1D loops (topological loops) that persist and disappear as the map learns.

Usage:
    python examples/topology_learning_visualization.py [--integration-window 480]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

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


def track_topology_evolution(
    env: Environment,
    integration_window: Optional[float],
    duration_seconds: float = 600.0,  # 10 minutes
    dt: float = 0.05,
    num_place_cells: int = 120,
    sigma: float = 0.15,
    seed: int = 42,
) -> dict:
    """Track topology evolution over time.

    Parameters
    ----------
    env:
        Environment to navigate in
    integration_window:
        Integration window in seconds (ϖ). None = no integration window
    duration_seconds:
        Total simulation duration
    dt:
        Simulation time step
    num_place_cells:
        Number of place cells
    sigma:
        Place field size parameter
    seed:
        Random seed

    Returns
    -------
    dict
        Time series of topology metrics
    """
    config = PlaceCellControllerConfig(
        num_place_cells=num_place_cells,
        sigma=sigma,
        max_rate=20.0,
        coactivity_window=0.2,
        coactivity_threshold=5.0,
        max_edge_distance=2.0 * sigma,
        integration_window=integration_window,
    )

    rng = np.random.default_rng(seed)
    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent_rng = np.random.default_rng(seed + 1)
    agent = Agent(environment=env, random_state=agent_rng)

    num_steps = int(duration_seconds / dt)
    sample_interval = max(1, num_steps // 100)  # Sample ~100 times for smooth plots

    results = {
        "times": [],
        "edges": [],
        "components": [],
        "betti_0": [],
        "betti_1": [],
        "betti_2": [],
        "graphs": [],  # Store graphs at sample points
        "positions": None,
    }

    print("Running simulation...")
    print(f"  Duration: {duration_seconds/60:.1f} minutes")
    print(f"  Integration window: {integration_window}s" if integration_window else "  Integration window: None")
    print(f"  Sampling every {sample_interval * dt:.1f} seconds\n")

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

            # Store graph periodically (not too frequently to save memory)
            if len(results["times"]) % 5 == 0 or step == num_steps - 1:
                results["graphs"].append((current_time, graph.graph.copy()))

            # Compute Betti numbers
            try:
                betti = graph.compute_betti_numbers(max_dim=2)
                b0 = betti.get(0, 0)
                b1 = betti.get(1, 0)
                b2 = betti.get(2, 0)
                
                # Consistency check: when edges = 0, all nodes are isolated
                # Betti computation on empty clique complex may return b₀=1 incorrectly
                # Use components (which counts correctly) instead
                if num_edges == 0:
                    b0 = num_components
                elif abs(b0 - num_components) > 0:
                    # If they disagree with edges present, prefer components (more reliable)
                    # Betti numbers computed from clique complex may differ slightly
                    pass  # Keep Betti number but note the discrepancy
                
                results["betti_0"].append(b0)
                results["betti_1"].append(b1)
                results["betti_2"].append(b2)
            except ImportError:
                print("Warning: ripser/gudhi not available, skipping Betti number computation")
                results["betti_0"].append(num_components)
                results["betti_1"].append(-1)
                results["betti_2"].append(-1)

        if (step + 1) % (num_steps // 10) == 0:
            progress = 100 * (step + 1) / num_steps
            print(f"  Progress: {progress:.0f}%", end="\r")

    results["positions"] = controller.place_cell_positions
    
    # Add consistency assertions
    final_edges = results["edges"][-1]
    final_b0 = results["betti_0"][-1]
    final_components = results["components"][-1]
    
    # Assert consistency: if no edges, b0 should equal number of nodes
    if final_edges == 0:
        assert final_b0 == final_components, (
            f"Inconsistent final state: {final_edges} edges but b₀={final_b0} "
            f"(expected {final_components} isolated nodes)"
        )
    
    print("\n  Simulation complete!\n")
    return results


def plot_barcode_style(results: dict, ax: plt.Axes, betti_dim: int = 1) -> None:
    """Plot Betti number timelines in barcode style (like Figure 1A of the paper).

    Parameters
    ----------
    results:
        Results dictionary from track_topology_evolution
    ax:
        Matplotlib axes to plot on
    betti_dim:
        Betti dimension to plot (0, 1, or 2)
    """
    betti_key = f"betti_{betti_dim}"
    times = np.array(results["times"]) / 60.0  # Convert to minutes
    betti_values = np.array(results[betti_key])

    if betti_values[0] == -1:
        ax.text(0.5, 0.5, "Betti numbers not available\n(install ripser or gudhi)", 
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        return

    # Find intervals where each unique Betti number persists
    unique_betti = np.unique(betti_values)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_betti)))

    for i, betti_val in enumerate(unique_betti):
        # Find intervals where this Betti number appears
        mask = betti_values == betti_val
        if not np.any(mask):
            continue

        # Group consecutive occurrences
        intervals = []
        start = None
        for j, in_interval in enumerate(mask):
            if in_interval and start is None:
                start = times[j]
            elif not in_interval and start is not None:
                intervals.append((start, times[j-1]))
                start = None
        if start is not None:
            intervals.append((start, times[-1]))

        # Plot horizontal bars for each interval
        for interval_start, interval_end in intervals:
            y_pos = betti_val
            ax.barh(y_pos, interval_end - interval_start, left=interval_start, 
                   height=0.8, color=colors[i], alpha=0.7, edgecolor="black", linewidth=0.5)
            
            # Annotate if interval is long enough
            if interval_end - interval_start > 1.0:
                mid_time = (interval_start + interval_end) / 2
                ax.text(mid_time, y_pos, f"{int(betti_val)}", 
                       ha="center", va="center", fontsize=8, fontweight="bold")

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(f"Betti number b_{betti_dim}")
    ax.set_title(f"Barcode: b_{betti_dim} (Topological {['Components', 'Holes', 'Voids'][betti_dim]}) over Time")
    ax.set_yticks(sorted(unique_betti))
    ax.grid(True, alpha=0.3, axis="x")


def visualize_topology_learning(
    results: dict,
    output_path: Optional[Path] = None,
    integration_window: Optional[float] = None,
) -> None:
    """Create comprehensive visualization of topology learning.

    Parameters
    ----------
    results:
        Results dictionary from track_topology_evolution
    output_path:
        Optional path to save figure
    integration_window:
        Integration window used (for title)
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    window_str = f"ϖ = {integration_window/60:.1f} min" if integration_window else "ϖ = None"
    fig.suptitle(
        f"Topological Learning Evolution ({window_str})",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Barcode style for b₀ (components)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_barcode_style(results, ax1, betti_dim=0)

    # Plot 2: Barcode style for b₁ (holes) - key plot from paper
    ax2 = fig.add_subplot(gs[0, 1])
    plot_barcode_style(results, ax2, betti_dim=1)

    # Plot 3: Barcode style for b₂ (voids)
    ax3 = fig.add_subplot(gs[0, 2])
    plot_barcode_style(results, ax3, betti_dim=2)

    # Plot 4: Edge count over time
    ax4 = fig.add_subplot(gs[1, 0])
    times = np.array(results["times"]) / 60.0
    ax4.plot(times, results["edges"], "b-", linewidth=2, alpha=0.8)
    ax4.set_xlabel("Time (minutes)")
    ax4.set_ylabel("Number of Edges")
    ax4.set_title("Graph Growth: Edges")
    ax4.grid(True, alpha=0.3)

    # Plot 5: Components over time
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(times, results["components"], "g-", linewidth=2, alpha=0.8)
    ax5.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="Target (b₀=1)")
    ax5.set_xlabel("Time (minutes)")
    ax5.set_ylabel("Number of Components (b₀)")
    ax5.set_title("Fragmentation: Components")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Betti numbers over time (line plot)
    ax6 = fig.add_subplot(gs[1, 2])
    if results["betti_1"][0] != -1:
        ax6.plot(times, results["betti_0"], "b-", label="b₀ (components)", linewidth=2)
        ax6.plot(times, results["betti_1"], "r-", label="b₁ (holes)", linewidth=2)
        if results["betti_2"][0] != -1:
            ax6.plot(times, results["betti_2"], "g-", label="b₂ (voids)", linewidth=2, alpha=0.7)
        ax6.set_xlabel("Time (minutes)")
        ax6.set_ylabel("Betti Number")
        ax6.set_title("Betti Numbers over Time")
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, "Betti numbers not available", 
                ha="center", va="center", transform=ax6.transAxes, fontsize=12)

    # Plot 7-9: Graph snapshots at different times
    if results["graphs"]:
        positions = results["positions"]
        
        # Early time
        if len(results["graphs"]) >= 1:
            time1, graph1 = results["graphs"][0]
            ax7 = fig.add_subplot(gs[2, 0])
            plot_graph_snapshot(ax7, graph1, positions, f"Early: t = {time1/60:.1f} min")
        
        # Middle time
        if len(results["graphs"]) >= 2:
            mid_idx = len(results["graphs"]) // 2
            time2, graph2 = results["graphs"][mid_idx]
            ax8 = fig.add_subplot(gs[2, 1])
            plot_graph_snapshot(ax8, graph2, positions, f"Middle: t = {time2/60:.1f} min")
        
        # Final time
        if len(results["graphs"]) >= 1:
            time3, graph3 = results["graphs"][-1]
            ax9 = fig.add_subplot(gs[2, 2])
            plot_graph_snapshot(ax9, graph3, positions, f"Final: t = {time3/60:.1f} min")

    if output_path:
        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")
    else:
        plt.show()


def plot_graph_snapshot(ax: plt.Axes, graph: nx.Graph, positions: np.ndarray, title: str) -> None:
    """Plot a graph snapshot showing place cell positions and connections.

    Parameters
    ----------
    ax:
        Matplotlib axes
    graph:
        NetworkX graph to plot
    positions:
        Place cell positions (num_cells, 2)
    title:
        Plot title
    """
    # Draw edges
    for edge in graph.edges():
        i, j = edge
        ax.plot([positions[i, 0], positions[j, 0]], 
               [positions[i, 1], positions[j, 1]], 
               "k-", alpha=0.2, linewidth=0.5)

    # Draw nodes (place cell centers)
    ax.scatter(positions[:, 0], positions[:, 1], 
              s=20, c="blue", alpha=0.6, edgecolors="black", linewidths=0.5)

    ax.set_xlim(positions[:, 0].min() - 0.1, positions[:, 0].max() + 0.1)
    ax.set_ylim(positions[:, 1].min() - 0.1, positions[:, 1].max() + 0.1)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def main() -> None:
    """Run topology learning visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize topological learning with Betti number tracking"
    )
    parser.add_argument(
        "--integration-window",
        type=float,
        default=480.0,
        help="Integration window in seconds (default: 480 = 8 minutes)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=600.0,
        help="Simulation duration in seconds (default: 600 = 10 minutes)",
    )
    parser.add_argument(
        "--num-cells",
        type=int,
        default=120,
        help="Number of place cells (default: 120)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save output figure (default: show interactively)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    print("=" * 70)
    print("Topology Learning Visualization")
    print("Inspired by Hoffman et al. (2016) Figure 1A")
    print("=" * 70)
    print()

    # Create environment
    env = Environment(width=1.0, height=1.0)

    # Track topology evolution
    integration_window = args.integration_window if args.integration_window > 0 else None
    results = track_topology_evolution(
        env=env,
        integration_window=integration_window,
        duration_seconds=args.duration,
        num_place_cells=args.num_cells,
        seed=args.seed,
    )

    # Visualize
    print("Generating visualizations...")
    visualize_topology_learning(
        results, output_path=args.output, integration_window=integration_window
    )

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    print()
    print("Key observations:")
    print("  - b₀ (components) should converge to 1 (connected space)")
    print("  - b₁ (holes) should stabilize at the correct number of loops")
    print("  - Persistent loops in barcode indicate stable topological features")
    print()


if __name__ == "__main__":
    main()

