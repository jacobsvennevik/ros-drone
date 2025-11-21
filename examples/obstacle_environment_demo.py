#!/usr/bin/env python3
"""Demonstration of topological mapping with obstacle environments.

This script shows how obstacles create holes (b₁ > 0) in the learned topological map.
With a central obstacle, the graph should correctly identify b₁ = 1 (one hole
encircling the obstacle), matching the paper's findings.

Run with: python3 examples/obstacle_environment_demo.py
Run with bat controller: python3 examples/obstacle_environment_demo.py --controller bat
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np

from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)
try:
    from hippocampus_core.controllers.bat_navigation_controller import (
        BatNavigationController,
        BatNavigationControllerConfig,
    )
    BAT_AVAILABLE = True
except ImportError:
    BAT_AVAILABLE = False
    BatNavigationController = None
    BatNavigationControllerConfig = None
from hippocampus_core.env import Agent, CircularObstacle, Environment


def main() -> None:
    """Demonstrate obstacle environment with topological mapping."""
    parser = argparse.ArgumentParser(
        description="Obstacle environment topological mapping demo"
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="place",
        choices=["place", "bat"],
        help="Controller type: 'place' (default) or 'bat' (requires heading data)",
    )
    args = parser.parse_args()
    
    controller_type = args.controller.lower()
    if controller_type == "bat" and not BAT_AVAILABLE:
        print("ERROR: Bat navigation controller not available.")
        print("Ensure hippocampus_core is properly installed.")
        sys.exit(1)
    
    print("=" * 70)
    print("Obstacle Environment Topological Mapping Demo")
    print("=" * 70)
    print()
    print(f"Controller: {controller_type.upper()}")
    if controller_type == "bat":
        print("  Using BatNavigationController with HD/grid attractors")
    else:
        print("  Using PlaceCellController")
    print()
    print("This demonstrates how obstacles create holes in the learned map.")
    print("With a central obstacle, we expect b₁ = 1 (one hole encircling obstacle).")
    print()

    # Create environment with central obstacle
    # Similar to paper's setup with a central column
    central_obstacle = CircularObstacle(center_x=0.5, center_y=0.5, radius=0.15)
    env_with_obstacle = Environment(
        width=1.0,
        height=1.0,
        obstacles=[central_obstacle],
    )

    # Create environment without obstacle for comparison
    env_no_obstacle = Environment(width=1.0, height=1.0)

    rng = np.random.default_rng(42)
    seed_offset = 100

    # Configure controller based on type
    if controller_type == "bat":
        config = BatNavigationControllerConfig(
            num_place_cells=120,
            sigma=0.15,
            coactivity_window=0.2,
            coactivity_threshold=12.0,
            max_edge_distance=0.3,
            integration_window=None,
            hd_num_neurons=72,
            grid_size=(16, 16),
            calibration_interval=250,
        )
        controller_obstacle = BatNavigationController(
            environment=env_with_obstacle, config=config, rng=rng
        )
        controller_no_obstacle = BatNavigationController(
            environment=env_no_obstacle, config=config, rng=np.random.default_rng(43)
        )
        agent_obstacle = Agent(
            environment=env_with_obstacle,
            random_state=np.random.default_rng(42 + seed_offset),
            track_heading=True,
        )
        agent_no_obstacle = Agent(
            environment=env_no_obstacle,
            random_state=np.random.default_rng(43 + seed_offset),
            track_heading=True,
        )
        include_theta = True
    else:
        config = PlaceCellControllerConfig(
            num_place_cells=120,
            sigma=0.15,
            max_rate=20.0,
            coactivity_window=0.2,
            coactivity_threshold=12.0,
            max_edge_distance=0.3,
            integration_window=None,
        )
        controller_obstacle = PlaceCellController(
            environment=env_with_obstacle, config=config, rng=rng
        )
        controller_no_obstacle = PlaceCellController(
            environment=env_no_obstacle, config=config, rng=np.random.default_rng(43)
        )
        agent_obstacle = Agent(
            environment=env_with_obstacle,
            random_state=np.random.default_rng(42 + seed_offset),
        )
        agent_no_obstacle = Agent(
            environment=env_no_obstacle,
            random_state=np.random.default_rng(43 + seed_offset),
        )
        include_theta = False

    print("Running simulation WITH obstacle...")
    print("Running simulation WITHOUT obstacle (for comparison)...")

    # Run simulations
    duration_seconds = 1200.0  # 20 minutes (longer for obstacle detection)
    dt = 0.05
    num_steps = int(duration_seconds / dt)
    sample_interval = max(1, num_steps // 50)

    results_obstacle = {
        "times": [],
        "edges": [],
        "components": [],
        "betti_0": [],
        "betti_1": [],
    }

    results_no_obstacle = {
        "times": [],
        "edges": [],
        "components": [],
        "betti_0": [],
        "betti_1": [],
    }

    print(f"\nRunning for {duration_seconds/60:.1f} minutes...")
    print("  Progress: ", end="", flush=True)

    for step in range(num_steps):
        # With obstacle
        obs_obstacle = agent_obstacle.step(dt, include_theta=include_theta)
        controller_obstacle.step(np.asarray(obs_obstacle), dt)

        # Without obstacle
        obs_no_obstacle = agent_no_obstacle.step(dt, include_theta=include_theta)
        controller_no_obstacle.step(np.asarray(obs_no_obstacle), dt)

        if step % sample_interval == 0 or step == num_steps - 1:
            # Sample with obstacle
            graph_obstacle = controller_obstacle.get_graph()
            time = controller_obstacle.current_time

            results_obstacle["times"].append(time)
            results_obstacle["edges"].append(graph_obstacle.num_edges())
            results_obstacle["components"].append(graph_obstacle.num_components())

            try:
                betti_obstacle = graph_obstacle.compute_betti_numbers(max_dim=1)
                results_obstacle["betti_0"].append(betti_obstacle.get(0, 0))
                results_obstacle["betti_1"].append(betti_obstacle.get(1, 0))
            except ImportError:
                results_obstacle["betti_0"].append(graph_obstacle.num_components())
                results_obstacle["betti_1"].append(-1)

            # Sample without obstacle
            graph_no_obstacle = controller_no_obstacle.get_graph()

            results_no_obstacle["times"].append(time)
            results_no_obstacle["edges"].append(graph_no_obstacle.num_edges())
            results_no_obstacle["components"].append(graph_no_obstacle.num_components())

            try:
                betti_no_obstacle = graph_no_obstacle.compute_betti_numbers(max_dim=1)
                results_no_obstacle["betti_0"].append(betti_no_obstacle.get(0, 0))
                results_no_obstacle["betti_1"].append(betti_no_obstacle.get(1, 0))
            except ImportError:
                results_no_obstacle["betti_0"].append(graph_no_obstacle.num_components())
                results_no_obstacle["betti_1"].append(-1)

        if (step + 1) % (num_steps // 10) == 0:
            progress = 100 * (step + 1) / num_steps
            print(f"{progress:.0f}% ", end="", flush=True)

    print("\n\nSimulation complete!")
    print()

    # Print bat controller stats if applicable
    if controller_type == "bat":
        print("=" * 70)
        print("Bat Controller Statistics")
        print("=" * 70)
        hd_estimate_obs = controller_obstacle.hd_attractor.estimate_heading()
        grid_drift_obs = controller_obstacle.grid_attractor.drift_metric()
        hd_estimate_no = controller_no_obstacle.hd_attractor.estimate_heading()
        grid_drift_no = controller_no_obstacle.grid_attractor.drift_metric()
        print(f"  HD estimate (with obstacle): {np.degrees(hd_estimate_obs):.1f}°")
        print(f"  HD estimate (without obstacle): {np.degrees(hd_estimate_no):.1f}°")
        print(f"  Grid drift (with obstacle): {grid_drift_obs:.4f}")
        print(f"  Grid drift (without obstacle): {grid_drift_no:.4f}")
        print()

    # Print comparison
    print("=" * 70)
    print("Results Comparison")
    print("=" * 70)
    print()
    print(f"{'Metric':<25} | {'With Obstacle':>15} | {'Without Obstacle':>15}")
    print("-" * 70)

    final_edges_obs = results_obstacle["edges"][-1]
    final_b0_obs = results_obstacle["betti_0"][-1]
    final_b1_obs = results_obstacle["betti_1"][-1] if results_obstacle["betti_1"][-1] != -1 else "N/A"

    final_edges_no = results_no_obstacle["edges"][-1]
    final_b0_no = results_no_obstacle["betti_0"][-1]
    final_b1_no = results_no_obstacle["betti_1"][-1] if results_no_obstacle["betti_1"][-1] != -1 else "N/A"

    print(f"{'Final Edges':<25} | {final_edges_obs:>15} | {final_edges_no:>15}")
    print(f"{'Final b₀ (components)':<25} | {final_b0_obs:>15} | {final_b0_no:>15}")
    print(f"{'Final b₁ (holes)':<25} | {final_b1_obs:>15} | {final_b1_no:>15}")
    print()

    # Expected results
    print("Expected Results:")
    print("  - With obstacle: b₁ = 1 (one hole encircling obstacle)")
    print("  - Without obstacle: b₁ = 0 (no holes)")
    print()

    if final_b1_obs != "N/A" and final_b1_no != "N/A":
        if final_b1_obs == 1 and final_b1_no == 0:
            print("✓ Topology correctly identifies the obstacle as a hole!")
        elif final_b1_obs == 1:
            print("✓ Obstacle correctly produces b₁ = 1")
        elif final_b1_obs == 0:
            print("⚠ Obstacle not detected as hole (may need longer simulation or different parameters)")
        else:
            print(f"⚠ Unexpected b₁ = {final_b1_obs} with obstacle (expected 1)")

    # Plot comparison
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        controller_label = "BatNavigationController" if controller_type == "bat" else "PlaceCellController"
        fig.suptitle(f"Obstacle Environment Comparison ({controller_label})", fontsize=14, fontweight="bold")

        times = np.array(results_obstacle["times"]) / 60.0  # Convert to minutes

        # Top row: With obstacle
        ax = axes[0, 0]
        ax.plot(times, results_obstacle["edges"], "b-", label="With obstacle", linewidth=2)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Number of Edges")
        ax.set_title("Graph Growth: With Obstacle")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(times, results_obstacle["betti_1"], "r-", label="b₁ (holes)", linewidth=2)
        ax.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="Expected (b₁=1)")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Betti number b₁")
        ax.set_title("Holes over Time: With Obstacle")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        graph_obs = controller_obstacle.get_graph()
        positions_obs = controller_obstacle.place_cell_positions
        from hippocampus_core.visualization import plot_graph, plot_obstacles

        plot_obstacles(env_with_obstacle, ax=ax)
        ax.scatter(positions_obs[:, 0], positions_obs[:, 1], s=20, color="tab:green", alpha=0.6)
        for i, j in graph_obs.graph.edges():
            ax.plot(
                [positions_obs[i, 0], positions_obs[j, 0]],
                [positions_obs[i, 1], positions_obs[j, 1]],
                "k-",
                alpha=0.2,
                linewidth=0.5,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title("Final Graph: With Obstacle")
        ax.grid(True, alpha=0.3)

        # Bottom row: Without obstacle
        ax = axes[1, 0]
        ax.plot(times, results_no_obstacle["edges"], "g-", label="Without obstacle", linewidth=2)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Number of Edges")
        ax.set_title("Graph Growth: Without Obstacle")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(times, results_no_obstacle["betti_1"], "r-", label="b₁ (holes)", linewidth=2)
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.5, label="Expected (b₁=0)")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Betti number b₁")
        ax.set_title("Holes over Time: Without Obstacle")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        graph_no = controller_no_obstacle.get_graph()
        positions_no = controller_no_obstacle.place_cell_positions

        ax.scatter(positions_no[:, 0], positions_no[:, 1], s=20, color="tab:green", alpha=0.6)
        for i, j in graph_no.graph.edges():
            ax.plot(
                [positions_no[i, 0], positions_no[j, 0]],
                [positions_no[i, 1], positions_no[j, 1]],
                "k-",
                alpha=0.2,
                linewidth=0.5,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title("Final Graph: Without Obstacle")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = Path("results") / "obstacle_comparison.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot to: {output_path}")
        print()

    except ImportError:
        print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()

