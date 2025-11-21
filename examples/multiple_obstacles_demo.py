#!/usr/bin/env python3
"""Demonstration of topological mapping with multiple obstacles.

This script shows how multiple obstacles create multiple holes (b₁ > 1) in the
learned topological map. With N well-separated obstacles, we expect b₁ = N
(one hole per obstacle), matching the paper's findings.

Run with: python3 examples/multiple_obstacles_demo.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import matplotlib.pyplot as plt

from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)
from hippocampus_core.env import Agent, CircularObstacle, Environment


def generate_random_obstacles(
    num_obstacles: int,
    env_width: float = 1.0,
    env_height: float = 1.0,
    min_radius: float = 0.08,
    max_radius: float = 0.12,
    min_separation: float = 0.3,
    rng: np.random.Generator = None,
    max_attempts: int = 1000,
) -> list[CircularObstacle]:
    """Generate random non-overlapping obstacles.

    Parameters
    ----------
    num_obstacles:
        Number of obstacles to generate.
    env_width:
        Environment width.
    env_height:
        Environment height.
    min_radius:
        Minimum obstacle radius.
    max_radius:
        Maximum obstacle radius.
    min_separation:
        Minimum distance between obstacle centers.
    rng:
        Random number generator.
    max_attempts:
        Maximum attempts to place obstacles.

    Returns
    -------
    list[CircularObstacle]
        List of non-overlapping obstacles.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    obstacles = []
    attempts = 0

    while len(obstacles) < num_obstacles and attempts < max_attempts:
        # Generate random position and radius
        radius = rng.uniform(min_radius, max_radius)
        center_x = rng.uniform(radius, env_width - radius)
        center_y = rng.uniform(radius, env_height - radius)

        candidate = CircularObstacle(center_x=center_x, center_y=center_y, radius=radius)

        # Check if it overlaps with existing obstacles
        overlaps = False
        for existing in obstacles:
            dist = np.sqrt(
                (candidate.center_x - existing.center_x) ** 2
                + (candidate.center_y - existing.center_y) ** 2
            )
            if dist < (candidate.radius + existing.radius + min_separation):
                overlaps = True
                break

        # Check if it fits in bounds
        if (
            candidate.center_x - candidate.radius < 0
            or candidate.center_x + candidate.radius > env_width
            or candidate.center_y - candidate.radius < 0
            or candidate.center_y + candidate.radius > env_height
        ):
            overlaps = True

        if not overlaps:
            obstacles.append(candidate)

        attempts += 1

    if len(obstacles) < num_obstacles:
        raise RuntimeError(
            f"Failed to place {num_obstacles} obstacles after {attempts} attempts. "
            f"Placed {len(obstacles)} obstacles."
        )

    return obstacles


def generate_grid_obstacles(
    num_obstacles: int,
    env_width: float = 1.0,
    env_height: float = 1.0,
    radius: float = 0.1,
    margin: float = 0.15,
) -> list[CircularObstacle]:
    """Generate obstacles in a grid layout.

    Parameters
    ----------
    num_obstacles:
        Number of obstacles (will use closest grid size).
    env_width:
        Environment width.
    env_height:
        Environment height.
    radius:
        Obstacle radius.
    margin:
        Margin from edges.

    Returns
    -------
    list[CircularObstacle]
        List of obstacles in grid layout.
    """
    # Determine grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_obstacles)))
    available_width = env_width - 2 * margin
    available_height = env_height - 2 * margin

    if grid_size > 1:
        spacing_x = available_width / (grid_size - 1) if grid_size > 1 else available_width
        spacing_y = available_height / (grid_size - 1) if grid_size > 1 else available_height
    else:
        spacing_x = spacing_y = 0

    obstacles = []
    count = 0

    for i in range(grid_size):
        for j in range(grid_size):
            if count >= num_obstacles:
                break

            center_x = margin + i * spacing_x
            center_y = margin + j * spacing_y

            # Ensure obstacles fit in bounds
            center_x = np.clip(center_x, radius, env_width - radius)
            center_y = np.clip(center_y, radius, env_height - radius)

            obstacles.append(CircularObstacle(center_x=center_x, center_y=center_y, radius=radius))
            count += 1

        if count >= num_obstacles:
            break

    return obstacles


def main() -> None:
    """Demonstrate multiple obstacles environment with topological mapping."""
    print("=" * 70)
    print("Multiple Obstacles Topological Mapping Demo")
    print("=" * 70)
    print()
    print("This demonstrates how multiple obstacles create multiple holes.")
    print("With N well-separated obstacles, we expect b₁ = N (one hole per obstacle).")
    print()

    # Configuration
    num_obstacles = 3
    layout = "random"  # "random" or "grid"
    duration = 600.0  # 10 minutes
    num_place_cells = 120

    print(f"Configuration:")
    print(f"  Number of obstacles: {num_obstacles}")
    print(f"  Layout: {layout}")
    print(f"  Duration: {duration} seconds ({duration/60:.1f} minutes)")
    print(f"  Place cells: {num_place_cells}")
    print()

    # Generate obstacles
    rng = np.random.default_rng(42)
    if layout == "random":
        obstacles = generate_random_obstacles(
            num_obstacles=num_obstacles,
            min_radius=0.08,
            max_radius=0.12,
            min_separation=0.25,
            rng=rng,
        )
    else:  # grid
        obstacles = generate_grid_obstacles(
            num_obstacles=num_obstacles,
            radius=0.1,
            margin=0.15,
        )

    print(f"Generated {len(obstacles)} obstacles:")
    for i, obs in enumerate(obstacles):
        print(f"  Obstacle {i+1}: center=({obs.center_x:.3f}, {obs.center_y:.3f}), radius={obs.radius:.3f}")

    # Create environment
    env = Environment(width=1.0, height=1.0, obstacles=obstacles)

    # Create controller
    config = PlaceCellControllerConfig(
        num_place_cells=num_place_cells,
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.2,
        coactivity_threshold=8.0,  # Moderate threshold
        max_edge_distance=0.3,
        integration_window=240.0,  # 4 minutes
    )

    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent = Agent(environment=env, random_state=rng)

    # Run simulation
    print()
    print("Running simulation...")
    dt = 0.05
    num_steps = int(duration / dt)
    sample_interval = max(1, num_steps // 50)

    times = []
    edges = []
    components = []
    betti_0 = []
    betti_1 = []

    for step in range(num_steps):
        position = agent.step(dt)
        controller.step(position, dt)

        if step % sample_interval == 0:
            graph = controller.get_graph()
            times.append(controller.current_time)
            edges.append(graph.num_edges())
            components.append(graph.num_components())

            # Compute Betti numbers if available
            try:
                betti = graph.compute_betti_numbers(max_dim=2)
                betti_0.append(betti.get(0, -1))
                betti_1.append(betti.get(1, -1))
            except (ImportError, Exception):
                betti_0.append(-1)
                betti_1.append(-1)

        if (step + 1) % (num_steps // 10) == 0:
            progress = 100 * (step + 1) / num_steps
            print(f"  {progress:.0f}% complete")

    # Final statistics
    graph = controller.get_graph()
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Final edges: {graph.num_edges()}")
    print(f"Final components: {graph.num_components()}")

    try:
        final_betti = graph.compute_betti_numbers(max_dim=2)
        print(f"Final b₀ (components): {final_betti.get(0, -1)}")
        print(f"Final b₁ (holes): {final_betti.get(1, -1)}")
        print(f"Expected b₁: {num_obstacles} (one hole per obstacle)")
        if final_betti.get(1, -1) >= 0:
            print(f"  Match: {'✓' if final_betti.get(1, -1) == num_obstacles else '✗ (clique complex may fill holes)'}")
    except (ImportError, Exception):
        print("Betti numbers not available (ripser/gudhi not installed)")

    # Visualization
    print()
    print("Creating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Multiple Obstacles Topological Mapping ({num_obstacles} obstacles)", fontsize=14, fontweight="bold")

    # Plot 1: Environment with obstacles and graph
    ax1 = axes[0, 0]
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")

    # Draw obstacles
    for obs in obstacles:
        circle = plt.Circle(
            (obs.center_x, obs.center_y),
            obs.radius,
            color="red",
            alpha=0.5,
            edgecolor="darkred",
            linewidth=2,
        )
        ax1.add_patch(circle)

    # Draw graph
    positions = controller.place_cell_positions
    for u, v in graph.graph.edges():
        pos_u = positions[u]
        pos_v = positions[v]
        ax1.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], "b-", alpha=0.3, linewidth=0.5)

    # Draw nodes
    ax1.scatter(positions[:, 0], positions[:, 1], c="blue", s=10, alpha=0.6, label="Place cells")

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Graph Structure with Obstacles")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Edges over time
    ax2 = axes[0, 1]
    if times and edges:
        ax2.plot(times, edges, "b-", linewidth=2, label="Edges")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Number of Edges")
    ax2.set_title("Edge Formation Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Betti numbers over time
    ax3 = axes[1, 0]
    if times and betti_0 and betti_0[0] >= 0:
        valid_b0 = [b for b in betti_0 if b >= 0]
        valid_times_b0 = [t for t, b in zip(times, betti_0) if b >= 0]
        if valid_b0:
            ax3.plot(valid_times_b0, valid_b0, "g-", linewidth=2, label="b₀ (Components)", marker="o", markersize=3)

    if times and betti_1 and betti_1[0] >= 0:
        valid_b1 = [b for b in betti_1 if b >= 0]
        valid_times_b1 = [t for t, b in zip(times, betti_1) if b >= 0]
        if valid_b1:
            ax3.plot(valid_times_b1, valid_b1, "r-", linewidth=2, label="b₁ (Holes)", marker="s", markersize=3)
            ax3.axhline(y=num_obstacles, color="r", linestyle="--", alpha=0.5, label=f"Target (b₁={num_obstacles})")

    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Betti Number")
    ax3.set_title("Topology Evolution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Components over time
    ax4 = axes[1, 1]
    if times and components:
        ax4.plot(times, components, "purple", linewidth=2, label="Components")
        ax4.axhline(y=1, color="g", linestyle="--", alpha=0.5, label="Target (b₀=1)")
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Number of Components")
    ax4.set_title("Graph Connectivity")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = Path("results/multiple_obstacles_demo.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")

    plt.close()


if __name__ == "__main__":
    main()

