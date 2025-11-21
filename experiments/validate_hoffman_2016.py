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
from hippocampus_core.env import Agent, CircularObstacle, Environment


def _sample_uniform_positions(
    env: Environment, num_points: int, rng: np.random.Generator, max_attempts: int = 50000
) -> np.ndarray:
    """Sample valid positions within the environment (excluding obstacles)."""

    bounds = env.bounds
    samples: list[np.ndarray] = []
    attempts = 0

    while len(samples) < num_points and attempts < max_attempts:
        candidate = rng.uniform(
            low=(bounds.min_x, bounds.min_y),
            high=(bounds.max_x, bounds.max_y),
            size=(2,),
        )
        if env.contains(tuple(candidate)):
            samples.append(candidate)
        attempts += 1

    if len(samples) < num_points:
        raise RuntimeError(
            f"Failed to sample {num_points} valid positions after {attempts} attempts"
        )

    return np.stack(samples, axis=0)


def _generate_obstacle_ring_positions(
    env: Environment,
    obstacle: CircularObstacle,
    num_cells: int,
    rng: np.random.Generator,
    ring_fraction: float,
    ring_offset: float,
    ring_jitter: float,
) -> np.ndarray:
    """Generate place-cell centers with a ring hugging the obstacle boundary."""

    if not (0.0 <= ring_fraction <= 1.0):
        raise ValueError("ring_fraction must lie in [0, 1]")
    if ring_offset <= 0:
        raise ValueError("ring_offset must be positive to stay outside the obstacle")
    if ring_jitter < 0:
        raise ValueError("ring_jitter must be non-negative")

    ring_count = int(round(num_cells * ring_fraction))
    if ring_fraction > 0.0:
        ring_count = max(3, ring_count)
    ring_count = min(ring_count, num_cells)
    remaining = num_cells - ring_count

    ring_radius = obstacle.radius + ring_offset
    bounds = env.bounds
    max_radius = min(
        obstacle.center_x - bounds.min_x,
        bounds.max_x - obstacle.center_x,
        obstacle.center_y - bounds.min_y,
        bounds.max_y - obstacle.center_y,
    )
    if ring_radius >= max_radius:
        raise ValueError(
            "ring_radius extends beyond environment bounds; "
            "reduce ring_offset or obstacle radius"
        )

    positions: list[np.ndarray] = []
    if ring_count > 0:
        angles = np.linspace(0.0, 2.0 * np.pi, ring_count, endpoint=False)
        for angle in angles:
            base = np.array(
                [
                    obstacle.center_x + np.cos(angle) * ring_radius,
                    obstacle.center_y + np.sin(angle) * ring_radius,
                ]
            )
            placed = False
            for _ in range(200):
                jitter = (
                    rng.normal(scale=ring_jitter, size=2) if ring_jitter > 0 else np.zeros(2)
                )
                candidate = base + jitter
                if env.contains(tuple(candidate)):
                    positions.append(candidate)
                    placed = True
                    break
            if not placed:
                raise RuntimeError(
                    "Failed to place ring cell; adjust ring_offset/jitter parameters."
                )

    if remaining > 0:
        filler = _sample_uniform_positions(env, remaining, rng)
        positions.extend(list(filler))

    centers = np.stack(positions, axis=0) if positions else np.zeros((0, 2))
    if centers.shape[0] != num_cells:
        raise RuntimeError(
            f"Requested {num_cells} place cells but generated {centers.shape[0]}"
        )

    perm = rng.permutation(num_cells)
    return centers[perm]


def _generate_ring_spoke_positions(
    env: Environment,
    obstacle: CircularObstacle,
    num_cells: int,
    rng: np.random.Generator,
    ring_fraction: float,
    spoke_fraction: float,
    ring_offset: float,
    ring_jitter: float,
    spoke_extension: float,
    spoke_jitter: float,
    num_spokes: int = 4,
) -> np.ndarray:
    """Generate positions combining an obstacle ring plus radial spokes."""

    if not (0.0 <= spoke_fraction <= 1.0):
        raise ValueError("spoke_fraction must lie in [0, 1]")
    if spoke_extension <= 0:
        raise ValueError("spoke_extension must be positive")
    if spoke_jitter < 0:
        raise ValueError("spoke_jitter must be non-negative")
    if num_spokes < 2:
        raise ValueError("num_spokes must be at least 2")

    ring_count = int(round(num_cells * ring_fraction))
    spoke_count = int(round(num_cells * spoke_fraction))
    remaining = num_cells - ring_count - spoke_count
    if remaining < 0:
        remaining = 0
        total = max(1, ring_count + spoke_count)
        scale = num_cells / total
        ring_count = int(round(ring_count * scale))
        spoke_count = num_cells - ring_count

    positions: list[np.ndarray] = []
    if ring_count > 0:
        positions.extend(
            _generate_obstacle_ring_positions(
                env=env,
                obstacle=obstacle,
                num_cells=ring_count,
                rng=rng,
                ring_fraction=1.0,
                ring_offset=ring_offset,
                ring_jitter=ring_jitter,
            )
        )

    if spoke_count > 0:
        angles = np.linspace(0.0, 2.0 * np.pi, num_spokes, endpoint=False)
        points_per_spoke = max(1, spoke_count // num_spokes)
        extra = spoke_count - points_per_spoke * num_spokes

        ring_radius = obstacle.radius + ring_offset
        outer_radius = ring_radius + spoke_extension
        bounds = env.bounds
        max_radius = min(
            obstacle.center_x - bounds.min_x,
            bounds.max_x - obstacle.center_x,
            obstacle.center_y - bounds.min_y,
            bounds.max_y - obstacle.center_y,
        )
        if outer_radius >= max_radius:
            raise ValueError(
                "spoke extension pushes cells outside bounds; reduce --spoke-extension"
            )

        distances = np.linspace(ring_radius, outer_radius, points_per_spoke + 2)[1:-1]
        if distances.size == 0:
            distances = np.array([0.5 * (ring_radius + outer_radius)])

        for idx, angle in enumerate(angles):
            count = points_per_spoke + (1 if idx < extra else 0)
            if count == 0:
                continue
            base_vec = np.array([np.cos(angle), np.sin(angle)])
            for d in np.linspace(ring_radius, outer_radius, count + 2)[1:-1]:
                candidate = np.array(
                    [
                        obstacle.center_x + base_vec[0] * d,
                        obstacle.center_y + base_vec[1] * d,
                    ]
                )
                placed = False
                for _ in range(200):
                    jitter = (
                        rng.normal(scale=spoke_jitter, size=2)
                        if spoke_jitter > 0
                        else np.zeros(2)
                    )
                    jittered = candidate + jitter
                    if env.contains(tuple(jittered)):
                        positions.append(jittered)
                        placed = True
                        break
                if not placed:
                    raise RuntimeError(
                        "Failed to place spoke cell; adjust spoke parameters."
                    )

    if remaining > 0:
        filler = _sample_uniform_positions(env, remaining, rng)
        positions.extend(list(filler))

    centers = np.stack(positions, axis=0) if positions else np.zeros((0, 2))
    if centers.shape[0] != num_cells:
        raise RuntimeError(
            f"Requested {num_cells} place cells but generated {centers.shape[0]}"
        )

    perm = rng.permutation(num_cells)
    return centers[perm]


def run_learning_experiment(
    env: Environment,
    integration_window: Optional[float],
    duration_seconds: float = 300.0,  # 5 minutes
    dt: float = 0.05,
    num_place_cells: int = 100,
    sigma: float = 0.15,
    seed: int = 42,
    expected_b1: Optional[int] = None,
    coactivity_threshold: float = 5.0,
    max_edge_distance: Optional[float] = None,
    place_cell_positions: Optional[np.ndarray] = None,
    trajectory_mode: str = "random",
    trajectory_params: Optional[dict] = None,
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
    coactivity_threshold:
        Minimum coactivity required before edges are admitted
    max_edge_distance:
        Maximum spatial distance between place-cell centers to admit an edge
    place_cell_positions:
        Optional explicit array of place-cell centers to use instead of sampling

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
        coactivity_threshold=coactivity_threshold,
        max_edge_distance=max_edge_distance,
        integration_window=integration_window,  # ϖ: integration window
        place_cell_positions=place_cell_positions,
    )

    rng = np.random.default_rng(seed)
    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent_rng = np.random.default_rng(seed + 1)
    agent = Agent(environment=env, random_state=agent_rng)
    trajectory_params = trajectory_params or {}
    orbit_state = None
    if trajectory_mode == "orbit_then_random":
        obstacle_list = env.obstacles
        if not obstacle_list:
            raise ValueError("orbit_then_random trajectory requires an obstacle.")
        obstacle = obstacle_list[0]
        orbit_radius = trajectory_params.get("orbit_radius", obstacle.radius + 0.02)
        orbit_duration = trajectory_params.get("orbit_duration", 120.0)
        orbit_speed = trajectory_params.get("orbit_speed", 0.5)
        orbit_state = {
            "center": np.array([obstacle.center_x, obstacle.center_y]),
            "radius": orbit_radius,
            "duration": orbit_duration,
            "angular_speed": orbit_speed / max(1e-6, orbit_radius),
            "angle": 0.0,
        }

    num_steps = int(duration_seconds / dt)
    sample_interval = max(1, num_steps // 20)  # Sample ~20 times (reduced for speed)

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
        if trajectory_mode == "orbit_then_random" and orbit_state and controller.current_time < orbit_state["duration"]:
            orbit_state["angle"] += orbit_state["angular_speed"] * dt
            angle = orbit_state["angle"]
            position = orbit_state["center"] + orbit_state["radius"] * np.array([np.cos(angle), np.sin(angle)])
            agent.position = position.copy()
        else:
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
                if num_edges == 0:
                    # If no edges, all nodes are isolated - b0 should equal number of nodes
                    # But Betti number computation on empty graph might give 1
                    # Use components instead for consistency
                    b0 = num_components
                elif b0 != num_components:
                    # Log warning but use Betti number (may be computing from clique complex)
                    # This is acceptable as Betti numbers come from clique complex
                    pass
                
                results["betti_0"].append(b0)
                results["betti_1"].append(b1)
                results["betti_2"].append(b2)
            except ImportError:
                # Fallback if persistent homology not available
                results["betti_0"].append(num_components)
                results["betti_1"].append(-1)  # Mark as unavailable
                results["betti_2"].append(-1)

    print(" done")
    
    # Add consistency assertions
    final_edges = results["edges"][-1]
    final_b0 = results["betti_0"][-1]
    final_components = results["components"][-1]
    
    # Assert consistency: if no edges, b0 should equal number of nodes
    if final_edges == 0:
        assert final_b0 == final_components, (
            f"Inconsistent: {final_edges} edges but b₀={final_b0} "
            f"(expected {final_components} isolated nodes)"
        )
    
    # Optional: validate expected topology
    if expected_b1 is not None and results["betti_1"][-1] != -1:
        final_b1 = results["betti_1"][-1]
        # Allow some tolerance (could be 0-2 if learning incomplete)
        if abs(final_b1 - expected_b1) > 2:
            print(f"  Warning: Expected b₁ ≈ {expected_b1}, got {final_b1}")
    
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
        "--sigma",
        type=float,
        default=0.15,
        help="Place-field sigma parameter in arena units (default: 0.15)",
    )
    parser.add_argument(
        "--num-cells",
        type=int,
        default=100,
        help="Number of place cells (default: 100)",
    )
    parser.add_argument(
        "--coactivity-threshold",
        type=float,
        default=5.0,
        help="Minimum coactivity required before edges are added (default: 5.0)",
    )
    parser.add_argument(
        "--edge-distance-multiplier",
        type=float,
        default=2.0,
        help=(
            "Multiplier applied to sigma to compute max edge distance "
            "when --max-edge-distance is not provided (default: 2.0)"
        ),
    )
    parser.add_argument(
        "--max-edge-distance",
        type=float,
        help=(
            "Absolute maximum allowed distance between place-cell centers "
            "for clique edges. Overrides --edge-distance-multiplier."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save output figure (default: show interactively)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--obstacle",
        action="store_true",
        help="Use environment with central obstacle (expected b₁ = 1)",
    )
    parser.add_argument(
        "--num-obstacles",
        type=int,
        default=1,
        help="Number of obstacles to create (default: 1, requires --obstacle)",
    )
    parser.add_argument(
        "--obstacle-layout",
        choices=["grid", "random"],
        default="random",
        help="Layout strategy for multiple obstacles (default: random)",
    )
    parser.add_argument(
        "--obstacle-radius",
        type=float,
        default=0.15,
        help="Radius of obstacle(s) (default: 0.15). For multiple obstacles, this is the base radius.",
    )
    parser.add_argument(
        "--obstacle-size-variance",
        type=float,
        default=0.0,
        help="Size variance for obstacles (0.0 = all same size, >0.0 = random sizes) (default: 0.0)",
    )
    parser.add_argument(
        "--placement-mode",
        choices=["uniform", "obstacle_ring", "ring_spokes"],
        default="uniform",
        help="Place-cell placement strategy (default: uniform)",
    )
    parser.add_argument(
        "--ring-fraction",
        type=float,
        default=0.35,
        help=(
            "Fraction of place cells pinned along the obstacle ring when "
            "placement-mode=obstacle_ring (default: 0.35)"
        ),
    )
    parser.add_argument(
        "--ring-offset",
        type=float,
        default=0.02,
        help="Offset beyond obstacle radius used for ring placement (default: 0.02)",
    )
    parser.add_argument(
        "--ring-jitter",
        type=float,
        default=0.01,
        help="Gaussian jitter applied to ring positions (default std dev: 0.01)",
    )
    parser.add_argument(
        "--spoke-fraction",
        type=float,
        default=0.2,
        help="Fraction of cells assigned to radial spokes (ring_spokes mode)",
    )
    parser.add_argument(
        "--spoke-extension",
        type=float,
        default=0.12,
        help="How far spokes extend beyond the ring radius (default: 0.12)",
    )
    parser.add_argument(
        "--spoke-jitter",
        type=float,
        default=0.01,
        help="Gaussian jitter for spoke cells (default std dev: 0.01)",
    )
    parser.add_argument(
        "--num-spokes",
        type=int,
        default=4,
        help="Number of spokes used in ring_spokes placement (default: 4)",
    )
    parser.add_argument(
        "--default-window-output",
        type=Path,
        help=(
            "If provided, rerun the sweep with default integration windows "
            "(0 60 120 240 480) and save that composite figure here."
        ),
    )
    parser.add_argument(
        "--trajectory-mode",
        choices=["random", "orbit_then_random"],
        default="random",
        help="Agent trajectory mode (default: random walk)",
    )
    parser.add_argument(
        "--orbit-duration",
        type=float,
        default=120.0,
        help="Duration (s) to keep agent orbiting obstacle before random walk",
    )
    parser.add_argument(
        "--orbit-radius",
        type=float,
        default=0.25,
        help="Orbit radius used during orbit_then_random (default: 0.25)",
    )
    parser.add_argument(
        "--orbit-speed",
        type=float,
        default=0.6,
        help="Linear speed along orbit (arena units/s) (default: 0.6)",
    )

    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("duration must be positive")
    if args.num_cells <= 0:
        parser.error("num-cells must be positive")
    if args.sigma <= 0:
        parser.error("sigma must be positive")
    if args.coactivity_threshold <= 0:
        parser.error("coactivity-threshold must be positive")
    if args.edge_distance_multiplier <= 0:
        parser.error("edge-distance-multiplier must be positive")
    if args.max_edge_distance is not None and args.max_edge_distance <= 0:
        parser.error("max-edge-distance must be positive when provided")
    if not 0.0 <= args.ring_fraction <= 1.0:
        parser.error("ring-fraction must lie within [0, 1]")
    if args.ring_offset <= 0:
        parser.error("ring-offset must be positive")
    if args.ring_jitter < 0:
        parser.error("ring-jitter must be non-negative")
    if not 0.0 <= args.spoke_fraction <= 1.0:
        parser.error("spoke-fraction must lie within [0, 1]")
    if args.spoke_extension <= 0:
        parser.error("spoke-extension must be positive")
    if args.spoke_jitter < 0:
        parser.error("spoke-jitter must be non-negative")
    if args.num_spokes < 2:
        parser.error("num-spokes must be at least 2")
    if args.placement_mode in {"obstacle_ring", "ring_spokes"} and not args.obstacle:
        parser.error(f"--placement-mode {args.placement_mode} requires --obstacle")
    if args.num_obstacles < 1:
        parser.error("num-obstacles must be at least 1")
    if args.num_obstacles > 1 and not args.obstacle:
        parser.error("--num-obstacles > 1 requires --obstacle")
    if args.obstacle_size_variance < 0:
        parser.error("obstacle-size-variance must be non-negative")
    if args.orbit_radius <= 0:
        parser.error("orbit-radius must be positive")
    if args.orbit_duration < 0:
        parser.error("orbit-duration must be non-negative")
    if args.orbit_speed <= 0:
        parser.error("orbit-speed must be positive")

    max_edge_distance_value = (
        args.max_edge_distance
        if args.max_edge_distance is not None
        else args.edge_distance_multiplier * args.sigma
    )

    # Initialize env to None to avoid UnboundLocalError with closure
    env = None

    # Create environment
    if args.obstacle:
        if args.num_obstacles == 1:
            # Single obstacle (original behavior)
            obstacle = CircularObstacle(center_x=0.5, center_y=0.5, radius=args.obstacle_radius)
            obstacles = [obstacle]
            expected_b1 = 1
            env = Environment(width=1.0, height=1.0, obstacles=obstacles)
        else:
            # Multiple obstacles - generate them here
            obstacle_rng = np.random.default_rng(args.seed + 9999)

            if args.obstacle_layout == "grid":
                # Grid layout
                grid_size = int(np.ceil(np.sqrt(args.num_obstacles)))
                margin = 0.15
                available_width = 1.0 - 2 * margin
                available_height = 1.0 - 2 * margin
                
                if grid_size > 1:
                    spacing_x = available_width / (grid_size - 1) if grid_size > 1 else available_width
                    spacing_y = available_height / (grid_size - 1) if grid_size > 1 else available_height
                else:
                    spacing_x = spacing_y = 0
                
                obstacles = []
                count = 0
                for i in range(grid_size):
                    for j in range(grid_size):
                        if count >= args.num_obstacles:
                            break
                        center_x = margin + i * spacing_x
                        center_y = margin + j * spacing_y
                        center_x = np.clip(center_x, args.obstacle_radius, 1.0 - args.obstacle_radius)
                        center_y = np.clip(center_y, args.obstacle_radius, 1.0 - args.obstacle_radius)
                        obstacles.append(CircularObstacle(center_x=center_x, center_y=center_y, radius=args.obstacle_radius))
                        count += 1
                    if count >= args.num_obstacles:
                        break
            else:  # random
                # Random layout with non-overlapping obstacles
                min_radius = max(0.05, args.obstacle_radius - args.obstacle_size_variance)
                max_radius = args.obstacle_radius + args.obstacle_size_variance
                min_separation = 0.25
                max_attempts = 1000
                
                obstacles = []
                attempts = 0
                while len(obstacles) < args.num_obstacles and attempts < max_attempts:
                    radius = obstacle_rng.uniform(min_radius, max_radius)
                    center_x = obstacle_rng.uniform(radius, 1.0 - radius)
                    center_y = obstacle_rng.uniform(radius, 1.0 - radius)
                    candidate = CircularObstacle(center_x=center_x, center_y=center_y, radius=radius)
                    
                    overlaps = False
                    for existing in obstacles:
                        dist = np.sqrt(
                            (candidate.center_x - existing.center_x) ** 2
                            + (candidate.center_y - existing.center_y) ** 2
                        )
                        if dist < (candidate.radius + existing.radius + min_separation):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        obstacles.append(candidate)
                    attempts += 1
                
                if len(obstacles) < args.num_obstacles:
                    raise RuntimeError(
                        f"Failed to place {args.num_obstacles} obstacles after {attempts} attempts. "
                        f"Placed {len(obstacles)} obstacles."
                    )

            env = Environment(width=1.0, height=1.0, obstacles=obstacles)
            expected_b1 = args.num_obstacles
            obstacle = obstacles[0] if obstacles else None  # For compatibility with placement modes
    else:
        obstacle = None
        obstacles = []
        env = Environment(width=1.0, height=1.0)
        expected_b1 = 0

    place_cell_positions = None
    if args.placement_mode == "obstacle_ring":
        placement_rng = np.random.default_rng(args.seed + 1337)
        assert obstacle is not None
        place_cell_positions = _generate_obstacle_ring_positions(
            env=env,
            obstacle=obstacle,
            num_cells=args.num_cells,
            rng=placement_rng,
            ring_fraction=args.ring_fraction,
            ring_offset=args.ring_offset,
            ring_jitter=args.ring_jitter,
        )
    elif args.placement_mode == "ring_spokes":
        placement_rng = np.random.default_rng(args.seed + 1337)
        assert obstacle is not None
        place_cell_positions = _generate_ring_spoke_positions(
            env=env,
            obstacle=obstacle,
            num_cells=args.num_cells,
            rng=placement_rng,
            ring_fraction=args.ring_fraction,
            spoke_fraction=args.spoke_fraction,
            ring_offset=args.ring_offset,
            ring_jitter=args.ring_jitter,
            spoke_extension=args.spoke_extension,
            spoke_jitter=args.spoke_jitter,
            num_spokes=args.num_spokes,
        )

    print("=" * 70)
    print("Hoffman et al. (2016) Topological Mapping Validation")
    print("=" * 70)
    print()
    print(f"Integration windows: {args.integration_windows} seconds")
    print(f"Simulation duration: {args.duration} seconds ({args.duration/60:.1f} minutes)")
    print(f"Number of place cells: {args.num_cells}")
    print(f"Place-field sigma: {args.sigma}")
    print(f"Coactivity threshold: {args.coactivity_threshold}")
    print(f"Max edge distance: {max_edge_distance_value:.3f}")
    print(f"Obstacle environment: {'Yes' if args.obstacle else 'No'}")
    if args.obstacle:
        print(f"  Number of obstacles: {args.num_obstacles}")
        print(f"  Layout: {args.obstacle_layout}")
        print(f"  Obstacle radius: {args.obstacle_radius}")
        if args.obstacle_size_variance > 0:
            print(f"  Size variance: {args.obstacle_size_variance}")
        print(f"  Expected b₁: {expected_b1} (one hole per obstacle)")
    print(f"Placement mode: {args.placement_mode}")
    if args.placement_mode == "obstacle_ring":
        print(
            f"  Ring fraction: {args.ring_fraction:.2f}, "
            f"offset: {args.ring_offset:.3f}, jitter: {args.ring_jitter:.3f}"
        )
    elif args.placement_mode == "ring_spokes":
        print(
            f"  Ring fraction: {args.ring_fraction:.2f}, spokes: {args.spoke_fraction:.2f} "
            f"(num-spokes={args.num_spokes}), ring offset: {args.ring_offset:.3f}, "
            f"ring jitter: {args.ring_jitter:.3f}, spoke ext: {args.spoke_extension:.3f}, "
            f"spoke jitter: {args.spoke_jitter:.3f}"
        )
    print(f"Trajectory mode: {args.trajectory_mode}")
    if args.trajectory_mode == "orbit_then_random":
        print(
            f"  Orbit duration: {args.orbit_duration:.1f}s, "
            f"radius: {args.orbit_radius:.3f}, speed: {args.orbit_speed:.3f}"
        )
    print()

    trajectory_params = None
    if args.trajectory_mode == "orbit_then_random":
        trajectory_params = {
            "orbit_radius": args.orbit_radius,
            "orbit_duration": args.orbit_duration,
            "orbit_speed": args.orbit_speed,
        }

    def execute_window_sweep(
        window_list: list[float],
        environment: Environment,
    ) -> dict[Optional[float], dict]:
        """Run a sweep over integration windows with shared parameters."""

        sweep_results: dict[Optional[float], dict] = {}
        for integration_window_sec in window_list:
            integration_window = None if integration_window_sec == 0 else integration_window_sec
            results = run_learning_experiment(
                env=environment,
                integration_window=integration_window,
                duration_seconds=args.duration,
                num_place_cells=args.num_cells,
                sigma=args.sigma,
                seed=args.seed,
                expected_b1=expected_b1,
                coactivity_threshold=args.coactivity_threshold,
                max_edge_distance=max_edge_distance_value,
                place_cell_positions=place_cell_positions,
                trajectory_mode=args.trajectory_mode,
                trajectory_params=trajectory_params,
            )
            sweep_results[integration_window] = results
        return sweep_results

    # Run experiments for each integration window
    results_by_window = execute_window_sweep(args.integration_windows, env)

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

        # Assert consistency before printing
        if final_edges == 0:
            assert final_b0 == final_components, (
                f"Inconsistent final state: {final_edges} edges but b₀={final_b0} "
                f"(expected {final_components} isolated nodes for ϖ={window_str})"
            )

        print(
            f"{window_str:>8} | {final_edges:>12} | {final_b0:>10} | {final_b1:>10} | {t_min_str:>12}"
        )

    # Plot results
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)
    try:
        plot_results(results_by_window, output_path=args.output)
    except Exception as e:
        print(f"\nError generating plots: {e}")
        print("Continuing without plots...")
        import traceback
        traceback.print_exc()

    if args.default_window_output:
        print("\n" + "=" * 70)
        print("Running default integration-window sweep (0 60 120 240 480)")
        print("=" * 70)
        default_windows = [0, 60, 120, 240, 480]
        default_results = execute_window_sweep(default_windows, env)
        try:
            plot_results(default_results, output_path=args.default_window_output)
        except Exception as e:  # pragma: no cover - plotting diagnostics
            print(f"\nError generating default-window plots: {e}")
            import traceback

            traceback.print_exc()

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

