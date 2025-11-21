#!/usr/bin/env python3
"""Replicate Hoffman et al. (2016) paper results.

This script runs experiments with exact paper parameters to enable
direct comparison with published results.

Paper: arXiv:1601.04253v1 [q-bio.NC] - "Topological mapping of space in bat hippocampus"

Key parameters from paper:
- 343 place cells (7×7×7 grid in 3D)
- Place field size: 95 cm (σ = 33.6 cm normalized)
- Mean speed: 66 cm/s
- Duration: 120 minutes
- Integration window: 8 minutes (480 seconds)
- Expected T_min: ~28 minutes for clique complex

Usage:
    python3 experiments/replicate_paper.py \
        --output results/paper_replication.png \
        --quick  # Use quick preset for faster testing
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

from hippocampus_core.controllers.place_cell_controller import PlaceCellController
from hippocampus_core.env import Agent, Environment
from hippocampus_core.presets import (
    PaperPreset,
    get_paper_preset,
    get_paper_preset_2d,
    get_paper_preset_quick,
)


def run_paper_experiment(
    config: PlaceCellControllerConfig,
    agent_params: dict,
    duration: float,
    seed: int = 42,
    quick: bool = False,
) -> dict:
    """Run experiment with paper parameters.

    Parameters
    ----------
    config:
        Place cell controller configuration.
    agent_params:
        Agent parameters dictionary.
    duration:
        Simulation duration in seconds.
    seed:
        Random seed.
    quick:
        If True, use reduced sampling for faster execution.

    Returns
    -------
    dict
        Results dictionary with times, edges, components, Betti numbers.
    """
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(seed)

    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent = Agent(
        environment=env,
        random_state=rng,
        base_speed=agent_params["base_speed"],
        max_speed=agent_params["max_speed"],
        velocity_noise=agent_params["velocity_noise"],
    )

    dt = 0.05
    num_steps = int(duration / dt)
    sample_interval = max(1, num_steps // 100) if not quick else max(1, num_steps // 50)

    results = {
        "times": [],
        "edges": [],
        "components": [],
        "betti_0": [],
        "betti_1": [],
    }

    print(f"Running paper replication experiment...")
    print(f"  Duration: {duration/60:.1f} minutes")
    print(f"  Place cells: {config.num_place_cells}")
    print(f"  Integration window: {config.integration_window}s ({config.integration_window/60:.1f} min)")
    print(f"  Sampling every {sample_interval} steps")

    for step in range(num_steps):
        position = agent.step(dt)
        controller.step(position, dt)

        if step % sample_interval == 0:
            graph = controller.get_graph()
            results["times"].append(controller.current_time)
            results["edges"].append(graph.num_edges())
            results["components"].append(graph.num_components())

            # Compute Betti numbers if available
            try:
                betti = graph.compute_betti_numbers(max_dim=2)
                results["betti_0"].append(betti.get(0, -1))
                results["betti_1"].append(betti.get(1, -1))
            except (ImportError, Exception):
                results["betti_0"].append(-1)
                results["betti_1"].append(-1)

        if (step + 1) % (num_steps // 10) == 0:
            progress = 100 * (step + 1) / num_steps
            print(f"  {progress:.0f}% complete")

    return results


def estimate_t_min(results: dict, target_betti_0: int = 1, target_betti_1: int = 0) -> float | None:
    """Estimate learning time T_min from results.

    Parameters
    ----------
    results:
        Results dictionary.
    target_betti_0:
        Target b₀ value.
    target_betti_1:
        Target b₁ value.

    Returns
    -------
    float | None
        T_min in seconds, or None if not reached.
    """
    times = np.array(results["times"])
    betti_0 = np.array(results["betti_0"])
    betti_1 = np.array(results["betti_1"])

    # Find first time when target is reached
    valid_mask = (betti_0 >= 0) & (betti_1 >= 0)
    if not np.any(valid_mask):
        return None

    target_mask = (betti_0[valid_mask] == target_betti_0) & (betti_1[valid_mask] == target_betti_1)
    if not np.any(target_mask):
        return None

    valid_times = times[valid_mask]
    first_idx = np.where(target_mask)[0][0]
    return float(valid_times[first_idx])


def create_paper_replication_plot(
    results: dict,
    t_min: float | None,
    preset: PaperPreset,
    output_path: Path | None = None,
) -> None:
    """Create publication-ready figure matching paper style.

    Parameters
    ----------
    results:
        Results dictionary.
    t_min:
        Estimated T_min value.
    preset:
        Paper preset configuration.
    output_path:
        Optional path to save figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Hoffman et al. (2016) Paper Replication",
        fontsize=16,
        fontweight="bold",
    )

    times = np.array(results["times"])

    # Plot 1: Edges over time
    ax1 = axes[0, 0]
    ax1.plot(times / 60, results["edges"], "b-", linewidth=2, label="Edges")
    if t_min:
        ax1.axvline(x=t_min / 60, color="r", linestyle="--", alpha=0.7, label=f"T_min = {t_min/60:.1f} min")
    ax1.set_xlabel("Time (minutes)")
    ax1.set_ylabel("Number of Edges")
    ax1.set_title("Edge Formation Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Betti numbers over time
    ax2 = axes[0, 1]
    valid_b0 = [b for b in results["betti_0"] if b >= 0]
    valid_times_b0 = [t / 60 for t, b in zip(times, results["betti_0"]) if b >= 0]
    if valid_b0:
        ax2.plot(valid_times_b0, valid_b0, "g-", linewidth=2, label="b₀ (Components)", marker="o", markersize=3)

    valid_b1 = [b for b in results["betti_1"] if b >= 0]
    valid_times_b1 = [t / 60 for t, b in zip(times, results["betti_1"]) if b >= 0]
    if valid_b1:
        ax2.plot(valid_times_b1, valid_b1, "r-", linewidth=2, label="b₁ (Holes)", marker="s", markersize=3)

    if t_min:
        ax2.axvline(x=t_min / 60, color="r", linestyle="--", alpha=0.7, label=f"T_min = {t_min/60:.1f} min")

    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("Betti Number")
    ax2.set_title("Topology Evolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Components over time
    ax3 = axes[1, 0]
    ax3.plot(times / 60, results["components"], "purple", linewidth=2, label="Components")
    ax3.axhline(y=1, color="g", linestyle="--", alpha=0.5, label="Target (b₀=1)")
    if t_min:
        ax3.axvline(x=t_min / 60, color="r", linestyle="--", alpha=0.7, label=f"T_min = {t_min/60:.1f} min")
    ax3.set_xlabel("Time (minutes)")
    ax3.set_ylabel("Number of Components")
    ax3.set_title("Graph Connectivity")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis("off")

    table_data = []
    table_data.append(["Parameter", "Value"])

    table_data.append(["Place cells", f"{preset.num_place_cells}"])
    table_data.append(["Sigma (σ)", f"{preset.sigma:.3f}"])
    table_data.append(["Coactivity window", f"{preset.coactivity_window*1000:.0f} ms"])
    table_data.append(["Integration window (ϖ)", f"{preset.integration_window/60:.1f} min"])
    table_data.append(["Duration", f"{preset.duration/60:.0f} min"])
    table_data.append(["Agent speed", f"{preset.agent_base_speed*100:.0f} cm/s"])

    if results["edges"]:
        table_data.append(["Final edges", f"{results['edges'][-1]}"])
    if results["components"]:
        table_data.append(["Final components", f"{results['components'][-1]}"])
    if results["betti_0"] and results["betti_0"][-1] >= 0:
        table_data.append(["Final b₀", f"{results['betti_0'][-1]}"])
    if results["betti_1"] and results["betti_1"][-1] >= 0:
        table_data.append(["Final b₁", f"{results['betti_1'][-1]}"])
    if t_min:
        table_data.append(["T_min", f"{t_min/60:.1f} min"])
        table_data.append(["Paper T_min", "~28 min"])
        if t_min / 60 < 50:
            table_data.append(["Match", "✓ Reasonable"])

    table = ax4.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax4.set_title("Parameters and Results", fontweight="bold", pad=20)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved figure to: {output_path}")
    else:
        plt.show()


def main() -> None:
    """Run paper replication experiment."""
    parser = argparse.ArgumentParser(
        description="Replicate Hoffman et al. (2016) paper results"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/paper_replication.png"),
        help="Output path for figure (default: results/paper_replication.png)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick preset (reduced duration and cells for faster testing)",
    )
    parser.add_argument(
        "--2d",
        action="store_true",
        help="Use 2D-optimized preset (100 cells instead of 343)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Override duration in seconds (default: use preset duration)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Hoffman et al. (2016) Paper Replication")
    print("=" * 70)
    print()

    # Get preset
    if args.quick:
        config, agent_params = get_paper_preset_quick()
        preset = PaperPreset()  # For display
        duration = args.duration or 600.0  # 10 minutes for quick test
        print("Using QUICK preset (reduced parameters for faster testing)")
    elif args.d2d:
        config, agent_params = get_paper_preset_2d()
        preset = PaperPreset()  # For display
        duration = args.duration or preset.duration
        print("Using 2D-optimized preset (100 cells)")
    else:
        config, agent_params = get_paper_preset()
        preset = PaperPreset()
        duration = args.duration or preset.duration
        print("Using FULL paper preset (343 cells, 120 minutes)")

    print()
    print("Paper Parameters:")
    print(f"  Place cells: {config.num_place_cells}")
    print(f"  Sigma (σ): {config.sigma:.3f}")
    print(f"  Coactivity window: {config.coactivity_window*1000:.0f} ms")
    print(f"  Coactivity threshold: {config.coactivity_threshold}")
    print(f"  Max edge distance: {config.max_edge_distance:.3f}")
    print(f"  Integration window (ϖ): {config.integration_window/60:.1f} minutes")
    print(f"  Agent speed: {agent_params['base_speed']*100:.0f} cm/s")
    print(f"  Duration: {duration/60:.1f} minutes")
    print()

    # Run experiment
    results = run_paper_experiment(
        config=config,
        agent_params=agent_params,
        duration=duration,
        seed=args.seed,
        quick=args.quick,
    )

    # Estimate T_min
    t_min = estimate_t_min(results, target_betti_0=1, target_betti_1=0)

    # Print summary
    print()
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    if results["edges"]:
        print(f"Final edges: {results['edges'][-1]}")
    if results["components"]:
        print(f"Final components: {results['components'][-1]}")
    if results["betti_0"] and results["betti_0"][-1] >= 0:
        print(f"Final b₀: {results['betti_0'][-1]}")
    if results["betti_1"] and results["betti_1"][-1] >= 0:
        print(f"Final b₁: {results['betti_1'][-1]}")
    if t_min:
        print(f"T_min: {t_min/60:.1f} minutes")
        print(f"Paper T_min: ~28 minutes")
        if abs(t_min / 60 - 28) < 10:
            print("  ✓ T_min is within reasonable range of paper value")
        else:
            print(f"  ⚠ T_min differs from paper (difference: {abs(t_min/60 - 28):.1f} min)")
    else:
        print("T_min: Not reached (target topology not achieved)")

    # Create plot
    print()
    create_paper_replication_plot(results, t_min, preset, args.output)

    print()
    print("=" * 70)
    print("Paper replication complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

