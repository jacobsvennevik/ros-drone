"""Run a simple random-walk simulation in the toy environment."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np

from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)
try:
    from hippocampus_core.controllers.snntorch_controller import (
        SnnTorchController,
        SnnTorchControllerConfig,
    )
except ImportError:
    SnnTorchController = None
    SnnTorchControllerConfig = None
from hippocampus_core.env import Agent, Environment
from hippocampus_core.visualization import plot_summary

# Simulation parameters (easy to tweak)
ARENA_WIDTH: float = 1.0
ARENA_HEIGHT: float = 1.0
DT: float = 0.01
NUM_STEPS: int = 10_000
SEED: int = 42
NUM_PLACE_CELLS: int = 120
PLACE_CELL_SIGMA: float = 0.1
PLACE_CELL_MAX_RATE: float = 15.0
COACTIVITY_WINDOW: float = 0.2
COACTIVITY_THRESHOLD: float = 5.0
MAX_EDGE_DISTANCE: float = 2.0 * PLACE_CELL_SIGMA
TOP_COACTIVE_PAIRS: int = 5
SHOW_PLOTS: bool = True
CONTROLLER_BACKEND: str = "place_cells"  # or "snntorch"
SNN_MODEL_PATH = PROJECT_ROOT / "models" / "snn_controller.pt"


def _init_snntorch_controller() -> SnnTorchController:
    if SnnTorchController is None or SnnTorchControllerConfig is None:
        raise RuntimeError(
            "snnTorch backend selected but snnTorch is not installed. "
            "Run 'python3 -m pip install torch snntorch' first."
        )

    if SNN_MODEL_PATH.exists():
        controller = SnnTorchController.from_checkpoint(SNN_MODEL_PATH, device="cpu")
        print(
            f"Loaded snnTorch checkpoint from '{SNN_MODEL_PATH}'. "
            f"time_steps={controller.config.time_steps}",
        )
        return controller

    print(
        "Warning: snnTorch checkpoint not found. "
        "Initialising controller with random weights for demonstration."
    )
    return SnnTorchController(
        config=SnnTorchControllerConfig(
            obs_dim=4,
            action_dim=2,
            hidden_dim=64,
            beta=0.9,
            time_steps=1,
            device="cpu",
        )
    )


def run_simulation(
    num_steps: int = NUM_STEPS,
    dt: float = DT,
    arena_size: Tuple[float, float] = (ARENA_WIDTH, ARENA_HEIGHT),
    seed: int | None = SEED,
) -> Tuple[np.ndarray, Environment, object, np.ndarray | None]:
    """Simulate an agent trajectory using the configured controller."""

    width, height = arena_size
    environment = Environment(width=width, height=height)
    agent_rng = np.random.default_rng(seed)
    agent = Agent(environment=environment, random_state=agent_rng)

    controller_rng = (
        np.random.default_rng(seed + 1) if seed is not None else np.random.default_rng()
    )
    actions: np.ndarray | None = None

    if CONTROLLER_BACKEND == "snntorch":
        if SnnTorchController is None:
            raise RuntimeError(
                "snnTorch backend selected but snnTorch is not installed. "
                "Run 'python3 -m pip install torch snntorch' first."
            )
        controller = _init_snntorch_controller()
        actions = np.zeros((num_steps, controller.config.action_dim), dtype=float)
    else:
        controller_config = PlaceCellControllerConfig(
            num_place_cells=NUM_PLACE_CELLS,
            sigma=PLACE_CELL_SIGMA,
            max_rate=PLACE_CELL_MAX_RATE,
            coactivity_window=COACTIVITY_WINDOW,
            coactivity_threshold=COACTIVITY_THRESHOLD,
            max_edge_distance=MAX_EDGE_DISTANCE,
            # TODO: Try enabling integration window to see edge gating effect
            integration_window=2.0,  # 2 second integration window (paper's ϖ parameter)
            #integration_window=None,  # Set to None to disable, or a float (seconds) to enable
        )
        controller = PlaceCellController(
            environment=environment,
            config=controller_config,
            rng=controller_rng,
        )
        actions = np.zeros((num_steps, 2), dtype=float)

    trajectory = np.zeros((num_steps, 2), dtype=float)

    for step_idx in range(num_steps):
        position = agent.step(dt)
        trajectory[step_idx] = position
        velocity = agent.velocity
        heading = float(np.arctan2(velocity[1], velocity[0]))
        observation = np.array(
            [position[0], position[1], np.cos(heading), np.sin(heading)],
            dtype=np.float32,
        )
        action = controller.step(observation, dt)
        if actions is not None:
            actions[step_idx] = action

    return trajectory, environment, controller, actions


def _format_top_coactive_pairs(coactivity_matrix: np.ndarray, top_n: int) -> str:
    """Return a formatted string describing the top-N coactive cell pairs."""

    if top_n <= 0:
        return "(no pairs requested)"

    upper = np.triu(coactivity_matrix, k=1)
    positive_mask = upper > 0
    if not np.any(positive_mask):
        return "No coactive cell pairs observed."

    counts = upper[positive_mask]
    pair_indices = np.argwhere(positive_mask)
    top_n = min(top_n, counts.size)
    order = np.argsort(counts)[-top_n:][::-1]

    lines = []
    for idx in order:
        i, j = pair_indices[idx]
        lines.append(f"  Cells ({i}, {j}) -> {int(counts[idx])} coactivity events")
    return "\n".join(lines)


def _summarize_cycles(cycles: list[list[int]]) -> str:
    """Summarize simple cycle statistics for display."""

    if not cycles:
        return "No simple cycles detected."

    lengths = np.array([len(cycle) for cycle in cycles], dtype=int)
    median_length = float(np.median(lengths))
    return (
        "Simple cycles: "
        f"{len(cycles)} (length min/median/max = {lengths.min()} / {median_length:.1f} / {lengths.max()})"
    )


def main() -> None:
    trajectory, environment, controller, actions = run_simulation()
    start = trajectory[0]
    end = trajectory[-1]
    displacement = np.linalg.norm(end - start)

    if isinstance(controller, PlaceCellController):
        avg_rate_per_cell = controller.average_rate_per_cell
        overall_mean_rate = controller.overall_mean_rate
        spike_counts = controller.spike_counts
        coactivity_matrix = controller.get_coactivity_matrix()
        positions = controller.place_cell_positions
        graph = controller.get_graph()
    else:
        avg_rate_per_cell = None
        overall_mean_rate = None
        spike_counts = None
        coactivity_matrix = None
        positions = None
        graph = None

    print("Simulation complete")
    print(f"Steps: {NUM_STEPS}")
    print(f"Time step dt: {DT}")
    print(f"Start position: ({start[0]:.3f}, {start[1]:.3f})")
    print(f"End position:   ({end[0]:.3f}, {end[1]:.3f})")
    print(f"Net displacement: {displacement:.3f}")

    if isinstance(controller, PlaceCellController):
        print("\nPlace-cell firing statistics")
        print(f"Number of cells: {NUM_PLACE_CELLS}")
        print(f"Overall mean firing rate: {overall_mean_rate:.3f} Hz")
        print(
            f"Per-cell mean rate min/max: {avg_rate_per_cell.min():.3f} Hz / "
            f"{avg_rate_per_cell.max():.3f} Hz"
        )
        steps = controller.steps if controller.steps > 0 else NUM_STEPS
        mean_spike_rate = spike_counts.mean() / (steps * DT)
        print(f"Average spike rate across cells: {mean_spike_rate:.3f} Hz")

        print("\nPlace-cell coactivity summary")
        upper = np.triu(coactivity_matrix, k=1)
        num_coactive_pairs = int(np.count_nonzero(upper))
        print(f"Window size w: {controller.config.coactivity_window} s")
        print(f"Number of coactive pairs: {num_coactive_pairs}")
        print(_format_top_coactive_pairs(coactivity_matrix, TOP_COACTIVE_PAIRS))

        print("\nTopological graph summary")
        print(f"Coactivity threshold C_min: {controller.config.coactivity_threshold}")
        print(f"Max edge distance: {controller.config.max_edge_distance}")
        if controller.config.integration_window is not None:
            print(f"Integration window: {controller.config.integration_window} s")
        print(f"Nodes (place cells): {graph.num_nodes()}")
        print(f"Edges (eligible pairs): {graph.num_edges()}")
        
        # Integration window statistics
        if controller.config.integration_window is not None:
            coactivity_matrix = controller.get_coactivity_matrix()
            threshold = controller.config.coactivity_threshold
            pairs_exceeding = np.sum(np.triu(coactivity_matrix, k=1) >= threshold)
            integration_times = controller.coactivity.check_threshold_exceeded(
                threshold=threshold,
                current_time=controller.current_time,
            )
            pairs_passing_window = len(integration_times)
            if pairs_passing_window > 0:
                elapsed_times = [
                    controller.current_time - t
                    for t in integration_times.values()
                    if controller.current_time - t >= controller.config.integration_window
                ]
                avg_time_to_admission = np.mean(elapsed_times) if elapsed_times else 0.0
                print(f"  - Pairs exceeding threshold: {pairs_exceeding}")
                print(f"  - Pairs passing integration window: {pairs_passing_window}")
                if elapsed_times:
                    print(f"  - Average time to admission: {avg_time_to_admission:.1f} s")
        
        # Degree statistics
        degree_stats = graph.get_degree_statistics()
        print(f"Average node degree: {degree_stats['mean']:.1f} (min: {degree_stats['min']}, max: {degree_stats['max']})")
        
        # Edge length statistics
        if graph.num_edges() > 0:
            length_stats = graph.get_edge_length_statistics()
            max_dist = controller.config.max_edge_distance
            all_below = length_stats['max'] <= max_dist
            checkmark = "✓" if all_below else "⚠"
            print(
                f"Edge lengths: min={length_stats['min']:.3f}, "
                f"mean={length_stats['mean']:.3f}, "
                f"max={length_stats['max']:.3f} "
                f"(all below {max_dist:.3f} {checkmark})"
            )
        
        print(f"Connected components: {graph.num_components()}")
        cycles = list(graph.cycle_basis())
        print(_summarize_cycles(cycles))

        if SHOW_PLOTS:
            # Show diagnostic plots if integration window is enabled
            show_diagnostics = controller.config.integration_window is not None
            fig, _ = plot_summary(
                environment,
                trajectory,
                positions,
                controller.config.sigma,
                graph,
                max_edge_distance=controller.config.max_edge_distance,
                show_diagnostics=show_diagnostics,
            )
            fig.tight_layout()
            try:
                import matplotlib.pyplot as plt
            except ImportError:  # pragma: no cover - optional dependency already guarded in visualization
                print(
                    "matplotlib is required for plotting. Install with 'python3 -m pip install matplotlib'."
                )
            else:
                plt.show()
    elif CONTROLLER_BACKEND == "snntorch" and actions is not None:
        print("\nsnnTorch controller diagnostics")
        action_min = actions.min(axis=0)
        action_max = actions.max(axis=0)
        action_mean = actions.mean(axis=0)
        print(f"Action mean: {action_mean}")
        print(f"Action min:  {action_min}")
        print(f"Action max:  {action_max}")
        print(f"Final action: {actions[-1]}")
    else:
        print("\nsnnTorch controller diagnostics")
        print("No actions recorded.")


def sanity_check_place_cell_controller() -> None:
    """Run a short place-cell simulation and assert basic invariants."""

    num_steps = 1_000
    dt = 0.01
    environment = Environment()
    agent_rng = np.random.default_rng(0)
    agent = Agent(environment=environment, random_state=agent_rng)

    # Test without integration window (backward compatibility)
    controller = PlaceCellController(
        environment=environment,
        config=PlaceCellControllerConfig(
            num_place_cells=50,
            sigma=PLACE_CELL_SIGMA,
            max_rate=PLACE_CELL_MAX_RATE,
            coactivity_window=COACTIVITY_WINDOW,
            coactivity_threshold=COACTIVITY_THRESHOLD,
            max_edge_distance=MAX_EDGE_DISTANCE,
            integration_window=None,  # No integration window
        ),
        rng=np.random.default_rng(1),
    )

    for _ in range(num_steps):
        position = agent.step(dt)
        controller.step(np.asarray(position), dt)

    coactivity = controller.get_coactivity_matrix()
    num_cells = controller.place_cell_positions.shape[0]
    assert coactivity.shape == (num_cells, num_cells), "Unexpected coactivity shape."
    assert np.allclose(coactivity, coactivity.T), "Coactivity matrix is not symmetric."

    upper = np.triu(coactivity, k=1)
    num_nonzero = int(np.count_nonzero(upper))
    max_pairs = num_cells * (num_cells - 1) // 2
    assert num_nonzero > 0, "No coactive pairs detected in sanity check."
    assert num_nonzero < max_pairs, "All cell pairs marked coactive in sanity check."

    graph = controller.get_graph()
    assert graph.num_nodes() == num_cells, "Graph node count mismatch."
    assert graph.num_edges() >= 0, "Graph edge count should be non-negative."
    edges_without_window = graph.num_edges()

    print("Place-cell sanity check (without integration window) passed.")

    # Test with integration window
    controller_with_window = PlaceCellController(
        environment=environment,
        config=PlaceCellControllerConfig(
            num_place_cells=50,
            sigma=PLACE_CELL_SIGMA,
            max_rate=PLACE_CELL_MAX_RATE,
            coactivity_window=COACTIVITY_WINDOW,
            coactivity_threshold=COACTIVITY_THRESHOLD,
            max_edge_distance=MAX_EDGE_DISTANCE,
            integration_window=2.0,  # 2 second integration window
        ),
        rng=np.random.default_rng(1),
    )

    # Reset agent for fair comparison
    agent_rng2 = np.random.default_rng(0)
    agent2 = Agent(environment=environment, random_state=agent_rng2)

    for _ in range(num_steps):
        position = agent2.step(dt)
        controller_with_window.step(np.asarray(position), dt)

    graph_with_window = controller_with_window.get_graph()
    assert graph_with_window.num_nodes() == num_cells, "Graph node count mismatch (with window)."
    assert graph_with_window.num_edges() >= 0, "Graph edge count should be non-negative (with window)."
    
    # With integration window, should have fewer or equal edges (gating effect)
    edges_with_window = graph_with_window.num_edges()
    assert edges_with_window <= edges_without_window, (
        f"Integration window should gate edges: {edges_with_window} <= {edges_without_window}"
    )

    print(f"Place-cell sanity check (with integration window) passed.")
    print(f"  Edges without window: {edges_without_window}")
    print(f"  Edges with window: {edges_with_window}")
    
    # Test Betti number computation if available
    try:
        from hippocampus_core.persistent_homology import is_persistent_homology_available
        
        if is_persistent_homology_available():
            betti = graph_with_window.compute_betti_numbers(max_dim=2)
            print(f"  Betti numbers: b_0={betti[0]}, b_1={betti[1]}, b_2={betti.get(2, 0)}")
            assert betti[0] >= 1, "b_0 should be at least 1 (at least one component)"
            assert betti[0] == graph_with_window.num_components(), (
                f"b_0 should equal number of components: {betti[0]} == {graph_with_window.num_components()}"
            )
            print("  ✓ Betti number computation validated")
        else:
            print("  (Betti number computation skipped: ripser/gudhi not available)")
    except ImportError:
        print("  (Betti number computation skipped: persistent_homology module not available)")


def sanity_check_snntorch_controller() -> None:
    """Run a snnTorch controller check if dependencies are available."""

    if SnnTorchController is None:
        print("snnTorch not installed; skipping snnTorch controller sanity check.")
        return

    controller = SnnTorchController(
        config=SnnTorchControllerConfig(
            obs_dim=2,
            action_dim=2,
            hidden_dim=16,
            beta=0.9,
            device="cpu",
        )
    )
    controller.reset()

    num_steps = 500
    dt = 0.01
    actions = np.zeros((num_steps, controller.config.action_dim), dtype=float)
    obs = np.zeros(controller.config.obs_dim, dtype=float)

    for step in range(num_steps):
        obs[:] = np.sin(2 * np.pi * step * dt)  # vary input a bit
        actions[step] = controller.step(obs, dt)

    assert actions.shape == (num_steps, controller.config.action_dim)
    assert np.all(np.isfinite(actions)), "Non-finite values produced by snnTorch controller."

    spread = np.abs(actions).sum(axis=1)
    assert np.any(spread > 0), "snnTorch controller actions are identically zero."

    print("snnTorch controller sanity check passed.")


if __name__ == "__main__":
    try:
        main()
    finally:
        sanity_check_place_cell_controller()
        sanity_check_snntorch_controller()
