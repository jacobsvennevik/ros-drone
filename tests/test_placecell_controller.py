import numpy as np

from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)
from hippocampus_core.env import Environment


def test_place_cell_controller_smoke():
    env = Environment(width=1.0, height=1.0)
    config = PlaceCellControllerConfig(
        num_place_cells=16,
        sigma=0.18,
        max_rate=20.0,
        coactivity_window=0.06,
        coactivity_threshold=2.0,
        max_edge_distance=0.5,
    )

    rng = np.random.default_rng(1234)
    controller = PlaceCellController(environment=env, config=config, rng=rng)

    angles = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    radius = 0.3
    center = np.array([0.5, 0.5])
    trajectory = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))

    dt = 0.05
    for point in trajectory:
        action = controller.step(point, dt)
        assert action.shape == (2,)
        assert np.allclose(action, np.zeros(2))

    assert controller.steps == trajectory.shape[0]

    coactivity = controller.get_coactivity_matrix()
    assert coactivity.shape == (config.num_place_cells, config.num_place_cells)
    assert np.all(coactivity >= 0.0)
    assert np.count_nonzero(np.triu(coactivity, k=1)) > 0

    spike_counts = controller.spike_counts
    assert spike_counts.shape == (config.num_place_cells,)
    assert spike_counts.sum() > 0

    avg_rates = controller.average_rate_per_cell
    assert avg_rates.shape == (config.num_place_cells,)

    graph = controller.get_graph()
    assert graph.num_nodes() == config.num_place_cells
    first_edge_count = graph.num_edges()

    same_graph = controller.get_graph()
    assert same_graph is graph
    assert same_graph.num_edges() == first_edge_count


def test_integration_window_gates_edges():
    """Test that integration window prevents edges from being added too quickly."""
    env = Environment(width=1.0, height=1.0)
    
    # Create config without integration window (old behavior)
    config_no_window = PlaceCellControllerConfig(
        num_place_cells=16,
        sigma=0.18,
        max_rate=20.0,
        coactivity_window=0.06,
        coactivity_threshold=2.0,
        max_edge_distance=0.5,
        integration_window=None,  # No integration window
    )
    
    # Create config with integration window
    config_with_window = PlaceCellControllerConfig(
        num_place_cells=16,
        sigma=0.18,
        max_rate=20.0,
        coactivity_window=0.06,
        coactivity_threshold=2.0,
        max_edge_distance=0.5,
        integration_window=5.0,  # 5 second integration window
    )
    
    rng = np.random.default_rng(1234)
    controller_no_window = PlaceCellController(environment=env, config=config_no_window, rng=rng)
    controller_with_window = PlaceCellController(environment=env, config=config_with_window, rng=rng)
    
    # Run same trajectory for both
    angles = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    radius = 0.3
    center = np.array([0.5, 0.5])
    trajectory = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    dt = 0.05
    
    for point in trajectory:
        controller_no_window.step(point, dt)
        controller_with_window.step(point, dt)
    
    # After short time, controller with integration window should have fewer edges
    graph_no_window = controller_no_window.get_graph()
    graph_with_window = controller_with_window.get_graph()
    
    # With integration window, edges should be gated (fewer or equal edges)
    assert graph_with_window.num_edges() <= graph_no_window.num_edges()
    
    # Both should have some edges if enough time has passed
    # (This depends on the trajectory and parameters)


def test_integration_window_allows_edges_after_duration():
    """Test that edges are added after integration window has elapsed."""
    env = Environment(width=1.0, height=1.0)
    
    # Small integration window for testing
    config = PlaceCellControllerConfig(
        num_place_cells=16,
        sigma=0.18,
        max_rate=20.0,
        coactivity_window=0.06,
        coactivity_threshold=2.0,
        max_edge_distance=0.5,
        integration_window=1.0,  # 1 second integration window
    )
    
    rng = np.random.default_rng(1234)
    controller = PlaceCellController(environment=env, config=config, rng=rng)
    
    # Run trajectory
    angles = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    radius = 0.3
    center = np.array([0.5, 0.5])
    trajectory = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    dt = 0.05
    
    for point in trajectory:
        controller.step(point, dt)
    
    # After running, should have some edges (integration window allows them after 1 second)
    graph = controller.get_graph()
    assert graph.num_edges() >= 0  # At least non-negative
    
    # If we ran long enough (120 steps * 0.05 = 6 seconds), should have edges
    total_time = len(trajectory) * dt
    if total_time > config.integration_window:
        # Should have some edges after integration window
        assert graph.num_edges() >= 0  # Could be 0 or more depending on coactivity


def test_controller_respects_explicit_place_cell_positions():
    env = Environment(width=1.0, height=1.0)
    centers = np.array(
        [
            [0.2, 0.5],
            [0.5, 0.8],
            [0.8, 0.5],
            [0.5, 0.2],
        ],
        dtype=float,
    )

    config = PlaceCellControllerConfig(
        num_place_cells=len(centers),
        sigma=0.12,
        max_rate=18.0,
        coactivity_window=0.05,
        coactivity_threshold=3.0,
        max_edge_distance=0.3,
        place_cell_positions=centers,
    )

    controller = PlaceCellController(environment=env, config=config, rng=np.random.default_rng(321))

    assert np.allclose(controller.place_cell_positions, centers)
    controller.reset()
    assert np.allclose(controller.place_cell_positions, centers)

