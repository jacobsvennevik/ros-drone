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

