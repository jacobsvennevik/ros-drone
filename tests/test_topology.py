import numpy as np
import pytest

from hippocampus_core.coactivity import CoactivityTracker
from hippocampus_core.env import Environment
from hippocampus_core.place_cells import PlaceCellPopulation
from hippocampus_core.topology import TopologicalGraph


@pytest.fixture(scope="module")
def topology_inputs():
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(777)
    num_cells = 9
    population = PlaceCellPopulation(
        environment=env,
        num_cells=num_cells,
        sigma=0.2,
        max_rate=18.0,
        rng=rng,
    )
    tracker = CoactivityTracker(num_cells=num_cells, window=0.08)

    dt = 0.02
    time = 0.0
    path = np.array(
        [
            [0.2, 0.2],
            [0.8, 0.2],
            [0.8, 0.8],
            [0.2, 0.8],
        ]
    )
    steps_per_segment = 50
    positions = np.vstack(
        [
            np.linspace(path[i], path[(i + 1) % len(path)], steps_per_segment, endpoint=False)
            for i in range(len(path))
        ]
    )

    for position in positions:
        x, y = position
        rates = population.get_rates(float(x), float(y))
        spikes = population.sample_spikes(rates, dt)
        time += dt
        tracker.register_spikes(time, spikes)

    coactivity_matrix = tracker.get_coactivity_matrix()
    return population.get_positions(), coactivity_matrix


def test_topological_graph_edges_increase_with_lower_threshold(topology_inputs):
    positions, coactivity = topology_inputs

    strict_graph = TopologicalGraph(positions)
    strict_graph.build_from_coactivity(coactivity, c_min=4.0, max_distance=1.5)

    lenient_graph = TopologicalGraph(positions)
    lenient_graph.build_from_coactivity(coactivity, c_min=1.0, max_distance=1.5)

    assert lenient_graph.num_edges() >= strict_graph.num_edges()
    assert lenient_graph.num_edges() > 0
    assert strict_graph.num_nodes() == lenient_graph.num_nodes() == positions.shape[0]
    assert strict_graph.num_components() >= 1
    assert lenient_graph.num_components() >= 1
    assert strict_graph.num_edges() < lenient_graph.num_edges()

