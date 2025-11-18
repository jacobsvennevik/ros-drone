import numpy as np
import pytest

from hippocampus_core.coactivity import CoactivityTracker
from hippocampus_core.env import Environment
from hippocampus_core.place_cells import PlaceCellPopulation


@pytest.fixture(scope="module")
def generated_coactivity_matrix():
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(2024)
    num_cells = 8
    population = PlaceCellPopulation(
        environment=env,
        num_cells=num_cells,
        sigma=0.18,
        max_rate=20.0,
        rng=rng,
    )
    tracker = CoactivityTracker(num_cells=num_cells, window=0.05)

    dt = 0.01
    time = 0.0

    angles = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    radius = 0.28
    center = np.array([0.5, 0.5])
    positions = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))

    for position in positions:
        x, y = position
        rates = population.get_rates(float(x), float(y))
        spikes = population.sample_spikes(rates, dt)
        time += dt
        tracker.register_spikes(time, spikes)

    return tracker.get_coactivity_matrix()


def test_coactivity_matrix_symmetry(generated_coactivity_matrix):
    matrix = generated_coactivity_matrix
    assert matrix.shape[0] == matrix.shape[1]
    assert np.allclose(matrix, matrix.T)


def test_coactivity_matrix_populates_partial_pairs(generated_coactivity_matrix):
    matrix = generated_coactivity_matrix
    num_cells = matrix.shape[0]
    total_pairs = num_cells * (num_cells - 1) // 2
    active_pairs = np.count_nonzero(np.triu(matrix, k=1))
    assert 0 < active_pairs < total_pairs


def test_integration_window_tracking():
    """Test that CoactivityTracker tracks when pairs first exceed threshold."""
    tracker = CoactivityTracker(num_cells=3, window=0.1)
    
    # Register some spikes to build up coactivity
    spikes_0 = np.array([True, False, False])
    spikes_1 = np.array([False, True, True])
    
    time = 0.0
    # Make cells 0 and 1 coactive
    tracker.register_spikes(time, spikes_0)
    tracker.register_spikes(time + 0.05, spikes_1)  # Within window
    
    # Check threshold exceeded (low threshold to ensure it triggers)
    threshold = 1.0
    times = tracker.check_threshold_exceeded(threshold, time + 0.1)
    
    # Pair (0, 1) should have exceeded threshold
    assert (0, 1) in times
    assert times[(0, 1)] == time + 0.1  # Time when check was called
    
    # Continue to exceed threshold, time should not change
    time += 0.2
    tracker.register_spikes(time, spikes_0)
    times2 = tracker.check_threshold_exceeded(threshold, time)
    assert times2[(0, 1)] == times[(0, 1)]  # Should still be original time
    
    # Reset should clear integration times
    tracker.reset()
    times3 = tracker.check_threshold_exceeded(threshold, time + 1.0)
    assert (0, 1) not in times3

