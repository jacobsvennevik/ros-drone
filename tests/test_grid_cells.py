import numpy as np

from hippocampus_core.grid_cells import GridAttractor, GridAttractorConfig


def test_grid_attractor_step_moves_state():
    attractor = GridAttractor(GridAttractorConfig(size=(5, 5), tau=0.1))
    initial_state = attractor.state.copy()
    attractor.step(np.array([0.1, 0.0]), dt=0.05)

    assert not np.allclose(attractor.state, initial_state)


def test_grid_attractor_estimate_position_bounds():
    attractor = GridAttractor(GridAttractorConfig(size=(4, 4)))
    attractor.state[2, 3] = 10.0
    estimate = attractor.estimate_position()

    assert estimate.shape == (2,)
    assert 0 <= estimate[0] <= 3
    assert 0 <= estimate[1] <= 3


def test_grid_attractor_drift_metric_non_negative():
    attractor = GridAttractor(GridAttractorConfig(size=(4, 4)))
    metric = attractor.drift_metric()
    assert metric >= 0.0

