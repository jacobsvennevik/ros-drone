import numpy as np

from hippocampus_core.calibration.phase_optimizer import PhaseOptimizer


def test_phase_optimizer_needs_samples():
    optimizer = PhaseOptimizer(max_history=5)
    assert optimizer.estimate_correction() is None


def test_phase_optimizer_returns_mean_offsets():
    optimizer = PhaseOptimizer(max_history=10)
    for _ in range(12):
        position = np.array([1.0, 1.0])
        grid = np.array([0.8, 0.9])
        optimizer.add_sample(position, heading=0.5, hd_estimate=0.3, grid_estimate=grid)

    correction = optimizer.estimate_correction()
    assert correction is not None
    assert np.allclose(correction.grid_translation, np.array([0.2, 0.1]))
    assert abs(correction.heading_delta - 0.2) < 1e-6

