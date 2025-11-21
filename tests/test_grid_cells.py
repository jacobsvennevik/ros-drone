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


def test_grid_attractor_normalization_maintains_stability():
    """Test that normalization (mean subtraction) maintains attractor stability."""
    attractor = GridAttractor(GridAttractorConfig(size=(10, 10), tau=0.05))
    
    # Run multiple steps
    initial_mean = attractor.state.mean()
    initial_norm = np.linalg.norm(attractor.state)
    
    for _ in range(100):
        attractor.step(velocity=np.array([0.1, 0.05]), dt=0.05)
        # After normalization, mean should be ~0
        assert abs(attractor.state.mean()) < 1e-6  # Mean should be approximately zero
        # State should remain bounded (not diverging)
        assert np.linalg.norm(attractor.state) < 100.0  # Reasonable bound
    
    # Final state should have mean ~0
    final_mean = attractor.state.mean()
    assert abs(final_mean) < 1e-6


def test_grid_attractor_normalization_preserves_bump():
    """Test that normalization preserves activity bump structure."""
    attractor = GridAttractor(GridAttractorConfig(size=(8, 8), tau=0.1))
    
    # Create a localized bump
    attractor.state[4, 4] = 5.0
    attractor.state[3, 4] = 3.0
    attractor.state[4, 3] = 3.0
    attractor.state[5, 4] = 3.0
    attractor.state[4, 5] = 3.0
    
    initial_peak = attractor.state[4, 4]
    initial_pattern = attractor.state.copy()
    
    # Step should preserve bump structure
    attractor.step(velocity=np.array([0.0, 0.0]), dt=0.05)
    
    # Bump should still be present (normalization subtracts mean, but relative structure preserved)
    assert np.max(attractor.state) > 0.0  # Should still have activity
    # Peak location should remain (or shift slightly due to dynamics)
    max_idx = np.unravel_index(np.argmax(attractor.state), attractor.state.shape)
    # Peak should be near original location (within 1-2 cells)
    assert abs(max_idx[0] - 4) <= 2
    assert abs(max_idx[1] - 4) <= 2

