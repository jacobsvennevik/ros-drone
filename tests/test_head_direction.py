import numpy as np

from hippocampus_core.head_direction import HeadDirectionAttractor, HeadDirectionConfig


def test_head_direction_step_advances_state():
    config = HeadDirectionConfig(num_neurons=8, tau=0.1, gamma=0.5)
    attractor = HeadDirectionAttractor(config=config)

    initial_state = attractor.state.copy()
    attractor.step(omega=0.2, dt=0.05)

    assert not np.allclose(attractor.state, initial_state)


def test_head_direction_heading_estimate_tracks_cue():
    attractor = HeadDirectionAttractor(HeadDirectionConfig(num_neurons=12))
    target_angle = np.pi / 4
    attractor.inject_cue(target_angle, gain=5.0)
    estimate = attractor.estimate_heading()

    assert np.isclose(estimate, target_angle, atol=0.2)


def test_head_direction_normalization_maintains_stability():
    """Test that normalization (mean subtraction) maintains HD attractor stability."""
    attractor = HeadDirectionAttractor(HeadDirectionConfig(num_neurons=60, tau=0.05))
    
    # Run multiple steps
    for _ in range(100):
        attractor.step(omega=0.1, dt=0.05)
        # After normalization, mean should be ~0
        assert abs(attractor.state.mean()) < 1e-6  # Mean should be approximately zero
        # State should remain bounded
        assert np.linalg.norm(attractor.state) < 100.0  # Reasonable bound
    
    # Final state should have mean ~0
    assert abs(attractor.state.mean()) < 1e-6


def test_head_direction_circular_continuity():
    """Test that HD estimation handles circular continuity correctly (0/2π boundary)."""
    attractor = HeadDirectionAttractor(HeadDirectionConfig(num_neurons=60))
    
    # Create bump near 0° (should also work near 2π)
    # Preferred angles are [0, 2π) with 60 neurons
    # Neuron 0 prefers 0°, neuron 59 prefers almost 2π
    
    # Inject cue at 0° (neuron 0)
    attractor.inject_cue(0.0, gain=5.0)
    estimate_0 = attractor.estimate_heading()
    assert abs(estimate_0) < 0.3 or abs(estimate_0 - 2*np.pi) < 0.3  # Should be near 0 or 2π
    
    # Inject cue near 2π
    attractor.inject_cue(2*np.pi - 0.1, gain=5.0)
    estimate_near_2pi = attractor.estimate_heading()
    # Should wrap correctly
    assert abs(estimate_near_2pi - (2*np.pi - 0.1)) < 0.3 or abs(estimate_near_2pi + 0.1) < 0.3

