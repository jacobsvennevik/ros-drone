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

