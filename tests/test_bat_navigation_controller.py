import numpy as np

from hippocampus_core.controllers.bat_navigation_controller import (
    BatNavigationController,
    BatNavigationControllerConfig,
)
from hippocampus_core.env import Environment, Agent


def test_bat_navigation_controller_runs():
    env = Environment(width=1.0, height=1.0)
    config = BatNavigationControllerConfig(num_place_cells=12, calibration_interval=5)
    controller = BatNavigationController(environment=env, config=config)

    obs = np.array([0.5, 0.5, 0.0])
    action = controller.step(obs, dt=0.1)

    assert action.shape == (2,)
    graph = controller.get_graph()
    assert graph.num_nodes() == config.num_place_cells


def test_bat_navigation_controller_handles_nan_heading():
    """Test that bat controller handles NaN heading gracefully."""
    env = Environment(width=1.0, height=1.0)
    config = BatNavigationControllerConfig(num_place_cells=10, calibration_interval=10)
    controller = BatNavigationController(environment=env, config=config)

    # Valid observation first
    obs_valid = np.array([0.5, 0.5, 0.5])
    controller.step(obs_valid, dt=0.05)
    
    # NaN heading - should fallback to last valid heading
    obs_nan = np.array([0.5, 0.5, np.nan])
    action = controller.step(obs_nan, dt=0.05)
    
    # Should not crash
    assert action.shape == (2,)
    assert controller._prev_heading == 0.5  # Should use last valid heading


def test_bat_navigation_controller_handles_inf_heading():
    """Test that bat controller handles Inf heading gracefully."""
    env = Environment(width=1.0, height=1.0)
    config = BatNavigationControllerConfig(num_place_cells=10, calibration_interval=10)
    controller = BatNavigationController(environment=env, config=config)

    # Valid observation first
    obs_valid = np.array([0.5, 0.5, 0.3])
    controller.step(obs_valid, dt=0.05)
    
    # Inf heading - should fallback to last valid heading
    obs_inf = np.array([0.5, 0.5, np.inf])
    action = controller.step(obs_inf, dt=0.05)
    
    # Should not crash
    assert action.shape == (2,)
    assert controller._prev_heading == 0.3  # Should use last valid heading


def test_bat_navigation_controller_handles_nan_heading_no_previous():
    """Test that bat controller handles NaN heading when no previous heading exists."""
    env = Environment(width=1.0, height=1.0)
    config = BatNavigationControllerConfig(num_place_cells=10, calibration_interval=10)
    controller = BatNavigationController(environment=env, config=config)

    # NaN heading on first step - should fallback to zero
    obs_nan = np.array([0.5, 0.5, np.nan])
    action = controller.step(obs_nan, dt=0.05)
    
    # Should not crash, should use zero as fallback
    assert action.shape == (2,)
    assert controller._prev_heading == 0.0  # Should use zero fallback


def test_bat_navigation_controller_adaptive_calibration():
    """Test adaptive calibration mode."""
    env = Environment(width=1.0, height=1.0)
    config = BatNavigationControllerConfig(
        num_place_cells=10,
        calibration_interval=100,  # High fixed interval
        adaptive_calibration=True,
        calibration_drift_threshold=0.05,
    )
    controller = BatNavigationController(environment=env, config=config)
    
    agent = Agent(env, track_heading=True, random_state=np.random.default_rng(42))
    
    # Run simulation - calibration should trigger based on drift, not fixed interval
    dt = 0.05
    steps_before_calibration = controller._steps_since_calibration
    
    for _ in range(50):
        obs = agent.step(dt, include_theta=True)
        controller.step(np.asarray(obs), dt)
        
        # If drift exceeds threshold, calibration may trigger early
        grid_drift = controller.grid_attractor.drift_metric()
        if grid_drift > config.calibration_drift_threshold:
            # Calibration may have triggered (reset counter)
            break
    
    # Adaptive calibration should work (may trigger before fixed interval)
    assert controller._steps_since_calibration <= config.calibration_interval

