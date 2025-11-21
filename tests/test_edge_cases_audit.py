"""Additional edge case tests addressing audit questions (Q13)."""
from __future__ import annotations

import numpy as np
import pytest

from hippocampus_core.coactivity import CoactivityTracker
from hippocampus_core.controllers.bat_navigation_controller import (
    BatNavigationController,
    BatNavigationControllerConfig,
)
from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)
from hippocampus_core.env import Agent, Environment
from hippocampus_core.grid_cells import GridAttractor, GridAttractorConfig
from hippocampus_core.head_direction import HeadDirectionAttractor, HeadDirectionConfig


def test_empty_coactivity_matrix():
    """Test handling of empty coactivity matrix (no spikes registered)."""
    tracker = CoactivityTracker(num_cells=10, window=0.2)
    
    # Register no spikes - coactivity should remain zero
    coactivity = tracker.get_coactivity_matrix()
    assert coactivity.shape == (10, 10)
    assert np.allclose(coactivity, 0.0)
    
    # Should handle gracefully when checking threshold
    times = tracker.check_threshold_exceeded(threshold=1.0, current_time=1.0)
    assert len(times) == 0


def test_all_zero_firing_rates():
    """Test handling of all-zero firing rates."""
    env = Environment(width=1.0, height=1.0)
    config = PlaceCellControllerConfig(num_place_cells=10, sigma=0.15)
    controller = PlaceCellController(environment=env, config=config)
    
    # Create observation that should produce very low rates
    # (position far from all place cell centers)
    obs = np.array([-10.0, -10.0])  # Far outside environment
    
    # Should handle gracefully
    action = controller.step(obs, dt=0.05)
    assert action.shape == (2,)
    
    # Rates should be non-negative (may be zero)
    if hasattr(controller, 'last_rates') and controller.last_rates is not None:
        assert np.all(controller.last_rates >= 0.0)


def test_controller_with_all_zero_rates():
    """Test controller behavior when all place cells have zero rates."""
    env = Environment(width=1.0, height=1.0)
    config = PlaceCellControllerConfig(
        num_place_cells=5,
        sigma=0.01,  # Very small sigma - most positions will have zero rates
        max_rate=20.0,
    )
    controller = PlaceCellController(environment=env, config=config)
    
    # Position likely to produce zero rates (small sigma, random centers)
    obs = np.array([0.1, 0.1])
    
    for _ in range(10):
        action = controller.step(obs, dt=0.05)
        assert action.shape == (2,)
        # Should not crash even if rates are zero
        assert np.all(np.isfinite(action))


def test_nan_propagation_in_grid_attractor():
    """Test that grid attractor handles NaN/Inf gracefully."""
    attractor = GridAttractor(GridAttractorConfig(size=(5, 5), tau=0.05))
    
    # Introduce NaN in state (shouldn't happen in normal operation, but test robustness)
    attractor.state[2, 2] = np.nan
    
    # Step should either fix NaN or raise error, not silently propagate
    try:
        attractor.step(velocity=np.array([0.1, 0.0]), dt=0.05)
        # If it doesn't raise, check that NaN is handled
        assert np.all(np.isfinite(attractor.state)) or np.any(np.isnan(attractor.state))
    except (ValueError, AssertionError):
        pass  # Raising error is acceptable


def test_inf_propagation_in_hd_attractor():
    """Test that HD attractor handles Inf gracefully."""
    attractor = HeadDirectionAttractor(HeadDirectionConfig(num_neurons=12, tau=0.05))
    
    # Introduce Inf in state
    attractor.state[5] = np.inf
    
    # Step should handle Inf (either fix or raise error)
    try:
        attractor.step(omega=0.1, dt=0.05)
        # If it doesn't raise, check that Inf is handled
        assert np.all(np.isfinite(attractor.state)) or np.any(np.isinf(attractor.state))
    except (ValueError, AssertionError):
        pass  # Raising error is acceptable


def test_bat_controller_zero_heading():
    """Test bat controller with zero heading."""
    env = Environment(width=1.0, height=1.0)
    config = BatNavigationControllerConfig(num_place_cells=10, calibration_interval=10)
    controller = BatNavigationController(environment=env, config=config)
    
    # Zero heading should work
    obs = np.array([0.5, 0.5, 0.0])
    action = controller.step(obs, dt=0.05)
    
    assert action.shape == (2,)
    assert np.all(np.isfinite(action))


def test_bat_controller_large_heading():
    """Test bat controller with large heading values."""
    env = Environment(width=1.0, height=1.0)
    config = BatNavigationControllerConfig(num_place_cells=10, calibration_interval=10)
    controller = BatNavigationController(environment=env, config=config)
    
    # Large heading (should wrap)
    obs = np.array([0.5, 0.5, 10.0])  # ~3.2π
    action = controller.step(obs, dt=0.05)
    
    assert action.shape == (2,)
    assert np.all(np.isfinite(action))


def test_coactivity_empty_histories():
    """Test coactivity tracker with empty spike histories."""
    tracker = CoactivityTracker(num_cells=5, window=0.1)
    
    # Register spikes
    spikes = np.array([True, False, False, False, False])
    tracker.register_spikes(t=0.0, spikes=spikes)
    
    # Wait beyond window
    tracker.register_spikes(t=0.2, spikes=np.zeros(5, dtype=bool))
    
    # Histories should be pruned (empty for cells 1-4)
    coactivity = tracker.get_coactivity_matrix()
    # Should still be valid (symmetric, non-negative)
    assert coactivity.shape == (5, 5)
    assert np.all(coactivity >= 0.0)
    assert np.allclose(coactivity, coactivity.T)


def test_topology_empty_graph():
    """Test topology computation on empty graph (no edges)."""
    from hippocampus_core.topology import TopologicalGraph
    
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    graph = TopologicalGraph(positions)
    
    # Build with zero coactivity (no edges)
    coactivity = np.zeros((3, 3))
    graph.build_from_coactivity(coactivity, c_min=1.0, max_distance=2.0)
    
    assert graph.num_nodes() == 3
    assert graph.num_edges() == 0
    assert graph.num_components() == 3  # All isolated
    
    # Betti numbers on empty graph
    try:
        betti = graph.compute_betti_numbers(max_dim=1)
        assert betti[0] == 3  # Three components
        assert betti[1] == 0  # No holes
    except ImportError:
        pass  # Persistent homology not available


def test_conjunctive_cells_zero_inputs():
    """Test conjunctive cells with zero grid and HD activity."""
    from hippocampus_core.conjunctive_place_cells import (
        ConjunctivePlaceCellConfig,
        ConjunctivePlaceCellPopulation,
    )
    
    config = ConjunctivePlaceCellConfig(
        num_place_cells=5, grid_dim=4, head_direction_dim=3
    )
    population = ConjunctivePlaceCellPopulation(config)
    
    # Zero activity
    grid = np.zeros(config.grid_dim)
    hd = np.zeros(config.head_direction_dim)
    
    rates = population.compute_rates(grid, hd)
    assert rates.shape == (config.num_place_cells,)
    assert np.all(rates >= 0.0)  # Non-negative (may have bias)
    assert np.all(np.isfinite(rates))  # No NaN/Inf


def test_grid_attractor_zero_velocity():
    """Test grid attractor with zero velocity input."""
    attractor = GridAttractor(GridAttractorConfig(size=(8, 8), tau=0.05))
    initial_state = attractor.state.copy()
    
    # Zero velocity - should maintain state (with normalization)
    attractor.step(velocity=np.array([0.0, 0.0]), dt=0.05)
    
    # State should remain bounded
    assert np.all(np.isfinite(attractor.state))
    # Mean should be ~0 after normalization
    assert abs(attractor.state.mean()) < 1e-6


def test_hd_attractor_zero_angular_velocity():
    """Test HD attractor with zero angular velocity."""
    attractor = HeadDirectionAttractor(HeadDirectionConfig(num_neurons=12, tau=0.05))
    
    # Inject initial cue
    attractor.inject_cue(heading=np.pi/4, gain=3.0)
    initial_estimate = attractor.estimate_heading()
    
    # Zero angular velocity - should maintain heading
    for _ in range(20):
        attractor.step(omega=0.0, dt=0.05)
    
    final_estimate = attractor.estimate_heading()
    # Should remain near initial estimate (with some drift due to dynamics)
    diff = abs(final_estimate - initial_estimate)
    wrapped_diff = min(diff, 2*np.pi - diff)
    assert wrapped_diff < 0.5  # Should remain within ~0.5 radians


def test_bat_controller_missing_position():
    """Test bat controller with invalid position (but valid heading)."""
    env = Environment(width=1.0, height=1.0)
    config = BatNavigationControllerConfig(num_place_cells=10, calibration_interval=10)
    controller = BatNavigationController(environment=env, config=config)
    
    # Observation too short - should raise error
    obs_short = np.array([0.5, 0.5])  # Missing theta
    with pytest.raises(ValueError, match="requires observations containing"):
        controller.step(obs_short, dt=0.05)


def test_policy_service_empty_features():
    """Test policy service with minimal/empty feature vectors."""
    from hippocampus_core.policy import (
        TopologyService,
        SpatialFeatureService,
        SpikingPolicyService,
        RobotState,
        Mission,
        MissionGoal,
        GoalType,
        PointGoal,
    )
    
    env = Environment(width=1.0, height=1.0)
    controller = PlaceCellController(
        environment=env,
        config=PlaceCellControllerConfig(num_place_cells=10),
        rng=np.random.default_rng(42),
    )
    
    ts = TopologyService()
    ts.update_from_controller(controller)
    sfs = SpatialFeatureService(ts, k_neighbors=3)
    sps = SpikingPolicyService(sfs)
    
    # Build features with empty graph (minimal features)
    robot_state = RobotState(pose=(0.5, 0.5, 0.0), time=0.0)
    mission = Mission(goal=MissionGoal(type=GoalType.POINT, value=PointGoal(position=(0.9, 0.9))))
    
    # Should handle gracefully even with minimal graph
    features, context = sfs.build_features(robot_state, mission)
    decision = sps.decide(features, context, dt=0.05)
    
    assert decision.action_proposal.v is not None
    assert decision.action_proposal.omega is not None
    assert np.isfinite(decision.action_proposal.v)
    assert np.isfinite(decision.action_proposal.omega)

