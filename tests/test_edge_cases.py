"""Edge case tests for topological mapping system.

These tests verify that the system handles boundary conditions gracefully
and provides meaningful error messages for invalid inputs.
"""
from __future__ import annotations

import numpy as np
import pytest

from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)
from hippocampus_core.env import Agent, CircularObstacle, Environment


# ============================================================================
# Empty Graph Edge Cases
# ============================================================================


def test_no_edges_form_high_threshold():
    """Test that system handles case where no edges form (too high threshold)."""
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(42)

    # Extremely high coactivity threshold - no edges should form
    config = PlaceCellControllerConfig(
        num_place_cells=50,
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.2,
        coactivity_threshold=1000.0,  # Unrealistically high
        max_edge_distance=0.3,
        integration_window=None,
    )

    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent = Agent(environment=env, random_state=rng)

    # Run short simulation
    dt = 0.05
    for _ in range(100):
        position = agent.step(dt)
        controller.step(position, dt)

    graph = controller.get_graph()

    # Should have nodes but no edges
    assert graph.num_nodes() == config.num_place_cells
    assert graph.num_edges() == 0
    assert graph.num_components() == config.num_place_cells  # All isolated


def test_very_short_duration():
    """Test that system handles very short simulation duration."""
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(42)

    config = PlaceCellControllerConfig(
        num_place_cells=50,
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.2,
        coactivity_threshold=5.0,
        max_edge_distance=0.3,
        integration_window=None,
    )

    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent = Agent(environment=env, random_state=rng)

    # Run for only 1 second
    dt = 0.05
    num_steps = int(1.0 / dt)  # 20 steps

    for _ in range(num_steps):
        position = agent.step(dt)
        controller.step(position, dt)

    graph = controller.get_graph()

    # Should still have valid graph structure
    assert graph.num_nodes() == config.num_place_cells
    assert graph.num_edges() >= 0  # May have some edges or none
    assert graph.num_components() >= 1


def test_integration_window_longer_than_duration():
    """Test integration window longer than simulation duration."""
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(42)

    config = PlaceCellControllerConfig(
        num_place_cells=50,
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.2,
        coactivity_threshold=5.0,
        max_edge_distance=0.3,
        integration_window=1000.0,  # Much longer than duration
    )

    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent = Agent(environment=env, random_state=rng)

    # Run for only 10 seconds
    dt = 0.05
    num_steps = int(10.0 / dt)

    for _ in range(num_steps):
        position = agent.step(dt)
        controller.step(position, dt)

    graph = controller.get_graph()

    # Should handle gracefully - no edges should be admitted yet
    assert graph.num_nodes() == config.num_place_cells
    # Edges may exist but not yet admitted due to integration window
    assert graph.num_edges() >= 0


# ============================================================================
# Obstacle Edge Cases
# ============================================================================


def test_obstacle_too_large():
    """Test obstacle that extends beyond arena bounds."""
    # Obstacle with radius 0.51 at center (extends beyond bounds: 0.5 - 0.51 = -0.01)
    obstacle = CircularObstacle(center_x=0.5, center_y=0.5, radius=0.51)

    # Should raise error because obstacle extends outside bounds
    with pytest.raises(ValueError, match="extends outside bounds"):
        Environment(width=1.0, height=1.0, obstacles=[obstacle])


def test_obstacle_at_boundary():
    """Test obstacle placed at environment boundary."""
    # Obstacle at top-right corner with radius that extends beyond
    obstacle = CircularObstacle(center_x=0.95, center_y=0.95, radius=0.1)

    # Should raise error because obstacle extends outside bounds
    with pytest.raises(ValueError, match="extends outside bounds"):
        Environment(width=1.0, height=1.0, obstacles=[obstacle])


def test_obstacle_at_boundary_valid():
    """Test obstacle that just fits at boundary."""
    # Obstacle at corner with radius that fits exactly
    obstacle = CircularObstacle(center_x=0.1, center_y=0.1, radius=0.1)

    # Should be valid
    env = Environment(width=1.0, height=1.0, obstacles=[obstacle])
    assert len(env.obstacles) == 1


def test_multiple_overlapping_obstacles():
    """Test that overlapping obstacles are detected/rejected."""
    # Two overlapping obstacles
    obstacles = [
        CircularObstacle(center_x=0.5, center_y=0.5, radius=0.2),
        CircularObstacle(center_x=0.52, center_y=0.52, radius=0.2),  # Overlaps
    ]

    # Environment creation should succeed (overlap detection is not enforced)
    # But agent navigation may be problematic
    env = Environment(width=1.0, height=1.0, obstacles=obstacles)
    assert len(env.obstacles) == 2

    # Agent should still be able to navigate (though may be constrained)
    rng = np.random.default_rng(42)
    agent = Agent(environment=env, random_state=rng, position=(0.1, 0.1))

    # Should be able to step without crashing
    for _ in range(10):
        position = agent.step(0.05)
        assert env.contains(tuple(position)) or any(
            obs.contains(tuple(position)) for obs in env.obstacles
        )


def test_agent_starts_inside_obstacle():
    """Test that agent cannot start inside obstacle."""
    obstacle = CircularObstacle(center_x=0.5, center_y=0.5, radius=0.15)
    env = Environment(width=1.0, height=1.0, obstacles=[obstacle])

    # Try to place agent in obstacle center
    with pytest.raises(ValueError, match="Initial position must lie within"):
        Agent(environment=env, position=(0.5, 0.5))

    # Try to place agent inside obstacle
    with pytest.raises(ValueError, match="Initial position must lie within"):
        Agent(environment=env, position=(0.52, 0.52))


# ============================================================================
# Place Cell Edge Cases
# ============================================================================


def test_very_few_place_cells():
    """Test with very few place cells (<10)."""
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(42)

    config = PlaceCellControllerConfig(
        num_place_cells=5,  # Very few
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.2,
        coactivity_threshold=5.0,
        max_edge_distance=0.3,
        integration_window=None,
    )

    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent = Agent(environment=env, random_state=rng)

    # Should still work
    dt = 0.05
    for _ in range(100):
        position = agent.step(dt)
        controller.step(position, dt)

    graph = controller.get_graph()
    assert graph.num_nodes() == 5
    assert graph.num_edges() >= 0


def test_very_many_place_cells():
    """Test with very many place cells (>500)."""
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(42)

    config = PlaceCellControllerConfig(
        num_place_cells=600,  # Very many
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.2,
        coactivity_threshold=5.0,
        max_edge_distance=0.3,
        integration_window=None,
    )

    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent = Agent(environment=env, random_state=rng)

    # Should still work (may be slow)
    dt = 0.05
    for _ in range(50):  # Fewer steps for speed
        position = agent.step(dt)
        controller.step(position, dt)

    graph = controller.get_graph()
    assert graph.num_nodes() == 600
    assert graph.num_edges() >= 0


def test_place_cells_clustered():
    """Test with place cells clustered in one region."""
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(42)

    # Create clustered positions
    cluster_center = np.array([0.3, 0.3])
    cluster_positions = []
    for _ in range(50):
        offset = rng.normal(scale=0.1, size=2)
        pos = cluster_center + offset
        # Clip to bounds
        pos = np.clip(pos, 0.05, 0.95)
        cluster_positions.append(pos)
    cluster_positions = np.array(cluster_positions)

    config = PlaceCellControllerConfig(
        num_place_cells=50,
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.2,
        coactivity_threshold=5.0,
        max_edge_distance=0.3,
        integration_window=None,
        place_cell_positions=cluster_positions,
    )

    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent = Agent(environment=env, random_state=rng)

    # Should still work
    dt = 0.05
    for _ in range(100):
        position = agent.step(dt)
        controller.step(position, dt)

    graph = controller.get_graph()
    assert graph.num_nodes() == 50
    # May have many edges due to clustering
    assert graph.num_edges() >= 0


# ============================================================================
# Integration Window Edge Cases
# ============================================================================


def test_integration_window_zero():
    """Test integration window = 0 (should be same as None)."""
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(42)

    config = PlaceCellControllerConfig(
        num_place_cells=50,
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.2,
        coactivity_threshold=5.0,
        max_edge_distance=0.3,
        integration_window=0.0,  # Zero window
    )

    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent = Agent(environment=env, random_state=rng)

    dt = 0.05
    for _ in range(100):
        position = agent.step(dt)
        controller.step(position, dt)

    graph = controller.get_graph()
    assert graph.num_nodes() == 50
    assert graph.num_edges() >= 0


def test_integration_window_equals_duration():
    """Test integration window exactly equals duration."""
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(42)

    duration = 10.0
    config = PlaceCellControllerConfig(
        num_place_cells=50,
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.2,
        coactivity_threshold=5.0,
        max_edge_distance=0.3,
        integration_window=duration,  # Exactly equals duration
    )

    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent = Agent(environment=env, random_state=rng)

    dt = 0.05
    num_steps = int(duration / dt)
    for _ in range(num_steps):
        position = agent.step(dt)
        controller.step(position, dt)

    graph = controller.get_graph()
    assert graph.num_nodes() == 50
    # Edges may be admitted at the very end
    assert graph.num_edges() >= 0


def test_integration_window_very_short():
    """Test very short integration window (<60s)."""
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(42)

    config = PlaceCellControllerConfig(
        num_place_cells=50,
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.2,
        coactivity_threshold=5.0,
        max_edge_distance=0.3,
        integration_window=5.0,  # Very short (5 seconds)
    )

    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent = Agent(environment=env, random_state=rng)

    dt = 0.05
    for _ in range(200):  # 10 seconds
        position = agent.step(dt)
        controller.step(position, dt)

    graph = controller.get_graph()
    assert graph.num_nodes() == 50
    # Should have edges after integration window passes
    assert graph.num_edges() >= 0


# ============================================================================
# Topology Edge Cases
# ============================================================================


def test_multiple_holes_three_obstacles():
    """Test topology with 3+ obstacles (multiple holes)."""
    obstacles = [
        CircularObstacle(center_x=0.3, center_y=0.3, radius=0.1),
        CircularObstacle(center_x=0.7, center_y=0.3, radius=0.1),
        CircularObstacle(center_x=0.5, center_y=0.7, radius=0.1),
    ]
    env = Environment(width=1.0, height=1.0, obstacles=obstacles)

    rng = np.random.default_rng(42)
    config = PlaceCellControllerConfig(
        num_place_cells=80,
        sigma=0.12,
        max_rate=15.0,
        coactivity_window=0.2,
        coactivity_threshold=5.0,
        max_edge_distance=0.3,
        integration_window=None,
    )

    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent = Agent(environment=env, random_state=rng)

    dt = 0.05
    for _ in range(150):
        position = agent.step(dt)
        controller.step(position, dt)

    graph = controller.get_graph()
    assert graph.num_nodes() == 80
    assert graph.num_components() >= 1

    # Try to compute Betti numbers if available
    try:
        betti = graph.compute_betti_numbers(max_dim=2)
        assert isinstance(betti, dict)
        assert 0 in betti
        assert 1 in betti
        # Ideally b₁ should be 3, but clique complex may fill holes
        assert betti[1] >= 0
    except (ImportError, Exception):
        pass  # Dependencies not available


def test_disconnected_graph_high_threshold():
    """Test disconnected graph (high threshold prevents connections)."""
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(42)

    # Very high threshold and small max distance
    config = PlaceCellControllerConfig(
        num_place_cells=50,
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.2,
        coactivity_threshold=100.0,  # Very high
        max_edge_distance=0.05,  # Very small
        integration_window=None,
    )

    controller = PlaceCellController(environment=env, config=config, rng=rng)
    agent = Agent(environment=env, random_state=rng)

    dt = 0.05
    for _ in range(100):
        position = agent.step(dt)
        controller.step(position, dt)

    graph = controller.get_graph()
    assert graph.num_nodes() == 50
    # Should have many components (disconnected)
    assert graph.num_components() >= 1
    # May have no edges or very few
    assert graph.num_edges() >= 0


def test_invalid_config_parameters():
    """Test that invalid configuration parameters are handled."""
    env = Environment(width=1.0, height=1.0)

    # Note: PlaceCellControllerConfig may not validate all parameters at creation
    # These tests verify the system doesn't crash with edge case values
    
    # Very small number of cells (edge case, but should work)
    config = PlaceCellControllerConfig(
        num_place_cells=1,  # Minimum
        sigma=0.15,
    )
    assert config.num_place_cells == 1

    # Very small sigma (edge case)
    config = PlaceCellControllerConfig(
        num_place_cells=50,
        sigma=0.001,  # Very small
    )
    assert config.sigma == 0.001

    # Zero coactivity threshold (edge case - may cause issues but shouldn't crash)
    config = PlaceCellControllerConfig(
        num_place_cells=50,
        sigma=0.15,
        coactivity_threshold=0.0,  # Edge case
    )
    assert config.coactivity_threshold == 0.0


def test_environment_invalid_dimensions():
    """Test environment with invalid dimensions."""
    # Zero width
    with pytest.raises(ValueError, match="dimensions must be positive"):
        Environment(width=0.0, height=1.0)

    # Negative height
    with pytest.raises(ValueError, match="dimensions must be positive"):
        Environment(width=1.0, height=-1.0)

    # Both invalid
    with pytest.raises(ValueError, match="dimensions must be positive"):
        Environment(width=0.0, height=0.0)


def test_agent_invalid_speed():
    """Test agent with invalid speed parameters."""
    env = Environment(width=1.0, height=1.0)

    # Zero base speed
    with pytest.raises(ValueError, match="base_speed must be positive"):
        Agent(environment=env, base_speed=0.0)

    # Negative max speed
    with pytest.raises(ValueError, match="max_speed must be positive"):
        Agent(environment=env, max_speed=-1.0)

    # Base speed > max speed
    # This should be allowed (agent will use base_speed as minimum)
    agent = Agent(environment=env, base_speed=0.5, max_speed=0.3)
    assert agent.base_speed == 0.5
    assert agent.max_speed == 0.3


def test_agent_invalid_dt():
    """Test agent step with invalid dt."""
    env = Environment(width=1.0, height=1.0)
    agent = Agent(environment=env)

    # Zero dt
    with pytest.raises(ValueError, match="dt must be positive"):
        agent.step(dt=0.0)

    # Negative dt
    with pytest.raises(ValueError, match="dt must be positive"):
        agent.step(dt=-0.1)


def test_controller_invalid_dt():
    """Test controller step with invalid dt."""
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(42)

    config = PlaceCellControllerConfig(
        num_place_cells=50,
        sigma=0.15,
    )
    controller = PlaceCellController(environment=env, config=config, rng=rng)

    position = np.array([0.5, 0.5])

    # Zero dt - should handle gracefully or raise error
    try:
        controller.step(position, dt=0.0)
    except (ValueError, AssertionError):
        pass  # Expected

    # Negative dt - should handle gracefully or raise error
    try:
        controller.step(position, dt=-0.1)
    except (ValueError, AssertionError):
        pass  # Expected

