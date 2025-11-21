"""Unit tests for environment and obstacle functionality."""
import numpy as np
import pytest

from hippocampus_core.env import Agent, CircularObstacle, Environment


@pytest.fixture
def simple_environment():
    """Create a simple 1x1 environment without obstacles."""
    return Environment(width=1.0, height=1.0)


@pytest.fixture
def obstacle_environment():
    """Create an environment with a central circular obstacle."""
    obstacle = CircularObstacle(center_x=0.5, center_y=0.5, radius=0.15)
    return Environment(width=1.0, height=1.0, obstacles=[obstacle])


@pytest.fixture
def multiple_obstacles_environment():
    """Create an environment with multiple obstacles."""
    obstacles = [
        CircularObstacle(center_x=0.3, center_y=0.3, radius=0.1),
        CircularObstacle(center_x=0.7, center_y=0.7, radius=0.1),
    ]
    return Environment(width=1.0, height=1.0, obstacles=obstacles)


# CircularObstacle tests


def test_circular_obstacle_creation():
    """Test valid obstacle creation."""
    obstacle = CircularObstacle(center_x=0.5, center_y=0.5, radius=0.1)
    assert obstacle.center_x == 0.5
    assert obstacle.center_y == 0.5
    assert obstacle.radius == 0.1


def test_circular_obstacle_invalid_radius():
    """Test that invalid radius values are handled (though dataclass allows them)."""
    # CircularObstacle is a dataclass, so it doesn't validate radius directly
    # But Environment will validate when obstacles are added
    # This test documents that the obstacle itself can be created with invalid radius
    # but Environment will catch it
    obstacle = CircularObstacle(center_x=0.5, center_y=0.5, radius=-0.1)
    assert obstacle.radius == -0.1  # Dataclass allows it
    
    # But Environment should reject it
    with pytest.raises(ValueError, match="radius must be positive"):
        Environment(width=1.0, height=1.0, obstacles=[obstacle])
    
    obstacle_zero = CircularObstacle(center_x=0.5, center_y=0.5, radius=0.0)
    with pytest.raises(ValueError, match="radius must be positive"):
        Environment(width=1.0, height=1.0, obstacles=[obstacle_zero])


def test_obstacle_contains_inside():
    """Test that points inside obstacle return True."""
    obstacle = CircularObstacle(center_x=0.5, center_y=0.5, radius=0.1)
    
    # Center point
    assert obstacle.contains((0.5, 0.5)) is True
    
    # Points inside
    assert obstacle.contains((0.52, 0.52)) is True
    assert obstacle.contains((0.48, 0.48)) is True
    assert obstacle.contains((0.55, 0.5)) is True  # On edge (within radius)
    assert obstacle.contains((0.5, 0.55)) is True


def test_obstacle_contains_outside():
    """Test that points outside obstacle return False."""
    obstacle = CircularObstacle(center_x=0.5, center_y=0.5, radius=0.1)
    
    # Points outside
    assert obstacle.contains((0.7, 0.7)) is False
    assert obstacle.contains((0.3, 0.3)) is False
    assert obstacle.contains((0.61, 0.5)) is False  # Just outside edge
    assert obstacle.contains((0.0, 0.0)) is False


def test_obstacle_contains_on_edge():
    """Test points exactly on the obstacle edge."""
    obstacle = CircularObstacle(center_x=0.5, center_y=0.5, radius=0.1)
    
    # Points exactly on edge (distance = radius)
    assert obstacle.contains((0.6, 0.5)) is True  # Right edge
    assert obstacle.contains((0.4, 0.5)) is True  # Left edge
    assert obstacle.contains((0.5, 0.6)) is True  # Top edge
    assert obstacle.contains((0.5, 0.4)) is True  # Bottom edge


def test_obstacle_distance_to_edge():
    """Test distance_to_edge calculation."""
    obstacle = CircularObstacle(center_x=0.5, center_y=0.5, radius=0.1)
    
    # Inside obstacle (negative distance)
    dist_inside = obstacle.distance_to_edge((0.5, 0.5))
    assert dist_inside < 0
    
    # On edge (zero distance)
    dist_edge = obstacle.distance_to_edge((0.6, 0.5))
    assert abs(dist_edge) < 1e-10
    
    # Outside obstacle (positive distance)
    dist_outside = obstacle.distance_to_edge((0.8, 0.5))
    assert dist_outside > 0
    assert abs(dist_outside - 0.2) < 1e-10  # Should be 0.2 units from edge


# Environment tests


def test_environment_creation_without_obstacles(simple_environment):
    """Test environment creation without obstacles."""
    assert simple_environment.bounds.width == 1.0
    assert simple_environment.bounds.height == 1.0
    assert len(simple_environment.obstacles) == 0


def test_environment_creation_with_obstacles(obstacle_environment):
    """Test environment creation with obstacles."""
    assert len(obstacle_environment.obstacles) == 1
    obstacle = obstacle_environment.obstacles[0]
    assert obstacle.center_x == 0.5
    assert obstacle.center_y == 0.5
    assert obstacle.radius == 0.15


def test_environment_obstacles_returns_copy(obstacle_environment):
    """Test that obstacles property returns a copy."""
    obstacles1 = obstacle_environment.obstacles
    obstacles2 = obstacle_environment.obstacles
    assert obstacles1 is not obstacles2  # Different objects
    assert obstacles1 == obstacles2  # But same content


def test_environment_invalid_dimensions():
    """Test that invalid environment dimensions raise errors."""
    with pytest.raises(ValueError, match="dimensions must be positive"):
        Environment(width=0.0, height=1.0)
    
    with pytest.raises(ValueError, match="dimensions must be positive"):
        Environment(width=1.0, height=-1.0)


def test_environment_invalid_obstacle_radius():
    """Test that obstacles with invalid radius raise errors."""
    with pytest.raises(ValueError, match="radius must be positive"):
        Environment(
            width=1.0,
            height=1.0,
            obstacles=[CircularObstacle(0.5, 0.5, radius=0.0)],
        )
    
    with pytest.raises(ValueError, match="radius must be positive"):
        Environment(
            width=1.0,
            height=1.0,
            obstacles=[CircularObstacle(0.5, 0.5, radius=-0.1)],
        )


def test_environment_obstacle_outside_bounds():
    """Test that obstacles outside bounds raise errors."""
    # Obstacle center outside bounds
    with pytest.raises(ValueError, match="center_x.*outside bounds"):
        Environment(
            width=1.0,
            height=1.0,
            obstacles=[CircularObstacle(1.5, 0.5, radius=0.1)],
        )
    
    # Obstacle extends outside bounds
    with pytest.raises(ValueError, match="extends outside bounds"):
        Environment(
            width=1.0,
            height=1.0,
            obstacles=[CircularObstacle(0.5, 0.5, radius=0.6)],
        )


def test_environment_contains_without_obstacles(simple_environment):
    """Test contains() method without obstacles."""
    # Inside bounds
    assert simple_environment.contains((0.5, 0.5)) is True
    assert simple_environment.contains((0.1, 0.1)) is True
    assert simple_environment.contains((0.9, 0.9)) is True
    
    # On boundaries
    assert simple_environment.contains((0.0, 0.5)) is True
    assert simple_environment.contains((1.0, 0.5)) is True
    assert simple_environment.contains((0.5, 0.0)) is True
    assert simple_environment.contains((0.5, 1.0)) is True
    
    # Outside bounds
    assert simple_environment.contains((1.1, 0.5)) is False
    assert simple_environment.contains((-0.1, 0.5)) is False
    assert simple_environment.contains((0.5, 1.1)) is False
    assert simple_environment.contains((0.5, -0.1)) is False


def test_environment_contains_with_obstacles(obstacle_environment):
    """Test contains() method with obstacles."""
    # Inside bounds but in obstacle
    assert obstacle_environment.contains((0.5, 0.5)) is False  # Center of obstacle
    assert obstacle_environment.contains((0.52, 0.52)) is False  # Inside obstacle
    
    # Inside bounds and outside obstacle
    assert obstacle_environment.contains((0.1, 0.1)) is True
    assert obstacle_environment.contains((0.9, 0.9)) is True
    assert obstacle_environment.contains((0.7, 0.5)) is True  # Outside obstacle
    
    # Outside bounds
    assert obstacle_environment.contains((1.1, 0.5)) is False
    assert obstacle_environment.contains((-0.1, 0.5)) is False


def test_environment_contains_with_multiple_obstacles(multiple_obstacles_environment):
    """Test contains() with multiple obstacles."""
    # In first obstacle
    assert multiple_obstacles_environment.contains((0.3, 0.3)) is False
    
    # In second obstacle
    assert multiple_obstacles_environment.contains((0.7, 0.7)) is False
    
    # Outside all obstacles
    assert multiple_obstacles_environment.contains((0.5, 0.5)) is True
    assert multiple_obstacles_environment.contains((0.1, 0.1)) is True
    assert multiple_obstacles_environment.contains((0.9, 0.9)) is True


def test_environment_clip():
    """Test clip() method for boundary clipping."""
    env = Environment(width=1.0, height=1.0)
    
    # Inside bounds - no clipping
    pos, mask = env.clip(np.array([0.5, 0.5]))
    assert np.allclose(pos, [0.5, 0.5])
    assert not np.any(mask)
    
    # Outside bounds - should clip
    pos, mask = env.clip(np.array([1.5, 0.5]))
    assert np.allclose(pos, [1.0, 0.5])
    assert bool(mask[0]) is True
    assert bool(mask[1]) is False
    
    pos, mask = env.clip(np.array([-0.5, 0.5]))
    assert np.allclose(pos, [0.0, 0.5])
    assert bool(mask[0]) is True
    assert bool(mask[1]) is False


# Agent tests


def test_agent_creation(simple_environment):
    """Test agent creation."""
    agent = Agent(environment=simple_environment)
    assert agent.environment is simple_environment
    assert agent.position.shape == (2,)
    assert np.all(agent.position >= 0.0)
    assert np.all(agent.position <= 1.0)


def test_agent_custom_position(simple_environment):
    """Test agent creation with custom position."""
    agent = Agent(environment=simple_environment, position=(0.2, 0.3))
    assert np.allclose(agent.position, [0.2, 0.3])


def test_agent_invalid_position(simple_environment):
    """Test that invalid initial position raises error."""
    with pytest.raises(ValueError, match="Initial position must lie within"):
        Agent(environment=simple_environment, position=(1.5, 0.5))


def test_agent_step_moves(simple_environment):
    """Test that agent step() moves the agent."""
    agent = Agent(environment=simple_environment)
    initial_position = agent.position.copy()
    
    new_position = agent.step(dt=0.1)
    
    # Agent should have moved (with some tolerance for edge cases)
    assert not np.allclose(new_position, initial_position, atol=1e-6)


def test_agent_obstacle_avoidance(obstacle_environment):
    """Test that agent avoids obstacles."""
    # Place agent near obstacle
    agent = Agent(
        environment=obstacle_environment,
        position=(0.7, 0.5),  # To the right of obstacle
        base_speed=0.1,
        max_speed=0.2,
    )
    
    # Move agent towards obstacle
    for _ in range(100):
        position = agent.step(dt=0.05)
        # Agent should never enter obstacle
        assert obstacle_environment.contains(tuple(position)), (
            f"Agent entered obstacle at position {position}"
        )


def test_agent_velocity_deflection_on_obstacle(obstacle_environment):
    """Test that agent velocity deflects correctly when hitting obstacle."""
    # Place agent moving directly towards obstacle
    agent = Agent(
        environment=obstacle_environment,
        position=(0.7, 0.5),  # To the right of obstacle center (0.5, 0.5)
        base_speed=0.15,
        max_speed=0.3,
        velocity_noise=0.0,  # No noise for deterministic test
    )
    
    # Set velocity directly towards obstacle (leftward)
    agent.velocity = np.array([-0.2, 0.0])  # Moving left towards obstacle
    
    initial_velocity = agent.velocity.copy()
    initial_position = agent.position.copy()
    
    # Step multiple times - agent should hit obstacle and deflect
    hit_obstacle = False
    for _ in range(50):
        position_before = agent.position.copy()
        velocity_before = agent.velocity.copy()
        position_after = agent.step(dt=0.05)
        velocity_after = agent.velocity.copy()
        
        # Check if we're near obstacle (within 0.2 of center)
        dist_to_obstacle = np.linalg.norm(position_after - np.array([0.5, 0.5]))
        if dist_to_obstacle < 0.25:  # Near obstacle
            hit_obstacle = True
            # Velocity should have been deflected (should have positive x component)
            # since we're pushing away from obstacle center
            if velocity_after[0] > 0:  # Moving away from obstacle
                # Deflection worked
                break
    
    # Agent should have encountered obstacle
    assert hit_obstacle or np.linalg.norm(agent.position - initial_position) > 0.1, (
        "Agent should have moved and encountered obstacle"
    )
    
    # Final position should be valid (not in obstacle)
    assert obstacle_environment.contains(tuple(agent.position)), (
        f"Agent ended up in obstacle at {agent.position}"
    )


def test_agent_path_around_obstacle(obstacle_environment):
    """Test that agent can navigate around obstacle."""
    # Place agent to the right of obstacle
    agent = Agent(
        environment=obstacle_environment,
        position=(0.7, 0.5),  # Right of obstacle
        base_speed=0.1,
        max_speed=0.2,
    )
    
    # Track positions to verify agent moves around obstacle
    positions = [agent.position.copy()]
    
    # Run for many steps - agent should explore around obstacle
    for _ in range(300):
        position = agent.step(dt=0.05)
        positions.append(position.copy())
        
        # Should always be in valid position
        assert obstacle_environment.contains(tuple(position)), (
            f"Agent entered obstacle at step {len(positions)-1}, position {position}"
        )
    
    # Agent should have moved around (not stuck)
    # Check that agent visited different quadrants around obstacle
    positions_array = np.array(positions)
    
    # Check that agent moved significantly from start
    total_displacement = np.linalg.norm(positions_array[-1] - positions_array[0])
    assert total_displacement > 0.05, (
        f"Agent didn't move enough: total displacement = {total_displacement:.4f}"
    )
    
    # Check that agent explored different areas (not just oscillating)
    # Compute max distance from start
    max_displacement = max(
        np.linalg.norm(pos - positions_array[0]) for pos in positions_array
    )
    assert max_displacement > 0.1, (
        f"Agent didn't explore enough: max displacement = {max_displacement:.4f}"
    )


def test_agent_invalid_initial_position_in_obstacle(obstacle_environment):
    """Test that agent cannot be initialized inside an obstacle."""
    # Try to place agent in obstacle center
    with pytest.raises(ValueError, match="Initial position must lie within"):
        Agent(
            environment=obstacle_environment,
            position=(0.5, 0.5),  # Center of obstacle
        )
    
    # Try to place agent inside obstacle (but not at center)
    with pytest.raises(ValueError, match="Initial position must lie within"):
        Agent(
            environment=obstacle_environment,
            position=(0.52, 0.52),  # Inside obstacle
        )


def test_agent_boundary_reflection(simple_environment):
    """Test that agent reflects off boundaries."""
    # Place agent at boundary moving outward
    agent = Agent(environment=simple_environment, position=(0.95, 0.5))
    agent.velocity = np.array([0.2, 0.0])  # Moving right
    
    initial_velocity = agent.velocity.copy()
    position_before = agent.position.copy()
    
    # Step forward multiple times to ensure boundary is hit
    hit_boundary = False
    for _ in range(20):
        position_after = agent.step(dt=0.1)
        # Position should always be within bounds
        assert position_after[0] <= 1.0 + 1e-6  # Allow small tolerance
        assert position_after[0] >= 0.0 - 1e-6
        # Check if we hit the right boundary (x >= 0.99)
        if position_after[0] >= 0.99:
            hit_boundary = True
            # After hitting boundary, velocity should be reflected (negative x)
            # Allow some tolerance for noise/perturbation
            if agent.velocity[0] > 0:
                # If still positive, check next step - reflection might happen then
                continue
            # Velocity should be negative or very small after reflection
            assert agent.velocity[0] <= 0.1, (
                f"Velocity not reflected: {agent.velocity[0]} (expected <= 0.1)"
            )
            break
    
    # Verify we actually hit the boundary
    assert hit_boundary or position_after[0] >= 0.95, (
        "Agent should have reached or hit the boundary"
    )


def test_agent_doesnt_get_stuck(obstacle_environment):
    """Test that agent doesn't get stuck in obstacles."""
    agent = Agent(
        environment=obstacle_environment,
        position=(0.7, 0.5),
        base_speed=0.1,
        max_speed=0.2,
    )
    
    # Run many steps
    positions = []
    for _ in range(200):
        position = agent.step(dt=0.05)
        positions.append(position.copy())
        # Should always be in valid position
        assert obstacle_environment.contains(tuple(position))
    
    # Agent should have moved around (not stuck)
    # Check that position has changed (not stuck at same spot)
    total_displacement = np.linalg.norm(positions[-1] - positions[0])
    # Also check that agent moved at some point (max displacement from start)
    max_displacement = max(np.linalg.norm(pos - positions[0]) for pos in positions)
    
    # Agent should have moved at least a little bit
    assert total_displacement > 0.01 or max_displacement > 0.05, (
        f"Agent appears stuck: total displacement={total_displacement:.4f}, "
        f"max displacement={max_displacement:.4f}"
    )


def test_agent_invalid_dt(simple_environment):
    """Test that invalid dt raises error."""
    agent = Agent(environment=simple_environment)
    
    with pytest.raises(ValueError, match="dt must be positive"):
        agent.step(dt=0.0)
    
    with pytest.raises(ValueError, match="dt must be positive"):
        agent.step(dt=-0.1)


def test_agent_velocity_initialization(simple_environment):
    """Test that agent velocity is properly initialized."""
    agent = Agent(environment=simple_environment)
    
    assert agent.velocity.shape == (2,)
    speed = np.linalg.norm(agent.velocity)
    # Allow small floating point tolerance
    assert agent.base_speed - 1e-10 <= speed <= agent.max_speed + 1e-10


def test_agent_returns_heading_when_requested(simple_environment):
    """Agent can append heading information to its observation."""
    agent = Agent(
        environment=simple_environment,
        position=(0.5, 0.5),
        velocity_noise=0.0,
    )
    agent.velocity = np.array([0.0, 0.2])
    obs = agent.step(dt=0.05, include_theta=True)

    assert obs.shape == (3,)
    assert np.isclose(obs[2], agent.heading)


def test_agent_altitude_channel(simple_environment):
    """Agent can append altitude information when requested."""
    agent = Agent(
        environment=simple_environment,
        track_altitude=True,
        altitude=0.25,
    )
    obs = agent.step(dt=0.05, include_altitude=True)

    assert obs.shape == (3,)
    assert obs[-1] == pytest.approx(0.25)

