"""Integration tests for SNN Policy Service."""
from __future__ import annotations

import numpy as np
import pytest

from hippocampus_core.env import Environment
from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)
from hippocampus_core.policy import (
    TopologyService,
    SpatialFeatureService,
    SpikingPolicyService,
    ActionArbitrationSafety,
    RobotState,
    Mission,
    MissionGoal,
    GoalType,
    PointGoal,
)


@pytest.fixture
def environment():
    """Create test environment."""
    return Environment(width=1.0, height=1.0)


@pytest.fixture
def place_controller(environment):
    """Create place cell controller."""
    config = PlaceCellControllerConfig(
        num_place_cells=50,
        coactivity_window=0.2,
        coactivity_threshold=5.0,
        integration_window=None,  # No integration window for faster testing
    )
    rng = np.random.default_rng(42)
    return PlaceCellController(environment=environment, config=config, rng=rng)


@pytest.fixture
def topology_service(place_controller):
    """Create topology service."""
    ts = TopologyService()
    ts.update_from_controller(place_controller)
    return ts


@pytest.fixture
def feature_service(topology_service):
    """Create feature service."""
    return SpatialFeatureService(topology_service, k_neighbors=8)


@pytest.fixture
def policy_service(feature_service):
    """Create policy service."""
    return SpikingPolicyService(feature_service)


@pytest.fixture
def safety_arbitrator():
    """Create safety arbitrator."""
    return ActionArbitrationSafety(max_linear=0.3, max_angular=1.0)


def test_topology_service_wraps_graph(place_controller, topology_service):
    """Test that TS correctly wraps TopologicalGraph."""
    snapshot = topology_service.get_graph_snapshot(0.0)
    graph = place_controller.get_graph()
    
    # Initially graph has no edges (no coactivity yet)
    assert len(snapshot.V) == graph.num_nodes()
    assert len(snapshot.E) == graph.num_edges()


def test_feature_service_builds_features(feature_service, place_controller):
    """Test that feature service builds features."""
    # Run controller for a few steps to build some graph
    for _ in range(100):
        position = np.array([0.5, 0.5])
        place_controller.step(position, dt=0.05)
    
    # Update topology service
    topology_service = feature_service.topology_service
    topology_service.update_from_controller(place_controller)
    
    # Build features
    robot_state = RobotState(
        pose=(0.5, 0.5, 0.0),
        time=5.0,
    )
    mission = Mission(
        goal=MissionGoal(
            type=GoalType.POINT,
            value=PointGoal(position=(0.9, 0.9)),
        )
    )
    
    features, context = feature_service.build_features(robot_state, mission)
    
    # Check feature structure
    assert len(features.goal_ego) == 3
    assert len(features.neighbors_k) == 8
    assert len(features.topo_ctx) == 3
    assert len(features.safety) == 4
    assert features.dim > 0


def test_policy_service_heuristic(feature_service, policy_service):
    """Test policy service heuristic decision."""
    # Build features
    robot_state = RobotState(
        pose=(0.5, 0.5, 0.0),
        time=0.0,
    )
    mission = Mission(
        goal=MissionGoal(
            type=GoalType.POINT,
            value=PointGoal(position=(0.9, 0.9)),
        )
    )
    
    features, context = feature_service.build_features(robot_state, mission)
    
    # Make decision
    decision = policy_service.decide(features, context, dt=0.1)
    
    # Check decision structure
    assert decision.action_proposal.v is not None
    assert decision.action_proposal.omega is not None
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.reason == "heuristic"


def test_policy_service_step(policy_service):
    """Test policy service step method."""
    # Create observation [x, y, cos(heading), sin(heading)]
    obs = np.array([0.5, 0.5, 1.0, 0.0], dtype=np.float32)
    
    # Step
    action = policy_service.step(obs, dt=0.1)
    
    # Check action
    assert action.shape == (2,)
    assert -0.3 <= action[0] <= 0.3  # v within limits
    assert -1.0 <= action[1] <= 1.0  # omega within limits


def test_safety_arbitration(safety_arbitrator, feature_service, place_controller):
    """Test safety arbitration."""
    # Run controller to build graph
    for _ in range(100):
        position = np.array([0.5, 0.5])
        place_controller.step(position, dt=0.05)
    
    topology_service = feature_service.topology_service
    topology_service.update_from_controller(place_controller)
    
    # Build features and decision
    robot_state = RobotState(
        pose=(0.5, 0.5, 0.0),
        time=5.0,
    )
    mission = Mission(
        goal=MissionGoal(
            type=GoalType.POINT,
            value=PointGoal(position=(0.9, 0.9)),
        )
    )
    
    features, context = feature_service.build_features(robot_state, mission)
    policy_service = SpikingPolicyService(feature_service)
    decision = policy_service.decide(features, context, dt=0.1)
    
    # Filter through safety
    graph_snapshot = topology_service.get_graph_snapshot(robot_state.time)
    safe_cmd = safety_arbitrator.filter(decision, robot_state, graph_snapshot, mission)
    
    # Check safe command
    assert safe_cmd.cmd[0] is not None  # v
    assert safe_cmd.cmd[1] is not None  # omega
    assert -0.3 <= safe_cmd.cmd[0] <= 0.3
    assert -1.0 <= safe_cmd.cmd[1] <= 1.0
    assert isinstance(safe_cmd.safety_flags, dict)


def test_end_to_end_pipeline(
    place_controller,
    topology_service,
    feature_service,
    policy_service,
    safety_arbitrator,
):
    """Test end-to-end policy pipeline."""
    # Run simulation
    for step in range(50):
        position = np.array([0.4 + step * 0.001, 0.4 + step * 0.001])
        place_controller.step(position, dt=0.05)
        
        # Update topology service periodically
        if step % 10 == 0:
            topology_service.update_from_controller(place_controller)
        
        # Build features
        robot_state = RobotState(
            pose=(position[0], position[1], 0.0),
            time=step * 0.05,
        )
        mission = Mission(
            goal=MissionGoal(
                type=GoalType.POINT,
                value=PointGoal(position=(0.9, 0.9)),
            )
        )
        
        features, context = feature_service.build_features(robot_state, mission)
        
        # Make decision
        decision = policy_service.decide(features, context, dt=0.05)
        
        # Filter through safety
        graph_snapshot = topology_service.get_graph_snapshot(robot_state.time)
        safe_cmd = safety_arbitrator.filter(decision, robot_state, graph_snapshot, mission)
        
        # Verify outputs are valid
        assert -0.3 <= safe_cmd.cmd[0] <= 0.3
        assert -1.0 <= safe_cmd.cmd[1] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

