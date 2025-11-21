"""Tests for policy freeze/unfreeze functionality during safety hold mode."""
from __future__ import annotations

import numpy as np
import pytest

from hippocampus_core.policy import (
    SpikingPolicyService,
    SpatialFeatureService,
    TopologyService,
    FeatureVector,
    LocalContext,
    RobotState,
    Mission,
    MissionGoal,
    GoalType,
    PointGoal,
)
from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)
from hippocampus_core.env import Environment


@pytest.fixture
def policy_service():
    """Create policy service for testing."""
    env = Environment(width=1.0, height=1.0)
    controller = PlaceCellController(
        environment=env,
        config=PlaceCellControllerConfig(num_place_cells=20),
        rng=np.random.default_rng(42),
    )
    
    # Run a few steps to build graph
    for _ in range(50):
        controller.step(np.array([0.5, 0.5]), dt=0.05)
    
    ts = TopologyService()
    ts.update_from_controller(controller)
    sfs = SpatialFeatureService(ts, k_neighbors=5)
    return SpikingPolicyService(sfs)


def test_policy_freezes_correctly(policy_service):
    """Test that freeze() sets frozen flag and resets internal state."""
    # Make a decision first to initialize state
    robot_state = RobotState(pose=(0.5, 0.5, 0.0), time=0.0)
    mission = Mission(goal=MissionGoal(type=GoalType.POINT, value=PointGoal(position=(0.9, 0.9))))
    
    features, context = policy_service._feature_service.build_features(robot_state, mission)
    decision1 = policy_service.decide(features, context, dt=0.05)
    
    # Freeze policy
    policy_service.freeze()
    assert policy_service._is_frozen
    
    # Try to make decision while frozen - should return zero action
    decision2 = policy_service.decide(features, context, dt=0.05)
    assert decision2.action_proposal.v == 0.0
    assert decision2.action_proposal.omega == 0.0
    assert decision2.confidence == 0.0
    assert decision2.reason == "policy_frozen"


def test_policy_unfreezes_correctly(policy_service):
    """Test that unfreeze() clears frozen flag."""
    policy_service.freeze()
    assert policy_service._is_frozen
    
    policy_service.unfreeze()
    assert not policy_service._is_frozen
    
    # Should be able to make decisions again
    robot_state = RobotState(pose=(0.5, 0.5, 0.0), time=0.0)
    mission = Mission(goal=MissionGoal(type=GoalType.POINT, value=PointGoal(position=(0.9, 0.9))))
    
    features, context = policy_service._feature_service.build_features(robot_state, mission)
    decision = policy_service.decide(features, context, dt=0.05)
    
    # Should not be frozen decision
    assert decision.reason != "policy_frozen"
    assert decision.confidence > 0.0 or decision.action_proposal.v != 0.0 or decision.action_proposal.omega != 0.0


def test_policy_reset_clears_frozen(policy_service):
    """Test that reset() clears frozen flag."""
    policy_service.freeze()
    assert policy_service._is_frozen
    
    policy_service.reset()
    assert not policy_service._is_frozen


def test_policy_step_when_frozen(policy_service):
    """Test that step() returns zero action when frozen."""
    policy_service.freeze()
    
    obs = np.array([0.5, 0.5, 1.0, 0.0])  # [x, y, cos(theta), sin(theta)]
    action = policy_service.step(obs, dt=0.05)
    
    # Should return zero action
    assert np.allclose(action, [0.0, 0.0])

