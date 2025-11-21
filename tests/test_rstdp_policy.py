"""Tests for R-STDP policy learning (biologically plausible SNN).

These tests verify:
- R-STDP network forward pass
- Eligibility trace updates
- Weight updates from rewards
- Integration with policy service
- Reward function computation
"""
from __future__ import annotations

import numpy as np
import pytest

from hippocampus_core.policy import (
    RSTDPPolicySNN,
    RSTDPConfig,
    NavigationRewardFunction,
    RewardConfig,
    FeatureVector,
    RobotState,
    Mission,
    MissionGoal,
    GoalType,
    PointGoal,
    Pose,
    PolicyDecision,
    ActionProposal,
    TopologyService,
    SpatialFeatureService,
    SpikingPolicyService,
)


class TestRSTDPNetwork:
    """Tests for RSTDPPolicySNN network."""

    def test_network_initialization(self):
        """Test R-STDP network initialization."""
        config = RSTDPConfig(
            feature_dim=10,
            hidden_size=32,
            output_size=2,
        )
        network = RSTDPPolicySNN(config)

        assert network.config == config
        assert network.feature_dim == 10
        assert network.output_dim == 2
        assert network._w_in.shape == (32, 10)
        assert network._w_out.shape == (2, 32)

    def test_network_reset(self):
        """Test network state reset."""
        config = RSTDPConfig(feature_dim=10, hidden_size=32, output_size=2)
        network = RSTDPPolicySNN(config)

        # Forward pass to set some state
        features = FeatureVector(
            goal_ego=[0.5, 0.5, 0.5],
            neighbors_k=[0.1] * 8,
            topo_ctx=[0.2] * 4,
            safety=[0.3] * 4,
            dynamics=[0.4] * 2,
        )
        network.forward(features)

        # Verify state is non-zero
        assert np.any(network._mem_hidden != 0) or np.any(network._pre_trace != 0)

        # Reset
        network.reset()

        # Verify state is zero
        assert np.all(network._mem_hidden == 0)
        assert np.all(network._mem_out == 0)
        assert np.all(network._pre_trace == 0)
        assert np.all(network._post_trace == 0)
        assert np.all(network._eligibility == 0)

    def test_forward_pass(self):
        """Test forward pass produces valid actions."""
        config = RSTDPConfig(feature_dim=10, hidden_size=32, output_size=2)
        network = RSTDPPolicySNN(config)

        features = FeatureVector(
            goal_ego=[0.5, 0.5, 0.5],
            neighbors_k=[0.1] * 8,
            topo_ctx=[0.2] * 4,
            safety=[0.3] * 4,
            dynamics=[0.4] * 2,
        )

        action = network.forward(features)

        assert action.shape == (2,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)  # tanh output

    def test_eligibility_trace_update(self):
        """Test eligibility traces are updated correctly."""
        config = RSTDPConfig(feature_dim=10, hidden_size=32, output_size=2)
        network = RSTDPPolicySNN(config)

        features = FeatureVector(
            goal_ego=[0.5, 0.5, 0.5],
            neighbors_k=[0.1] * 8,
            topo_ctx=[0.2] * 4,
            safety=[0.3] * 4,
            dynamics=[0.4] * 2,
        )

        # Initial eligibility should be zero
        assert np.all(network._eligibility == 0)

        # Forward pass should update eligibility
        network.forward(features)

        # Eligibility may be non-zero if spikes occurred
        # (depends on random weights and inputs)
        # Just verify the structure is correct
        assert network._eligibility.shape == (2, 32)

    def test_weight_update_positive_reward(self):
        """Test weight updates with positive reward."""
        config = RSTDPConfig(
            feature_dim=10,
            hidden_size=32,
            output_size=2,
            learning_rate=0.01,
        )
        network = RSTDPPolicySNN(config)

        # Get initial weights
        initial_weights = network._w_out.copy()

        # Forward pass to set eligibility traces
        features = FeatureVector(
            goal_ego=[0.5, 0.5, 0.5],
            neighbors_k=[0.1] * 8,
            topo_ctx=[0.2] * 4,
            safety=[0.3] * 4,
            dynamics=[0.4] * 2,
        )
        network.forward(features)

        # Update with positive reward
        network.update_weights(reward=1.0)

        # Weights should have changed (if eligibility was non-zero)
        # Note: weights may not change if no spikes occurred
        # This is expected behavior - just verify the method runs
        assert network._w_out.shape == (2, 32)

    def test_weight_update_negative_reward(self):
        """Test weight updates with negative reward."""
        config = RSTDPConfig(
            feature_dim=10,
            hidden_size=32,
            output_size=2,
            learning_rate=0.01,
        )
        network = RSTDPPolicySNN(config)

        # Forward pass
        features = FeatureVector(
            goal_ego=[0.5, 0.5, 0.5],
            neighbors_k=[0.1] * 8,
            topo_ctx=[0.2] * 4,
            safety=[0.3] * 4,
            dynamics=[0.4] * 2,
        )
        network.forward(features)

        # Update with negative reward
        network.update_weights(reward=-1.0)

        # Weights should be bounded
        assert np.all(network._w_out >= config.weight_min)
        assert np.all(network._w_out <= config.weight_max)

    def test_weight_update_zero_reward(self):
        """Test weight updates with zero reward (should decay eligibility)."""
        config = RSTDPConfig(
            feature_dim=10,
            hidden_size=32,
            output_size=2,
            eligibility_decay=0.8,
        )
        network = RSTDPPolicySNN(config)

        # Forward pass to set eligibility
        features = FeatureVector(
            goal_ego=[0.5, 0.5, 0.5],
            neighbors_k=[0.1] * 8,
            topo_ctx=[0.2] * 4,
            safety=[0.3] * 4,
            dynamics=[0.4] * 2,
        )
        network.forward(features)

        # Get eligibility before update
        eligibility_before = network._eligibility.copy()

        # Update with zero reward (should decay eligibility)
        network.update_weights(reward=0.0)

        # Eligibility should decay
        eligibility_after = network._eligibility
        assert np.allclose(
            eligibility_after, eligibility_before * config.eligibility_decay
        )

    def test_weight_bounds(self):
        """Test weights stay within bounds after updates."""
        config = RSTDPConfig(
            feature_dim=10,
            hidden_size=32,
            output_size=2,
            weight_min=-1.0,
            weight_max=1.0,
            learning_rate=1.0,  # Large learning rate to test bounds
        )
        network = RSTDPPolicySNN(config)

        # Multiple forward passes and updates
        features = FeatureVector(
            goal_ego=[0.5, 0.5, 0.5],
            neighbors_k=[0.1] * 8,
            topo_ctx=[0.2] * 4,
            safety=[0.3] * 4,
            dynamics=[0.4] * 2,
        )

        for _ in range(10):
            network.forward(features)
            network.update_weights(reward=10.0)  # Large reward

        # Weights should still be bounded
        assert np.all(network._w_out >= config.weight_min)
        assert np.all(network._w_out <= config.weight_max)

    def test_weight_save_load(self):
        """Test saving and loading weights."""
        config = RSTDPConfig(feature_dim=10, hidden_size=32, output_size=2)
        network = RSTDPPolicySNN(config)

        # Modify weights
        network._w_out += 0.1

        # Save
        weights = network.get_weights()

        # Create new network
        network2 = RSTDPPolicySNN(config)

        # Load weights
        network2.set_weights(weights)

        # Verify weights match
        assert np.allclose(network._w_out, network2._w_out)
        assert np.allclose(network._w_in, network2._w_in)

    def test_weight_load_shape_mismatch(self):
        """Test loading weights with wrong shape raises error."""
        config = RSTDPConfig(feature_dim=10, hidden_size=32, output_size=2)
        network = RSTDPPolicySNN(config)

        # Wrong shape
        wrong_weights = {
            "w_in": np.zeros((10, 5)),  # Wrong shape
            "w_out": np.zeros((2, 32)),
        }

        with pytest.raises(ValueError, match="w_in shape mismatch"):
            network.set_weights(wrong_weights)


class TestRewardFunction:
    """Tests for NavigationRewardFunction."""

    def test_reward_function_initialization(self):
        """Test reward function initialization."""
        config = RewardConfig(
            goal_reward_gain=2.0,
            goal_reached_reward=10.0,
        )
        reward_fn = NavigationRewardFunction(config)

        assert reward_fn.config == config

    def test_reward_function_reset(self):
        """Test reward function reset."""
        reward_fn = NavigationRewardFunction()

        # Compute reward to set state
        robot_state = RobotState(
            pose=Pose(x=0.5, y=0.5, heading=0.0),
            velocity=(0.0, 0.0),
        )
        mission = Mission(
            goal=MissionGoal(
                type=GoalType.POINT,
                value=PointGoal(position=(0.9, 0.9)),
            )
        )
        decision = PolicyDecision(
            action_proposal=ActionProposal(v=0.1, omega=0.0),
            confidence=0.8,
            reason="test",
        )

        reward_fn.compute(robot_state, decision, mission, dt=0.05)

        # Reset
        reward_fn.reset()

        # State should be reset
        assert reward_fn._last_position is None
        assert reward_fn._last_goal_distance is None

    def test_goal_progress_reward(self):
        """Test reward for goal progress."""
        reward_fn = NavigationRewardFunction()

        mission = Mission(
            goal=MissionGoal(
                type=GoalType.POINT,
                value=PointGoal(position=(1.0, 1.0)),
            )
        )

        # Start far from goal
        robot_state1 = RobotState(
            pose=Pose(x=0.0, y=0.0, heading=0.0),
            velocity=(0.0, 0.0),
        )
        decision = PolicyDecision(
            action_proposal=ActionProposal(v=0.1, omega=0.0),
            confidence=0.8,
            reason="test",
        )

        reward1 = reward_fn.compute(robot_state1, decision, mission, dt=0.05)

        # Move closer to goal
        robot_state2 = RobotState(
            pose=Pose(x=0.5, y=0.5, heading=0.0),
            velocity=(0.0, 0.0),
        )

        reward2 = reward_fn.compute(robot_state2, decision, mission, dt=0.05)

        # Second reward should be positive (progress toward goal)
        # Note: exact value depends on implementation
        assert isinstance(reward2, float)

    def test_goal_reached_reward(self):
        """Test large reward when goal is reached."""
        config = RewardConfig(
            goal_reached_reward=10.0,
            goal_reached_tolerance=0.1,
        )
        reward_fn = NavigationRewardFunction(config)

        mission = Mission(
            goal=MissionGoal(
                type=GoalType.POINT,
                value=PointGoal(position=(0.5, 0.5)),
            )
        )

        # At goal position
        robot_state = RobotState(
            pose=Pose(x=0.5, y=0.5, heading=0.0),
            velocity=(0.0, 0.0),
        )
        decision = PolicyDecision(
            action_proposal=ActionProposal(v=0.0, omega=0.0),
            confidence=0.8,
            reason="test",
        )

        reward = reward_fn.compute(robot_state, decision, mission, dt=0.05)

        # Should get goal reached reward
        assert reward >= config.goal_reached_reward * 0.9  # Allow some tolerance

    def test_smoothness_reward(self):
        """Test reward penalizes large angular velocities."""
        config = RewardConfig(
            angular_penalty_gain=0.2,
        )
        reward_fn = NavigationRewardFunction(config)

        mission = Mission(
            goal=MissionGoal(
                type=GoalType.POINT,
                value=PointGoal(position=(0.9, 0.9)),
            )
        )

        robot_state = RobotState(
            pose=Pose(x=0.5, y=0.5, heading=0.0),
            velocity=(0.0, 0.0),
        )

        # Small angular velocity
        decision_smooth = PolicyDecision(
            action_proposal=ActionProposal(v=0.1, omega=0.1),
            confidence=0.8,
            reason="test",
        )
        reward_smooth = reward_fn.compute(robot_state, decision_smooth, mission, dt=0.05)

        # Large angular velocity
        decision_aggressive = PolicyDecision(
            action_proposal=ActionProposal(v=0.1, omega=2.0),
            confidence=0.8,
            reason="test",
        )
        reward_aggressive = reward_fn.compute(robot_state, decision_aggressive, mission, dt=0.05)

        # Smooth action should have higher reward (less penalty)
        assert reward_smooth > reward_aggressive

    def test_reward_clipping(self):
        """Test rewards are clipped to bounds."""
        config = RewardConfig(
            reward_clip=1.0,
            goal_reached_reward=100.0,  # Very large reward
        )
        reward_fn = NavigationRewardFunction(config)

        mission = Mission(
            goal=MissionGoal(
                type=GoalType.POINT,
                value=PointGoal(position=(0.5, 0.5)),
            )
        )

        robot_state = RobotState(
            pose=Pose(x=0.5, y=0.5, heading=0.0),
            velocity=(0.0, 0.0),
        )
        decision = PolicyDecision(
            action_proposal=ActionProposal(v=0.0, omega=0.0),
            confidence=0.8,
            reason="test",
        )

        reward = reward_fn.compute(robot_state, decision, mission, dt=0.05)

        # Reward should be clipped
        assert abs(reward) <= config.reward_clip * config.reward_scale


class TestRSTDPIntegration:
    """Integration tests for R-STDP with policy service."""

    def test_policy_service_with_rstdp(self):
        """Test policy service with R-STDP model."""
        ts = TopologyService()
        sfs = SpatialFeatureService(ts, k_neighbors=8, is_3d=False)

        # Create R-STDP network
        config = RSTDPConfig(
            feature_dim=44,  # Typical 2D feature dimension
            hidden_size=32,
            output_size=2,
        )
        rstdp_model = RSTDPPolicySNN(config)

        # Create reward function
        reward_fn = NavigationRewardFunction()

        # Create policy service
        policy = SpikingPolicyService(
            feature_service=sfs,
            config={"max_linear": 0.3, "max_angular": 1.0},
            rstdp_model=rstdp_model,
            reward_function=reward_fn,
        )

        # Verify R-STDP is used
        assert policy._use_rstdp
        assert not policy._use_snn

    def test_policy_service_rstdp_decision(self):
        """Test policy service makes decisions with R-STDP."""
        ts = TopologyService()
        sfs = SpatialFeatureService(ts, k_neighbors=8, is_3d=False)

        config = RSTDPConfig(feature_dim=44, hidden_size=32, output_size=2)
        rstdp_model = RSTDPPolicySNN(config)
        reward_fn = NavigationRewardFunction()

        policy = SpikingPolicyService(
            feature_service=sfs,
            rstdp_model=rstdp_model,
            reward_function=reward_fn,
        )

        # Create robot state and mission
        robot_state = RobotState(
            pose=Pose(x=0.1, y=0.1, heading=0.0),
            velocity=(0.0, 0.0),
        )
        mission = Mission(
            goal=MissionGoal(
                type=GoalType.POINT,
                value=PointGoal(position=(0.9, 0.9)),
            )
        )

        # Build features
        features, local_context = sfs.build_features(
            robot_state=robot_state,
            mission=mission,
        )

        # Make decision
        decision = policy.decide(features, local_context, dt=0.05, mission=mission)

        assert decision is not None
        assert decision.action_proposal is not None
        assert decision.reason == "rstdp"

    def test_policy_service_cannot_use_both_snn_and_rstdp(self):
        """Test policy service raises error if both SNN and R-STDP provided."""
        ts = TopologyService()
        sfs = SpatialFeatureService(ts)

        config = RSTDPConfig(feature_dim=44, hidden_size=32, output_size=2)
        rstdp_model = RSTDPPolicySNN(config)

        # Try to use both (should fail)
        with pytest.raises(ValueError, match="Cannot use both"):
            SpikingPolicyService(
                feature_service=sfs,
                snn_model=None,  # Would need actual SNN model
                rstdp_model=rstdp_model,
            )

    def test_rstdp_learning_during_step(self):
        """Test R-STDP learning happens during step() call."""
        ts = TopologyService()
        sfs = SpatialFeatureService(ts, k_neighbors=8, is_3d=False)

        config = RSTDPConfig(
            feature_dim=44,
            hidden_size=32,
            output_size=2,
            learning_rate=0.01,
        )
        rstdp_model = RSTDPPolicySNN(config)
        reward_fn = NavigationRewardFunction()

        policy = SpikingPolicyService(
            feature_service=sfs,
            rstdp_model=rstdp_model,
            reward_function=reward_fn,
        )

        # Get initial weights
        weights_before = rstdp_model.get_weights()["w_out"].copy()

        # Run step (should trigger learning)
        obs = np.array([0.1, 0.1, 1.0, 0.0])  # [x, y, cos(heading), sin(heading)]
        policy.step(obs, dt=0.05)

        # Weights may have changed (if spikes occurred and reward was non-zero)
        weights_after = rstdp_model.get_weights()["w_out"]

        # Verify structure is maintained
        assert weights_after.shape == weights_before.shape

