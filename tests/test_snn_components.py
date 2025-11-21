"""Tests for SNN components."""
from __future__ import annotations

import pytest
import numpy as np

try:
    import torch
    import snntorch
    SNN_AVAILABLE = True
except ImportError:
    SNN_AVAILABLE = False

from hippocampus_core.policy.data_structures import FeatureVector
from hippocampus_core.policy.spike_encoding import FeatureEncoder, EncodingConfig
from hippocampus_core.policy.snn_network import PolicySNN, SNNConfig
from hippocampus_core.policy.decision_decoding import DecisionDecoder
from hippocampus_core.policy.temporal_context import TemporalContext


@pytest.mark.skipif(not SNN_AVAILABLE, reason="PyTorch/snnTorch not available")
def test_feature_encoder_rate():
    """Test rate encoding."""
    features = FeatureVector(
        goal_ego=[0.5, 0.8, 0.2],
        neighbors_k=[[0.1, 0.2, 0.3, 0.0]] * 8,
        topo_ctx=[0.5, 0.3, 0.2],
        safety=[1.0, 1.0, 1.0, 1.0],
    )
    
    config = EncodingConfig(encoding_scheme="rate", num_steps=1)
    encoder = FeatureEncoder(config)
    
    spike_train = encoder.encode(features)
    
    assert spike_train.shape[0] == 1  # num_steps
    assert spike_train.shape[1] == 1  # batch
    assert spike_train.dtype == torch.bool


@pytest.mark.skipif(not SNN_AVAILABLE, reason="PyTorch/snnTorch not available")
def test_policy_snn_forward():
    """Test PolicySNN forward pass."""
    feature_dim = 44  # 2D feature dimension
    model = PolicySNN(
        feature_dim=feature_dim,
        hidden_dim=32,
        output_dim=2,
        beta=0.9,
    )
    
    # Single step
    spike_input = torch.zeros(1, feature_dim)
    membrane = model.init_state(1, torch.device("cpu"))
    
    action, next_membrane = model.forward_step(spike_input, membrane)
    
    assert action.shape == (1, 2)
    assert next_membrane.shape == (1, 32)
    assert torch.all(torch.abs(action) <= 1.0)  # tanh output


@pytest.mark.skipif(not SNN_AVAILABLE, reason="PyTorch/snnTorch not available")
def test_policy_snn_sequence():
    """Test PolicySNN sequence forward."""
    feature_dim = 44
    model = PolicySNN(feature_dim=feature_dim, hidden_dim=32, output_dim=2)
    
    spike_train = torch.zeros(5, 1, feature_dim)
    actions, final_membrane = model.forward_sequence(spike_train)
    
    assert actions.shape == (5, 1, 2)
    assert final_membrane.shape == (1, 32)


@pytest.mark.skipif(not SNN_AVAILABLE, reason="PyTorch/snnTorch not available")
def test_decision_decoder():
    """Test decision decoding."""
    decoder = DecisionDecoder(max_linear=0.3, max_angular=1.0)
    
    # SNN output in [-1, 1] from tanh
    snn_output = torch.tensor([[0.5, -0.3]])
    
    features = FeatureVector(
        goal_ego=[0.5, 0.8, 0.2],
        neighbors_k=[[0.1, 0.2, 0.3, 0.0]] * 8,
        topo_ctx=[0.5, 0.3, 0.2],
        safety=[1.0, 1.0, 1.0, 1.0],
    )
    
    from hippocampus_core.policy.data_structures import LocalContext, Mission, MissionGoal, GoalType, PointGoal
    
    local_context = LocalContext()
    mission = Mission(goal=MissionGoal(type=GoalType.POINT, value=PointGoal((0.9, 0.9))))
    
    decision = decoder.decode(snn_output, features, local_context, mission)
    
    assert decision.action_proposal.v == 0.15  # 0.5 * 0.3
    assert decision.action_proposal.omega == -0.3  # -0.3 * 1.0
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.reason == "snn"


@pytest.mark.skipif(not SNN_AVAILABLE, reason="PyTorch/snnTorch not available")
def test_temporal_context():
    """Test temporal context."""
    context = TemporalContext(history_length=5)
    
    features = FeatureVector(
        goal_ego=[0.5, 0.8, 0.2],
        neighbors_k=[[0.1, 0.2, 0.3, 0.0]] * 8,
        topo_ctx=[0.5, 0.3, 0.2],
        safety=[1.0, 1.0, 1.0, 1.0],
    )
    
    from hippocampus_core.policy.data_structures import PolicyDecision, ActionProposal
    
    decision = PolicyDecision(
        action_proposal=ActionProposal(v=0.1, omega=0.2),
        confidence=0.8,
    )
    
    membrane = torch.zeros(1, 32)
    
    # Update context
    context.update(features, decision, membrane)
    
    # Get temporal features
    temporal_features = context.get_temporal_features()
    assert temporal_features is not None
    assert temporal_features.shape[0] == 1  # time_steps
    assert temporal_features.shape[1] == features.dim  # feature_dim
    
    # Reset
    context.reset()
    assert len(context.feature_history) == 0


@pytest.mark.skipif(not SNN_AVAILABLE, reason="PyTorch/snnTorch not available")
def test_snn_policy_integration():
    """Test SNN integration in policy service."""
    from hippocampus_core.env import Environment
    from hippocampus_core.controllers.place_cell_controller import (
        PlaceCellController,
        PlaceCellControllerConfig,
    )
    from hippocampus_core.policy import (
        TopologyService,
        SpatialFeatureService,
        SpikingPolicyService,
    )
    
    # Setup
    env = Environment(width=1.0, height=1.0)
    config = PlaceCellControllerConfig(num_place_cells=50)
    rng = np.random.default_rng(42)
    place_controller = PlaceCellController(env, config, rng)
    
    # Build some graph
    for _ in range(50):
        place_controller.step(np.array([0.5, 0.5]), dt=0.05)
    
    ts = TopologyService()
    ts.update_from_controller(place_controller)
    sfs = SpatialFeatureService(ts)
    
    # Create SNN model
    feature_dim = 44  # Would be computed from features
    snn_model = PolicySNN(feature_dim=feature_dim, hidden_dim=32, output_dim=2)
    
    # Create policy service with SNN
    sps = SpikingPolicyService(
        sfs,
        config={"feature_dim": feature_dim},
        snn_model=snn_model,
    )
    
    # Test decision
    from hippocampus_core.policy.data_structures import RobotState, Mission, MissionGoal, GoalType, PointGoal
    
    robot_state = RobotState(pose=(0.5, 0.5, 0.0), time=0.0)
    mission = Mission(goal=MissionGoal(type=GoalType.POINT, value=PointGoal((0.9, 0.9))))
    
    features, local_context = sfs.build_features(robot_state, mission)
    decision = sps.decide(features, local_context, dt=0.1)
    
    # Should use SNN (not heuristic)
    assert decision.reason == "snn"
    assert decision.action_proposal.v is not None
    assert decision.action_proposal.omega is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

