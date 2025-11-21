"""Sanity checks for policy system - can run without pytest."""
from __future__ import annotations

import sys
import numpy as np

# Test imports
print("Testing imports...")
try:
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
        FeatureVector,
        GraphSnapshot,
    )
    print("✅ Basic imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test SNN imports (optional)
try:
    from hippocampus_core.policy import PolicySNN, FeatureEncoder, EncodingConfig
    SNN_AVAILABLE = True
    print("✅ SNN imports successful")
except ImportError:
    SNN_AVAILABLE = False
    print("⚠️  SNN imports not available (PyTorch/snnTorch not installed)")

# Test data structures
print("\nTesting data structures...")
try:
    goal = PointGoal(position=(0.9, 0.9))
    mission_goal = MissionGoal(type=GoalType.POINT, value=goal)
    mission = Mission(goal=mission_goal)
    assert mission.goal.type == GoalType.POINT
    print("✅ Mission data structures work")
except Exception as e:
    print(f"❌ Mission data structure failed: {e}")
    sys.exit(1)

try:
    robot_state = RobotState(pose=(0.5, 0.5, 0.0), time=0.0)
    assert robot_state.pose == (0.5, 0.5, 0.0)
    print("✅ RobotState works")
except Exception as e:
    print(f"❌ RobotState failed: {e}")
    sys.exit(1)

try:
    features = FeatureVector(
        goal_ego=[0.5, 0.8, 0.2],
        neighbors_k=[[0.1, 0.2, 0.3, 0.0]] * 8,
        topo_ctx=[0.5, 0.3, 0.2],
        safety=[1.0, 1.0, 1.0, 1.0],
    )
    assert features.dim > 0
    feature_array = features.to_array()
    assert feature_array.shape[0] == features.dim
    print("✅ FeatureVector works")
except Exception as e:
    print(f"❌ FeatureVector failed: {e}")
    sys.exit(1)

# Test topology service
print("\nTesting topology service...")
try:
    from hippocampus_core.env import Environment
    from hippocampus_core.controllers.place_cell_controller import (
        PlaceCellController,
        PlaceCellControllerConfig,
    )
    
    env = Environment(width=1.0, height=1.0)
    config = PlaceCellControllerConfig(num_place_cells=20)
    rng = np.random.default_rng(42)
    place_controller = PlaceCellController(env, config, rng)
    
    ts = TopologyService()
    ts.update_from_controller(place_controller)
    snapshot = ts.get_graph_snapshot(0.0)
    
    assert len(snapshot.V) == place_controller.get_graph().num_nodes()
    print("✅ TopologyService works")
except Exception as e:
    print(f"❌ TopologyService failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test feature service
print("\nTesting feature service...")
try:
    sfs = SpatialFeatureService(ts, k_neighbors=8)
    robot_state = RobotState(pose=(0.5, 0.5, 0.0), time=0.0)
    mission = Mission(goal=MissionGoal(type=GoalType.POINT, value=PointGoal((0.9, 0.9))))
    
    features, context = sfs.build_features(robot_state, mission)
    
    assert len(features.goal_ego) == 3
    assert len(features.neighbors_k) == 8
    assert len(features.topo_ctx) == 3
    assert len(features.safety) == 4
    print("✅ SpatialFeatureService works")
except Exception as e:
    print(f"❌ SpatialFeatureService failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test policy service (heuristic)
print("\nTesting policy service (heuristic)...")
try:
    sps = SpikingPolicyService(sfs)
    decision = sps.decide(features, context, dt=0.1)
    
    assert decision.action_proposal.v is not None
    assert decision.action_proposal.omega is not None
    assert -0.3 <= decision.action_proposal.v <= 0.3
    assert -1.0 <= decision.action_proposal.omega <= 1.0
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.reason == "heuristic"
    print("✅ SpikingPolicyService (heuristic) works")
except Exception as e:
    print(f"❌ SpikingPolicyService failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test safety arbitration
print("\nTesting safety arbitration...")
try:
    aas = ActionArbitrationSafety(max_linear=0.3, max_angular=1.0)
    graph_snapshot = ts.get_graph_snapshot(robot_state.time)
    safe_cmd = aas.filter(decision, robot_state, graph_snapshot, mission)
    
    assert safe_cmd.cmd[0] is not None
    assert safe_cmd.cmd[1] is not None
    assert -0.3 <= safe_cmd.cmd[0] <= 0.3
    assert -1.0 <= safe_cmd.cmd[1] <= 1.0
    assert isinstance(safe_cmd.safety_flags, dict)
    print("✅ ActionArbitrationSafety works")
except Exception as e:
    print(f"❌ ActionArbitrationSafety failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test SNN components (if available)
if SNN_AVAILABLE:
    print("\nTesting SNN components...")
    try:
        # Test encoder
        encoder = FeatureEncoder(EncodingConfig(encoding_scheme="rate", num_steps=1))
        spike_train = encoder.encode(features)
        assert spike_train.dtype == torch.bool or spike_train.dtype == torch.uint8
        print("✅ FeatureEncoder works")
        
        # Test SNN network
        feature_dim = features.dim
        snn_model = PolicySNN(feature_dim=feature_dim, hidden_dim=32, output_dim=2)
        spike_input = torch.zeros(1, feature_dim)
        membrane = snn_model.init_state(1, torch.device("cpu"))
        action, next_membrane = snn_model.forward_step(spike_input, membrane)
        assert action.shape == (1, 2)
        print("✅ PolicySNN works")
        
        # Test policy service with SNN
        sps_snn = SpikingPolicyService(sfs, snn_model=snn_model)
        decision_snn = sps_snn.decide(features, context, dt=0.1)
        assert decision_snn.reason == "snn"
        print("✅ SpikingPolicyService (SNN) works")
        
    except Exception as e:
        print(f"⚠️  SNN components test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n⚠️  Skipping SNN tests (not available)")

# Test end-to-end pipeline
print("\nTesting end-to-end pipeline...")
try:
    # Run a short simulation
    position = np.array([0.1, 0.1])
    heading = 0.0
    dt = 0.05
    
    for step in range(10):
        # Update controller
        place_controller.step(position, dt)
        if step % 5 == 0:
            ts.update_from_controller(place_controller)
        
        # Policy decision
        robot_state = RobotState(
            pose=(position[0], position[1], heading),
            time=step * dt,
        )
        features, context = sfs.build_features(robot_state, mission)
        decision = sps.decide(features, context, dt)
        
        # Safety filter
        graph_snapshot = ts.get_graph_snapshot(robot_state.time)
        safe_cmd = aas.filter(decision, robot_state, graph_snapshot, mission)
        
        # Apply action (simplified)
        v, omega = safe_cmd.cmd
        heading += omega * dt
        position[0] += v * np.cos(heading) * dt
        position[1] += v * np.sin(heading) * dt
    
    print("✅ End-to-end pipeline works")
except Exception as e:
    print(f"❌ End-to-end pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All sanity checks passed!")
print("=" * 60)

