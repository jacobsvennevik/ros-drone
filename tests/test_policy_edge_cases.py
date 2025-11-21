"""Edge case and validation tests for policy system."""
from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_empty_graph_snapshot():
    """Test handling of empty graph."""
    from hippocampus_core.policy import (
        TopologyService,
        GraphSnapshotMetadata,
        GraphSnapshot,
    )
    
    # Create empty snapshot
    snapshot = GraphSnapshot(
        V=[],
        E=[],
        meta=GraphSnapshotMetadata(
            epoch_id=0,
            frame_id="map",
            stamp=0.0,
            last_updated=0.0,
            update_rate=0.0,
            staleness_warning=False,
        ),
    )
    
    assert len(snapshot.V) == 0
    assert len(snapshot.E) == 0
    return True


def test_feature_vector_edge_cases():
    """Test feature vector edge cases."""
    from hippocampus_core.policy import FeatureVector
    
    # Empty neighbors
    features = FeatureVector(
        goal_ego=[0.5, 0.8, 0.2],
        neighbors_k=[],  # Empty
        topo_ctx=[0.5, 0.3, 0.2],
        safety=[1.0, 1.0, 1.0, 1.0],
    )
    assert features.dim == 3 + 0 + 3 + 4  # 10 features
    
    # With dynamics
    features_dyn = FeatureVector(
        goal_ego=[0.5, 0.8, 0.2],
        neighbors_k=[[0.1, 0.2, 0.3, 0.0]] * 8,
        topo_ctx=[0.5, 0.3, 0.2],
        safety=[1.0, 1.0, 1.0, 1.0],
        dynamics=[0.5, 0.3],
    )
    assert features_dyn.dim > features.dim
    
    return True


def test_goal_validation_edge_cases():
    """Test goal validation edge cases."""
    from hippocampus_core.policy import Mission, MissionGoal, GoalType, PointGoal
    
    # Zero tolerance
    goal = PointGoal(position=(0.9, 0.9), tolerance=0.0)
    mission_goal = MissionGoal(type=GoalType.POINT, value=goal)
    mission = Mission(goal=mission_goal)
    is_valid, error = mission.validate()
    # Zero tolerance might be valid (exact match required)
    
    # Negative timeout
    mission_goal.timeout = -1.0
    is_valid, error = mission.validate()
    assert not is_valid
    assert "timeout" in error.lower()
    
    return True


def test_robot_state_edge_cases():
    """Test robot state edge cases."""
    from hippocampus_core.policy import RobotState
    
    # Minimal state
    state = RobotState(pose=(0.0, 0.0, 0.0))
    assert state.pose == (0.0, 0.0, 0.0)
    assert state.time == 0.0
    
    # With all fields
    state_full = RobotState(
        pose=(0.5, 0.5, 1.57),
        twist=(0.1, 0.0, 0.5),
        health={"sensor_ok": True, "localization_ok": True},
        time=10.0,
        current_node=5,
    )
    assert state_full.current_node == 5
    
    return True


def test_action_proposal_validation():
    """Test action proposal validation."""
    from hippocampus_core.policy import ActionProposal, PolicyDecision
    
    # Valid action
    action = ActionProposal(v=0.2, omega=0.5)
    decision = PolicyDecision(
        action_proposal=action,
        confidence=0.8,
    )
    assert decision.action_proposal.v == 0.2
    assert decision.action_proposal.omega == 0.5
    
    # 3D action
    action_3d = ActionProposal(v=0.2, omega=0.5, vz=0.1)
    assert action_3d.vz == 0.1
    
    return True


def test_safety_flags():
    """Test safety flag combinations."""
    from hippocampus_core.policy import SafeCommand
    
    # All flags false
    cmd1 = SafeCommand(
        cmd=(0.1, 0.2),
        safety_flags={"clamped": False, "slowed": False, "stop": False},
    )
    assert not cmd1.safety_flags["stop"]
    
    # Stop flag
    cmd2 = SafeCommand(
        cmd=(0.0, 0.0),
        safety_flags={"clamped": False, "slowed": False, "stop": True},
    )
    assert cmd2.safety_flags["stop"]
    assert cmd2.cmd == (0.0, 0.0)
    
    return True


def test_staleness_levels():
    """Test staleness detection levels."""
    from hippocampus_core.policy import (
        GraphSnapshot,
        GraphSnapshotMetadata,
        GraphStalenessDetector,
    )
    
    detector = GraphStalenessDetector(
        warning_threshold=2.0,
        stale_threshold=5.0,
        critical_threshold=10.0,
    )
    
    # Fresh
    snapshot_fresh = GraphSnapshot(
        V=[],
        E=[],
        meta=GraphSnapshotMetadata(
            epoch_id=0,
            frame_id="map",
            stamp=1.0,
            last_updated=0.5,  # 0.5s ago
            update_rate=2.0,
            staleness_warning=False,
        ),
    )
    staleness = detector.check_staleness(snapshot_fresh, current_time=1.0)
    assert staleness["staleness_level"] == "fresh"
    
    # Warning
    snapshot_warning = GraphSnapshot(
        V=[],
        E=[],
        meta=GraphSnapshotMetadata(
            epoch_id=0,
            frame_id="map",
            stamp=5.0,
            last_updated=1.0,  # 4s ago
            update_rate=0.25,
            staleness_warning=True,
        ),
    )
    staleness = detector.check_staleness(snapshot_warning, current_time=5.0)
    assert staleness["staleness_level"] == "warning"
    
    # Stale
    snapshot_stale = GraphSnapshot(
        V=[],
        E=[],
        meta=GraphSnapshotMetadata(
            epoch_id=0,
            frame_id="map",
            stamp=10.0,
            last_updated=3.0,  # 7s ago
            update_rate=0.14,
            staleness_warning=True,
        ),
    )
    staleness = detector.check_staleness(snapshot_stale, current_time=10.0)
    assert staleness["staleness_level"] == "stale"
    
    return True


def main():
    """Run edge case tests."""
    print("Policy System Edge Case Tests")
    print("=" * 60)
    
    tests = [
        ("Empty Graph Snapshot", test_empty_graph_snapshot),
        ("Feature Vector Edge Cases", test_feature_vector_edge_cases),
        ("Goal Validation Edge Cases", test_goal_validation_edge_cases),
        ("Robot State Edge Cases", test_robot_state_edge_cases),
        ("Action Proposal Validation", test_action_proposal_validation),
        ("Safety Flags", test_safety_flags),
        ("Staleness Levels", test_staleness_levels),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        try:
            result = test_func()
            if result:
                print(f"  ✅ {name} passed")
                results.append((name, True))
            else:
                print(f"  ❌ {name} failed")
                results.append((name, False))
        except Exception as e:
            print(f"  ❌ {name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Summary:")
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    print("=" * 60)
    
    if all_passed:
        print("✅ All edge case tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

