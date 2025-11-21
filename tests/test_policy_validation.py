"""Validation tests for policy system - basic checks."""
from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all imports work."""
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
        )
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def test_data_structures():
    """Test data structure creation."""
    try:
        from hippocampus_core.policy import (
            MissionGoal, GoalType, PointGoal, Mission,
            RobotState, FeatureVector,
        )
        
        # Test mission
        goal = PointGoal(position=(0.9, 0.9))
        mission_goal = MissionGoal(type=GoalType.POINT, value=goal)
        mission = Mission(goal=mission_goal)
        
        assert mission.goal.type == GoalType.POINT
        assert mission.goal.value.position == (0.9, 0.9)
        
        # Test robot state
        robot_state = RobotState(pose=(0.5, 0.5, 0.0), time=0.0)
        assert robot_state.pose == (0.5, 0.5, 0.0)
        
        # Test feature vector
        features = FeatureVector(
            goal_ego=[0.5, 0.8, 0.2],
            neighbors_k=[[0.1, 0.2, 0.3, 0.0]] * 8,
            topo_ctx=[0.5, 0.3, 0.2],
            safety=[1.0, 1.0, 1.0, 1.0],
        )
        assert features.dim == 3 + 8*4 + 3 + 4  # 42 features
        
        return True
    except Exception as e:
        print(f"Data structure error: {e}")
        return False

def test_goal_validation():
    """Test goal validation."""
    try:
        from hippocampus_core.policy import Mission, MissionGoal, GoalType, PointGoal
        
        # Valid goal
        goal = PointGoal(position=(0.9, 0.9), tolerance=0.1)
        mission_goal = MissionGoal(type=GoalType.POINT, value=goal)
        mission = Mission(goal=mission_goal)
        
        is_valid, error = mission.validate()
        assert is_valid, f"Mission should be valid: {error}"
        
        # Test goal reached
        current_pose = (0.9, 0.9, 0.0)
        assert mission.goal.is_reached(current_pose)
        
        current_pose_far = (0.0, 0.0, 0.0)
        assert not mission.goal.is_reached(current_pose_far)
        
        return True
    except Exception as e:
        print(f"Goal validation error: {e}")
        return False

def test_feature_vector_operations():
    """Test feature vector operations."""
    try:
        from hippocampus_core.policy import FeatureVector
        import numpy as np
        
        features = FeatureVector(
            goal_ego=[0.5, 0.8, 0.2],
            neighbors_k=[[0.1, 0.2, 0.3, 0.0]] * 8,
            topo_ctx=[0.5, 0.3, 0.2],
            safety=[1.0, 1.0, 1.0, 1.0],
        )
        
        # Test to_array
        feature_array = features.to_array()
        assert feature_array.shape[0] == features.dim
        assert isinstance(feature_array, np.ndarray)
        
        # Test with dynamics
        features_with_dynamics = FeatureVector(
            goal_ego=[0.5, 0.8, 0.2],
            neighbors_k=[[0.1, 0.2, 0.3, 0.0]] * 8,
            topo_ctx=[0.5, 0.3, 0.2],
            safety=[1.0, 1.0, 1.0, 1.0],
            dynamics=[0.5, 0.3],
        )
        assert features_with_dynamics.dim == features.dim + 2
        
        return True
    except Exception as e:
        print(f"Feature vector error: {e}")
        return False

def test_service_initialization():
    """Test that services can be initialized."""
    try:
        from hippocampus_core.policy import (
            TopologyService,
            SpatialFeatureService,
            SpikingPolicyService,
            ActionArbitrationSafety,
        )
        
        # Can create services
        ts = TopologyService()
        sfs = SpatialFeatureService(ts)
        sps = SpikingPolicyService(sfs)
        aas = ActionArbitrationSafety()
        
        assert ts is not None
        assert sfs is not None
        assert sps is not None
        assert aas is not None
        
        return True
    except Exception as e:
        print(f"Service initialization error: {e}")
        return False

def main():
    """Run all validation tests."""
    print("Policy System Validation Tests")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Data Structures", test_data_structures),
        ("Goal Validation", test_goal_validation),
        ("Feature Vector Operations", test_feature_vector_operations),
        ("Service Initialization", test_service_initialization),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        try:
            result = test_func()
            if result:
                print(f"✅ {name} passed")
                results.append((name, True))
            else:
                print(f"❌ {name} failed")
                results.append((name, False))
        except Exception as e:
            print(f"❌ {name} raised exception: {e}")
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
        print("✅ All validation tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

