"""Check compatibility between policy system and ROS integration."""
from __future__ import annotations

import sys
from pathlib import Path

def check_import_compatibility():
    """Check that policy system can be imported from ROS node."""
    print("Checking import compatibility...")
    
    # Add paths
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        # Try importing policy components
        from hippocampus_core.policy import (
            SpikingPolicyService,
            TopologyService,
            SpatialFeatureService,
            ActionArbitrationSafety,
            GraphNavigationService,
            RobotState,
            Mission,
        )
        print("  ✅ Policy system imports successfully")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def check_interface_compliance():
    """Check that SpikingPolicyService implements SNNController interface."""
    print("\nChecking interface compliance...")
    
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        from hippocampus_core.policy import SpikingPolicyService
        from hippocampus_core.controllers.base import SNNController
        
        # Check inheritance
        if issubclass(SpikingPolicyService, SNNController):
            print("  ✅ SpikingPolicyService inherits from SNNController")
        else:
            print("  ❌ SpikingPolicyService does not inherit from SNNController")
            return False
        
        # Check required methods
        required_methods = ["step", "reset", "decide"]
        for method in required_methods:
            if hasattr(SpikingPolicyService, method):
                print(f"  ✅ Has {method}() method")
            else:
                print(f"  ❌ Missing {method}() method")
                return False
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def check_ros_node_can_import():
    """Check that ROS node can import policy components."""
    print("\nChecking ROS node import capability...")
    
    policy_node_path = Path(__file__).parent.parent / "ros2_ws" / "src" / "hippocampus_ros2" / "hippocampus_ros2" / "nodes" / "policy_node.py"
    
    if not policy_node_path.exists():
        print("  ⚠️  policy_node.py not found (skipping)")
        return True
    
    with open(policy_node_path, 'r') as f:
        content = f.read()
    
    # Check that imports are present
    required_imports = [
        "from hippocampus_core.policy import",
    ]
    
    all_ok = True
    for import_line in required_imports:
        if import_line in content:
            print(f"  ✅ Has policy imports")
        else:
            print(f"  ⚠️  Policy imports not found (may use different pattern)")
            all_ok = False
    
    return all_ok

def check_data_structure_compatibility():
    """Check that data structures are compatible with ROS usage."""
    print("\nChecking data structure compatibility...")
    
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        from hippocampus_core.policy import (
            RobotState,
            Mission,
            MissionGoal,
            GoalType,
            PointGoal,
            PolicyDecision,
            ActionProposal,
            SafeCommand,
        )
        
        # Check that we can create instances
        robot_state = RobotState(pose=(0.0, 0.0, 0.0))
        assert robot_state.pose == (0.0, 0.0, 0.0)
        print("  ✅ RobotState can be created")
        
        goal = PointGoal(position=(1.0, 1.0))
        mission = Mission(goal=MissionGoal(type=GoalType.POINT, value=goal))
        assert mission.goal.type == GoalType.POINT
        print("  ✅ Mission can be created")
        
        action = ActionProposal(v=0.1, omega=0.2)
        assert action.v == 0.1
        print("  ✅ ActionProposal can be created")
        
        decision = PolicyDecision(action_proposal=action, confidence=0.8)
        assert decision.confidence == 0.8
        print("  ✅ PolicyDecision can be created")
        
        safe_cmd = SafeCommand(cmd=(0.1, 0.2))
        assert safe_cmd.cmd == (0.1, 0.2)
        print("  ✅ SafeCommand can be created")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run compatibility checks."""
    print("Policy-ROS Compatibility Checks")
    print("=" * 60)
    
    tests = [
        ("Import Compatibility", check_import_compatibility),
        ("Interface Compliance", check_interface_compliance),
        ("ROS Node Imports", check_ros_node_can_import),
        ("Data Structure Compatibility", check_data_structure_compatibility),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} raised exception: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Summary:")
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    print("=" * 60)
    
    if all_passed:
        print("✅ All compatibility checks passed!")
        return 0
    else:
        print("❌ Some checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

