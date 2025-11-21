#!/usr/bin/env python3
"""Test script to verify ROS 2 integration without full ROS environment."""
from __future__ import annotations

import sys
from pathlib import Path

def test_message_files():
    """Check that message files exist and are valid."""
    print("Testing message files...")
    
    msg_dir = Path(__file__).parent.parent / "ros2_ws" / "src" / "hippocampus_ros2_msgs" / "msg"
    
    expected_msgs = [
        "PolicyDecision.msg",
        "PolicyStatus.msg",
        "MissionGoal.msg",
        "GraphSnapshot.msg",
    ]
    
    all_ok = True
    for msg_file in expected_msgs:
        msg_path = msg_dir / msg_file
        if msg_path.exists():
            # Check basic structure
            with open(msg_path, 'r') as f:
                content = f.read()
                if "#" in content or "float64" in content or "string" in content:
                    print(f"  ✅ {msg_file}")
                else:
                    print(f"  ⚠️  {msg_file} (unusual content)")
                    all_ok = False
        else:
            print(f"  ❌ {msg_file} not found")
            all_ok = False
    
    return all_ok

def test_node_syntax():
    """Test that nodes have valid Python syntax."""
    print("\nTesting node syntax...")
    
    nodes_dir = Path(__file__).parent.parent / "ros2_ws" / "src" / "hippocampus_ros2" / "hippocampus_ros2" / "nodes"
    
    nodes = ["policy_node.py", "mission_publisher.py"]
    all_ok = True
    
    for node_file in nodes:
        node_path = nodes_dir / node_file
        if not node_path.exists():
            print(f"  ❌ {node_file} not found")
            all_ok = False
            continue
        
        try:
            with open(node_path, 'r') as f:
                code = f.read()
            compile(code, node_path, 'exec')
            print(f"  ✅ {node_file} syntax valid")
        except SyntaxError as e:
            print(f"  ❌ {node_file} syntax error: {e}")
            all_ok = False
        except Exception as e:
            print(f"  ⚠️  {node_file} error: {e}")
            all_ok = False
    
    return all_ok

def test_launch_files():
    """Test that launch files exist."""
    print("\nTesting launch files...")
    
    launch_dir = Path(__file__).parent.parent / "ros2_ws" / "src" / "hippocampus_ros2" / "launch"
    
    expected_launches = [
        "policy.launch.py",
        "mission_publisher.launch.py",
    ]
    
    all_ok = True
    for launch_file in expected_launches:
        launch_path = launch_dir / launch_file
        if launch_path.exists():
            print(f"  ✅ {launch_file}")
        else:
            print(f"  ❌ {launch_file} not found")
            all_ok = False
    
    return all_ok

def test_package_structure():
    """Test package structure."""
    print("\nTesting package structure...")
    
    base = Path(__file__).parent.parent
    
    checks = [
        ("Message package", base / "ros2_ws" / "src" / "hippocampus_ros2_msgs"),
        ("Message CMakeLists", base / "ros2_ws" / "src" / "hippocampus_ros2_msgs" / "CMakeLists.txt"),
        ("Message package.xml", base / "ros2_ws" / "src" / "hippocampus_ros2_msgs" / "package.xml"),
        ("Policy node", base / "ros2_ws" / "src" / "hippocampus_ros2" / "hippocampus_ros2" / "nodes" / "policy_node.py"),
        ("Mission publisher", base / "ros2_ws" / "src" / "hippocampus_ros2" / "hippocampus_ros2" / "nodes" / "mission_publisher.py"),
    ]
    
    all_ok = True
    for name, path in checks:
        if path.exists():
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name} not found: {path}")
            all_ok = False
    
    return all_ok

def main():
    """Run all tests."""
    print("ROS 2 Integration Test")
    print("=" * 60)
    
    tests = [
        ("Message Files", test_message_files),
        ("Node Syntax", test_node_syntax),
        ("Launch Files", test_launch_files),
        ("Package Structure", test_package_structure),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} raised exception: {e}")
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
        print("✅ All integration tests passed!")
        print("\nTo build and run:")
        print("  cd ros2_ws")
        print("  colcon build --packages-select hippocampus_ros2_msgs hippocampus_ros2")
        print("  source install/setup.bash")
        print("  ros2 launch hippocampus_ros2 policy.launch.py")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

