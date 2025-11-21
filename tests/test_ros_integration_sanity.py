"""Sanity checks for ROS 2 integration - can run without ROS environment."""
from __future__ import annotations

import sys
import ast
from pathlib import Path

def check_ros_node_syntax():
    """Check that ROS nodes have valid Python syntax."""
    print("Checking ROS node syntax...")
    
    ros_nodes_dir = Path(__file__).parent.parent / "ros2_ws" / "src" / "hippocampus_ros2" / "hippocampus_ros2" / "nodes"
    
    if not ros_nodes_dir.exists():
        print(f"  ⚠️  ROS nodes directory not found: {ros_nodes_dir}")
        return False
    
    python_files = list(ros_nodes_dir.glob("*.py"))
    if not python_files:
        print(f"  ⚠️  No Python files found in {ros_nodes_dir}")
        return False
    
    all_ok = True
    for py_file in sorted(python_files):
        try:
            with open(py_file, 'r') as f:
                source = f.read()
            ast.parse(source)
            print(f"  ✅ {py_file.name}")
        except SyntaxError as e:
            print(f"  ❌ {py_file.name}: Syntax error - {e}")
            all_ok = False
        except Exception as e:
            print(f"  ❌ {py_file.name}: Error - {e}")
            all_ok = False
    
    return all_ok

def check_ros_node_structure():
    """Check that ROS nodes have required structure."""
    print("\nChecking ROS node structure...")
    
    ros_nodes_dir = Path(__file__).parent.parent / "ros2_ws" / "src" / "hippocampus_ros2" / "hippocampus_ros2" / "nodes"
    
    expected_nodes = ["brain_node.py", "policy_node.py"]
    all_ok = True
    
    for node_file in expected_nodes:
        node_path = ros_nodes_dir / node_file
        if not node_path.exists():
            print(f"  ❌ {node_file} not found")
            all_ok = False
            continue
        
        with open(node_path, 'r') as f:
            content = f.read()
        
        checks = [
            ("class", "Has class definition"),
            ("Node", "Inherits from Node"),
            ("def __init__", "Has __init__ method"),
            ("def main", "Has main function"),
            ("rclpy.init", "Initializes rclpy"),
            ("rclpy.spin", "Spins node"),
        ]
        
        node_ok = True
        for check_str, description in checks:
            if check_str in content:
                print(f"  ✅ {node_file}: {description}")
            else:
                print(f"  ⚠️  {node_file}: {description} (not found)")
                node_ok = False
        
        if not node_ok:
            all_ok = False
    
    return all_ok

def check_policy_node_imports():
    """Check that policy_node imports required components."""
    print("\nChecking policy_node imports...")
    
    policy_node_path = Path(__file__).parent.parent / "ros2_ws" / "src" / "hippocampus_ros2" / "hippocampus_ros2" / "nodes" / "policy_node.py"
    
    if not policy_node_path.exists():
        print(f"  ❌ policy_node.py not found")
        return False
    
    with open(policy_node_path, 'r') as f:
        content = f.read()
    
    required_imports = [
        "SpikingPolicyService",
        "TopologyService",
        "SpatialFeatureService",
        "ActionArbitrationSafety",
        "GraphNavigationService",
        "RobotState",
        "Mission",
        "rclpy",
        "Node",
    ]
    
    all_ok = True
    for import_name in required_imports:
        if import_name in content:
            print(f"  ✅ Imports {import_name}")
        else:
            print(f"  ❌ Missing import: {import_name}")
            all_ok = False
    
    return all_ok

def check_launch_files():
    """Check that launch files exist and have correct structure."""
    print("\nChecking launch files...")
    
    launch_dir = Path(__file__).parent.parent / "ros2_ws" / "src" / "hippocampus_ros2" / "launch"
    
    expected_launches = ["brain.launch.py", "policy.launch.py"]
    all_ok = True
    
    for launch_file in expected_launches:
        launch_path = launch_dir / launch_file
        if not launch_path.exists():
            print(f"  ❌ {launch_file} not found")
            all_ok = False
            continue
        
        with open(launch_path, 'r') as f:
            content = f.read()
        
        checks = [
            ("generate_launch_description", "Has launch description function"),
            ("LaunchDescription", "Returns LaunchDescription"),
            ("Node", "Creates Node"),
        ]
        
        launch_ok = True
        for check_str, description in checks:
            if check_str in content:
                print(f"  ✅ {launch_file}: {description}")
            else:
                print(f"  ⚠️  {launch_file}: {description} (not found)")
                launch_ok = False
        
        if not launch_ok:
            all_ok = False
    
    return all_ok

def check_config_files():
    """Check that config files exist."""
    print("\nChecking config files...")
    
    config_dir = Path(__file__).parent.parent / "ros2_ws" / "src" / "hippocampus_ros2" / "config"
    
    expected_configs = ["brain.yaml", "policy.yaml"]
    all_ok = True
    
    for config_file in expected_configs:
        config_path = config_dir / config_file
        if config_path.exists():
            print(f"  ✅ {config_file} exists")
        else:
            print(f"  ❌ {config_file} not found")
            all_ok = False
    
    return all_ok

def check_setup_py_entry_points():
    """Check that setup.py has entry points for both nodes."""
    print("\nChecking setup.py entry points...")
    
    setup_py_path = Path(__file__).parent.parent / "ros2_ws" / "src" / "hippocampus_ros2" / "setup.py"
    
    if not setup_py_path.exists():
        print(f"  ❌ setup.py not found")
        return False
    
    with open(setup_py_path, 'r') as f:
        content = f.read()
    
    required_entry_points = [
        "snn_brain_node",
        "policy_node",
    ]
    
    all_ok = True
    for entry_point in required_entry_points:
        if entry_point in content:
            print(f"  ✅ Entry point: {entry_point}")
        else:
            print(f"  ❌ Missing entry point: {entry_point}")
            all_ok = False
    
    return all_ok

def check_policy_node_integration():
    """Check that policy_node integrates with policy system correctly."""
    print("\nChecking policy_node integration...")
    
    policy_node_path = Path(__file__).parent.parent / "ros2_ws" / "src" / "hippocampus_ros2" / "hippocampus_ros2" / "nodes" / "policy_node.py"
    
    if not policy_node_path.exists():
        print(f"  ❌ policy_node.py not found")
        return False
    
    with open(policy_node_path, 'r') as f:
        content = f.read()
    
    integration_checks = [
        ("PlaceCellController", "Uses PlaceCellController"),
        ("TopologyService", "Uses TopologyService"),
        ("SpatialFeatureService", "Uses SpatialFeatureService"),
        ("SpikingPolicyService", "Uses SpikingPolicyService"),
        ("ActionArbitrationSafety", "Uses ActionArbitrationSafety"),
        ("build_features", "Calls build_features"),
        ("decide", "Calls decide"),
        ("filter", "Calls safety filter"),
        ("/odom", "Subscribes to /odom"),
        ("/cmd_vel", "Publishes /cmd_vel"),
    ]
    
    all_ok = True
    for check_str, description in integration_checks:
        if check_str in content:
            print(f"  ✅ {description}")
        else:
            print(f"  ⚠️  {description} (not found)")
            all_ok = False
    
    return all_ok

def main():
    """Run all ROS integration sanity checks."""
    print("ROS 2 Integration Sanity Checks")
    print("=" * 60)
    
    tests = [
        ("ROS Node Syntax", check_ros_node_syntax),
        ("ROS Node Structure", check_ros_node_structure),
        ("Policy Node Imports", check_policy_node_imports),
        ("Launch Files", check_launch_files),
        ("Config Files", check_config_files),
        ("Setup.py Entry Points", check_setup_py_entry_points),
        ("Policy Node Integration", check_policy_node_integration),
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
        print("✅ All ROS integration sanity checks passed!")
        return 0
    else:
        print("❌ Some checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

