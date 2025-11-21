"""Quick check script - verifies basic structure without dependencies."""
from __future__ import annotations

import sys
from pathlib import Path

def check_file_structure():
    """Check that all expected files exist."""
    print("Checking file structure...")
    
    base_dir = Path(__file__).parent.parent
    policy_dir = base_dir / "src" / "hippocampus_core" / "policy"
    
    expected_files = [
        "__init__.py",
        "data_structures.py",
        "topology_service.py",
        "feature_service.py",
        "policy_service.py",
        "safety.py",
        "spike_encoding.py",
        "snn_network.py",
        "decision_decoding.py",
        "temporal_context.py",
    ]
    
    missing = []
    for filename in expected_files:
        file_path = policy_dir / filename
        if file_path.exists():
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename} (missing)")
            missing.append(filename)
    
    return len(missing) == 0, missing

def check_class_definitions():
    """Check that key classes are defined."""
    print("\nChecking class definitions...")
    
    policy_dir = Path(__file__).parent.parent / "src" / "hippocampus_core" / "policy"
    
    expected_classes = {
        "data_structures.py": [
            "FeatureVector",
            "RobotState",
            "Mission",
            "MissionGoal",
            "PolicyDecision",
            "ActionProposal",
            "SafeCommand",
        ],
        "topology_service.py": ["TopologyService"],
        "feature_service.py": ["SpatialFeatureService"],
        "policy_service.py": ["SpikingPolicyService"],
        "safety.py": ["ActionArbitrationSafety", "GraphStalenessDetector"],
    }
    
    all_ok = True
    for filename, classes in expected_classes.items():
        file_path = policy_dir / filename
        if not file_path.exists():
            continue
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        for class_name in classes:
            if f"class {class_name}" in content:
                print(f"  ✅ {filename}: {class_name}")
            else:
                print(f"  ❌ {filename}: {class_name} (not found)")
                all_ok = False
    
    return all_ok

def check_interface_compliance():
    """Check that SpikingPolicyService implements SNNController interface."""
    print("\nChecking interface compliance...")
    
    policy_dir = Path(__file__).parent.parent / "src" / "hippocampus_core" / "policy"
    policy_service_file = policy_dir / "policy_service.py"
    
    if not policy_service_file.exists():
        print("  ❌ policy_service.py not found")
        return False
    
    with open(policy_service_file, 'r') as f:
        content = f.read()
    
    checks = [
        ("class SpikingPolicyService", "Class definition"),
        ("SNNController", "Inherits from SNNController"),
        ("def reset", "reset() method"),
        ("def step", "step() method"),
        ("def decide", "decide() method"),
    ]
    
    all_ok = True
    for check_str, description in checks:
        if check_str in content:
            print(f"  ✅ {description}")
        else:
            print(f"  ❌ {description} (not found)")
            all_ok = False
    
    return all_ok

def main():
    """Run quick checks."""
    print("Policy System Quick Check")
    print("=" * 60)
    
    # File structure
    files_ok, missing = check_file_structure()
    
    # Class definitions
    classes_ok = check_class_definitions()
    
    # Interface compliance
    interface_ok = check_interface_compliance()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Files: {'✅' if files_ok else '❌'}")
    print(f"  Classes: {'✅' if classes_ok else '❌'}")
    print(f"  Interface: {'✅' if interface_ok else '❌'}")
    print("=" * 60)
    
    if files_ok and classes_ok and interface_ok:
        print("✅ All quick checks passed!")
        print("\nNote: For full functionality tests, run with proper environment:")
        print("  python3 tests/test_policy_integration.py")
        return 0
    else:
        print("❌ Some checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

