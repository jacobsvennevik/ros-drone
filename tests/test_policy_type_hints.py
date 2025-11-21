"""Type hint validation for policy system."""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Tuple

def check_type_hints(file_path: Path) -> Tuple[bool, List[str]]:
    """Check for type hints in function signatures."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        tree = ast.parse(source)
        
        issues = []
        functions_without_hints = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function has return type hint
                has_return_hint = node.returns is not None
                
                # Check if parameters have type hints
                params_without_hints = []
                for arg in node.args.args:
                    if arg.annotation is None:
                        params_without_hints.append(arg.arg)
                
                # Public functions (not starting with _) should have hints
                if not node.name.startswith("_"):
                    if not has_return_hint:
                        functions_without_hints.append(f"{node.name} (missing return type)")
                    if params_without_hints:
                        functions_without_hints.append(
                            f"{node.name} (missing param hints: {', '.join(params_without_hints)})"
                        )
        
        return len(functions_without_hints) == 0, functions_without_hints
    except Exception as e:
        return False, [f"Error: {e}"]

def main():
    """Run type hint checks."""
    print("Policy System Type Hint Validation")
    print("=" * 60)
    print("(Checking public functions for type hints)")
    print()
    
    policy_dir = Path(__file__).parent.parent / "src" / "hippocampus_core" / "policy"
    
    if not policy_dir.exists():
        print(f"❌ Policy directory not found: {policy_dir}")
        return 1
    
    python_files = list(policy_dir.glob("*.py"))
    
    print(f"Checking {len(python_files)} files...")
    print("-" * 60)
    
    all_ok = True
    for py_file in sorted(python_files):
        if py_file.name.startswith("__"):
            continue
        
        ok, issues = check_type_hints(py_file)
        if ok:
            print(f"✅ {py_file.name}")
        else:
            print(f"⚠️  {py_file.name}:")
            for issue in issues[:3]:  # Show first 3 issues
                print(f"     - {issue}")
            if len(issues) > 3:
                print(f"     ... and {len(issues) - 3} more")
            all_ok = False
    
    print("=" * 60)
    if all_ok:
        print("✅ All type hint checks passed!")
    else:
        print("⚠️  Some functions missing type hints (non-critical)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

