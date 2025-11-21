"""Syntax and basic structure validation for policy system."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

def check_syntax(file_path: Path) -> tuple[bool, str]:
    """Check Python syntax of a file."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_imports(file_path: Path) -> tuple[bool, list[str]]:
    """Check that imports are valid (syntax-wise)."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        tree = ast.parse(source)
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return True, imports
    except Exception as e:
        return False, [f"Error: {e}"]

def main():
    """Run syntax checks."""
    print("Policy System Syntax Validation")
    print("=" * 60)
    
    policy_dir = Path(__file__).parent.parent / "src" / "hippocampus_core" / "policy"
    
    if not policy_dir.exists():
        print(f"❌ Policy directory not found: {policy_dir}")
        return 1
    
    python_files = list(policy_dir.glob("*.py"))
    
    if not python_files:
        print(f"❌ No Python files found in {policy_dir}")
        return 1
    
    print(f"\nFound {len(python_files)} Python files")
    print("-" * 60)
    
    results = []
    for py_file in sorted(python_files):
        if py_file.name == "__pycache__":
            continue
        
        print(f"\nChecking {py_file.name}...")
        
        # Syntax check
        syntax_ok, syntax_msg = check_syntax(py_file)
        if syntax_ok:
            print(f"  ✅ Syntax: OK")
        else:
            print(f"  ❌ Syntax: {syntax_msg}")
        
        # Import check
        imports_ok, imports = check_imports(py_file)
        if imports_ok:
            print(f"  ✅ Imports: {len(imports)} found")
        else:
            print(f"  ❌ Imports: {imports}")
        
        results.append((py_file.name, syntax_ok, imports_ok))
    
    print("\n" + "=" * 60)
    print("Summary:")
    all_ok = True
    for name, syntax_ok, imports_ok in results:
        if syntax_ok and imports_ok:
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name} (syntax: {syntax_ok}, imports: {imports_ok})")
            all_ok = False
    
    print("=" * 60)
    
    if all_ok:
        print("✅ All syntax checks passed!")
        return 0
    else:
        print("❌ Some syntax checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

