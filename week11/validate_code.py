#!/usr/bin/env python3
"""
Code Validation Script for Week11
Validates Python syntax and code structure without requiring full dependencies
"""

import os
import py_compile
import sys
from pathlib import Path

def validate_syntax(file_path):
    """Validate Python syntax of a file"""
    try:
        py_compile.compile(file_path, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)

def main():
    """Main validation function"""
    week11_dir = Path(__file__).parent
    python_files = list(week11_dir.glob("*.py"))

    print("="*60)
    print("WEEK11 CODE VALIDATION")
    print("="*60)
    print(f"Validating {len(python_files)} Python files...\n")

    errors = []
    success = []

    for py_file in sorted(python_files):
        file_name = py_file.name
        valid, error = validate_syntax(str(py_file))

        if valid:
            success.append(file_name)
            print(f"✓ {file_name:<40} PASS")
        else:
            errors.append((file_name, error))
            print(f"✗ {file_name:<40} FAIL")
            print(f"  Error: {error}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total files:    {len(python_files)}")
    print(f"Passed:         {len(success)}")
    print(f"Failed:         {len(errors)}")

    if errors:
        print("\nFailed files:")
        for file_name, error in errors:
            print(f"  - {file_name}")
        sys.exit(1)
    else:
        print("\n✓ All files passed syntax validation!")
        sys.exit(0)

if __name__ == "__main__":
    main()
