#!/usr/bin/env python3
"""
Master Validation Script for Week11
Runs all validation tests in sequence
"""

import subprocess
import sys
from pathlib import Path

def run_test(script_name, description):
    """Run a test script and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=Path(__file__).parent,
            capture_output=False,
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Run all validation tests"""
    print("="*60)
    print("WEEK11 COMPLETE VALIDATION SUITE")
    print("="*60)

    tests = [
        ("validate_code.py", "Syntax Validation"),
        ("test_structure.py", "Structure Validation"),
    ]

    results = {}

    for script, description in tests:
        success = run_test(script, description)
        results[description] = success

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:<40} {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓ All validation tests passed!")
        print("\nNext steps:")
        print("1. Set up the conda environment: conda env create -f ../environment.yml")
        print("2. Activate environment: conda activate cuda_pyt")
        print("3. Run model tests following TESTING_GUIDE.md")
        return 0
    else:
        print("\n✗ Some validation tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
