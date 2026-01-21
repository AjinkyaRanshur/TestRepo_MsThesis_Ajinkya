#!/usr/bin/env python3
"""
Structure Test for Week11 Code
Validates code organization and key functions exist
"""

import os
import ast
import sys
from pathlib import Path

def extract_functions(file_path):
    """Extract function names from a Python file"""
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)

        return functions, classes
    except Exception as e:
        return None, None

def check_file_structure(file_path, expected_functions=None, expected_classes=None):
    """Check if a file has expected functions and classes"""
    functions, classes = extract_functions(file_path)

    if functions is None:
        return False, "Failed to parse file"

    missing_functions = []
    missing_classes = []

    if expected_functions:
        for func in expected_functions:
            if func not in functions:
                missing_functions.append(func)

    if expected_classes:
        for cls in expected_classes:
            if cls not in classes:
                missing_classes.append(cls)

    if missing_functions or missing_classes:
        errors = []
        if missing_functions:
            errors.append(f"Missing functions: {', '.join(missing_functions)}")
        if missing_classes:
            errors.append(f"Missing classes: {', '.join(missing_classes)}")
        return False, '; '.join(errors)

    return True, None

def main():
    """Main test function"""
    week11_dir = Path(__file__).parent

    print("="*60)
    print("WEEK11 CODE STRUCTURE TEST")
    print("="*60)
    print()

    tests = [
        {
            'file': 'run_trajectory_test.py',
            'classes': ['TestConfig'],
            'description': 'Trajectory test runner'
        },
        {
            'file': 'run_pattern_test.py',
            'classes': ['TestConfig'],
            'description': 'Pattern test runner'
        },
        {
            'file': 'run_grid_search_test.py',
            'classes': ['TestConfig'],
            'description': 'Grid search test runner'
        },
        {
            'file': 'test_workflow.py',
            'functions': ['run_trajectory_test'],
            'description': 'Main test workflow'
        },
        {
            'file': 'pattern_testing.py',
            'functions': ['run_pattern_testing', 'print_pattern_testing_summary'],
            'description': 'Pattern testing module'
        },
        {
            'file': 'grid_search_testing.py',
            'functions': ['run_grid_search', 'print_grid_search_summary'],
            'description': 'Grid search testing module'
        },
        {
            'file': 'network.py',
            'classes': ['Net'],
            'description': 'Neural network model'
        },
        {
            'file': 'model_tracking.py',
            'functions': ['get_tracker'],
            'description': 'Model registry tracker'
        },
        {
            'file': 'utils.py',
            'functions': ['clear', 'banner'],
            'description': 'Utility functions'
        },
        {
            'file': 'main.py',
            'functions': ['train_test_loader'],
            'description': 'Main entry point'
        }
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        file_path = week11_dir / test['file']

        if not file_path.exists():
            print(f"✗ {test['file']:<40} MISSING")
            failed += 1
            errors.append((test['file'], "File not found"))
            continue

        expected_functions = test.get('functions', None)
        expected_classes = test.get('classes', None)

        success, error = check_file_structure(
            file_path,
            expected_functions=expected_functions,
            expected_classes=expected_classes
        )

        if success:
            print(f"✓ {test['file']:<40} PASS")
            passed += 1
        else:
            print(f"✗ {test['file']:<40} FAIL")
            print(f"  {error}")
            failed += 1
            errors.append((test['file'], error))

    # Check for data directories
    print()
    print("Data Directories:")
    data_dir = week11_dir / 'data'
    if data_dir.exists():
        print(f"✓ data/                                    EXISTS")
        for item in data_dir.iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
    else:
        print(f"✗ data/                                    MISSING")

    # Check for config directories
    configs_dir = week11_dir / 'configs'
    if configs_dir.exists():
        print(f"✓ configs/                                 EXISTS")
    else:
        print(f"✗ configs/                                 MISSING")

    # Check for plots directory
    plots_dir = week11_dir / 'plots'
    if plots_dir.exists():
        print(f"✓ plots/                                   EXISTS")
    else:
        print(f"  plots/                                   (will be created)")

    # Summary
    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tests:    {len(tests)}")
    print(f"Passed:         {passed}")
    print(f"Failed:         {failed}")

    if errors:
        print("\nFailed tests:")
        for file_name, error in errors:
            print(f"  - {file_name}: {error}")
        sys.exit(1)
    else:
        print("\n✓ All structure tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
