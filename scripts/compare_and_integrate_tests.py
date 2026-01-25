#!/usr/bin/env python3
"""
Compare test files between KmiDi and KmiDi-1 and identify unique tests to integrate.
"""
import os
import sys
from pathlib import Path
from typing import Set, List, Tuple

def get_test_files(directory: Path) -> Set[str]:
    """Get all test files from a directory."""
    test_files = set()
    if directory.exists():
        for test_file in directory.rglob("test_*.py"):
            # Get relative path from tests directory
            rel_path = test_file.relative_to(directory)
            test_files.add(str(rel_path))
    return test_files

def main():
    """Compare test files and generate integration report."""
    kmidi_tests_dir = Path("/Users/seanburdges/KmiDi/tests/music_brain")
    kmidi1_tests_dir = Path("/Users/seanburdges/KmiDi-1/tests")
    
    kmidi_tests = get_test_files(kmidi_tests_dir)
    kmidi1_tests = get_test_files(kmidi1_tests_dir)
    
    # Normalize paths - KmiDi-1 has different structure
    kmidi1_normalized = set()
    for test in kmidi1_tests:
        # Convert unit/test_*.py and integration/test_*.py to just test_*.py for comparison
        if "/" in test:
            kmidi1_normalized.add(test.split("/")[-1])
        else:
            kmidi1_normalized.add(test)
    
    kmidi_normalized = {t.split("/")[-1] for t in kmidi_tests}
    
    # Find unique tests in KmiDi
    unique_tests = kmidi_normalized - kmidi1_normalized
    
    print("=" * 80)
    print("Test File Comparison Report")
    print("=" * 80)
    print(f"\nKmiDi test files: {len(kmidi_tests)}")
    print(f"KmiDi-1 test files: {len(kmidi1_tests)}")
    print(f"Unique tests in KmiDi: {len(unique_tests)}")
    
    print("\n" + "=" * 80)
    print("Unique Test Files to Integrate:")
    print("=" * 80)
    for test in sorted(unique_tests):
        print(f"  - {test}")
    
    # Find full paths for unique tests
    print("\n" + "=" * 80)
    print("Full Paths for Integration:")
    print("=" * 80)
    for test_file in sorted(kmidi_tests):
        if test_file.split("/")[-1] in unique_tests:
            full_path = kmidi_tests_dir / test_file
            print(f"  {full_path}")
    
    return len(unique_tests)

if __name__ == "__main__":
    unique_count = main()
    sys.exit(0 if unique_count > 0 else 1)
