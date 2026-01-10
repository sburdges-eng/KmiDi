#!/usr/bin/env python3
"""
Fast test runner that runs tests in parallel and provides a summary.

Usage:
    python3 scripts/run_tests_fast.py [--quick] [--module MODULE]
"""

import sys
import subprocess
import time
from pathlib import Path

TEST_MODULES = [
    ("ML Tests", ["tests/ml/"]),
    ("Music Brain Core", ["tests/music_brain/"]),
    ("DSP Tests", ["tests/dsp/"]),
    ("API Tests", ["tests/api/"]),
    ("Penta Core", ["tests_penta-core/"]),
]


def run_tests(name, paths, quick=False):
    """Run a test module and return results."""
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"{'='*70}")
    
    cmd = ["python3", "-m", "pytest"] + paths + ["-v", "--tb=line"]
    
    if quick:
        cmd.extend(["--maxfail=5", "-x"])  # Stop after 5 failures
    
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per module
        )
        elapsed = time.time() - start
        
        # Parse output
        output = result.stdout + result.stderr
        
        if result.returncode == 0:
            # Extract summary
            for line in output.split('\n'):
                if 'passed' in line or 'failed' in line:
                    print(f"  {line.strip()}")
            print(f"  ✓ Completed in {elapsed:.1f}s")
            return True, elapsed
        else:
            # Show failures
            print(output[-1000:])  # Last 1000 chars
            print(f"  ✗ Failed after {elapsed:.1f}s")
            return False, elapsed
    except subprocess.TimeoutExpired:
        print(f"  ✗ TIMEOUT after 5 minutes")
        return False, 300.0
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False, 0.0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast test runner")
    parser.add_argument("--quick", action="store_true", help="Quick mode (stop after 5 failures)")
    parser.add_argument("--module", help="Run specific module only")
    args = parser.parse_args()
    
    print("="*70)
    print("FAST TEST RUNNER")
    print("="*70)
    print(f"Quick mode: {args.quick}")
    print(f"Python: {sys.version.split()[0]}")
    
    results = []
    total_start = time.time()
    
    for name, paths in TEST_MODULES:
        # Check if paths exist
        existing_paths = [p for p in paths if Path(p).exists()]
        if not existing_paths:
            print(f"\n⚠ Skipping {name} - paths not found")
            continue
        
        # Filter by module if specified
        if args.module and args.module.lower() not in name.lower():
            continue
        
        success, elapsed = run_tests(name, existing_paths, args.quick)
        results.append((name, success, elapsed))
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, s, _ in results if s)
    failed = len(results) - passed
    
    for name, success, elapsed in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}  {name:30s}  {elapsed:6.1f}s")
    
    print(f"\n  Total: {len(results)} modules, {passed} passed, {failed} failed")
    print(f"  Total time: {total_time:.1f}s")
    
    if failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} module(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

