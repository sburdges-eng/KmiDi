#!/usr/bin/env python3
"""
Build Verification Script for KmiDi-1

Verifies CMake configuration and build status for C++ components.
"""

import subprocess
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
build_dir = project_root / "build"

def check_cmake():
    """Check if CMake is available."""
    print("Checking CMake...")
    try:
        result = subprocess.run(
            ["cmake", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.split('\n')[0]
        print(f"  ✅ {version}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ❌ CMake not found")
        return False

def check_python_dev():
    """Check if Python development headers are available."""
    print("\nChecking Python development headers...")
    try:
        import sysconfig
        include_dir = sysconfig.get_path('include')
        if Path(include_dir).exists():
            print(f"  ✅ Python headers found at: {include_dir}")
            return True
        else:
            print(f"  ⚠️  Python headers not found at: {include_dir}")
            return False
    except Exception as e:
        print(f"  ⚠️  Could not check Python headers: {e}")
        return False

def check_pybind11():
    """Check if pybind11 is available."""
    print("\nChecking pybind11...")
    try:
        import pybind11
        print(f"  ✅ pybind11 found: {pybind11.__version__}")
        return True
    except ImportError:
        print("  ⚠️  pybind11 not found (pip install pybind11)")
        return False

def check_build_dir():
    """Check if build directory exists and is configured."""
    print("\nChecking build directory...")
    if not build_dir.exists():
        print("  ⚠️  Build directory doesn't exist")
        return False
    
    cmake_cache = build_dir / "CMakeCache.txt"
    if cmake_cache.exists():
        print("  ✅ Build directory is configured")
        
        # Check if penta_core target exists
        compile_commands = build_dir / "compile_commands.json"
        if compile_commands.exists():
            print("  ✅ compile_commands.json found")
        
        return True
    else:
        print("  ⚠️  Build directory exists but not configured")
        return False

def check_penta_core_sources():
    """Check if penta-core source files exist."""
    print("\nChecking penta-core sources...")
    src_dir = project_root / "src_penta-core"
    
    required_dirs = [
        "common",
        "diagnostics",
        "groove",
        "harmony",
        "mixer",
        "osc"
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = src_dir / dir_name
        if dir_path.exists():
            cpp_files = list(dir_path.glob("*.cpp"))
            print(f"  ✅ {dir_name}/ ({len(cpp_files)} .cpp files)")
        else:
            print(f"  ❌ {dir_name}/ not found")
            all_exist = False
    
    return all_exist

def check_bindings():
    """Check if Python bindings source files exist."""
    print("\nChecking Python bindings...")
    bindings_dir = project_root / "bindings"
    
    required_files = [
        "bindings.cpp",
        "harmony_bindings.cpp",
        "groove_bindings.cpp",
        "diagnostics_bindings.cpp",
        "osc_bindings.cpp",
        "ml_bindings.cpp",
    ]
    
    all_exist = True
    for file_name in required_files:
        file_path = bindings_dir / file_name
        if file_path.exists():
            print(f"  ✅ {file_name}")
        else:
            print(f"  ⚠️  {file_name} not found")
            all_exist = False
    
    return all_exist

def check_include_headers():
    """Check if required header files exist."""
    print("\nChecking include headers...")
    include_dir = project_root / "include" / "penta"
    
    required_dirs = [
        "common",
        "diagnostics",
        "groove",
        "harmony",
        "mixer",
        "osc",
        "ml"
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = include_dir / dir_name
        if dir_path.exists():
            h_files = list(dir_path.glob("*.h"))
            print(f"  ✅ penta/{dir_name}/ ({len(h_files)} headers)")
        else:
            print(f"  ⚠️  penta/{dir_name}/ not found")
            all_exist = False
    
    return all_exist

def suggest_build_commands():
    """Print suggested build commands."""
    print("\n" + "=" * 80)
    print("Suggested Build Commands")
    print("=" * 80)
    print()
    print("1. Configure CMake:")
    print("   cd /path/to/<repo-root>")
    print("   cmake -B build -S .")
    print()
    print("2. Build the project:")
    print("   cmake --build build")
    print()
    print("3. Build only penta-core:")
    print("   cmake --build build --target penta_core")
    print()
    print("4. Build Python bindings:")
    print("   cmake --build build --target penta_core_native")
    print()

def main():
    """Run all build verification checks."""
    print("=" * 80)
    print("KmiDi-1 Build Verification")
    print("=" * 80)
    print()
    
    checks = [
        ("CMake", check_cmake),
        ("Python Dev Headers", check_python_dev),
        ("pybind11", check_pybind11),
        ("Build Directory", check_build_dir),
        ("Penta-Core Sources", check_penta_core_sources),
        ("Python Bindings", check_bindings),
        ("Include Headers", check_include_headers),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            success = check_func()
            results.append((check_name, success))
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append((check_name, False))
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for check_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:10} {check_name}")
    
    print()
    print(f"Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("✅ All build prerequisites met!")
        print("\nYou can now run:")
        print("  cmake -B build -S .")
        print("  cmake --build build")
    else:
        print(f"❌ {total - passed} check(s) failed")
        suggest_build_commands()
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
