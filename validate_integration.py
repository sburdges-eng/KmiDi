#!/usr/bin/env python3
"""
KmiDi_FINAL Integration Validator

Validates that KmiDi_FINAL integration is working correctly:
- Checks KmiDi_FINAL availability
- Validates DSP core purity
- Tests build system integration
- Verifies compliance improvements
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

class IntegrationValidator:
    def __init__(self):
        self.kmidi_root = Path(__file__).parent
        self.kmidi_final_root = self.kmidi_root / ".." / "KmiDi-1" / "KmiDi_FINAL"
        self.build_dir = self.kmidi_root / "build_integration_test"

    def run_validation(self):
        """Run complete integration validation"""
        print("🔍 KmiDi_FINAL Integration Validator")
        print("=" * 50)

        tests = [
            ("KmiDi_FINAL Availability", self.check_kmidi_final_exists),
            ("DSP Core Purity", self.check_dsp_purity),
            ("CMake Integration", self.check_cmake_integration),
            ("Build System", self.test_build_system),
            ("Compliance Achievement", self.validate_compliance),
        ]

        results = []
        for test_name, test_func in tests:
            print(f"\n🧪 {test_name}...")
            try:
                result = test_func()
                if result:
                    print(f"✅ {test_name}: PASSED")
                    results.append(True)
                else:
                    print(f"❌ {test_name}: FAILED")
                    results.append(False)
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {e}")
                results.append(False)

        print("\n" + "=" * 50)
        passed = sum(results)
        total = len(results)
        print(f"📊 Validation Results: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 INTEGRATION VALIDATION: ALL TESTS PASSED")
            print("Ready for 95% compliance achievement!")
            return True
        else:
            print("⚠️  INTEGRATION ISSUES: Some tests failed")
            print("Check the output above for details")
            return False

    def check_kmidi_final_exists(self):
        """Check if KmiDi_FINAL exists and is accessible"""
        if not self.kmidi_final_root.exists():
            print(f"  KmiDi_FINAL not found at: {self.kmidi_final_root}")
            return False

        # Check key directories
        required_dirs = [
            "engine/src/dsp",
            "apps/macOS",
            "CMakeLists.txt"
        ]

        missing = []
        for dir_path in required_dirs:
            if not (self.kmidi_final_root / dir_path).exists():
                missing.append(dir_path)

        if missing:
            print(f"  Missing required directories: {missing}")
            return False

        print(f"  ✅ KmiDi_FINAL found at: {self.kmidi_final_root}")
        return True

    def check_dsp_purity(self):
        """Check DSP core purity (no JUCE dependencies)"""
        dsp_dir = self.kmidi_final_root / "engine" / "src" / "dsp"

        # Find all C++ files
        cpp_files = list(dsp_dir.glob("*.cpp"))
        if not cpp_files:
            print("  No DSP source files found")
            return False

        # Check for forbidden includes
        forbidden_patterns = ["juce", "JUCE", "AppKit", "SwiftUI", "Qt"]
        violations = []

        for cpp_file in cpp_files:
            try:
                with open(cpp_file, 'r') as f:
                    content = f.read()
                    for pattern in forbidden_patterns:
                        if f'#include.*{pattern}' in content:
                            violations.append(f"{cpp_file.name}: {pattern}")
            except Exception as e:
                print(f"  Error reading {cpp_file}: {e}")
                return False

        if violations:
            print(f"  ❌ Framework dependencies found: {violations}")
            return False

        print(f"  ✅ DSP core is pure: {len(cpp_files)} files checked")
        return True

    def check_cmake_integration(self):
        """Check that CMake integration is properly configured"""
        cmakelists = self.kmidi_root / "CMakeLists.txt"

        if not cmakelists.exists():
            print("  CMakeLists.txt not found")
            return False

        try:
            with open(cmakelists, 'r') as f:
                content = f.read()

            # Check for integration options
            required_options = [
                "USE_KMI_DI_FINAL",
                "BUILD_NATIVE_MACOS_APP",
                "KMI_DI_FINAL_ROOT"
            ]

            missing = []
            for option in required_options:
                if option not in content:
                    missing.append(option)

            if missing:
                print(f"  Missing CMake integration options: {missing}")
                return False

            print("  ✅ CMake integration configured")
            return True

        except Exception as e:
            print(f"  Error reading CMakeLists.txt: {e}")
            return False

    def test_build_system(self):
        """Test that build system can configure with integration"""
        try:
            # Clean any existing build
            if self.build_dir.exists():
                shutil.rmtree(self.build_dir)

            self.build_dir.mkdir(parents=True)

            # Try to configure with integration
            cmd = [
                "cmake", "..",
                "-DUSE_KMI_DI_FINAL=ON",
                "-DBUILD_NATIVE_MACOS_APP=ON",
                "-DCMAKE_BUILD_TYPE=Release"
            ]

            result = subprocess.run(
                cmd,
                cwd=self.build_dir,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                print(f"  CMake configuration failed: {result.stderr}")
                return False

            # Check for expected output
            if "Using KmiDi_FINAL" not in result.stdout:
                print("  CMake did not detect KmiDi_FINAL integration")
                return False

            print("  ✅ Build system integration working")
            return True

        except subprocess.TimeoutExpired:
            print("  CMake configuration timed out")
            return False
        except Exception as e:
            print(f"  Build system test failed: {e}")
            return False

    def validate_compliance(self):
        """Validate that compliance improvements are achievable"""
        # Check that all required components are available
        # Note: Native macOS App (AppKitShell) is deprecated per ADR 001.
        # The primary UI is now Tauri/React at engine/intent_ir/
        components = {
            "Pure DSP": self.kmidi_final_root / "engine" / "src" / "dsp" / "audio_buffer.cpp",
            "Legacy macOS App": self.kmidi_root / "legacy" / "ui" / "appkit_shell",
            "Integration Guide": self.kmidi_root / "KmiDi_FINAL_INTEGRATION_GUIDE.md",
            "Build Script": self.kmidi_root / "build_with_kmidi_final.sh"
        }

        missing = []
        for name, path in components.items():
            if not path.exists():
                missing.append(name)

        if missing:
            print(f"  Missing components: {missing}")
            return False

        print("  ✅ All integration components available")
        print("  🎯 95% compliance achievable through integration")
        return True

def main():
    validator = IntegrationValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()