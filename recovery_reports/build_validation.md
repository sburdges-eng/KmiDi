# Build Validation

## Environment
- workspace: /tmp/KmiDi_recovery_20260218-043218
- build_dir: /tmp/KmiDi_recovery_20260218-043218/build_out
- timestamp: 2026-02-18T11:41:08Z

## Gate Status
- configure: fail
- juce_vendored_present: present
- juce_gate: pass
- python_step: fail
- node_step: fail

## Target Results
- KellyCore: skip (configure failed)
- KellyPlugin: skip (configure failed)
- KellyApp: skip (configure failed)
- KellyTests: skip (configure failed)
- ctest: skip (configure failed)

## Failure Causes
- CMake configure failed because vendored JUCE is incomplete:
  - missing include target: `external/JUCE/extras/Build/CMake/JUCEModuleSupport.cmake`
  - subsequent CMake command missing: `juce_add_modules`
- Python step (`pip install -e .`) failed due offline dependency resolution (`setuptools>=65` not reachable from package index).
- Node step (`npm install --offline`) failed because required package cache entries were unavailable.

## Notes
- Online fetch to origin/main failed in this environment (DNS resolution for github.com).
- Reconciliation auto-apply of recovered files remains blocked by remote safety gate.

## Command Logs
- configure stdout: /tmp/kmidi_cmake_configure.out
- configure stderr: /tmp/kmidi_cmake_configure.err
- build KellyCore: /tmp/kmidi_build_KellyCore.out /tmp/kmidi_build_KellyCore.err
- build KellyPlugin: /tmp/kmidi_build_KellyPlugin.out /tmp/kmidi_build_KellyPlugin.err
- build KellyApp: /tmp/kmidi_build_KellyApp.out /tmp/kmidi_build_KellyApp.err
- build KellyTests: /tmp/kmidi_build_KellyTests.out /tmp/kmidi_build_KellyTests.err
- ctest: /tmp/kmidi_ctest.out /tmp/kmidi_ctest.err
- python venv/pip: /tmp/kmidi_py_venv.out /tmp/kmidi_py_venv.err /tmp/kmidi_py_pip.out /tmp/kmidi_py_pip.err
- npm install/build: /tmp/kmidi_npm_install.out /tmp/kmidi_npm_install.err /tmp/kmidi_npm_build.out /tmp/kmidi_npm_build.err
