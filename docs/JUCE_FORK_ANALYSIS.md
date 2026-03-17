# JUCE Dependency Analysis

Analysis of the JUCE dependency in KmiDi: source, version, patches, and build integration.

---

## 1. Source: Submodule, Not a Fork

JUCE is **not** a fork. It is the **official upstream** repo as a git submodule:

| Item | Value |
|------|--------|
| **Location** | `external/JUCE` |
| **Remote** | `https://github.com/juce-framework/JUCE.git` |
| **Type** | Git submodule (`.gitmodules`: `path = external/JUCE`, `url = https://github.com/juce-framework/JUCE.git`) |
| **Pinned commit** | `0c3fa0a` — *Cleanup: Remove obsolete Android Gradle configuration files from multiple demo projects* (2026-03-10) |
| **Branch context** | Detached from `501c076`; upstream `origin/HEAD` → `origin/master` |
| **Version** | JUCE 8.0.12 (from `juce_StandardHeader.h`: `JUCE_MAJOR_VERSION 8`, `JUCE_BUILDNUMBER 12`) |

No separate fork repo is used; the project tracks a specific commit of upstream JUCE.

---

## 2. Repo Layout (external/JUCE)

- **Root**: `CMakeLists.txt`, `README.md`, `LICENSE.md`, `extras/`, `modules/`, `examples/`, `docs/`, `.github/`
- **Required for KmiDi**: Full clone including **`extras/Build/CMake`** (see root `CMakeLists.txt` and `BUILD.md`). Bootstrap ensures `external/JUCE/CMakeLists.txt` exists after `git submodule update --init --recursive`.

---

## 3. KmiDi-Specific Patches

### 3.1 Git patch (macOS 15 window snapshot)

| Item | Detail |
|------|--------|
| **File** | `third_party/patches/juce/0001-macos15-window-snapshot.patch` |
| **Applicator** | `scripts/juce/apply_patches.sh` (must be run manually; not part of `bootstrap.sh` or `dev-setup.sh`) |
| **Purpose** | macOS 15 SDK compatibility: `CGWindowListCreateImage` is unavailable in macOS 15 SDK. Patch adds `createImageFromCachedContentView()` (view-backed capture) and uses it when `MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_15_0`, otherwise keeps `CGWindowListCreateImage` with deprecation warnings. |
| **Status** | **Applies cleanly** to current submodule commit `0c3fa0a` (verified with `git apply --check`). |
| **If not applied** | Builds against macOS 15 SDK can fail or emit errors where window snapshot is used. |

Apply with:

```bash
./scripts/juce/apply_patches.sh
```

Check without modifying:

```bash
./scripts/juce/apply_patches.sh --check
```

### 3.2 Python patch script (legacy)

| Item | Detail |
|------|--------|
| **File** | `cmake/patch_juce_macos15.py` |
| **Purpose** | In-place edit of `createNSWindowSnapshot` to return `{}` when `__MAC_OS_X_VERSION_MAX_ALLOWED >= 150000` (macOS 15 SDK), avoiding use of `CGWindowListCreateImage`. |
| **Status** | **Obsolete for current JUCE.** The script searches for an older code shape (single block using `CGWindowListCreateImage` without ScreenCaptureKit). Current JUCE at `0c3fa0a` has already been refactored to the ScreenCaptureKit path and a different `#if` layout, so the script’s search string does not match and it exits without changing anything. |
| **Recommendation** | Rely on the git patch above; consider removing or archiving `cmake/patch_juce_macos15.py` to avoid confusion. |

---

## 4. Build Integration

- **CMake** (`CMakeLists.txt`):
  - If `USE_KMI_DI_FINAL` and KmiDi_FINAL path exist: can use JUCE from `KmiDi_FINAL/build/external/JUCE`.
  - Else if `external/JUCE/CMakeLists.txt` exists: uses **local JUCE** with `add_subdirectory(external/JUCE EXCLUDE_FROM_ALL)` and `JUCE_DISABLE_JUCEAIDE_BUILD ON` (skip juceaide for macOS 15 compatibility).
  - Else: configure fails with instructions to clone full JUCE into `external/JUCE`.
- **Options**: `KMIDI_BUILD_JUCE_UI` (default OFF) enables legacy JUCE UI and allows `BUILD_PLUGINS` (VST3/CLAP). With `KMIDI_BUILD_JUCE_UI=OFF`, `BUILD_PLUGINS` is forced OFF.
- **macOS SDK**: Root CMake sets `CMAKE_OSX_DEPLOYMENT_TARGET`, `MAC_OS_X_VERSION_MIN_REQUIRED`, and availability macros; comments reference a fix in `external/JUCE/modules/juce_core/system/juce_StandardHeader.h` for SDK 26.2+ compatibility.

---

## 5. Bootstrap and Patch Workflow

- **Bootstrap** (`bootstrap.sh`): Runs `git submodule update --init --recursive` and checks for `external/JUCE/CMakeLists.txt`. It does **not** run `apply_patches.sh` or the Python patch.
- **Result**: After a fresh clone/setup, JUCE is vanilla at `0c3fa0a` with **no patches applied** unless the user runs `./scripts/juce/apply_patches.sh` manually.
- **Recommendation**: If the project intends to support macOS 15 SDK builds by default, either:
  - Run `scripts/juce/apply_patches.sh` from `bootstrap.sh` after submodule init, or
  - Document in AGENTS.md/BUILD.md that macOS 15 SDK builds require running `./scripts/juce/apply_patches.sh` once after submodule init.

---

## 6. Summary

| Aspect | Conclusion |
|--------|------------|
| **Fork vs upstream** | Upstream JUCE submodule at a single commit; no fork repo. |
| **Version** | JUCE 8.0.12, commit `0c3fa0a`. |
| **Patches** | One relevant git patch (macOS 15 window snapshot); applies cleanly; applied only if user runs `apply_patches.sh`. Python script is legacy and ineffective on current JUCE. |
| **Build** | CMake uses local `external/JUCE` or optional KmiDi_FINAL path; juceaide disabled. |
| **Gap** | Patch application is optional and not automated in bootstrap; consider wiring or documenting it. |
