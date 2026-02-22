# Building with Xcode

Generate an Xcode project from CMake to build using Xcode's SDK and toolchain. This can help avoid SDK 26.2+ compatibility issues.

---

## Prerequisites

**Xcode must be installed** (not just Command Line Tools):

1. Install Xcode from the App Store (free, ~15GB download)
2. Open Xcode once to accept the license
3. Switch developer directory to Xcode:
   ```bash
   sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
   ```

Verify:
```bash
xcode-select -p
# Should show: /Applications/Xcode.app/Contents/Developer
# NOT: /Library/Developer/CommandLineTools
```

Initialize JUCE submodule:
```bash
cd /Users/seanburdges/Dev/KmiDi
git submodule sync --recursive
git submodule update --init --recursive external/JUCE
```

---

## Generate Xcode Project

```bash
cd KmiDi-compile
./scripts/generate_xcode_project.sh Release
```

This creates `build-xcode/Kelly.xcodeproj`.
Both `./scripts/generate_xcode_project.sh` and `./scripts/configure_xcode.sh` now run
`./scripts/juce/apply_patches.sh` before CMake, so the JUCE macOS window snapshot patch
is applied deterministically.

If you need to apply JUCE patches manually:
```bash
cd /Users/seanburdges/Dev/KmiDi
./scripts/juce/apply_patches.sh
```

---

## Open in Xcode

```bash
open build-xcode/Kelly.xcodeproj
```

Then:
- Select scheme: **Kelly** (or **KellyPlugin_VST3**, **KellyPlugin_CLAP**)
- Select configuration: **Release** or **Debug**
- Press **⌘B** to build, or **⌘R** to run

---

## Build from Command Line

```bash
cd build-xcode

# Build everything
xcodebuild -project Kelly.xcodeproj -scheme Kelly -configuration Release

# Build specific targets
xcodebuild -project Kelly.xcodeproj -target KellyPlugin_VST3 -configuration Release
xcodebuild -project Kelly.xcodeproj -target KellyPlugin_CLAP -configuration Release
xcodebuild -project Kelly.xcodeproj -target KellyApp -configuration Release
```

---

## Why Use Xcode?

- **Better SDK compatibility** — Xcode's SDK may handle `__CLOCK_AVAILABILITY` / `_wchar.h` issues better than Command Line Tools
- **Integrated debugging** — Breakpoints, variable inspection, etc.
- **Visual project management** — See all targets, files, build settings in one place
- **Code signing** — Easier to configure signing for distribution

---

## If Xcode Isn't Installed

You can still build with **CMake + Make/Ninja** (Command Line Tools):

```bash
cd KmiDi-compile
export CMAKE_PREFIX_PATH="$(brew --prefix qt@6)"
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DBUILD_PLUGINS=ON
cmake --build build -j8
```

This uses the Command Line Tools SDK, which may have the SDK 26.2+ issues we've been working around with the `_time.h` wrapper.

---

## Clean-State Checks

Run these checks before submitting build fixes:

```bash
cd /Users/seanburdges/Dev/KmiDi
git status --short
git submodule status
git ls-files -s | rg "KmiDi_PROJECT/external/JUCE" || true
```

Expected:
- no unexpected tracked changes in `git status`
- `git submodule status` succeeds
- no output for `KmiDi_PROJECT/external/JUCE` (legacy gitlink removed)
