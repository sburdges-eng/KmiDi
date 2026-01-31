# KmiDi MIDI Companion — Create DMG & Install Plugins

How to build the macOS app DMG and install VST3/CLAP plugins on your machine.

---

## One command: DMG + plugins

From the **workspace root** (`KmiDi MIDI Companion`):

```bash
chmod +x create_dmg_and_install_plugins.sh
./create_dmg_and_install_plugins.sh
```

This will:

1. **Build the Kelly - iDAW app** (Tauri + React) and create a **DMG** in `KmiDi-compile/dist/`.
2. **Build Kelly VST3/CLAP plugins** and install them into your user plug-in folders:
   - **VST3:** `~/Library/Audio/Plug-Ins/VST3/`
   - **CLAP:** `~/Library/Audio/Plug-Ins/CLAP/`

Options:

- `./create_dmg_and_install_plugins.sh --dmg-only` — only build and create the app DMG (skip plugins).
- `./create_dmg_and_install_plugins.sh --plugins-only` — only build and install plugins (skip DMG).

---

## Step-by-step (run scripts yourself)

### 1) Create the app DMG

```bash
cd KmiDi-compile
chmod +x scripts/create_app_dmg.sh
./scripts/create_app_dmg.sh
```

- Builds the Tauri app and produces a `.dmg` under `src-tauri/target/release/bundle/` (and optionally `dist/` if `COPY_DMG=1`).
- To copy the DMG into `KmiDi-compile/dist/`:

  ```bash
  COPY_DMG=1 ./scripts/create_app_dmg.sh
  ```

### 2) Build and install plugins (VST3 / CLAP)

```bash
cd KmiDi-compile
chmod +x scripts/build_plugins_and_install.sh
./scripts/build_plugins_and_install.sh Release
```

- Requires a CMake build (and Qt6). If `build/` is missing or not set up for plugins, the script configures with `BUILD_PLUGINS=ON` and then builds.
- Installs into:
  - `~/Library/Audio/Plug-Ins/VST3/`
  - `~/Library/Audio/Plug-Ins/CLAP/`
- Removes quarantine and applies an ad-hoc signature so DAWs load the plugins.

---

## Prerequisites

- **DMG:** Node 18+, npm, Rust, and a working Tauri build (see [INSTALL_ALL.md](INSTALL_ALL.md)).
- **Plugins:** CMake 3.27+, Qt6, and a configured CMake build with `BUILD_PLUGINS=ON` (C++/JUCE).

Example (macOS):

```bash
brew install cmake qt@6
export CMAKE_PREFIX_PATH="$(brew --prefix qt@6)"
cd KmiDi-compile
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DBUILD_PLUGINS=ON -DBUILD_DESKTOP=OFF
```

Then run `./scripts/build_plugins_and_install.sh Release`.

---

## Where things end up

| Output          | Location |
|-----------------|----------|
| App DMG         | `KmiDi-compile/dist/*.dmg` (or `src-tauri/target/release/bundle/dmg/*.dmg`) |
| VST3 plugin     | `~/Library/Audio/Plug-Ins/VST3/` (e.g. `Kelly Plugin.vst3`) |
| CLAP plugin     | `~/Library/Audio/Plug-Ins/CLAP/` (e.g. `Kelly Plugin.clap`) |

After installing plugins, rescan your DAW’s plug-in list so it picks them up.

---

## Other DMGs in this repo

- **create_dmg.sh** (workspace root) — Creates a DMG of the **entire project folder** “KmiDi MIDI Companion” (source tree, not the app). Output: `~/KmiDi_MIDI_Companion.dmg`. Use this for backups or sharing the project, not for end-user install.
- **macOS/build_macos_app.sh** (KmiDi-compile) — Builds the **PyInstaller iDAW.app** and can create a DMG if `create-dmg` is installed. That flow is separate from the Tauri “Kelly - iDAW” app.
