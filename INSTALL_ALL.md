# KmiDi MIDI Companion — Install Everything

One-place reference for **all** installs: Tauri desktop app, Music Brain API, C++/JUCE build, and Kelly ML.

**Create DMG and install plugins:** see [DISTRIBUTE.md](DISTRIBUTE.md) and run `./create_dmg_and_install_plugins.sh`.

---

## Quick path: run the script

From the **workspace root** (`KmiDi MIDI Companion`):

```bash
chmod +x install_all.sh
./install_all.sh
```

Options:
- `./install_all.sh --skip-system` — you already have Homebrew, Node, Rust, CMake, Qt6
- `./install_all.sh --skip-ml` — you use conda or another Python env for ML

---

## What gets installed (full checklist)

| # | What | How |
|---|------|-----|
| 1 | **Xcode Command Line Tools** | Prompts `xcode-select --install` if missing |
| 2 | **Homebrew** | From official install script |
| 3 | **CMake** (3.27+) | `brew install cmake` |
| 4 | **Qt6** | `brew install qt@6` |
| 5 | **Node.js** (18+) | `brew install node` |
| 6 | **Rust** (rustup + cargo) | `curl ... \| sh` (rustup.rs) |
| 7 | **Python** (3.10+) | `brew install python@3.12` if needed |
| 8 | **KmiDi-compile** | `./scripts/setup_ml_env.sh` + `pip install -e ".[dev]"` + `npm install` |
| 9 | **KmiDi** | `npm install` (+ optional `requirements.txt` in venv) |
| 10 | **C++ / JUCE** | `cmake -B build -S .` (optional configure step) |

---

## After install: how to run everything

### 1) Music Brain API (backend for the desktop app)

```bash
cd KmiDi-compile
source venv/bin/activate    # if you use the venv from setup_ml_env
./scripts/start_music_brain_api.sh
```

Expect it on **http://127.0.0.1:8000**. Check with:

```bash
curl http://127.0.0.1:8000/emotions
```

### 2) Desktop app (Tauri + React)

In a **second** terminal:

```bash
cd KmiDi-compile
npm run tauri dev
```

Or from `KmiDi` if that’s your main app folder:

```bash
cd KmiDi
npm run tauri dev
```

The UI talks to the Music Brain API at `127.0.0.1:8000`.

### 3) C++ / JUCE build (optional)

```bash
cd KmiDi-compile
export CMAKE_PREFIX_PATH="$(brew --prefix qt@6)"
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### 4) Kelly ML training (optional)

```bash
cd KmiDi-compile
source venv/bin/activate
python scripts/train.py --list
# then run your chosen training command
```

---

## Manual install (if you prefer not to use the script)

1. **System (macOS)**  
   - `xcode-select --install`  
   - Install [Homebrew](https://brew.sh), then:  
     `brew install cmake qt@6 node`

2. **Rust**  
   - https://rustup.rs → `curl ... \| sh`

3. **Python**  
   - `brew install python@3.12`  
   - `python3 -m venv venv && source venv/bin/activate`

4. **KmiDi-compile**  
   - `cd KmiDi-compile`  
   - `./scripts/setup_ml_env.sh`  
   - `source venv/bin/activate && pip install -e ".[dev]"`  
   - `npm install`

5. **KmiDi**  
   - `cd KmiDi && npm install`  
   - Optional: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

---

## Troubleshooting

- **“API Offline” in the app** — Start the Music Brain API first (`./scripts/start_music_brain_api.sh` in `KmiDi-compile`).
- **Tauri / Rust errors** — Ensure Xcode CLT and Rust are installed: `xcode-select -p` and `cargo --version`.
- **CMake can’t find Qt6** — Set `CMAKE_PREFIX_PATH`:
  ```bash
  export CMAKE_PREFIX_PATH="$(brew --prefix qt@6)"
  ```
- **Python / torch on Apple Silicon** — The repo’s `requirements.txt` and `setup_ml_env.sh` use CPU/MPS-only PyTorch; avoid installing a CUDA build on macOS.
