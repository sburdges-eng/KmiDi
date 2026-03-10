# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

KmiDi / iDAW is an AI-powered music creation platform (monorepo). The two **required** services for development are:

| Service | Command | URL |
|---|---|---|
| React frontend (Vite) | `npm run dev` | http://localhost:1420 |
| Music Brain Python API | `npm run dev:python` (or `python3 -m uvicorn music_brain.api:app --reload --port 8000 --host 0.0.0.0` if you need access from outside this VM/container) | http://localhost:8000 |

The Tauri desktop shell, C++ KellyCore engine, Streamlit mixer panel, and Android app are **optional** and require native toolchains not present in the cloud VM.

### Running services

- **Local (with Tauri installed)**: you can start everything together with `npm run dev:all` (uses `concurrently -k`; if the Tauri process exits or fails, the React and Python processes will also be terminated).
- **Cloud VM (no Tauri)**: start the required services separately with `npm run dev` and `npm run dev:python` (do **not** use `npm run dev:all`, because the `-k` flag will stop all processes when Tauri fails in this environment).
- The Python API requires `fastapi`, `uvicorn`, and `pydantic` in addition to the base `pip install -e .` dependencies.
- `$HOME/.local/bin` must be on `PATH` for `uvicorn` to be found when installed with `--user`.

### Lint / test / build

- **TypeScript type-check**: `npx tsc --noEmit`
- **Frontend build**: `npm run build` (runs `tsc && vite build`)
- **Python lint**: `python3 -m flake8 music_brain/ --max-line-length 100` (flake8 is enforced in CI; ensure your changes pass with no new issues, respecting the repo's flake8 configuration and ignore set)
- **Python tests**: `python3 -m pytest tests/` (configured via `pytest.ini`)
- **pytest-timeout** is not installed; do not pass `--timeout` to pytest.

### Gotchas

- No lockfile exists (`package-lock.json`, `yarn.lock`, etc.). `npm install` is the canonical command.
- The `/generate` endpoint uses `GenerateRequest` → `EmotionalIntent` (defined inline in `music_brain/api.py`), **not** `CompleteSongIntentRequest`. At minimum you must provide:
  - `intent.emotional_intent` (required string — mood/emotion description)
  - `intent.technical.genre` (string)
  - `intent.technical.key` (string with space, e.g. `"C major"`)
  - `intent.technical.structure` — list of dicts with `name` matching `^(intro|verse|chorus|bridge|outro|build|drop)$` and `bars` (int)
  - `intent.technical.instruments` — list of **dicts** with an `instrument` key (not plain strings)
  - Minimal working example:
    ```json
    {
      "intent": {
        "emotional_intent": "intimate and powerful pop ballad",
        "core_desire": "emotional pop ballad",
        "technical": {
          "genre": "pop",
          "key": "C major",
          "bpm": 120,
          "structure": [
            { "name": "intro", "bars": 4 },
            { "name": "verse", "bars": 8 },
            { "name": "chorus", "bars": 8 }
          ],
          "instruments": [
            { "instrument": "piano" },
            { "instrument": "bass" },
            { "instrument": "drums" }
          ]
        }
      }
    }
    ```
  - Note: `CompleteSongIntentRequest` in `music_brain/engine_api/schema.py` is a separate strict schema used at the engine boundary, not the `/generate` API payload.
- `bootstrap.sh` tries to `git submodule update --init --recursive` and checks for `external/JUCE/CMakeLists.txt`. The JUCE submodule is large; in cloud VMs the C++ build path is not required so this step can be skipped.
- Vite is configured to bind to `0.0.0.0` (see `vite.config.ts`), so the frontend is accessible from outside the container.

### Reference docs

- `docs/DEVELOPMENT.md` — comprehensive dev guide, workflow details, and debugging tips.
- `docs/ENVIRONMENT.md` — all environment variables and config file layout.
- `BUILD.md` — C++ / CMake / Tauri build instructions.
