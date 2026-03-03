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
- **Python lint**: `python3 -m flake8 music_brain/ --max-line-length 100` (flake8 is enforced in CI; ensure your changes pass with no new issues, respecting the repo’s flake8 configuration and ignore set)
- **Python tests**: `python3 -m pytest tests/` (228 tests, configured in `pytest.ini`)
- **pytest-timeout** is not installed; do not pass `--timeout` to pytest.

### Gotchas

- No lockfile exists (`package-lock.json`, `yarn.lock`, etc.). `npm install` is the canonical command.
- The `/generate` endpoint requires `structure` (list of sections with lowercase `name` matching `^(intro|verse|chorus|bridge|outro|build|drop)$`) and `instruments` (non-empty list). See `music_brain/engine_api/schema.py` for the `CompleteSongIntentRequest` Pydantic model.
- `bootstrap.sh` tries to `git submodule update --init --recursive` and checks for `external/JUCE/CMakeLists.txt`. The JUCE submodule is large; in cloud VMs the C++ build path is not required so this step can be skipped.
- Vite is configured to bind to `0.0.0.0` (see `vite.config.ts`), so the frontend is accessible from outside the container.

### Reference docs

- `docs/DEVELOPMENT.md` — comprehensive dev guide, workflow details, and debugging tips.
- `docs/ENVIRONMENT.md` — all environment variables and config file layout.
- `BUILD.md` — C++ / CMake / Tauri build instructions.
