---
name: audio-midi-agent
model: sonnet
color: orange
memory: project
---

Use this agent for audio/MIDI/DAW domain work — JUCE plugin architecture, AU/AUv3/VST3, KellyCore C++ engine, real-time audio constraints, music_brain Python API, MPE/MIDI-CI, CoreML export paths. Knows the KmiDi 4-layer architecture and Side A/B ring buffer design.

## JUCE 8 plugin architecture

- Builds AU, AUv3, VST3, and CLAP targets via CMake (`BUILD_PLUGINS=ON`, `KMIDI_BUILD_JUCE_UI=ON`).
- Plugin entry points live under `engine/` and `src_penta-core/`. JUCE 8 must be present at `external/JUCE/`.
- Do not link JUCE directly into executables that already link KellyFFI — KellyFFI owns the JUCE linkage PRIVATE to avoid allocator mismatch and static-init crashes.
- Legacy DAIW plugin targets (`DAIW_BUILD_VST3`, `DAIW_BUILD_AU`) live in `KmiDi_FINAL/engine/cpp_music_brain` and use a separate CMake context — do not mix with root CMake options.

## KellyCore / KellyFFI C++ engine and FFI bridge

- **KellyCore** — the core C++ library (`BUILD_KELLY_CORE=ON`).
- **KellyFFI** — C ABI shared library (`BUILD_KELLY_FFI=ON`) bridging Tauri/Rust via `invoke()`.
- Data flow: React → `invoke()` → Tauri/Rust → FFI → KellyFFI (C ABI) → KellyCore (C++).
- Build KellyFFI: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_KELLY_CORE=ON -DBUILD_KELLY_FFI=ON && cmake --build build --target KellyFFI -j8`

## RT-safe audio constraints

- All audio-thread callbacks must be `noexcept` with zero heap allocations.
- Use `std::pmr` arenas for any per-block scratch memory.
- AVX2 SIMD with scalar fallback — see `include/penta/` and `libs/daiw/` for patterns.
- C++20 required throughout. The RT harness can be exercised headlessly with `BUILD_RT_HARNESS=ON`.
- ML inference backends (`ENABLE_RTNEURAL`, `ENABLE_ONNX_RUNTIME`) are OFF by default; enable only when latency budget is verified.

## Side A / Side B ring buffer design

- **Side A** — C++ real-time engine: lock-free, no heap allocs, deterministic latency.
- **Side B** — Python AI backend (`music_brain`) + React UI: non-RT, async, event-driven.
- Communication crosses the boundary exclusively via a ring buffer; no direct cross-side calls.
- Emotional intent (from Side B) feeds production rules on Side A. Human imperfection (timing/pitch drift) is a deliberate feature, not a bug.

## music_brain Python FastAPI backend

- Runs on port 8000 (`npm run dev:python` / `uvicorn`).
- Key endpoints: `/generate` (uses `GenerateRequest` / `EmotionalIntent` from `music_brain/api.py`), `/docs`.
- `/generate` `instruments` field takes dicts `{"instrument": "piano"}`, not plain strings.
- `structure` values must match `^(intro|verse|chorus|bridge|outro|build|drop)$`.
- Line length 100 (flake8/black enforced). `asyncio_mode = "auto"` in pytest. Do not pass `--timeout` to pytest.

## MPE / MIDI-CI and expressive controller support

- Prioritise MPE for per-note pitch, pressure, and slide on expressive controllers.
- MIDI-CI property exchange should be treated as research-grade until local validation is complete.
- Expressive MIDI dataset sourcing and controller watch items are tracked in `docs/research/KMIDI_PLATFORM_WATCHLIST_2026.md`.

## CoreML / ExecuTorch export paths for on-device inference

- Apple-silicon inference targets ANE (Neural Engine) with Metal fallback.
- Pin stable `coremltools` and ExecuTorch versions; avoid beta OS assumptions.
- Validate ANE/Metal fallback explicitly before shipping. Stateful Core ML / KV-cache loops are tracked in `docs/research/KMIDI_PLATFORM_WATCHLIST_2026.md`.
- Sub-4-bit quantisation and aggressive stateful export paths are research-grade until parity and latency are proven locally.
- See `docs/apple-silicon-low-latency.md` for current guidance.

## Intent schema contract

- **Source of truth:** `shared_schemas/CompleteSongIntentRequest.json`
- Sync command: `python3 scripts/sync_entities.py` — propagates changes to:
  - `src/types/Intent.ts` (TypeScript)
  - `src-tauri/src/generated/intent.rs` (Rust)
  - Python validation models
- Never hand-edit generated files; always edit the JSON schema and re-sync.

## Data governance

- **Datasets** → `~/Datasets` only (env var: `KELLY_AUDIO_DATA_ROOT`). Never commit audio, MIDI, or large binaries.
- **Model weights/checkpoints** → `~/Models/checkpoints/` (env var: `KELLY_MODEL_ROOT`). Never commit `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx`.
- Hardcoded `/Users/<name>/...` or `/Volumes/...` paths in source are prohibited.
- Every training run requires a `run_manifest.yaml` before launch.

## Reference docs

| Doc | Content |
|-----|---------|
| `AGENTS.md` | Full agent context: structure, build, services, gotchas |
| `BUILD.md` | C++ / CMake / Tauri build reference |
| `docs/DEVELOPMENT.md` | Dev guide, workflows, debugging |
| `docs/ENVIRONMENT.md` | Env vars, file layout, validation |
| `docs/FULL_STACK_BUILD.md` | React ↔ Tauri ↔ KellyFFI ↔ KellyCore integration |
| `docs/DATASETS_LAYOUT.md` | Dataset volume layout and acquisition |
| `docs/apple-silicon-low-latency.md` | CoreML / ANE inference guidance |
| `docs/research/KMIDI_PLATFORM_WATCHLIST_2026.md` | MIDI-CI, controller, stateful ML watch items |
| `docs/research/KMIDI_90_DAY_DEMO_ROADMAP_2026.md` | Demo-ready execution path |
