# Local Orchestration (Metal, Mistral 7B, Tier-1 MIDI)

## Roles and boundaries
- **Text / Brain layer:** Mistral 7B (GGUF Q4_K_M or Q5_K_M) via `llama.cpp` Metal build (`music_brain.intelligence.BrainController`). Responsibilities: intent parsing → structured objects, prompt expansion, MIDI/image/audio prompt scaffolding, explicit explanations, optional rule-breaking suggestions. No media generation.
- **MIDI generation:** Tier-1 deterministic stack (`music_brain.tier1.midi_pipeline.MidiGenerationPipeline`) for melody/harmony/groove/dynamics/humanization. Runs locally, seeded for repeatability. No diffusion/hallucinated audio.
- **Image generation (optional):** Stable Diffusion 1.5 local backend (SDXL excluded by default). Off unless explicitly enabled.
- **Audio texture (optional):** Local diffusion for beds/drones only; offline, non-real-time; never co-scheduled with LLM on 16 GB.
- **Orchestration loop:** `music_brain.orchestration.local_orchestrator.LocalMultiModelOrchestrator` drives sequential execution and explicit load/unload of the LLM to respect 16 GB constraints.

## Run it locally (Metal / llama.cpp)
1) Install `llama-cpp-python` built with Metal:
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install --upgrade --force-reinstall llama-cpp-python
```
2) Download a Mistral 7B GGUF (Q4_K_M or Q5_K_M), e.g. `mistral-7b-instruct-v0.2.Q4_K_M.gguf`.
3) Use the helper script (sequential, local-only):
```bash
python KmiDi_PROJECT/scripts/run_local_orchestrator.py \
  --gguf-path ~/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  --intent "moody triphop at 90 bpm with fragile vocals" \
  --export-midi ~/Music/iDAW_Output/midi/triphop.mid \
  --mistral-ctx 3072 --llama-threads 4 --n-gpu-layers 35 --midi-length 16 --midi-device mps
```
- Optional YAML config: `--config KmiDi_PROJECT/config/orchestration_local_metal.yaml` (fields override defaults; CLI flags override config).
- Default toggles keep images/audio/voice **off**; enable with `--enable-images` / `--enable-audio-texture` / `--enable-voice` if you have local generators ready.

## Resource policy (16 GB Apple Silicon)
- Sequential by default; no background services; no network calls.
- Set thread caps before launching (recommended):
  - `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_MAX_THREADS=4`
  - `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8` to keep Metal VRAM under control.
- Keep batch sizes small; don’t co-run diffusion with the LLM on 16 GB unless scheduled.
 - Prefer `mistral_ctx=3072`, `llama_threads=4`, `n_gpu_layers=35`, `midi_length=16`, `midi_device=mps` for low-heat runs.

## Outputs
- MIDI: exported via `LocalMultiModelOrchestrator.export_midi()` to `~/Music/iDAW_Output/midi/` by default (or `--export-midi` path).
- Images / audio textures / voice outputs: written under `output_root` in their respective subfolders when enabled.

## File map
- Code: `music_brain/orchestration/local_orchestrator.py`, `music_brain/intelligence/brain_controller.py`, `music_brain/tier1/midi_pipeline.py`
- Runner: `KmiDi_PROJECT/scripts/run_local_orchestrator.py`
- Config example: `KmiDi_PROJECT/config/orchestration_local_metal.yaml`
