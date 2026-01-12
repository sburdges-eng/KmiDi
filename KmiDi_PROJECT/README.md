# Kelly - Therapeutic iDAW (Desktop)

Kelly is a therapeutic desktop app that turns emotional intent into music. The current stack pairs a React + Tauri shell (UI + desktop bridge) with a Python “Music Brain” API that serves generation and interrogation endpoints.

## What’s working
- React + Tauri UI renders and routes buttons (Load Emotions, Generate Music, Start Interrogation, Side A/B toggle) to Tauri commands.
- Tauri Rust commands forward to the Music Brain API at `http://127.0.0.1:8000`.
- Error boundary and API status indicator surface connectivity issues.

## Architecture (high level)
- **Frontend:** React (Vite) bundled by Tauri. Lives in `src/` with hooks such as `useMusicBrain`.
- **Desktop bridge:** Tauri 2 Rust commands (`get_emotions`, `generate_music`, `interrogate`) forward HTTP calls to the Music Brain API.
- **Music Brain API (Python):** Expected to run locally on `127.0.0.1:8000`, exposing `/emotions`, `/generate`, and `/interrogate`.

Flow:
```
React UI → Tauri command → HTTP → Music Brain API → JSON response → UI
```

## Prerequisites
- Node 18+ and npm
- Rust toolchain with Cargo (required by Tauri 2 CLI)
- Python 3.9+ with `pip` (virtualenv recommended)

## Setup

From the repository root:

```bash
npm install
python -m pip install -e ".[dev]"
```

## Run (development)
1) Start the Music Brain API server  
   - Preferred: `./scripts/start_music_brain_api.sh`  
   - Default host/port: `127.0.0.1:8000`

2) Launch the desktop app  
   - `npm run tauri dev` (opens the Tauri window; proxies to the dev server on http://localhost:1420)  
   - UI-only iteration (no Tauri shell): `npm run dev` (API calls still target `127.0.0.1:8000`)

3) Smoke-test the API  
```bash
curl http://127.0.0.1:8000/emotions
```
If the call fails, the UI will show “API Offline.”

## Tauri command → API contract
The UI uses the hook from `.agents/handoffs/CURRENT_STATE.md` (mirrored here for quick reference):

| Tauri command       | HTTP call                     | Purpose                       |
|---------------------|-------------------------------|-------------------------------|
| `get_emotions`      | `GET /emotions`               | List available emotions/presets |
| `generate_music`    | `POST /generate`              | Generate music for an intent  |
| `interrogate`       | `POST /interrogate`           | Ask follow-ups / refine intent |

Example payloads:
- `POST /generate`
```json
{
  "intent": {
    "core_wound": "fear of being forgotten",
    "core_desire": "to feel seen",
    "emotional_intent": "anxious but hopeful",
    "technical": {
      "key": "C",
      "bpm": 90,
      "progression": ["I", "V", "vi", "IV"],
      "genre": "indie"
    }
  },
  "output_format": "midi"
}
```

- `POST /interrogate`
```json
{
  "message": "Make it feel more grounded",
  "session_id": "optional-session-id",
  "context": {}
}
```

- `GET /emotions`  
Returns a JSON list/dictionary of available emotions.

## Troubleshooting
- If the UI shows “API Offline,” ensure the Music Brain API server is running on `127.0.0.1:8000`.
- Use `curl http://127.0.0.1:8000/emotions` to verify availability.
- Tauri CLI needs Rust + system toolchain; on macOS install Xcode command line tools (`xcode-select --install`).

## Local Metal AI Orchestrator

This section describes how to set up and run the local AI orchestration layer that integrates Mistral 7B (GGUF), KmiDi Tier-1 MIDI generation, Stable Diffusion 1.5 (MPS), and optional audio diffusion.

### Prerequisites

-   **Python 3.9+** with `pip` (virtualenv recommended)
-   **`llama-cpp-python` with Metal support**:
    ```bash
    CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
    ```
-   (Optional for Image Generation) **`diffusers` and `torch` with MPS support**:
    ```bash
    pip install diffusers transformers torch
    ```
-   (Optional for Audio Generation) **`audiocraft`**:
    ```bash
    pip install audiocraft
    ```
-   **Mistral 7B GGUF model**: Download a `Q4_K_M` or `Q5_K_M` quantized Mistral 7B model (e.g., `mistral-7b-instruct-v0.2.Q5_K_M.gguf`) and place it in a known location (e.g., `./models/mistral-7b-q5_k_m.gguf`).
-   **Stable Diffusion 1.5 model**: The system will attempt to download `runwayml/stable-diffusion-v1-5` if not present.

### Usage

The orchestrator is run via a command-line interface.

```bash
python3 KmiDi_PROJECT/source/python/mcp_workstation/orchestrator.py \
    --llm_model_path "./models/mistral-7b-q5_k_m.gguf" \
    --prompt "Generate a joyful synthwave track with a neon cityscape image and an ethereal pad audio texture." \
    --output_dir "./orchestrator_outputs" \
    --enable_audio_gen
```

**Arguments:**

-   `--llm_model_path`: **(Required)** Path to the Mistral 7B GGUF model file.
-   `--prompt`: **(Required)** Natural language user intent.
-   `--no_image_gen`: **(Optional)** Add this flag to disable image generation.
-   `--enable_audio_gen`: **(Optional)** Add this flag to enable optional audio texture generation.
-   `--output_dir`: **(Optional)** Directory for all generated outputs. Defaults to `./orchestrator_outputs`.

**Example Output Structure:**

Upon successful execution, the `--output_dir` will contain:

-   `midi_outputs/`: Generated MIDI files (e.g., `kmidi_generated_*.mid`)
-   `stable_diffusion_v1_5/`: Downloaded Stable Diffusion model files.
-   `audio_textures/`: Generated audio texture files (if enabled).
-   `final_intent.json`: A JSON file containing the complete `CompleteSongIntent` object with all generated prompts and results.

## License
MIT
## SpectoCloud UI (dev)
- Added a SpectoCloud render panel in the app (calls `/spectocloud/render`).
- Presets: preview/standard/high; controls for mode (static/animation), fps, rotate, anchor density, particles, duration.
- Inputs: JSON MIDI events (textarea or upload a JSON file) or a MIDI file path (backend parses). Shows output path and frames.
- Humanizer config can be fetched via the panel (uses `/config/humanizer`).
- Make sure `python -m music_brain.api` is running locally on 127.0.0.1:8000 when using the panel.
