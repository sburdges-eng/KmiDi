# KmiDi 101 — Sections 2–9: By Area

This file walks through each major part of the project: the Brain, the Intent, the path from click to music, the Brain’s inner folders, the desktop app, the C++ engine, the web UI, and how to build and run.

---

## Section 2: The Brain (Python) — Where Your Request Lands

The **Music Brain** is the part of KmiDi that receives your request and turns it into music. It is written in **Python**. Python is used because it is easy to change quickly and works well with AI and data (libraries for emotion, harmony, and so on).

**The front door:** The file **`music_brain/api.py`** is the entry point. It defines one web API (using a library called FastAPI). All the routes (the different “pages” or actions the API offers) live in this same file. There are no separate route modules.

**The main action:** The most important route is **`POST /generate`**. When you click “Generate” in the UI, the request goes to this route. The Brain then checks the body of the request, turns it into an internal “Intent,” and runs the generation logic.

**How the Brain is started:** In development, you run the Brain with:  
`npm run dev:python`  
which runs:  
`python3 -m uvicorn music_brain.api:app --reload --port 8000`  
So the Brain listens on port 8000. The file that gets loaded is **`music_brain/api.py`**, and the object that represents the app is **`app`** inside that file.

---

## Section 3: The Intent — The Form Your Request Must Have

Your request must have a specific shape so that the web UI, the desktop app, and the Python Brain all agree on what you asked for. That shape is called the **Intent**.

**Single source of truth:** The canonical description lives in **`shared_schemas/CompleteSongIntentRequest.json`**. That JSON file describes the fields (e.g. emotional goal, genre, key, structure, instruments) and their types.

**How it gets everywhere:** One script keeps the Intent in sync across three places:

- **TypeScript (web UI):** **`src/types/Intent.ts`** — so the React app knows the shape and can build the request and show errors.
- **Rust (desktop):** **`src-tauri/src/generated/intent.rs`** — so the Tauri app can build the same request when running as a desktop window.
- **Python (Brain):** **`music_brain/engine_api/schema.py`** — defines a Pydantic model **`CompleteSongIntentRequest`** used to validate the body of **`POST /generate`**.

**The script:** **`scripts/sync_entities.py`** reads the Pydantic model and writes the JSON schema, then generates the TypeScript and Rust code. You run it after changing the Intent shape. Tests (e.g. **`tests/unit/test_sync_entities.py`**, **`tests/test_input_validation.py`**) check that the JSON, TypeScript, and Python do not drift apart.

**Why this matters:** So that one change to “what the user can ask for” is done in one place (the schema / Pydantic model), and the rest is generated. That avoids the web, desktop, and Brain getting out of sync.

---

## Section 4: From Click to Music — The Generate Path

This is the step-by-step path from the moment you click Generate to the moment the Brain returns music.

1. **You click Generate** in the web UI. The component that handles the intent form and the button is **`src/components/IntentBuilder.tsx`** (React, TypeScript).

2. **The hook builds the request.** The React app uses **`src/hooks/useMusicBrain.ts`**. The function **`generateFromIntent(intent)`** is called with your intent (of type **`CompleteSongIntentRequest`** from **`src/types/Intent.ts`**). Inside, **`buildGeneratePayload(intent)`** turns that into the body the API expects, then **`generateMusic(payload)`** sends **`POST /generate`** to the Brain (by default `http://127.0.0.1:8000`).

3. **The request hits the Brain.** The Python file **`music_brain/api.py`** defines the route **`@app.post("/generate")`**. The handler receives the body and validates it with **`CompleteSongIntentRequest`** from **`music_brain/engine_api/schema.py`**. If validation fails, the API returns an error (e.g. 422).

4. **Conversion to internal Intent.** The API converts the request into an internal object **`CompleteSongIntent`** (from **`music_brain/session/intent_schema.py`**) via a conversion function, then calls **`api.process_song_intent(complete_intent)`** (where **`api`** is the **`DAiWAPI`** instance in **`music_brain/api.py`**).

5. **Core logic.** **`process_song_intent`** eventually calls **`process_intent(intent)`** in **`music_brain/session/intent_processor.py`**. That function produces chord progressions (with optional rule-breaking), grooves, arrangement, and production guidelines. It uses other parts of **`music_brain`** (structure, harmony, groove, emotion, kelly_companion engines, etc.).

6. **Result.** The result is returned as a dictionary (and then as JSON over HTTP). It contains the generated musical elements (chords, MIDI-ready data, etc.). The UI can then display or play the result.

**In short:** IntentBuilder → useMusicBrain.generateFromIntent → POST /generate → api.py (validate with CompleteSongIntentRequest) → convert to CompleteSongIntent → process_song_intent → process_intent (intent_processor.py) → harmony/groove/engines → MIDI (and related data) back to the client.

---

## Section 5: The Brain’s Inner Rooms (music_brain Folders)

Under the folder **`music_brain/`** there are many subfolders. Each is like a “room” that handles one kind of task. Below is a short list: what the folder is for, and who typically uses it.

| Folder | What it does | Who calls it |
|--------|----------------|--------------|
| **api.py** | The FastAPI app and all routes; front door of the Brain. | Started by uvicorn; called by the web UI and desktop via HTTP. |
| **engine_api** | Request/schema validation at the boundary (e.g. CompleteSongIntentRequest). | api.py (for /generate and other routes). |
| **session** | Intent schema (CompleteSongIntent), intent processor (process_intent), generator, interrogator, teaching. | api.py, intent_bridge, intent.py. |
| **structure** | Song structure: sections, chord progressions, chord analysis, tension, comprehensive engine. | session (intent_processor), api.py. |
| **harmony_utils** / **harmony** | Harmony generation (chord progressions, voicings). | api.py, session, structure. |
| **groove** / **groove_kmidi** | Groove templates, humanization, drum humanizer. | api.py, session. |
| **emotion** / **emotion_kmidi** | Emotion classification, thesaurus, production mapping (emotion → production choices). | session, api.py. |
| **penta_core** | Core rules and engines: harmony rules, groove, DSP, ML, phases; the “rule book” and low-level music logic. | session, structure, other engines. |
| **kelly_companion** | Kelly’s generation engines (bass, melody, pads, strings, rhythm, fills, transitions, etc.). | session, orchestrator. |
| **orchestrator** | Pipeline and multi-engine coordination. | api.py, session. |
| **audio** | Audio analysis, features, I/O. | api.py, emotion, misc_code. |
| **voice** / **vocal** | Voice classification, synthesis, phonemes. | api.py, tier1. |
| **lyrics** | Lyrics handling. | api.py. |
| **visualization** / **video** | Spectocloud, emotion trajectory, video/scene composition. | api.py (e.g. /spectocloud routes). |
| **tier1** / **tier2** | Tiered generators (e.g. voice generator, MIDI generator, LoRA fine-tuner). | session, orchestrator. |
| **utils** | Shared utilities (MIDI I/O, instruments, etc.). | Many modules. |
| **arrangement**, **editing**, **export** | Arrangement logic, editing helpers, export. | session, api.py. |
| **intent_ir** | Intent intermediate representation. | session, engine_api. |
| **learning**, **teaching** | Learning/teaching components. | session. |
| **data_utils** | Data helpers (e.g. emotional mapping presets). | api.py, emotion. |

This list is a map: when you hear “the Brain does harmony,” the code lives in **structure**, **harmony_utils**, and **penta_core**; when you hear “the Brain generates bass and melody,” that is **kelly_companion** and the session’s use of **process_intent**.

---

## Section 6: The Desktop App (Tauri + Rust)

You can run KmiDi in a **browser** (just the web page) or as a **desktop app** (a window on your computer). The desktop app is built with **Tauri**. Tauri is a way to make a small, secure desktop application that uses a web-style UI (here, the same React app) but runs in a native window and can call native code.

**Why Tauri:** It keeps the desktop app small and secure. The UI is still the React app; the “host” that opens the window and talks to the C++ engine is written in **Rust**.

**Where the code lives:** In the folder **`src-tauri/`**. Important files:

- **`src-tauri/src/main.rs`** — Entry point. Registers commands (the actions the React app can ask the desktop to do) and sets up the app.
- **`src-tauri/src/commands.rs`** — Defines the Tauri commands. These include Kelly Brain commands (e.g. generate from text, from emotion, get version) that go through the Rust–C++ bridge, and fallback commands (e.g. generate_music, interrogate) that call the Music Brain API over HTTP.
- **`src-tauri/src/bridge/kelly_ffi.rs`** — Rust side of the bridge to the C++ engine. It talks to the **KellyFFI** shared library (C ABI). So: React → Tauri command → Rust (commands.rs) → kelly_ffi.rs → KellyFFI (C++) → KellyCore (C++ engine).
- **`src-tauri/build.rs`** — Build script. Tells Rust where to find and link the **KellyFFI** library (e.g. **`build/libKellyFFI.dylib`** on macOS).

**When the desktop is used vs when the browser is used:** If you run **`npm run dev`** and open http://localhost:1420 in a browser, only the React app and the Music Brain API (if you started it) are involved; there is no Tauri and no C++ engine. If you run **`npm run dev:tauri`** (or **`npm run dev:all`**), the Tauri desktop window opens; it can still talk to the Music Brain API over HTTP for **`/generate`**, and it can also call the C++ Kelly engine via the Rust bridge for real-time or native features.

---

## Section 7: The C++ Engine (KellyCore)

Some of the real-time audio work is done in **C++**. C++ is used because it is fast and predictable: no automatic “garbage collection” on the audio thread, so timing stays reliable.

**Where the code lives:**

- **`src_penta-core/`** — C++ real-time engine code: groove, harmony, mixer, ML interface, diagnostics, common utilities, OSC. This is the “penta_core” engine.
- **`include/penta/`** — Public C++ headers for that engine.
- **`src/bridge/kelly_ffi.cpp`** and **`src/bridge/kelly_ffi.h`** — The **C** boundary (FFI = Foreign Function Interface). They expose a small C API so that Rust (in **`src-tauri`**) can call into the C++ engine without linking to C++ directly. The Tauri build links to the **KellyFFI** shared library produced from this code.

**How it fits:** The React + Tauri app can call the Music Brain (Python) over HTTP for full song generation. When the desktop app needs the **real-time** C++ engine (e.g. for low-latency playback or plugin-style processing), it goes: Tauri → Rust **kelly_ffi.rs** → **KellyFFI** (C) → **kelly_ffi.cpp** → KellyCore (C++ in **src_penta-core** / **include/penta**). So the C++ engine is optional for the “generate a song” path; it is required for the real-time/desktop/plugin path.

**Build:** The root **`CMakeLists.txt`** defines options like **BUILD_KELLY_CORE**, **BUILD_PLUGINS**, **BUILD_RT_HARNESS**. Building the **KellyFFI** target produces the shared library. See **`docs/FULL_STACK_BUILD.md`** for exact commands.

---

## Section 8: The Web UI (React + TypeScript)

The part you see in the browser (or inside the Tauri window) is the **web UI**. It is built with **React** (a library for building interfaces with components) and **TypeScript** (JavaScript with types). The build tool is **Vite**, which gives fast startup and hot reload during development.

**Why React and Vite:** So developers can change the UI quickly and see updates without a full rebuild. TypeScript helps catch mismatches with the Intent and API (e.g. **`CompleteSongIntentRequest`** in **`src/types/Intent.ts`**).

**Where the code lives:** In the folder **`src/`**. Important pieces:

- **`src/App.tsx`** — Top-level React component; ties the app together.
- **`src/main.tsx`** — Entry point that mounts the React app.
- **`src/components/IntentBuilder.tsx`** — The form where you describe your intent and click Generate. It uses the Music Brain hook to send the request.
- **`src/hooks/useMusicBrain.ts`** — Hook that talks to the Music Brain API: **buildGeneratePayload**, **generateFromIntent**, **POST /generate**, plus emotions, interrogate, humanizer config, spectocloud, lyrics, etc.
- **`src/types/Intent.ts`** — TypeScript types for the Intent (kept in sync with the JSON schema by **scripts/sync_entities.py**).
- **SideA / SideB:** Components under **`src/components/SideA/`** (e.g. VUMeter, Timeline, Mixer, Transport) and **`src/components/SideB/`** (e.g. Interrogator, GhostWriter, EmotionWheel) for the two-sided layout.

**How you run it:** From the repo root, **`npm run dev`** (or **`npm run dev:react`**) starts the Vite dev server. The app is at http://localhost:1420. The **`package.json`** in the root defines **dev**, **dev:react**, **dev:python**, **dev:tauri**, and **dev:all** (all three: React, Tauri, and Music Brain API together).

---

## Section 9: Build and Run — What You Need to Do

**One-time setup (from repo root):**

1. Have **CMake** (3.27+), **Node** (20+), **Rust**, and **Python** (3.11+) installed.
2. Run **`./scripts/dev-setup.sh`**. This runs bootstrap (e.g. JUCE submodule if needed), **npm install**, and **pip install -e .** so the **music_brain** package and tools (like **sync_entities**) are available.

**To run the app:**

- **Web UI only:** **`npm run dev`**. Open http://localhost:1420. (The Brain is not running; some features need the API.)
- **Music Brain API only:** **`npm run dev:python`**. The API runs at http://localhost:8000; docs at http://localhost:8000/docs.
- **Both (typical development):** Run **`npm run dev`** in one terminal and **`npm run dev:python`** in another. Or use **`npm run dev:all`** to start React, Tauri, and the Python API together (if Tauri fails in your environment, start **dev** and **dev:python** separately as in **AGENTS.md**).
- **Desktop app:** **`npm run dev:tauri`** (after the C++ KellyFFI library is built; see **docs/FULL_STACK_BUILD.md**).

**Useful references:**

- **`docs/DEVELOPMENT.md`** — Full development guide, workflows, and troubleshooting.
- **`docs/FULL_STACK_BUILD.md`** — How to build the C++ KellyFFI, plugin, and run the full stack (React, Tauri, C++).
- **`AGENTS.md`** (repo root) — Quick table of required services (React + Music Brain), commands, and gotchas (e.g. **/generate** expects a strict Intent shape; see **CompleteSongIntentRequest**).

That is enough to get the project running and to know where to look when you need more detail.
