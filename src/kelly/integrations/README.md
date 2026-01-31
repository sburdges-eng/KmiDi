# Kelly / KmiDi — Integration Points

ML ↔ DSP and external-model integration. Single source of truth for what lives here and how it connects to the Brain.

## Modules

| Module | Purpose | Dependencies |
|--------|---------|--------------|
| **magenta_integration.py** | Google Magenta bridge: NoteSequence ↔ mido, VAD → MusicVAE latent, GrooVAE humanization, real-time MIDI. | `note-seq`, `magenta`, `mido`, `numpy`; Python 3.7–3.9 for full Magenta compatibility. |
| **stem_jepa_integration.py** | Stem-JEPA + VAD: emotion-conditioned stem retrieval, FiLM conditioning from VAD, MusicVAE latent compatibility, emotion→stem scoring. | Uses `magenta_integration.VADState` / `VADLatentMapper` when available; numpy. |

## Usage

- Import from `src.kelly.integrations` (or `kelly.integrations` when run from repo root with path set). Magenta is optional; modules guard imports and set `MAGENTA_AVAILABLE` / equivalent.
- Model paths and checkpoints: per governance, use `~/Models` (e.g. `~/Models/checkpoints`); see `docs/DATA_AND_TRAINING.md`.

## Roadmap

Documented in `docs/PROJECT_ROADMAP.md`. Bridge priority: unify ML ↔ DSP and Python ↔ C++ at these boundaries; avoid silos.

### Bridge Opportunities (Next 90 Days)

**1. ML → DSP: Realtime VAD-conditioned synthesis**
- **Current:** VADLatentMapper (magenta_integration) produces emotion latents; Parrot (vocal/parrot.py) and voice synth (voice/synthesizer.py) operate independently.
- **Opportunity:** Wire VAD latents into voice synthesizer for emotion-conditioned realtime vocal synthesis; bridge: `magenta_integration.VADLatentMapper` → `voice.synthesizer.EmotionConditionedSynth`.
- **Benefit:** Unified emotion → voice path; reduce redundant emotion models.

**2. Python → C++: Phoneme/Voice bridge (prrot)**
- **Current:** Python voice modules (vocal/parrot, voice/cpp_bridge) send OSC to C++ prrot engine.
- **Opportunity:** Formalize contract for Python → C++ phoneme/voice data; add type-safe bindings or FFI for low-latency voice control.
- **Benefit:** Reduce OSC latency; stronger contract enforcement; enable bidirectional voice data flow.

**3. ML → DSP: Stem-JEPA emotion retrieval into music_brain arrangement**
- **Current:** stem_jepa_integration scores stems by emotion; music_brain arrangement (session/intent_processor/arrangement_processor) builds sections independently.
- **Opportunity:** Feed Stem-JEPA scored stems into arrangement as emotion-conditioned audio clips; bridge: `stem_jepa_integration.retrieve_stems_for_emotion` → `arrangement_processor.add_emotion_stem_layer`.
- **Benefit:** Emotion-aware stem layering; unify stem retrieval with generative arrangement.

**Document bridge completion:** When a bridge is implemented, update this README with "Bridge: [name] — COMPLETE (YYYY-MM-DD)" and reference target modules.
