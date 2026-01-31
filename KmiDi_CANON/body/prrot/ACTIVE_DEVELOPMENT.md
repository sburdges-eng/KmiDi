# 🟡 ACTIVE DEVELOPMENT — prrot / Parrot completeness

**Status:** 🟡 ACTIVE  
**Directory:** `KmiDi_CANON/body/prrot`  
**Color Code:** Yellow/Gold (🟡)

## Purpose

**prrot** — C++ phoneme/voice engine (Kelly body): VoiceProfile, PhonemeSegmenter, ArticulationAnalyzer, SpectralAnalyzer, PitchTracker, MidiShaper, VarianceModeler, BreathDetector, EnvelopeGenerator, AudioValidator. PRROTEngine orchestrates analysis and MIDI shaping for real-time voice synthesis.

## prrot / Parrot module completeness

| Layer | Location | Status | Notes |
|-------|----------|--------|-------|
| **prrot** (C++) | `body/prrot/` | Complete | Full engine: phoneme segmentation, pitch, formants, MIDI shaping, voice profile. |
| **Parrot** (Python) | `brain/music_brain/vocal/parrot.py` | Complete | Voice learning, formant analysis, train/synthesize; used by synthesis, neural_voice, MCP server. |
| **Bridge** | `brain/music_brain/voice/cpp_bridge.py` | Complete | Python → C++ via OSC: load voice model, speak text, phoneme queue, real-time vowel/pitch/formant. |

Parrot (brain) learns voices from audio; prrot (body) renders phonemes/voice profile in real time. Bridge sends Parrot models to C++ for low-latency playback. See `music_brain/vocal/README.md` for Parrot API.

## Migration

Migrated from `KmiDi_FINAL/engine/src/prrot/` on 2026-01-21.
