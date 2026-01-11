# AI-Enhanced AU Plugins: Architecture, ML Tooling, and Roadmap

This document captures the AI-AU plugin guidance and references the ML bundle located at:

`/Users/seanburdges/Downloads/KELLY_ML_COMPLETE.zip`

Zip contents (unextracted): C++ ML utilities (`src/ml/*.h`), an ML-specific CMake fragment (`CMakeLists_ML.cmake`), an emotion model (`Resources/emotion_model.json`), training scripts (`ml_training/train_emotion_model.py`, requirements), and a teaching guide (`KELLY_ML_TEACHING_GUIDE.md`).

## Summary (from provided write-up)
- **Formats/Frameworks:** AU via JUCE/iPlug2; VST3/CLAP optional. Tauri as companion app (not as plugin GUI). NIH-plug lacks AU; use hybrid Rust DSP via FFI if needed.
- **Plugin targets:** App + shared core + plugins (AU/VST3/CLAP). Use `juce_add_plugin()` with appropriate codes.
- **ML inference:** Real-time-safe engines (RTNeural, Neutone, ONNX, CoreML). Audio thread must be lock-free; use ring buffers to inference threads. PDC for lookahead.
- **MIDI generation:** Compound Word Transformer with valence-arousal conditioning; EMOPIA dataset; continuous conditioning preferred. MidiTok for tokenization.
- **Neural synthesis:** DDSP for fast timbre transfer; RAVE for latent audio FX; diffusion offline only.
- **Recommended stack:** JUCE 7+, RTNeural + ONNX Runtime, DDSP, EMOPIA fine-tuned CWT, Tauri companion app for training/presets.
- **Threading pattern:** Audio Thread ↔ Lock-free ring buffers ↔ Inference threads; avoid allocations/locks on audio thread.
- **Build targets:** AUv2 vs AUv3; ship both if needed. AU component codes: generators (augn), instruments (aumu), effects (aufx), music effects (aufm).

## Suggested integration steps
1) Extract `KELLY_ML_COMPLETE.zip` into a sandbox (e.g., `third_party/ml_bundle/`) and wire `CMakeLists_ML.cmake` for ML sources/resources.
2) Connect ML inference layer (RTNeuralProcessor, EmotionFusionLayer) to plugin DSP; ensure lock-free buffers and PDC reporting.
3) Hook training artifacts: point to `Resources/emotion_model.json` for shipped model; Tauri app for training/export of replacements.
4) Validate real-time behavior: small buffer sizes, no allocations on audio thread; ONNX/CoreML only if RT-safe configuration is proven.

## Key repos (as referenced)
- JUCE, iPlug2, RTNeural, Neutone SDK, anira, DDSP-VST, RAVE, Neural Amp Modeler, NIH-plug (for Rust VST3/CLAP).

Keep this file as reference; do not commit the zip. When ready, extract the ML bundle into a dedicated vendor folder and add it to CMake with explicit include/link paths.


