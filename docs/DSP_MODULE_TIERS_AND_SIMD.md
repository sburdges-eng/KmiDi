# DSP module tiers and SIMD kernels

This document maps **code-size and CPU-heavy** subsystems (embedded-style layering, per Kuan et al., *J. Signal Process. Syst.* 2014) and lists **where SIMD lives** so hot paths stay isolated and optional features can be stripped or disabled at build time.

---

## Tier 0 — Audio callback hard real time

**Rules:** No blocking, no heap growth, no contended locks. Prefer atomics, lock-free queues, and preallocation in `prepareToPlay`.

| Area | Location | Notes |
|------|-----------|--------|
| Callback body | `src/plugin/PluginProcessor.cpp` (`processBlock`) | `try_lock` on MIDI path; lookahead + mono mix preallocated |
| Master EQ | `src/plugin/MasterEQProcessor.cpp` | In-place biquad coefficients; smoothed parameters |
| Latency probe | `include/penta/diagnostics/BlockLatencyInstrument.h` (usage in `PluginProcessor`) | Measures overrun vs buffer budget |

See also: [apple-silicon-low-latency.md](apple-silicon-low-latency.md).

---

## Tier 1 — Lightweight DSP primitives (linked with core)

Small, RT-usable building blocks; keep logic here so Tier 2/3 does not bloat the callback.

| Component | Location | SIMD |
|-----------|----------|------|
| Buffer gain, mix, peak, envelope, ramps | `include/daiw/simd.hpp` (`daiw::simd`) | AVX2 / SSE4.2 / NEON / scalar |
| Planar stereo → mono | `daiw::simd::stereo_planar_to_mono` | Same (used from plugin for emotion runner input) |
| Interleaved stereo → mono | `daiw::simd::stereo_to_mono` | Scalar (interleaved) |
| Scalar DSP helpers | `src/dsp/dsp.cpp` | Uses `daiw/simd.hpp` where included |

**Harmony / chord SIMD (AVX2 when available):** `src_penta-core/harmony/ChordAnalyzerSIMD.cpp` — template scoring and batch match; scalar paths in `ChordAnalyzer.cpp`.

---

## Tier 2 — Heavy logic, not in the innermost sample loop

Large state, parsing, generation, UI. Run on the message thread or dedicated workers; communicate with Tier 0 via atomics and bounded queues.

| Subsystem | Typical location | Guidance |
|-----------|------------------|----------|
| Intent / Kelly brain | `src/engine/`, `IntentPipeline` | Do not call into ML-heavy stacks from `processBlock` without an async boundary |
| MIDI generation | `MidiGenerator`, `midiGenerator_` | Same |
| Project / file I/O | `src/project/` | Never on audio thread |

---

## Tier 3 — Largest binary and runtime cost (optional at CMake level)

These dominate **linked size** and **load time**; treat as optional features where product requirements allow.

| Feature | CMake / code flags | Libraries |
|---------|--------------------|-----------|
| ONNX inference | `ENABLE_ONNX_RUNTIME`, `ONNXRuntime_FOUND` | ONNX Runtime |
| RTNeural | `ENABLE_RTNEURAL` | RTNeural |
| Emotion / JEPA models | `AudioEmotionRunner`, `MultiModelProcessor` | Bundled `.onnx` under `models/` or app Resources |
| Python bridge | `pybind11_FOUND`, `PYTHON_AVAILABLE` | pybind11 + interpreter assumptions |

When targeting **embedded or minimal plug-ins**, disable Tier 3 and keep Tier 0–1 only; keep SIMD kernels in headers or small `.cpp` units so the linker can drop unused objects where applicable.

---

## Adding a new hot kernel

1. Implement in **`daiw::simd`** (sample buffers) or a dedicated `*SIMD.cpp` next to the scalar implementation (domain-specific, e.g. harmony).
2. Provide a **scalar fallback** on the same prototype so all architectures build.
3. Call only from **Tier 0 or Tier 1** code paths that already satisfy RT rules.
4. Document the function in this file or in the header above the function.

---

## References

- Gruhl, *libDsp* — MIT thesis PDF in-repo: `docs/33350097-MIT.pdf`
- Kuan et al., “C++ Support and Applications for Embedded Multicore DSP Systems,” *J. Signal Process. Syst.* 75(2), 2014 (layered libraries, SIMD/DSP APIs)
