# Exporting and quantizing a JEPA encoder for Core ML

**Item:** Exporting and quantizing a JEPA encoder for Core ML  
**Task:** Plan export and quantization of a JEPA encoder for Core ML with explicit state-handling audit.  
**Secondary tasks:** Document encoder source, quantization levels, runtime path, and how mutable state is lowered during export.  
**Context:** Core ML export/quantization; WavJEPA frozen encoder or in-repo JEPA; repo already contains Core ML export wrappers and Apple-silicon notes.  
**Verification status:** VERIFIED_WITH_PRIMARY_SOURCES  
**Verification basis:** official Core ML Tools docs, official PyTorch/ExecuTorch issue trackers, plus local repo scripts/docs.  
**Known facts:**
- Repo already has a stateful Core ML export lane for LLMs in `scripts/export_llm_coreml.py`, and it explicitly forces `--coreml-enable-state` with `--disable_dynamic_shape`.
- Repo also has JEPA-related Core ML export hooks (`scripts/export_audio_jepa.py`) and existing Apple-silicon guidance in `docs/FULL_STACK_BUILD.md` and `docs/apple-silicon-low-latency.md`.
- Core ML Tools now supports stateful models, which changes how iterative state such as KV caches is represented and updated across inference calls.
- Export semantics for mutable state are still an active-risk area: do not assume eager in-place buffer updates survive conversion unchanged.
- Open PyTorch export/compiler issues show dynamic-shape and cache-style mutation paths remain fragile; symbolic-position and slice-update patterns should be treated as export blockers until proven otherwise on the exact toolchain tuple.
- Open ExecuTorch/Core ML issues show continuing friction around dynamic shapes and Apple backend export behavior.
**Unknowns:** Which encoder should be exported first (`wavjepa`, `audio_jepa`, or another JEPA); whether the encoder path needs true mutable state or can stay stateless; acceptable quantization mode; exact encoder and tool licensing; known-good version tuple of PyTorch, Core ML Tools, ExecuTorch, macOS, and Xcode.  
**Assumptions:** Freeze-ready export should prefer fixed shapes, offline compilation, and parity-checked artifacts over flexible runtime graph generation.  
**Constraints:** Encoder remains frozen; no runtime cloud dependency; exports must be deterministic and reproducible; parity tests must compare PyTorch vs exported Core ML on fixed fixtures before release use.  
**Ambiguities:** ANE vs GPU target; ONNX intermediate vs direct conversion; whether KV-cache/stateful export applies only to token/LLM paths or also to the chosen JEPA encoder.  
**Source text / data:**
- Core ML Tools docs: Unified Conversion API and stateful-model guidance
- PyTorch issue tracker: dynamic-shape/export limitations and correctness risks around cache-style mutation
- ExecuTorch issue tracker: dynamic-shape/Core ML backend export limitations
- Local repo: `scripts/export_llm_coreml.py`, `scripts/export_audio_jepa.py`, `docs/FULL_STACK_BUILD.md`, `docs/apple-silicon-low-latency.md`
**Captured guidance:**
- Audit the exported graph for every stateful path; do not trust naive in-place updates to lower correctly.
- Prefer fixed-shape exports and explicit state contracts over dynamic slice-write patterns.
- Pin one known-good PyTorch/Core ML Tools/ExecuTorch/macOS tuple in CI and replay parity fixtures on every toolchain change.
- Treat export failures as both hard-error and silent-correctness risks; benchmark latency only after parity passes.
**Output format:** Design doc, version-pinned export manifest, parity fixture set, benchmark report, and artifact storage outside repo.
