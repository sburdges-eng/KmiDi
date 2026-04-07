# Microbench for one-way latency on Apple silicon

**Item:** Microbench for one-way latency on Apple silicon  
**Task:** Plan Apple-silicon microbenches that catch latency, stateful-export, and KV-cache regressions early.  
**Secondary tasks:** Tie to `apple-silicon-low-latency.md`, `FULL_STACK_BUILD.md`, and Core ML/ExecuTorch export scripts; inform buffer, QoS, state handling, and cache strategy.  
**Context:** Apple silicon latency; docs/apple-silicon-low-latency.md.  
**Verification status:** PARTIALLY_VERIFIED  
**Verification basis:** repo_scan_only + user_pasted_briefing; repo already has Core ML docs/scripts, but the new KV-cache strategy and regression-harness guidance arrives as briefing text rather than implemented tests.  
**Known facts:**
- Doc exists; sub-10 ms target
- 64/128 samples at 48 kHz; Instruments/Xcode
- Repo already has a stateful Core ML export lane (`scripts/export_llm_coreml.py`) and Core ML export/benchmark helpers for JEPA (`scripts/export_audio_jepa.py`).
- Repo docs already treat KV-cache/stateful MLState paths as important for low-latency Apple-silicon inference.
- User briefing says paged/block KV layout, low-bit KV quantization, priority-aware eviction, and tiered RAM/SSD cache persistence are now major practical levers for long-context on-device inference.
- User briefing includes two concrete regression-harness sketches:
  - a TinyBlock-style ExecuTorch/Core ML state-update loop that checks cache mutation step-to-step and watches for copy-like timing spikes
  - a macOS 15 packaging/load/signing harness that inspects `.mlmodel`, compiles `.mlmodelc`, and verifies runtime load plus codesign/notarization expectations
**Unknowns:** Precise pass/fail thresholds for hidden-copy detection; which CI runners can support macOS 15 hardware validation; whether the first implementation should target LLM KV cache, JEPA state, or both; what "one-way latency" should mean in each harness.  
**Assumptions:** Cache/state regressions should be caught with tiny deterministic fixtures before large-model export or ANE performance work begins.  
**Constraints:** Realtime safety per existing docs; no hidden cloud dependency; export tests should pin shapes and version tuples; packaging tests must be gated on the right macOS runner instead of silently skipping critical coverage.  
**Ambiguities:** Scope of "one-way"; whether timing spikes alone are enough to infer memcpy; how much attestation logic belongs in CI versus org-specific security tooling.  
**Source text / data:** User briefing pasted in full; local repo docs/scripts provide only partial implementation context.  
**Captured guidance:**
- KV-cache management is now a first-class optimization surface, not just a backend detail.
- For small devices, paged KV plus low-bit cache quantization and utility-aware eviction may matter more than further weight quantization.
- Regression harnesses should exercise state mutation correctness before they chase throughput wins.
- Packaging/load verification on macOS 15 should be treated as part of the export lane, not an afterthought.
**Output format:** Doc (procedure/spec), optional pytest harnesses, optional Swift/XCTest load check, and CI gating notes for macOS 15 / Apple-silicon runners.
