# ODR Consolidation — Behavioral-Equivalence Caveats

Companion to `CODEAUDIT_ODR_PAIR_DIFF_2026Q2.md`. The build is provably
ODR-clean after the 2026-04-13 consolidation (no duplicate symbols in the
final dylib, `nm -U | uniq -c | awk '$1>1'` empty). But the consolidation
picked a canonical side per pair — the losing side's implementation is no
longer compiled. For pairs where the two implementations diverged
algorithmically (not just stylistically), switching canonical sides changes
runtime behaviour even though the API is identical.

These are **behavioral-equivalence checks**, not link-correctness checks.
Run them before declaring the consolidation ready for a training freeze or
a plugin release.

---

## 1. `GrooveEngine::processAudio` — analysis cadence

Canonical: `src_penta-core/`.

| Aspect | `src/` (excluded) | `src_penta-core/` (live) |
|---|---|---|
| Tempo update | every ~100 ms (`kAnalysisInterval = 4410`) | on every onset detection |
| Time-sig detection | every ~100 ms | on every onset detection |
| Config race guard | ❌ none | ✅ `configUpdating_` atomic (`memory_order_acquire`) |
| Sub-object construction | default-constructed | Config-forwarded from parent |

**Risk:** penta-core's per-onset analysis is more responsive but also more
CPU-intensive under dense onset streams. A caller relying on the throttled
cadence to reduce jitter will see noisier tempo estimates.

**Spot-check:** feed a 120 BPM click track through `GrooveEngine::processAudio`.
Verify `analysis.currentTempo` converges to 120 ± 2 BPM within 2 seconds.
The penta version should converge faster but potentially with more jitter.

---

## 2. `OnsetDetector::computeSpectralFlux` — windowing & buffer layout

Canonical: `src_penta-core/`.

| Aspect | `src/` (excluded) | `src_penta-core/` (live) |
|---|---|---|
| Windowing | manual `readPtr[i] * window_[i]` | `SIMDKernels::applyWindow()` |
| Flux computation | manual loop with `diff > 0` | `SIMDKernels::spectralFlux()` |
| FFT buffer layout | last `fftSize` samples from buffer | first `min(frames, fftSize)` samples |
| Flux history | ring buffer (`fluxHistoryIndex_++`) | `std::rotate` (shift left) |
| Strength clamp | `juce::jlimit(0, 1, …)` | `std::min(1.0f, …)` |

**Risk:** penta reads samples from the **start** of the buffer; src/ reads from
the **end**. For buffers larger than `fftSize` this changes which audio segment
is analyzed. Onset *timing* stays correct (governed by `sampleCounter_`) but
spectral content may differ subtly for large block sizes.

**Spot-check:** feed a single transient at sample 4095 of a 4096-sample buffer.
Confirm penta's `computeSpectralFlux` still detects it. If `hopSize < frames`,
hop-based chunking covers it; if `hopSize >= frames`, the transient may be
missed.

---

## 3. `HarmonyEngine::updateChordAnalysis` — history dedup logic

Canonical: `src_penta-core/`.

| Aspect | `src/` (excluded) | `src_penta-core/` (live) |
|---|---|---|
| Dedup check | root + quality + full `pitchClass` array | root + quality + `confidence > 0.7` |
| `currentChord_` store order | before history check | after history check |

**Risk:** penta appends a chord to history even if identical to the previous
one, as long as `confidence > 0.7`. The ring buffer fills faster and biases
toward high-confidence repeated chords.

**Spot-check:** play a sustained C major chord for 5 s. Verify
`getChordHistory(10)` does not return 10 identical entries. If it does,
decide whether that's acceptable for downstream emotion/intent consumers.

---

## 4. `OSCServer::OSCListener` — callback threading model

Canonical: `src/`. **No action needed** — caveat listed for completeness.

| Aspect | `src/` (live) | `src_penta-core/` (excluded) |
|---|---|---|
| JUCE callback type | `RealtimeCallback` | `MessageLoopCallback` |
| Loopback validation | ✅ `isLoopbackAddress()` | ❌ none |

src/ is canonical here because it has the loopback check and the RT-safe
callback. Consolidation correctly retained it.

---

## Python bindings smoke test

Minimum post-consolidation check that bindings still import and function:

```python
import penta_core_native as pcn
import numpy as np

engine = pcn.GrooveEngine()
buf = np.zeros(4096, dtype=np.float32)
engine.process_audio(buf)
a = engine.get_analysis()
assert a.current_tempo == 120.0  # default
engine.reset()
```

If any of these fail, the consolidation broke the Python surface — revert
the relevant `list(FILTER … EXCLUDE)` and investigate.

---

## Note on the original caller-driven audit

The `odr_pair_diff.sh` regex was originally too loose — `Class::method(`
matched stdlib template instantiations (`std::sort`, `std::clamp`,
`juce::jlimit`, `juce::MemoryBlock::copyFrom`, `Logger::writeToLog`). For
GrooveEngine and OnsetDetector the unscoped match flagged `std::sort`
instantiations as "src/-only methods," making it look like src/ was
canonical. Manual per-class review (above) showed the real class APIs are
identical — the script bug is now fixed (scoped to filename-derived class),
and only `harmony/VoiceLeading.cpp` remains legitimately caller-driven.
