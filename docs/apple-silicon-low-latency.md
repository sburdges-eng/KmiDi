# Low-latency tuning on Apple Silicon

Practical plan for ultra-low-latency audio/ML loops on M-series Macs, tuned for KmiDi/JEPA-style streaming (I/O → encode → DSP → out). Target: sub-10 ms end-to-end; OS-level scheduling and inference placement usually dominate over micro-optimizing DSP. For the **encode** step (frozen audio front-end), see [WAVJEPA_LATENT_PIPELINE.md](WAVJEPA_LATENT_PIPELINE.md).

---

## Step 1 — Scheduler lanes (Audio Workgroup + QoS)

**Goal:** Reduce timer jitter, avoid E-core confinement, avoid deadline misses.

- **Join the device Audio Workgroup** so CoreAudio and your threads share a realtime budget.
- **Promote QoS on the critical path** so the audio thread doesn’t land on slow cores.
- **Pin non-realtime helpers** (logging, UI, prefetch) to background/utility QoS so they don’t steal headroom.

### KmiDi integration

- **JUCE:** The host (CoreAudio, AUv3) already calls `AudioProcessor::audioWorkgroupContextChanged(workgroup)` when the device/workgroup is known. Override it in `PluginProcessor`, store the workgroup, and use it for any render-adjacent worker (e.g. a thread that joins via `workgroup.join(token)`).
- **QoS:** Use `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` in render-adjacent workers if you spawn them; the host’s audio callback thread is often already high priority. For helpers (e.g. `RTLogger::processingThread`), use `QOS_CLASS_UTILITY` or `QOS_CLASS_BACKGROUND`.
- **PluginProcessor** already uses `juce::ScopedNoDenormals` in `processBlock` (denormal handling).

### Checklist

- [ ] Audio render callback does **no allocations** and **no locks** (see [EQ note](#realtime-safety-master-eq-implemented) below).
- [ ] UI timers use `NSBackgroundActivityScheduler` or background QoS where applicable.
- [ ] Any `std::future`/thread pools: set explicit QoS per worker (utility/background for non-audio).

---

## Step 2 — Two weekend benchmarks

### (A) Scheduling + buffer sweep

**Vary:**

- I/O buffer: **64 vs 128** samples at 48 kHz (≈ 1.33 ms vs 2.67 ms one-way).
- JEPA hop/stride: **128 vs 256** (when applicable).
- With/without **Audio Workgroup** + QoS promotion.

**Record per take (60–120 s):**

- Median latency and **P95/P99** (render callback start→finish / deadline slack).
- XRuns/dropouts count.
- CPU % per core (watch E-core scheduling).
- **Instruments:** Xcode → Audio template; annotate takes e.g. “64/128, workgroup on/off”.

**Gates:**

- Tail ≤ 8–10 ms at 48 kHz; **dropouts = 0** at 128-sample; “rare but audible” acceptable only at 64-sample.

### (B) Inference offload (ANE vs GPU)

**Prep:**

- Convert streaming model to **Core ML** (coremltools graph fusions + palettization/quantization).
- Export two variants: **ANE-preferred** and **GPU/Metal**.

**Measure (end-to-end):**

- Callback-to-callback latency contribution.
- CPU load and thermals.
- Tail percentiles and dropout incidence.

**Rule of thumb:** Compact streaming nets (JEPA-style) often win on **ANE** for tail stability and CPU relief; validate with the full audio loop, not only microbench.

**KmiDi:** `MLConfig::use_coreml` and penta-core ML bindings already support Core ML; ensure the streaming path uses it and compare ANE vs GPU variants.

## 2026 export/runtime watchlist

Use this as a pinned-risk list for Core ML and ExecuTorch work:

- Prefer stable macOS/iOS builds for validation. User briefings noted Core ML regressions on beta OS builds that break previously working models.
- Pin known-good versions of `coremltools`, ExecuTorch, and deployment targets in CI. Stateful export paths are still sensitive to converter and runtime drift.
- Treat rotary-attention and complex `einsum` lowering as a known risk area. If a model uses those patterns, expect graph rewrites or decomposition before Core ML conversion succeeds.
- Validate both delegated and fallback paths. ANE/Metal fallback behavior can differ when mutable buffers or KV-cache state are involved.
- Keep export tests around exact output parity, not only compile success.

## 2026 optimization levers

- **BNNSGraphBuilder:** prebuild and optimize compute graphs ahead of runtime when the deployment path supports it.
- **Metal 4 explicit timing tools:** use command-buffer groups, events, and timestamp queries so GPU work can be measured against audio deadlines instead of guessed.
- **Activation quantization:** newer Core ML tooling supports calibration-based activation quantization in addition to weight compression; use representative clips to reduce peak memory before shipping.
- **Sub-4-bit compression:** promising for offline experiments and LLM-style helper models, but keep the plugin/runtime path conservative until calibration, accuracy, and export stability are proven on-device.

## GPU scheduling caveat

Audio Workgroups and QoS help the CPU side honor deadlines, but Apple does not expose a hard real-time GPU guarantee. Design for graceful degradation:

- keep CPU or ANE fallback paths available,
- instrument GPU work with timestamps,
- budget for contention from unrelated GPU activity,
- prefer bounded precompiled graphs over ad hoc dispatch in the hot loop.

---

## Step 3 — Housekeeping that protects tails

| Area | Action |
|------|--------|
| **Memory** | Pre-allocate model tensors and scratch; lock pages if possible; avoid ARC/allocation churn on the render path. |
| **I/O** | Move disk/network off the critical path; double-buffer feature frames. |
| **Denormals** | Flush-to-zero; no accidental `powf`/`expf` in the hot loop. (`ScopedNoDenormals` is already used in `processBlock`.) |
| **Vectorization** | Use **Accelerate/vDSP** for small FIR/IIR blocks; keep SIMD-friendly (e.g. 16-byte alignment). |

### vDSP biquad (denormal-safe) sketch

```cpp
#include <Accelerate/Accelerate.h>
void process(float* in, float* out, int n, vDSP_biquad_Setup s, float* z) {
  vDSP_deq22(in, 1, s->biquadCoefficients, out, 1, n);
  // or vDSP_biquadm for multi-section; ensure FTZ/DAZ set.
}
```

### Pin helper thread to background QoS

```cpp
pthread_t t;
pthread_attr_t a;
pthread_attr_init(&a);
pthread_attr_set_qos_class_np(&a, QOS_CLASS_UTILITY, 0);
pthread_create(&t, &a, &helperMain, nullptr);
```

Or from inside an already-created thread (e.g. `std::thread`): call `pthread_set_qos_class_self_np(pthread_self(), QOS_CLASS_UTILITY, 0)` at thread start (macOS).

---

## Minimal “prove it” plan (1 afternoon)

1. Add **Audio Workgroup** handling + QoS bump → run **(A)** at 128-sample → expect fewer dropouts and tighter P99.
2. Convert model to **Core ML**, run **(B)** with ANE → expect lower CPU and steadier tails; keep 128-sample stable, then try 64-sample.
3. Pin one stable OS + `coremltools` + ExecuTorch tuple in CI and replay a parity fixture before testing betas or alternate delegates.

---

## What “good” looks like on M-series

- **48 kHz, 128-sample I/O, JEPA hop 128:** P95 ≤ ~6–8 ms, P99 ≤ ~9–10 ms, **0 dropouts** over 2 min.
- **64-sample:** P95 ≤ ~4–6 ms with occasional safe headroom ≥ 0.5 ms.

---

## KmiDi-specific notes

### Realtime safety: Master EQ (implemented)

`MasterEQProcessor::processBlock` now updates biquad coefficients **in place** via pre-allocated `bandCoefficients_` and helpers `fillHighPass`, `fillLowShelf`, `fillPeakFilter`, `fillHighShelf` that write into `getRawCoefficients()`. No allocation or locks in the audio callback; formulas match JUCE’s IIR::ArrayCoefficients.

### JUCE Audio Workgroup API

- `juce::AudioWorkgroup`, `juce::WorkgroupToken`: see `external/JUCE/modules/juce_audio_basics/utilities/juce_AudioWorkgroup.h`.
- Host calls `audioWorkgroupContextChanged(workgroup)` when the device provides a workgroup (CoreAudio, AUv3). Override in `PluginProcessor` to store it and pass to any worker that should join (e.g. a dedicated realtime helper thread).

### References

- JUCE: `AudioWorkgroup`, `Thread::RealtimeOptions`, `juce_Threads_mac.mm` (QoS mapping).
- Apple: Audio Work Interval API, `pthread_set_qos_class_self_np`, `QOS_CLASS_*`.
- Module tiers and SIMD entry points: [DSP_MODULE_TIERS_AND_SIMD.md](DSP_MODULE_TIERS_AND_SIMD.md).
