# UMP → JEPA Expressive Conditioning

**Date:** 2026-02-28  
**Purpose:** Architecture for conditioning JEPA on high-resolution UMP controller data via a parallel expressive channel (no raw CC tokens in the note stream).  
**Related:** [STEM_JEPA_INTEGRATION.md](research/STEM_JEPA_INTEGRATION.md), [REMI_BPE_TOKENIZATION.md](REMI_BPE_TOKENIZATION.md), [apple-silicon-low-latency.md](apple-silicon-low-latency.md), [WAVJEPA_LATENT_PIPELINE.md](WAVJEPA_LATENT_PIPELINE.md)

---

## Core idea

- High-res controller data carries emotional/expressive signal.
- That signal should **condition** latent prediction, not inflate the note sequence.
- JEPA benefits from expressive state awareness; design so that if controller entropy is low or noisy, the system degrades gracefully.

We do **not** stuff raw CC tokens into the same stream as note tokens. We build a **parallel expressive channel**.

---

## Data flow

```
Raw UMP Stream
   ↓
32-bit CC extraction
   ↓
Perceptual quantization (14-bit effective)
   ↓
Temporal aggregation (bar-level or windowed)
   ↓
UMP control state (mean, std, slope per CC per window)
   ↓
Control embedding layer
   ↓
JEPA context encoder (audio/MIDI structure) ← control bias injected here
   ↓
Latent projection head
   ↓
Predictor network
   ↓
Target encoder (EMA)  ← no control
   ↓
Latent alignment loss
```

---

## 1. UMP → aggregated control state

Do **not** feed per-event CC tokens; JEPA is sensitive to noise.

Compute **windowed** expressive state per CC:

- **Perceptual quantization:** map 32-bit value to ~14-bit effective (e.g. sqrt scaling) so resolution matches perceptual sensitivity.
- **Per window:** mean, std, slope (or similar) → 3 integers per CC. Dense, not spam.

Reference implementation: `experiments/ump_jepa/ump_aggregate.py` (perceptual_quantize, aggregate_window).

---

## 2. Control embedding layer

- Embed the aggregated control state **numerically** (embedding over binned values), not as giant discrete vocab tokens.
- Per CC: embed mean, std, slope; combine by **sum** (or concat if you want a larger projection). Output: one vector per window.

Reference: `experiments/ump_jepa/control_embedding.py` (UMPControlEmbedding).

---

## 3. JEPA integration

- **Context encoder:** receives note (or audio) tokens **and** control embedding. Inject control as a **bias** (e.g. linear projection of control embedding added to encoder hidden states). Do not concatenate to every token; do not inflate sequence length. A frozen audio front-end (e.g. WavJEPA) can feed the context encoder; see [WAVJEPA_LATENT_PIPELINE.md](WAVJEPA_LATENT_PIPELINE.md).
- **Target encoder:** does **not** see control. It encodes future tokens only.
- **Loss:** standard JEPA latent alignment (e.g. cosine similarity between predicted and target latents).

This forces the model to learn how expressive state **predicts** future structure.

---

## 4. Optional: latent-level emotional conditioning

- Map controller stats → small vector (e.g. VAD-like: Valence–Arousal–Dominance).
- Inject this vector at the **latent bottleneck** (e.g. into context encoder output before predictor). Then the JEPA is conditioned on “intensity state” rather than raw brightness/CC.

Reference: ExpressiveMapper in doc (experiments can implement when needed).

---

## 5. What can break

- **Assumption:** Controller curves correlate with musical structure.  
  **Counter:** They may correlate with performer nuance but not macro-structure. If controllers don’t predict structure, they become useless bias.

- **Assumption:** Higher resolution always helps.  
  **Counter:** Past ~12–14 bits you may be modeling jitter. Run entropy analysis; if effective bit depth &lt; 16, consider reducing bins.

---

## 6. Evaluation matrix

Train four variants:

| Model | Control injection | Aggregation | Expected |
|-------|-------------------|-------------|----------|
| A | None | — | Baseline |
| B | Linear quant | Window mean | Weak |
| C | Perceptual quant | Mean + slope (e.g. + std) | Strong |
| D | Perceptual + VAD mapper | Latent-level | Strongest |

Measure:

- Latent prediction cosine similarity
- Expressive reconstruction MSE (if auxiliary head)
- Human expressive quality rating
- Structural coherence under expressive shifts

- If **D** wins: architecture is valid.
- If **B** wins: may have overengineered.
- If **A** wins: data may have no expressive signal — that’s the uncomfortable possibility.

---

## 7. Reality check

MIDI 2.0 / UMP resolution is a tool. JEPA is about predictive abstraction. If expressive state genuinely helps predict future musical structure, this pipeline will use it. If not, it will tend to ignore it. Design for graceful degradation.
