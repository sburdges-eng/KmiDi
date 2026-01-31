# AI Types → KmiDi Project Mapping (Research)

**Status:** Research / stub. Not active implementation.  
**Purpose:** Map broad AI model types to where they could theoretically fit in KmiDi; document reasoning and future potential.  
**Created:** 2026-01-31  
**Refs:** [PROJECT_ROADMAP_REIMPLEMENTATION.md](../../docs/PROJECT_ROADMAP_REIMPLEMENTATION.md), [CONTRACTS.md](../../docs/CONTRACTS.md), [local_vs_cloud_deployment.md](local_vs_cloud_deployment.md)

---

## 1. Scope and reasoning requirement

We map: **Neural** (substrate), **AR/LLM**, **Diffusion**, **VAE**, **Flow**, **GAN**, **JEPA-style**, **Energy/score**. For each we state *what it can do*, *where in the project it could go*, and *why* — so future experiments can test assumptions rather than cargo-cult.

---

## 2. Reasoning per type

### Neural (deep learning)

- **What it can do:** Any learned function: classification, regression, sequence prediction, embeddings, control. It is the substrate; other types are instances built on it.
- **Where in project:** Everywhere ML is used: penta_core inference, LLM backend, image/audio engines, future melody/groove/harmony models.
- **Why:** The spine (Brain → music_brain → tier1 → orchestrator) already assumes neural inference in penta_core and optional LLM/image/audio. No new “type” here — it’s the shared machinery. Reasoning: keep a single substrate so we don’t fragment tooling (training, checkpointing, deployment); see [local_vs_cloud_deployment.md](local_vs_cloud_deployment.md) for which neural workloads run locally vs cloud.

### Autoregressive (AR) / LLM

- **What it can do:** Next-token prediction over sequences (text, MIDI tokens, code). Good for parsing user text → structured intent, generating prompts, chat, and discrete sequence generation.
- **Where in project:** LLM → Intent (`mcp_workstation/llm_reasoning_engine.py`), music_brain chatbot (`music_brain/chatbot/agent.py`), future text→MIDI token stream if we model MIDI as tokens.
- **Why:** User input is text; intent is structured but can be generated or parsed by a single model. AR is the dominant paradigm for discrete sequences and fits the contract “user text → CompleteSongIntent.” Reasoning: we don’t need to generate pixels or waveforms here — we need symbols and structure; AR gives one coherent API (prompt in, tokens out) that can drive both parsing and generation. Latency and cost drive whether this runs locally (small model on 16 GB Mac) or in cloud; see deployment doc.

### Diffusion

- **What it can do:** Generate high-dimensional continuous data by iterative denoising: images, audio waveforms, sometimes symbolic sequences in continuous latent space.
- **Where in project:** Image generation (`mcp_workstation/image_generation_engine.py`), audio generation (`mcp_workstation/audio_generation_engine.py`); future latent diffusion over learned MIDI/harmony representation if we introduce one.
- **Why:** The orchestrator already has optional image/audio phases; diffusion is the current best practice for quality and stability. Reasoning: diffusion trades sampling steps for quality and avoids GAN instability; for “prompt → asset” we don’t need real-time latency, so multi-step sampling is acceptable. These workloads are good candidates for cloud (or future self-hosted GPU) because they are not in the critical low-latency path of the DAW/plugin; see deployment doc.

### VAE

- **What it can do:** Encode → compact latent; decode → reconstruct or generate. Fast single-pass decode; good for structured latent spaces and compression.
- **Where in project:** Inside latent diffusion stacks (e.g. SD VAE for image). Future: encode “song slice” or “groove” for similarity, interpolation, or as input to a small AR/diffusion model.
- **Why:** VAEs are already embedded in SD-style pipelines; we don’t add them as a separate surface. Reasoning: for future “song/groove latent” we want one-pass encode/decode so plugin or DAW can run it locally without iterative denoising; VAE fits that. Such a model would be trained locally first (small data, 16 GB Mac), then optionally refined in cloud; see deployment doc.

### Flow (normalizing flow)

- **What it can do:** Exact density in a learned space; fast sampling; good for anomaly detection and anything needing likelihoods.
- **Where in project:** Future only: “Is this intent/harmony in-distribution?” for safety or filtering; density-based ranking of generated options; compression or representation in low-dimensional space (e.g. chord or groove).
- **Why:** We don’t need exact density today; rules and heuristics cover intent validation. Reasoning: if we later add learned intent/harmony models, flows give a principled way to reject out-of-distribution inputs and to rank candidates. That would be a small, local model (plugin/DAW friendly); see deployment doc.

### GAN

- **What it can do:** Generate sharp samples (image/audio); no direct likelihood; training can be brittle.
- **Where in project:** Theoretically alternative to diffusion for image/audio; not primary.
- **Why:** Diffusion is preferred for stability and quality; we don’t adopt GAN as default. Reasoning: only consider GAN if we have a specific asset pipeline (e.g. a pre-trained GAN) or a research experiment comparing GAN vs diffusion; otherwise keep a single generative paradigm per modality to reduce maintenance.

### JEPA-style (joint embedding predictive)

- **What it can do:** Learn representations by predicting in latent space (no pixel/token generation); good for “what comes next?” in abstract space, alignment of modalities, and robust features.
- **Where in project:** `src/kelly/integrations/stem_jepa_integration.py` (stems/audio representation); future: intent alignment (text ↔ CompleteSongIntent in shared space), “next section” prediction in latent space, multimodal alignment.
- **Why:** JEPA gives representation and prediction without the cost of generative decoding — useful for DAW/plugin features that need “understanding” or continuity, not synthesis. Reasoning: low-latency path (e.g. “next bar” or “match intent”) can be a small encoder + predictor running locally; training can start local, then scale in cloud; see deployment doc.

### Energy / score-based

- **What it can do:** Model score (gradient of log-density); used inside diffusion; general but often implemented via diffusion.
- **Where in project:** Already inside diffusion stacks; standalone score model only if we add “refine this latent” or guided generation later.
- **Why:** No separate surface today. Reasoning: document for completeness; any diffusion-based experiment implicitly uses score-based learning.

---

## 3. Summary table (supporting; reasoning above)

| AI type | Can do | Where in project (theoretical) | Status |
|--------|--------|--------------------------------|--------|
| **Neural** | Substrate | penta_core, LLM, image/audio, melody/groove | Current (inference, stubs) |
| **AR/LLM** | Next-token; intent parse/generate | llm_reasoning_engine, chatbot, future text→MIDI | Stub → reimpl |
| **Diffusion** | Images, audio | image_generation_engine, audio_generation_engine | Stub / optional pipeline |
| **VAE** | Encode/decode; latent | Inside SD/audio; future song/groove latent | Future |
| **Flow** | Exact density; anomaly | Future: intent/harmony sanity | Future |
| **GAN** | Generate image/audio | Alternative; not primary | Future / niche |
| **JEPA-style** | Predict in latent; representation | stem_jepa_integration; intent; next section | Stub (stem); rest future |
| **Energy/score** | Score = ∇log p | Inside diffusion; standalone later | Future |

---

## 4. By project area (supporting)

| Area | Best-fit types | Notes |
|------|----------------|-------|
| LLM → Intent | AR/LLM | Parse/generate CompleteSongIntent, prompts. |
| Intent → MIDI | Neural (+ optional AR over tokens) | Today: rules + harmony; later: AR MIDI or small neural blocks. |
| Image engine | Diffusion (VAE in SD) | Stub → load SD; return asset. |
| Audio engine | Diffusion / AR | Stub → document; optional Audiocraft-style. |
| Stem / audio repr. | JEPA | stem_jepa_integration; latent only. |
| Chatbot | AR/LLM | Deferred; when implemented. |
| penta_core | Neural | Whatever models are registered. |
| Future: “next section” | JEPA | Predict next latent (section/groove). |
| Future: density / anomaly | Flow | Intent/harmony in-distribution. |

---

## 5. Implementation status

- This doc is **research only**. No code changes; no new stubs in spine.
- When a type is implemented, update this table and add a one-line note in CONTRACTS or the owning module.
- Experiments that use one of these types should reference this file and state which assumption they are testing.
- Deployment (local vs cloud, training path) is in [local_vs_cloud_deployment.md](local_vs_cloud_deployment.md).
