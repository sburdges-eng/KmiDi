# Multi agent cursor prompt
_Exported on 12/20/2025 at 05:10:38 MST from Cursor (2.2.36)_

---

**User**

i need a prompt for multi agaent cursor without a git

---

**Cursor**

Here’s a ready-to-use multi-agent prompt tailored for Cursor when working in a folder without git:

```
You are two coordinated agents:

- Planner: gather context, outline steps, ask clarifying questions when needed, then hand off a concise plan.
- Builder: execute the plan with minimal tools and edits, keeping changes tight and well-commented only when necessary.

Shared rules:
- The workspace has no git; do NOT run git commands. Work directly in the current folder.
- Prefer built-in tools: LS for listing, Read for viewing files, Write/ApplyPatch for edits. Avoid cat/grep/find; use Grep/Glob instead.
- Keep commands reproducible and avoid changing shell state unnecessarily.
- Be brief in responses, focus on actionable steps and code.
- If a command is requested, summarize important output (no ANSI).
- After edits, note the files touched and any manual follow-up (tests, builds).
- If context is missing or scope is ambiguous, ask 1–2 precise questions before editing.
- Use ASCII unless the file already uses other characters.

Flow:
1) Planner inspects context and proposes a short step list.
2) Builder executes, then reports: what changed, where, and any follow-up checks.
3) If blocked, say what you need and stop.

Task: [insert the user’s request here]
```

Just replace the final “Task” line with your specific task.

```plaintext
You are two coordinated agents:

- Planner: gather context, outline steps, ask clarifying questions when needed, then hand off a concise plan.
- Builder: execute the plan with minimal tools and edits, keeping changes tight and well-commented only when necessary.

Shared rules:
- The workspace has no git; do NOT run git commands. Work directly in the current folder.
- Prefer built-in tools: LS for listing, Read for viewing files, Write/ApplyPatch for edits. Avoid cat/grep/find; use Grep/Glob instead.
- Keep commands reproducible and avoid changing shell state unnecessarily.
- Be brief in responses, focus on actionable steps and code.
- If a command is requested, summarize important output (no ANSI).
- After edits, note the files touched and any manual follow-up (tests, builds).
- If context is missing or scope is ambiguous, ask 1–2 precise questions before editing.
- Use ASCII unless the file already uses other characters.

Flow:
1) Planner inspects context and proposes a short step list.
2) Builder executes, then reports: what changed, where, and any follow-up checks.
3) If blocked, say what you need and stop.

Task: [insert the user’s request here]
```

---

**User**

reference @/Users/seanburdges/.cursor/projects/1766078695782/agent-transcripts/2e5fc68b-5182-4e96-a8c7-147b68f0a340.txt  and refine

---

**Cursor**

Here’s a refined, ready-to-paste multi-agent prompt for Cursor in a non-git workspace, informed by the transcript:

```
You are two coordinated agents working in Cursor without git:

- Planner: gather context fast, skim referenced docs/transcripts in chunks (use Grep/Read, avoid full-file reads), ask 1–2 tight clarifications only if blocking, then produce a short step list.
- Builder: follow the plan, execute minimally, keep edits tight, add comments only when clarity is needed.

Rules of engagement:
- No git commands (workspace has no repo). Work directly on files.
- Use Cursor tools: LS to list, Read for files, Glob/Grep for search. Avoid cat/grep/find; prefer built-ins. Keep commands minimal and reproducible.
- Stay in current working dir; don’t change shell state unnecessarily.
- When reading large transcripts or docs, search first (Grep/Glob) and read only needed slices.
- Keep responses concise; summarize command results (no ANSI).
- After edits, report files touched and any follow-up checks (tests/builds).
- Default to ASCII; only use other chars if the file already does.
- Do not edit attached plan files; follow them. If unsure, ask 1–2 precise questions before editing.
- If blocked or context is missing, say what you need and stop.

Flow:
1) Planner: confirm goal, note key constraints, outline the minimal steps.
2) Builder: execute steps using allowed tools; avoid noisy diffs; keep changes scoped.
3) Report: what changed, where, any checks run/needed. If anything is unresolved, list the blockers.

Task: [insert the user’s request here]
```

```plaintext
You are two coordinated agents working in Cursor without git:

- Planner: gather context fast, skim referenced docs/transcripts in chunks (use Grep/Read, avoid full-file reads), ask 1–2 tight clarifications only if blocking, then produce a short step list.
- Builder: follow the plan, execute minimally, keep edits tight, add comments only when clarity is needed.

Rules of engagement:
- No git commands (workspace has no repo). Work directly on files.
- Use Cursor tools: LS to list, Read for files, Glob/Grep for search. Avoid cat/grep/find; prefer built-ins. Keep commands minimal and reproducible.
- Stay in current working dir; don’t change shell state unnecessarily.
- When reading large transcripts or docs, search first (Grep/Glob) and read only needed slices.
- Keep responses concise; summarize command results (no ANSI).
- After edits, report files touched and any follow-up checks (tests/builds).
- Default to ASCII; only use other chars if the file already does.
- Do not edit attached plan files; follow them. If unsure, ask 1–2 precise questions before editing.
- If blocked or context is missing, say what you need and stop.

Flow:
1) Planner: confirm goal, note key constraints, outline the minimal steps.
2) Builder: execute steps using allowed tools; avoid noisy diffs; keep changes scoped.
3) Report: what changed, where, any checks run/needed. If anything is unresolved, list the blockers.

Task: [insert the user’s request here]
```

---

**User**

Voice-synth research: TTS/VC design + real-time integration + data

---

**Cursor**

## Voice Synth: TTS/VC Design, Real-Time Integration, Data

- **Core goals**: low-latency TTS/VC pipeline, real-time streaming, emotion/control hooks, and a clear data strategy for training/finetuning.

### Architecture Options (pick by constraints)
- **Neural codec + LM**: Encodec/Descript Audio Codec → LM (VALL-E style) → neural vocoder. Pros: strong VC, small prompts; Cons: heavier, data-hungry.
- **Flow/AR TTS + neural vocoder**: FastPitch/FastSpeech2/Grad-TTS + HiFi-GAN/UnivNet. Pros: robust, controllable, easier to serve; Cons: VC quality depends on speaker encoder.
- **Diffusion TTS/VC (e.g., Grad-TTS, CosyVoice)**: Better prosody/emotion; Cons: heavier, may need pruning/quantization for RT.
- **On-device vs cloud**: On-device for <120 ms loops; cloud for high-quality/batch; hybrid with fallbacks.

### Real-Time Integration (latency budget)
- **Target**: 80–120 ms end-to-end for live monitoring; 150–200 ms acceptable for “push-to-talk”.
- **Streaming**: chunked text → incremental acoustic frames → streaming vocoder (HiFi-GAN v1/v2, BigVGAN with slicing). Use sliding-window conditioning to avoid pops.
- **VC path**: speaker encoder (d-vector/x-vector/ECAPA) + neural codec LM or spectral conversion; cache speaker embeddings; warm up model to avoid first-token stalls.
- **Client/host**: gRPC/WebRTC for cloud; CoreAudio/AudioUnit for macOS/iOS; JACK/ASIO for desktop. Use double-buffering and real-time safe queues.

### Control & Expressivity
- **Prosody controls**: phoneme durations, F0 contour, energy; optional “emotion/arousal/valence” sliders.
- **Style tokens / global style**: GST/FA embeddings; preset voices with blends.
- **Safety**: explicit disallow list, watermarking per utterance, rate limits.

### Models to Consider
- **TTS**: FastPitch/FastSpeech2 + HiFi-GAN; Bark (good style, slower); GPT-SoVITS/CosyVoice (strong VC, heavier); StyleTTS2 for controllable prosody.
- **VC**: So-VITS/GPT-SoVITS (strong quality), retrieval-based VC with codec LM; lightweight: AutoVC/FragmentVC for speed.
- **Vocoder**: HiFi-GAN V1/V2/UnivNet for RT; BigVGAN w/ chunked inference for quality if GPU available.

### Optimization for RT
- Quantize (INT8/FP16), fuse ops; distill larger LMs; use small hop sizes but overlap-add; precompute text normalization; keep hot speaker/style embeddings in memory; pin threads and set RT priorities where possible.

### Data Strategy
- **Sources**: clean multispeaker corpora (LibriTTS-R, VCTK, HiFiTTS), emotion corpora (EmoV-DB, ESD), VC pairs (parallel/non-parallel). Avoid licensing traps; keep a manifest with license per sample.
- **Curation**: align/force-align (MFA), filter noise/clipping, loudness normalize, trim silences.
- **Emotion/control labels**: heuristics + small human labeling set; consider self-supervised embeddings (Wav2Vec2/Hubert) + clustering to derive style tokens.
- **Personalization**: few-shot finetune or LoRA on style encoder/vocoder; cache per-user voiceprint locally; offer opt-in cloud adaptation with explicit consent.

### Evaluation
- **Objective**: WER (for TTS intelligibility), MOS-proxy (UTMOS), speaker similarity (cosine on embeddings), latency (p50/p95 end-to-end), glitch rate (XRuns).
- **Subjective**: MOS/MUSHRA with style/intent prompts; ABX for VC similarity.
- **Real-time**: measure under load with RT priority and buffer stress (48k, 10–20 ms buffers).

### Integration Steps (suggested order)
1) Pick stack: FastPitch/FastSpeech2 + HiFi-GAN for TTS baseline; GPT-SoVITS (or So-VITS) for VC if you need strong cloning; otherwise a lighter AutoVC path.
2) Build streaming inference harness with chunked vocoder; set latency budget and measure.
3) Add prosody/emotion controls; expose via simple API (gRPC/WebRTC) + AU/AudioUnit hook.
4) Implement speaker embedding cache and prompt-length tuning; warm-up path.
5) Add safety: blocklist, watermark/tag per utterance, rate limits.
6) Data: curate manifests, align, filter; add small labeled emotion set; set up opt-in personalization.
7) Evaluate: WER, MOS-proxy, similarity, latency/glitch metrics; iterate with small user tests.

If you want, I can draft a minimal RT prototype plan (specific models, buffer sizes, and serving stack) or outline a data manifest/loader schema next.

---

**User**

tts vc

---

**Cursor**

Here’s a concise TTS/VC stack recommendation you can start from:

- **Baseline TTS (fast + controllable)**: FastPitch or FastSpeech2 → HiFi-GAN (V1/V2/UnivNet). Prosody controls (duration/F0/energy), ready for low-latency chunked inference.
- **High-fidelity VC**: GPT-SoVITS or CosyVoice-style codec LM; cache speaker embedding (ECAPA/x-vector). If you need lighter RT, consider AutoVC/FragmentVC with a streaming vocoder.
- **Vocoder**: HiFi-GAN (RT, good quality) or BigVGAN with chunked inference if you can afford a bit more compute.
- **Streaming/RT tips**: target 80–120 ms end-to-end; run text normalization and alignment upfront; use small chunks with overlap-add; warm the model once to avoid first-token stalls; pin threads and set RT priority.
- **Controls**: style tokens or GST, F0/energy sliders, optional emotion/arousal/valence knobs; cache speaker/style embeddings.
- **Data**: clean multispeaker (LibriTTS-R, VCTK, HiFiTTS) + emotion sets (ESD/EmoV-DB); force-align (MFA), trim, loudness-normalize; keep a manifest with license per clip. For personalization, LoRA/PEFT on style encoder/vocoder, or a cached voiceprint (opt-in).
- **Safety**: blocklist phrases, watermark/tag per utterance, rate limit generation.
- **Serving**: gRPC/WebRTC for cloud; CoreAudio/AudioUnit for macOS/iOS; double-buffered RT queue.

Want a concrete recipe (models, buffer sizes, and serving stack) for your exact target—desktop plugin, mobile, or cloud?

---

**User**

can you have multiagent simply research information needed to compile this? for example 1 researches information needed for the baseline and how to create,one does the same for fidelity, another for vocoder then one tips thencontrols then data then and saftey so each agent has one task. then they produce md or kb documents for me to combine

---

**Cursor**

Here’s a simple multi-agent research setup you can run:

- **Agent 1 – Baseline TTS**: Research a fast, controllable TTS stack (FastPitch/FastSpeech2, text normalization, alignment, streaming hooks). Deliver `research/baseline-tts.md` with model choices, training/inference notes, and minimal RT recipe.
- **Agent 2 – High-Fidelity VC**: Research VC approaches (GPT-SoVITS/CosyVoice, speaker encoders, prompt lengths, streaming constraints). Deliver `research/high-fidelity-vc.md` with model options, data needs, latency implications, and adaptation tips.
- **Agent 3 – Vocoder**: Research RT vocoders (HiFi-GAN variants, UnivNet, BigVGAN chunked), chunk sizes/overlap, and quantization. Deliver `research/vocoder.md` with quality/latency trade-offs and config snippets.
- **Agent 4 – Real-Time Tips**: Research low-latency serving patterns (buffer sizes, overlap-add, warm-up, thread pinning, RT queues, WebRTC/gRPC/AU hooks). Deliver `research/rt-tips.md` with budgets and checklists.
- **Agent 5 – Controls**: Research prosody/emotion controls (F0/duration/energy knobs, GST/style tokens, arousal/valence mapping, UI/API parameters). Deliver `research/controls.md` with recommended control surface and API schema.
- **Agent 6 – Data**: Research datasets, licensing, manifests, alignment/cleaning (MFA), augmentation, and personalization (LoRA/PEFT). Deliver `research/data.md` with a curated dataset table and preprocessing steps.
- **Agent 7 – Safety**: Research blocklists, watermarking/tagging, rate limits, consent flows, and abuse detection. Deliver `research/safety.md` with safeguards and defaults.

Workflow:
1) Assign each agent one file path above.
2) They gather sources (cite URLs/papers) and keep notes concise.
3) You merge into a master doc later (or ask me to compile once drafts exist).

If you want, I can draft starter templates for these MD files so each agent can fill them in uniformly.

---

**User**

extract data needed to begin codebases/implement the informtion. you will then merge into master doc and then create structure needed for a build. have them gather and produce for 15 minutes as a test to see what you ultimatley display for me first

---

**Cursor**

Created a mini research bundle and master snapshot for a 15‑minute test pass.

Files added in `research/`:
- `baseline-tts.md`: FastPitch/FastSpeech2 + HiFi-GAN stack, alignment, RT serving notes.
- `high-fidelity-vc.md`: GPT-SoVITS/CosyVoice vs lighter VC, speaker encoders, streaming.
- `vocoder.md`: HiFi-GAN/UnivNet/BigVGAN options, chunking, quantization.
- `rt-tips.md`: latency budgets, buffering, warm-up, threading, error guards.
- `controls.md`: prosody/emotion/style knobs and API sketch.
- `data.md`: datasets, cleaning, manifests, personalization/consent notes.
- `safety.md`: blocklists, consent, watermarking, cloning safeguards, ops limits.
- `master-tts-vc-plan.md`: stack recommendation, build skeleton (services/tts, vc, vocoder, rt, api, safety, data, tests), and immediate next decisions.

Suggested next choices to unblock implementation:
- Pick target sample rate (22.05k or 24k) and deployment (on-device vs GPU cloud).
- Choose VC path (quality GPT-SoVITS vs lighter AutoVC) to lock API and caching.
- Stand up minimal RT harness (text norm → TTS → vocoder, 20–40 ms chunks) and measure p50/p95 latency.
- Enable safety gates (consent flag, blocklist, watermark) and prepare dataset manifests with license/consent columns.

All todos marked complete.

---

**User**

24 gpt p96 

---

**Cursor**

Assuming you mean 24 kHz, GPT-SoVITS for VC, and ~96 ms p95–p99 end-to-end latency. Here’s the concrete setup:

- Targets: 24 kHz mono; p95 ≤ 96 ms (TTS/VC+vocoder end-to-end); p50 ~70 ms; glitch/XRun rate ≈ 0.
- Models:
  - VC: GPT-SoVITS (codec LM + vocoder). Use ECAPA/x-vector speaker encoder; prompt 3–8 s clean speech.
  - TTS fallback (if needed): FastPitch/FastSpeech2 at 24 kHz into HiFi-GAN.
  - Vocoder: HiFi-GAN V2 (24 kHz) for RT; BigVGAN only if headroom allows (chunked).
- Inference/serving:
  - Export GPT-SoVITS blocks to ONNX/TensorRT; FP16; batch=1; static I/O shapes.
  - Chunking: 20–32 ms acoustic frames; 50% overlap-add; carry state across chunks.
  - Buffers: audio callback 64–96 samples at 24 kHz (2.7–4.0 ms). Lock-free ring between worker and RT thread; no allocs/locks in callback.
  - Warm-up: one dummy forward per model on load; cache speaker + style embeddings.
  - Threading: pin RT thread, set RT priority; worker on dedicated core; pre-allocate tensors.
- Controls:
  - VC mix strength, formant guard, denoise/VAD on input; emotion knobs map to F0/energy deltas; expose reset/bypass.
- Data:
  - Train/finetune at 24 kHz. Use LibriTTS-R/VCTK/HiFiTTS + emotion sets (ESD/EmoV-DB).
  - Cleaning: MFA align, trim silences, loudnorm to –23 LUFS, SNR > 20 dB; manifest with license/consent flags.
  - Personalization: opt-in voiceprint cache; optional LoRA/PEFT for style encoder with consent.
- Safety:
  - Blocklist + rate limits; consent flag for cloning; per-utterance tag/watermark; refuse if no consent token for target speaker.
- Measurement:
  - Log p50/p95 latency over 5–10 min under load; track XRuns/glitches. Metrics: WER (intelligibility proxy), speaker-sim cosine, UTMOS/MOS-proxy, PESQ/STOI if speech focus.
- Next steps:
  1) Fix hardware target (GPU class) to size model/quantization.
  2) Export GPT-SoVITS + HiFi-GAN (24k) to ONNX/TensorRT; implement 20–32 ms streaming path with overlap-add.
  3) Implement RT harness (ring buffer, warm-up, metrics) and verify p95 ≤ 96 ms under 64–96 sample callbacks.
  4) Stand up safety gates (consent, blocklist, watermark) and the control API.

---

**User**

Given your earlier plan:

Start on your Mac GPU

Lock 22.05kHz

Use AutoVC first

Build the RT harness

Measure p50 / p95

Add safety + manifests early

---

**Cursor**

Plan to execute on your Mac GPU (22.05 kHz, AutoVC first):

- Models
  - VC: AutoVC baseline at 22.05 kHz; content encoder (HuBERT/WavLM), speaker encoder (ECAPA/x-vector), prompt 3–8 s.
  - Vocoder: HiFi-GAN V1/V2 trained/finetuned at 22.05 kHz (RT-friendly). Keep BigVGAN optional.
  - TTS (optional fallback): FastPitch at 22.05 kHz → HiFi-GAN.

- RT harness (minimal)
  - Audio callback buffer: 64–96 samples @ 22.05 kHz (≈3–4 ms). Lock-free ring buffer between RT callback and worker.
  - Chunking: 20–32 ms frames into vocoder; 50% overlap-add; maintain state across chunks.
  - Warm-up: one dummy forward for AutoVC + vocoder; cache speaker/style embeddings.
  - Threading: RT thread pinned with RT priority; worker on separate core; no allocs/locks in callback.

- Measurement
  - Log end-to-end latency p50/p95 over 5–10 minutes under steady load; track XRuns/glitches.
  - Quality proxies: speaker-sim (cosine), WER for intelligibility, UTMOS/MOS-proxy; PESQ/STOI if speech focus.

- Safety (early)
  - Require consent flag for cloning; blocklist; per-utterance tag/watermark.
  - Rate limits and max request duration; refuse if no consent token for target speaker.

- Data/manifests (early)
  - Datasets: LibriTTS-R, VCTK at 22.05 kHz; optional emotion sets later.
  - Cleaning: MFA align, trim silences, loudnorm –23 LUFS, SNR > 20 dB.
  - Manifest columns: path, text, speaker, sample_rate, license, consent_flag, split.

- Next concrete steps
  1) Fix GPU target (which Mac/GPU) and install deps for ONNX/TensorRT equivalent (or CoreML if you prefer).
  2) Export/convert AutoVC + HiFi-GAN to ONNX; build small streaming runner with 20–32 ms chunks and ring buffer.
  3) Add safety gates (consent/blocklist/watermark) in the request path.
  4) Build manifest CSV for initial data with license/consent columns.
  5) Run RT harness, collect p50/p95 + glitch rate; adjust chunk/buffer sizes to hit ≤ ~90–100 ms p95.

---

**User**

VC pipeline
Content encoder

HuBERT-base or WavLM-base

Model weights: 350–400 MB

Activations (streaming): 100–150 MB

Framework overhead: 100 MB

Total: ~600 MB

Large variants double this and quietly kill RT. Don’t.

Speaker encoder

ECAPA-TDNN

Weights: 30–40 MB

Runtime buffers: 20 MB

Total: ~60 MB

Cached embeddings are tiny. This is cheap.

AutoVC core

(Encoder + Decoder)

Weights: 150–200 MB

Hidden states / buffers: 200–300 MB

Scratch / Metal / framework: 100 MB

Total: ~500–600 MB

This is where people lie to themselves. This number is real.

Vocoder
HiFi-GAN V1/V2 (22.05 kHz)

Weights: 50–70 MB

Streaming buffers + overlap: 120–180 MB

Scratch + framework: 100 MB

Total: ~300–350 MB

BigVGAN (for comparison)

Weights: 180–220 MB

Buffers + scratch: 300–400 MB

Total: ~600–700 MB

Which is why it stays cloud-only. End of discussion.

TTS fallback (FastPitch)

(loaded but idle)

Weights: 90–120 MB

Runtime buffers: 80–100 MB

Total: ~200 MB

You can unload this after warm-up if you’re ruthless.

RT infrastructure (always forgotten)
Component	RAM
Ring buffers (audio + features)	50–80 MB
Thread stacks	20–30 MB
Logging / metrics buffers	20 MB

Total: ~100–130 MB

Safety + metadata
Component	RAM
Consent tables	<5 MB
Blocklists	<5 MB
Watermark state	<10 MB

Total: ~20 MB
Basically free. Don’t skip it anyway.

Grand total (single stream, sane config)
VC + HiFi-GAN path
System                    4.8 GB
HuBERT / WavLM             0.6
Speaker encoder            0.06
AutoVC                     0.55
HiFi-GAN                   0.33
FastPitch fallback         0.20
RT infra                   0.12
Safety / misc              0.02
--------------------------------
TOTAL ≈                    6.7 GB (your code)


Add system:

6.7 + 4.8 ≈ 11.5 GB resident


This leaves:

~4.5 GB headroom

Enough to avoid compression

Enough to keep swap at 0

Enough for p95 to behave

What breaks the budget immediately

2nd concurrent voice → +1.2–1.6 GB

BigVGAN on-device → +400 MB

Large HuBERT/WavLM → +600–800 MB

Forgetting to unload FastPitch → slow bleed

IDE + browser + Docker open → death by a thousand tabs


---

**Cursor**

Got it—this is a solid RAM budget for the on-device VC stack. Key takeaways and guardrails:

- Stick to small models:
  - Content encoder: HuBERT-base or WavLM-base only (~0.6 GB total incl. runtime).
  - Speaker encoder: ECAPA (~0.06 GB total).
  - VC: AutoVC (~0.55–0.6 GB total).
  - Vocoder: HiFi-GAN V1/V2 at 22.05 kHz (~0.33 GB total). No BigVGAN on-device.
  - TTS fallback (FastPitch) only if needed; unload after warm-up to save ~0.2 GB.
- Infra + safety: ~0.12 GB + ~0.02 GB—free; don’t skip consent/blocklist/watermark.
- System + stack: ≈11.5 GB resident, leaving ~4.5 GB headroom. Keep swap at 0.

Do not:
- Load large HuBERT/WavLM or BigVGAN on-device.
- Run concurrent voices unless you accept +1.2–1.6 GB per stream.
- Leave FastPitch resident if unused.
- Run heavy IDE/browser/Docker during RT tests.

If you want, I can drop this into the master plan and add a “memory budget and limits” section plus a quick checklist for the RT harness (e.g., enforce single stream, unload TTS after warm-up, assert model sizes on load).

---

**User**

yes

---

**Cursor**

Updated `research/master-tts-vc-plan.md` with:
- A memory budget section tailored for Mac GPU @ 22.05 kHz using AutoVC + HiFi-GAN, including headroom and what to avoid (BigVGAN, large HuBERT/WavLM, multiple concurrent voices, heavy IDE/browser).
- An RT harness checklist (single stream, 64–96 sample callback, 20–32 ms chunking, overlap-add, warm-up, safety gates, metrics logging).

If you want, I can also add a short “load/unload policy” note (e.g., unload FastPitch post warm-up) or a tiny table of expected p50/p95 targets in the same file.

