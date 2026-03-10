# Perch + REMI-BPE pipeline (experiment)

Minimal pipeline for audio embeddings (Perch-style) and REMI-BPE symbolic tokenization. See [docs/research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md](../../docs/research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md).

## Quick test (audio + symbolic)

From repo root:

```bash
# Create fixtures (1s WAV + minimal MIDI)
python3 experiments/perch_remi_pipeline/make_fixtures.py

# Audio side: stub Perch embeddings → JSONL + .npy
python3 experiments/perch_remi_pipeline/embed_perch.py \
  --audio-root experiments/perch_remi_pipeline/fixtures \
  --pattern "*.wav" \
  --window-seconds 0.5 \
  --stride-seconds 0.5 \
  --output experiments/perch_remi_pipeline/out/embeddings.jsonl

# Symbolic side: tokenize-only round-trip (no model download)
pip install miditok   # if needed
python3 experiments/perch_remi_pipeline/remi_bpe_demo.py \
  --midi-path experiments/perch_remi_pipeline/fixtures/sample.mid \
  --output experiments/perch_remi_pipeline/out/generated.mid \
  --tokenize-only
```

The minimal `fixtures/sample.mid` may yield 0 tokens with default REMI; for a full tokenizer test use a real MAESTRO MIDI and optional `--model NathanFradet/Maestro-REMI-bpe20k`. Full generation (with model) omits `--tokenize-only`.

## Scripts

| Script | Purpose |
|--------|--------|
| `make_fixtures.py` | Create `fixtures/sample.wav` and `fixtures/sample.mid`. |
| `embed_perch.py` | Compute (stub) embeddings per window; writes JSONL + `.npy`. |
| `remi_bpe_demo.py` | REMI-BPE tokenize and optionally generate; `--tokenize-only` for round-trip without model. |
