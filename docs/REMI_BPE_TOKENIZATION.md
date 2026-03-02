# REMI-BPE Tokenization for Expressive MIDI

**Date:** 2026-02-28  
**Purpose:** Compact, expressive MIDI tokenization for KmiDi/Kelly-Brain via MidiTok REMI + BPE. Pipeline, validation, and baseline choices.  
**Related:** [STEM_JEPA_INTEGRATION.md](research/STEM_JEPA_INTEGRATION.md), [UMP_JEPA_EXPRESSIVE_CONDITIONING.md](UMP_JEPA_EXPRESSIVE_CONDITIONING.md)

---

## Why REMI + BPE

- **REMI** encodes bars, positions, notes, velocities in a music-aware way that works well with LMs.
- **BPE** learns frequent multi-event patterns (e.g. rhythm+velocity chunks) to shorten sequences and often improve downstream quality.

Target: shorter sequences without losing expressive detail (velocity nuance, tempo, chords, bar structure).

---

## Minimal Pipeline

```bash
pip install miditok
```

```python
from pathlib import Path
from miditok import REMI, TokenizerConfig

TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 64,
    "use_chords": True,
    "use_tempos": True,
    "special_tokens": ["PAD", "BOS", "EOS"],
}

cfg = TokenizerConfig(**TOKENIZER_PARAMS)
tokenizer = REMI(cfg)

midi_paths = list(Path("data/giga_expressive").glob("**/*.mid"))
tokenizer.learn_bpe(vocab_size=30_000, files_paths=midi_paths)
tokenizer.apply_bpe_to_dataset(
    files_paths=Path("data/giga_expressive"),
    out_dir=Path("tokens"),
    midi_paths_exts=[".mid", ".midi"],
)
```

Config for training: [configs/train_remi_bpe_30k.json](../configs/train_remi_bpe_30k.json).  
Dataset split by work ID: `scripts/split_midi_by_work.py`.  
Sequence length histograms: `scripts/seq_length_histogram.py`.

---

## Validation (Before Committing to REMI+BPE)

### 1. Length histograms

Compare:

- REMI (no BPE)
- REMI + 30k BPE
- REMI + 50k BPE

**Target:** Median sequence length under ~1k tokens for your window size (single-GPU / M-series friendly).  
Run `scripts/seq_length_histogram.py` and inspect outputs.

### 2. Expressive fidelity

After decoding samples, check:

- **Velocity** — Are bins too coarse?
- **Tempo** — Are tempo events merged in musically weird ways?
- **Chords** — Chord tokens intact?
- **Bar structure** — Long rests collapsing into strange tokens?

If the tokenizer erases expressive granularity, Valence–Arousal–Dominance / emotion mappings get fuzzier.

### 3. Vocabulary sanity

Inspect top ~200 merged BPE tokens.  
If BPE merges across musically meaningless boundaries (e.g. Position + unrelated velocity), that’s compression at the expense of structure and controllability.

### 4. Training constraint sanity

KmiDi uses JEPA layers and emotion mappings. If the tokenizer reduces expressive granularity, conditioning and controllability suffer. Validate that REMI+BPE preserves the signal you need for conditioning.

---

## Practical settings

| Setting | Recommendation |
|--------|------------------|
| Vocab size | Start 30k; if sequences still long (1k–2k tokens/window), try 40–60k. |
| Max sequence length | 1024 to start (single-GPU, M-series). |
| Dataset | GigaMIDI expressive subset (heuristics to separate expressive vs non-expressive tracks). Research-only access via Hugging Face. |

---

## Baseline question: REMI vs CPWord / Octuple

Before committing to REMI+BPE only:

- **CPWord** and **Octuple** are alternative tokenization schemes. Consider running the same validation (length histograms, expressive fidelity, vocab sanity) for at least one baseline so the choice is data-driven, not assumed.

If REMI+BPE does not clearly win on length + expressive fidelity + controllability, re-evaluate.

---

## Dataset & licensing

- **GigaMIDI:** Largest symbolic set; expressive subset aligns with emotion/JEPA layers. Research-only access via HF; keep non-commercial in this phase.
- **Time signatures:** REMI is often tuned for 4/x bars; check MidiTok docs if you rely on many time-signature changes.
- **Track-aware futures:** For arrangement/drum-only or per-track control, consider REMI+ (e.g. REMI-z) or extra program/time-signature tokens.

---

## Suggested branch

`experiment/miditok-remi-bpe` — keep tokenization experiments isolated until validation passes.
