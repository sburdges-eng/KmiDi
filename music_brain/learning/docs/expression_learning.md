# Expression Engine – Learning Guide

## What this engine does
- Shapes dynamics, articulation, and micro-variations (velocity, timing, timbre) to add human feel.
- Maps emotion/section energy to loudness, brightness, and phrasing.
- Enhances otherwise quantized notes with life and intention.

## Core basics (quick fundamentals)
- Velocity curves: phrase-level swells; accent placement on offbeats/backbeat; ghost notes for groove.
- Timing nuance: slight push/pull tied to groove; length variation (staccato vs. legato) for clarity or urgency.
- Timbre/dynamics link: brighter + louder for intensity; darker + softer for intimacy; automate filters and expression (CC11/CC1).
- Phrase shaping: crescendo into cadences/choruses, relax on resolutions.
- Register & density: fewer simultaneous high-velocity notes to avoid harshness; leave space around vocals/lead.

## Public repos to study (with advanced angles)
- **magenta/ddsp** – Neural timbre/dynamics modeling; see how amplitude envelopes and harmonic/excitation control timbre.
- **magenta/musicVAE & Music Transformer examples** – Look at humanization and dynamics tokens/embeddings.
- **microsoft/muzic** – Multi-track/dynamics conditioning ideas in their papers.
- **rainerKel/dl4m** (deep learning for music) – Tutorials on expressive performance generation.
- **ai-music/Performance-RNN resources** – Older but clear expressive MIDI baselines.

## Two recursive study questions
1) How do dynamics envelopes differ across genres and sections (verse vs. chorus)? → Extract velocity/time histograms from multi-track MIDIs; cluster by section label.
2) How does micro-timing variation correlate with dynamics in human performances? → Analyze Groove MIDI Dataset or MT3-extracted performances for timing vs. velocity correlations.

## Advanced techniques to notice
- DDSP-style control of loudness/brightness; learnable envelopes for section-aware expression.
- Token or continuous embeddings for dynamics/humanization in transformers (velocity bins, timing offsets as regression targets).
- Cross-part coherence: align dynamics swells across sections/instruments; avoid over-dense peaks.

## Next steps
- Task: From 50 expressive MIDIs, derive per-section velocity envelopes and timing jitter stats; apply to a quantized lead or chords.
- Where to look: DDSP loudness controls, Performance RNN/expressive MIDI datasets, MT3 for performance extraction.
