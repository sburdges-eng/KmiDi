# Bass Engine – Learning Guide

## What this engine does
- Lays down low-end foundation: rhythm tracker, harmonic anchor, and groove glue.
- Typically single-note lines, tightly locked to kick and groove.
- Bridges harmony and drums; reinforces chord roots and feel.

## Core basics (quick fundamentals)
- Lock with kick: align root/landing notes to kick pattern; anticipate/lag for feel.
- Note choice: roots on downbeats; fifths/octaves for movement; approach notes (chromatic/scale) for transitions.
- Rhythm shapes style: straight 8ths (rock), walking 4s (jazz), syncopated 16ths (funk/R&B), sparse subs (trap/EDM).
- Articulation: legato vs. staccato vs. ghost notes; slides for genre flavor (funk, hip-hop, DnB).
- Register matters: too high = loses weight; too low = mud; keep headroom for kick.

## Public repos to study (with advanced angles)
- **magenta/magenta (Groove/Bass examples)** – Baselines for rhythmic conditioning; adapt groove_rnn timing to bass.
- **microsoft/muzic** – Multi-task music generation; inspect bass/track-specific conditioning in their papers/code.
- **matthewjay/bassline_generator** – Rule/ML hybrid bassline ideas; good for pattern heuristics.
- **AI-Guru/bassline_extractor** – Extracts bass from mixes; useful for building a bass corpus and style stats.
- **sigsep/open-unmix** – Stem separation; mine real bass lines for timing/note distributions by genre.

## Two recursive study questions
1) How do genre-specific bass pockets differ (rock vs. funk vs. trap)? → Extract timing/velocity from stems (open-unmix) and cluster by genre.
2) How do approach notes and enclosures affect perceived forward motion? → Analyze chromatic vs. diatonic approaches in walking vs. pop lines.

## Advanced techniques to notice
- Groove-conditioned bass generation: tie note onsets to drum micro-timing and swing curves.
- Approach-note grammars (enclosures, chromatic steps) learned from real bass corpora.
- Dynamics shaping: ghost notes, accent placement, and articulation cues to keep weight without clutter.

## Next steps
- Task: Separate 50 tracks with open-unmix, extract bass MIDI (basic-pitch/mt3), compute onset histograms vs. kick, and derive a pocket template per genre.
- Where to look: open-unmix for stems, basic-pitch or mt3 for transcription, groove_rnn timing ideas for humanization.
