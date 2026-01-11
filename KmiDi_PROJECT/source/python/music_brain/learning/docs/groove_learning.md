# Groove Engine – Learning Guide

## What this engine does
- Shapes timing, swing, micro-timing offsets, and velocity curves to create “feel.”
- Aligns rhythmic pocket across drums, bass, and comping parts.
- Maps emotional intent to groove archetypes (tight/quantized vs. loose/humanized).

## Core basics (quick fundamentals)
- Swing vs. straight vs. shuffle: triplet vs. duplet grids; percent swing matters by tempo.
- Micro-timing: push (ahead) for urgency, pull (behind) for laid-back; keep kick/snare vs. hat relationships intentional.
- Velocity shaping: accent patterns (2 & 4, offbeats) define genre pocket; ghost notes add motion.
- Grid selection: 8th-note vs. 16th-note vs. ternary grids; groove is grid + offsets.
- Consistency: lock kick, snare, bass; let hats/percussion carry most humanization.

## Public repos to study (with advanced angles)
- **magenta/magenta (groove_rnn + Groove MIDI Dataset)** – Strong groove modeling baseline; inspect timing/velocity targets.
- **magenta/groove-dataset** – Labeled human drumming with micro-timing; great for statistical priors.
- **magenta/ddsp** – For expressive synthesis; combine groove timing with timbre controls.
- **harritaylor/BeatNet** – Beat/downbeat tracking; useful for aligning learned groove to tempo grids.
- **microsoft/muzic** – Contains rhythm-related tasks; review how rhythm conditioning works in generation.

## Two recursive study questions
1) How does swing percentage interact with tempo to preserve feel? → Plot human swing vs. BPM from Groove MIDI Dataset; derive a tempo-dependent swing curve.
2) How do genre pockets differ in micro-timing (e.g., neo-soul vs. funk vs. EDM)? → Cluster timing offsets/velocity curves per genre; compare kick/snare vs. hat placement.

## Advanced techniques to notice
- Micro-timing distributions modeled as continuous offsets (Gaussian mixtures) rather than fixed swing values.
- Velocity curve modeling with accent templates + stochastic ghost notes.
- Conditional groove: tie offsets to emotion (arousal) and tempo; adaptive swing per section.

## Next steps
- Task: Extract timing/velocity stats from 100 Groove MIDI files; build a small lookup table (genre→offset/velocity curves) and apply to a quantized drum pattern.
- Where to look: magenta groove_rnn scripts, Groove MIDI Dataset, BeatNet for beat alignment.
