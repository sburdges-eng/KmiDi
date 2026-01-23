# Melody Engine – Learning Guide

## What this engine does
- Generates melodic lines that carry the emotional intent of a track.
- Balances contour, interval choices, rhythm, and tonal center.
- Interfaces with harmony, groove, and expression layers for cohesive output.

## Core basics (quick fundamentals)
- Contour matters: rising = tension/energy, falling = release, arches = storytelling.
- Intervals set character: steps feel connected; leaps feel dramatic; repeated tones feel static.
- Scale/Mode choice encodes mood (major/lydian for bright, dorian/mixolydian for cool, phrygian/locrian for dark).
- Rhythm drives memorability: syncopation, pickups, and motifs over strict time.
- Motif development: repeat, invert, augment/diminish rhythms to build coherence.

## Public repos to study (with advanced angles)
- **magenta/magenta** – MelodyRNN, Music Transformer; strong baselines + datasets for melody modeling.
- **lucidrains/music-transformer-pytorch** – Clean Music Transformer implementation; inspect relative attention for long melodic arcs.
- **microsoft/muzic** – Collection of music generation models (MIDI GPT-style); see melody & lead-sheet tasks.
- **salu133445/mt3** – Multi-task transcription; study how melody is extracted for training data creation.
- **YCZhou2019/Pop-Melody-Generation (POP909 related repos)** – Melody datasets + generation scripts for pop idioms.

## Two recursive study questions
1) How do different attention mechanisms affect long-form melodic coherence? → Compare Music Transformer vs. decoder-only GPT (Muzic) on 16–32 bar phrases.
2) How do genre-specific interval/contour priors shift melody style? → Analyze POP909 melodies vs. folk/JSB datasets; compute interval histograms and contour patterns.

## Advanced techniques to notice
- Transformer-based melody modeling (relative position, ALiBi/RoPE) for long context.
- Conditioning on chords/groove/lyrics for controllable melody (lead-sheet or multi-track conditioning).
- Data curation: phrase-level segmentation, motif extraction, and augmentation (transposition, rhythm scaling).

## Next steps
- Task: Extract 50 melodies from POP909, build interval/contour histograms, and recreate 8-bar melodies using those stats.
- Where to look: POP909 repos, magenta melody scripts, music-transformer-pytorch for decoding scaffolds.
