# Arrangement Engine – Learning Guide

## What this engine does
- Designs section flow, energy arcs, and instrumentation over time.
- Places instruments, entries/exits, and texture changes to support emotion and narrative.
- Aligns harmony, melody, groove, and dynamics into a coherent song form.

## Core basics (quick fundamentals)
- Section archetypes: intro, verse, pre-chorus, chorus, bridge, outro; know their energy roles.
- Energy arcs: rise/fall, double-peak, long build; map to emotion and lyrics.
- Instrumentation layers: rhythm (drums/bass), harmony (keys/guitars), hook/lead, pads/FX.
- Density control: add/subtract parts per section; automate filters, space, and register.
- Transitions: fills, risers, drops, harmonic pivots to glue sections.

## Public repos to study (with advanced angles)
- **microsoft/muzic (SongMASS/Arranger)** – Sequence-to-sequence arrangement tasks; review attention over multi-track.
- **salu133445/mt3** – For deriving multi-instrument stems/MIDI to study real arrangements.
- **facebookresearch/musicgen** – Multi-track conditioning ideas; study how context windows affect arrangement consistency.
- **pop909-related repos (lead-sheet/arrangement)** – Real pop arrangements; mine section lengths and instrument entries.
- **magenta/ddsp** – Useful for expressive layering and timbral morphing across sections.

## Two recursive study questions
1) How do section length distributions differ by genre (pop vs. EDM vs. rock)? → Extract section markers from POP909/other MIDI sets; histogram lengths and transition patterns.
2) How does instrumentation density map to energy curves in modern mixes? → Count active tracks per bar from multi-track MIDIs; correlate with loudness/energy proxies.

## Advanced techniques to notice
- Attention over long context for section consistency (MusicGen/transformers with sliding windows or ALiBi/RoPE).
- Energy-aware arrangement: coupling dynamics/velocity curves with section energy targets.
- Instrument entry/exit scheduling learned from real stems/MIDI (probabilistic placement per section type).

## Next steps
- Task: Parse 100 pop MIDIs with section labels, compute section transition matrix and instrument entry frequencies; generate a 32-bar arrangement following those stats.
- Where to look: POP909 arrangement annotations, SongMASS/Muzic papers/repos, MusicGen code for long-context conditioning.
