# Harmony Engine – Learning Guide

## What this engine does
- Generates chord progressions/voicings aligned to emotion, style, and rule-breaking choices.
- Manages tonal center, modal interchange, cadences, and voice leading for smooth movement.
- Feeds downstream melody, bass, and arrangement decisions.

## Core basics (quick fundamentals)
- Roman numerals = transposable grammar; learn I–V–vi–IV (pop), ii–V–I (jazz), iv–V–I (folk), bVII (rock/mixolydian).
- Cadences shape tension: authentic (V–I), plagal (IV–I), deceptive (V–vi), half (…–V).
- Modal interchange: borrow iv, bVII, bVI, ii° for color; keep voice leading smooth.
- Voice leading: retain common tones, move by step, avoid parallel perfects in tonal contexts.
- Rhythm of harmony: bar-level vs. half-bar changes; anticipate/delay for push/pull.

## Public repos to study (with advanced angles)
- **cuthbertLab/music21** – Robust harmonic analysis tools; inspect roman numeral and voice-leading utilities.
- **microsoft/muzic** – Chord/lead-sheet tasks; see GPT-style conditioning on chords.
- **bzamecnik/harmcalc** – Harmony analysis/generation experiments; good for rule-based grounding.
- **openai/musenet (archived info)** – Not open weights, but read paper/blog for multi-instrument harmonic conditioning ideas.
- **ybayle/Pop909-dataset and related chord labeling repos** – Real pop progressions for empirical priors.

## Two recursive study questions
1) How do different cadential densities affect perceived resolution in pop vs. jazz? → Analyze cadence frequency in POP909 vs. Real Book leadsheets using music21.
2) Which borrowed-chord patterns recur in modern film/game scores? → Extract modal interchange counts from a small score set; cluster by emotion tags.

## Advanced techniques to notice
- Conditioning chords on emotion embeddings and groove context (tempo, swing) for coherence.
- Learned voice-leading: neural voicing models vs. rule-based spreading; inversion selection to minimize motion.
- Multi-scale harmony: local cadences + global key modulations (secondary dominants, pivot chords).

## Next steps
- Task: Parse 200 pop progressions (POP909), compute borrowed-chord rates and cadence types; generate 16-bar progressions matching those stats.
- Where to look: music21 roman numeral tools; POP909 chord annotations; muzic chord/lead-sheet examples.
