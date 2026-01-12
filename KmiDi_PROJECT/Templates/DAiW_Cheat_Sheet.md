# DAiW Quick Reference Cheat Sheet

## Emotion → Parameters (30-Second Lookup)

| Emotion | BPM | Mode | Timing | Dissonance | Density |
|---------|-----|------|--------|------------|---------|
| **Grief** | 60-82 | Minor/Dorian | Behind | 30% | Sparse |
| **Anxiety** | 100-140 | Phrygian/Minor | Ahead | 60% | Busy |
| **Nostalgia** | 70-90 | Mixolydian/Major | Behind | 25% | Moderate |
| **Anger** | 120-160 | Phrygian/Minor | Ahead | 50% | Dense |
| **Calm** | 60-80 | Major/Lydian | Behind | 10% | Sparse |

---

## Interval Tension (Semitones)

```
m2(1)  = 90%  → dread, clash
M2(2)  = 40%  → yearning, suspension  
m3(3)  = 30%  → sadness
M3(4)  = 20%  → brightness
P4(5)  = 30%  → questioning
TT(6)  = 100% → wrongness, danger
P5(7)  = 10%  → stability
m6(8)  = 60%  → anguish
M6(9)  = 25%  → warmth
m7(10) = 50%  → bluesy longing
M7(11) = 55%  → bittersweet
```

---

## Timing Feel

| Feel | Where Notes Land | Use For |
|------|------------------|---------|
| **Ahead** | Before beat | Anxiety, anger, urgency |
| **On** | Precisely on beat | Neutral, steady, marching |
| **Behind** | After beat | Grief, calm, blues, resignation |

---

## Compound Emotion Modifiers

| Modifier | Add To | Effect |
|----------|--------|--------|
| **PTSD Intrusion** | Grief | 15% chance: register spike, harmonic rush, unresolved dissonance |
| **Misdirection** | Any | Surface positive, undertow negative |
| **Dissociation** | Any | Oddly smooth, detached, higher register |
| **Suppressed** | Any | Controlled dynamics, tension underneath |

---

## Rule-Breaking Quick Reference

| Want This Feel? | Break This Rule | Example |
|-----------------|-----------------|---------|
| Raw/powerful | Parallel fifths | Power chords, Beethoven |
| Chaos/primal | Polytonality | Stravinsky Rite |
| Meaningful wrongness | Unresolved dissonance | Monk |
| Happy↔sad ambiguity | Modal mixture (iv) | Radiohead Creep |
| Anthemic yearning | ♭VII borrowed | Oasis Wonderwall |
| Floating/dreamy | ♭VII in pop | Rihanna Diamonds |
| Gut-punch sadness | ♭VI borrowed | Taylor Swift All Too Well |
| Desperate searching | ♭VI in minor | R.E.M. Losing My Religion |
| Epic dark power | Stacked borrowing | Imagine Dragons Radioactive |
| Floating/unsettled | Ambiguous meter | Radiohead Pyramid Song |
| Haunting/unfinished | Non-resolution | Chopin Prelude |
| Dread/evil | Unresolved tritone | Black Sabbath |

---

## Modal Mixture Cheat Sheet

| Borrowed Chord | From Mode | Effect | Example |
|----------------|-----------|--------|---------|
| ♭VII | Mixolydian | Yearning, anthemic | Wonderwall |
| ♭VI | Aeolian | Gut-punch, sudden dark | All Too Well |
| iv | Parallel minor | Happy→sad pivot | Creep |
| ♭III | Aeolian | Epic, powerful | Radioactive |

**For Kelly Song (F major):**
- ♭VII = E♭ major (yearning)
- ♭VI = D♭ major (gut-punch reveal)
- iv = Bb minor (sad color)

---

## Kelly Song Settings

```
Preset:      grief + misdirection + ptsd_intrusion
Progression: F - C - Am - Dm (or upgraded: Fmaj7 - C/E - Am7 - Dm7)
BPM:         82
Key:         F major surface / Am undertow
Feel:        Behind the beat
Technique:   Every line sounds like love until reveal

CHORD OPTIONS:
• Reveal moment: F → Fm → C (Elliott Smith gut-punch)
• Add gravity: F - F/E - Dm7 - Dm/C (descending bass)
• Gut-punch: F - C - D♭ - Dm (♭VI borrowed)

PRODUCTION (Elliott Smith style):
• Double-track vocals: Record twice, pan 30% L/R
• Close mic: 3-6" from source
• Tune down whole step: D-G-C-F-A-D
• Keep finger noise and breath

AVOID:
✗ Perfect cadences
✗ Root position chords
✗ Technical perfection
✗ Over-polishing
✗ Artificial reverb (use double-tracking instead)
```

---

## Interrogation Prompts (Copy-Paste)

**Before suggesting tempo:**
> Does [X] BPM feel right, or does this need to breathe more/less?

**Before timing feel:**
> The timing maps to 'behind the beat' - heavy, resigned. Does that match?

**For high dissonance:**
> This maps to lots of unresolved tension. Is that the texture you want?

**For compound emotions:**
> Is this pure [emotion] or is there something else underneath?

**For Kelly song:**
> What's the lie this song tells before the reveal?

---

## File Locations

| File | Purpose | Upload to GPT? |
|------|---------|----------------|
| `daiw_knowledge_base.json` | Complete presets & mappings | ✅ Yes |
| `rule_breaking_database.json` | All rule-break examples | ✅ Yes |
| `emotional_mapping.py` | Code implementation | ✅ Yes |
| `Integration_Architecture.md` | Workflow diagram | Optional |
| `Custom_GPT_Build_Script.md` | Setup instructions | No (for you) |

---

## The One Rule

> "Well, who has forbidden them? Well, I allow them!" — Beethoven

Every rule-break needs emotional justification. No justification = follow the rule.
