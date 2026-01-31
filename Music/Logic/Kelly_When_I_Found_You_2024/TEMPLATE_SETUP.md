# WHEN I FOUND YOU SLEEPING - Logic Pro Template
## Lo-Fi Bedroom Emo Setup

---

## PROJECT SETTINGS
- **Tempo:** 82 BPM
- **Time Signature:** 4/4
- **Key:** D minor (Dm)
- **Sample Rate:** 48kHz
- **Bit Depth:** 24-bit

---

## TRACK LAYOUT

### 1. GUITAR - Fingerpicking (Track 1)
**Instrument:** Steel String Acoustic (Sampler)
**Pan:** Center
**Volume:** 0 dB

**Plugin Chain:**
1. **BritPre** (Analog Obsession) - Subtle warmth
   - Drive: 2-3
   - Output: matched
2. **Rare** (Analog Obsession) - Pultec EQ
   - Low Boost: 2-3 at 100Hz
   - High Boost: 2-3 at 10kHz
   - Atten: 1-2 at 4kHz
3. **TDR Nova** - Dynamic EQ
   - Gentle compression on mids
   - Tame harsh frequencies

**Send:** Room Reverb (15-20%)

---

### 2. GUITAR - Strumming (Track 2)
**Instrument:** Steel String Acoustic (Sampler)
**Pan:** -15 Left
**Volume:** -4 dB

**Plugin Chain:**
1. **PreBOX** (Analog Obsession)
2. **RareSE** - Different EQ curve
3. **Britpressor** - Light compression (2:1)

**Send:** Room Reverb (10%)

---

### 3. VOCAL - Main (Track 3)
**Input:** Microphone
**Pan:** Center
**Volume:** 0 dB

**Plugin Chain:**
1. **BritPre** - Preamp color
2. **Channel EQ** - High-pass 80Hz
3. **FETish** - Vocal compression
4. **DeEsser** - If needed
5. **Rare** - Air boost at 12kHz

**Send:** Room Reverb (20%), Plate Reverb (10%)

---

### 4. VOCAL - Whisper/Double (Track 4)
**Pan:** +20 Right
**Volume:** -8 dB (sits behind main)

**Plugin Chain:**
- Similar to main but more lo-fi
- **MSaturator** for grit
- Higher reverb send

---

### 5. AMBIENT PAD (Track 5)
**Instrument:** Alchemy - Mood Acoustic or pad
**Pan:** Wide stereo
**Volume:** -12 dB (barely there)

**Plugin Chain:**
1. **Valhalla Supermassive** - Long reverb/delay
2. **Low-pass filter** at 3kHz

---

## BUS/AUX TRACKS

### Room Reverb Bus
**Plugin:** Space Designer or MReverb
- Small room impulse
- Decay: 0.8-1.2s
- Pre-delay: 10-20ms
- Mix: 100% wet (send)

### Plate Reverb Bus
**Plugin:** MReverb or ChromaVerb
- Plate setting
- Decay: 1.5-2s
- For vocals

### Master Bus
**Plugin Chain:**
1. **MStereoProcessor** - Subtle widening
2. **MEqualizer** - Final tone shaping
3. **COMBOX** or **BusterSE** - Glue compression
4. **MAutoVolume** - Level consistency
5. **Limiter** - -1dB ceiling

---

## MARKERS

| Bar | Time | Section |
|-----|------|---------|
| 1 | 0:00 | INTRO |
| 7 | 0:18 | VERSE 1 |
| 23 | 0:55 | CHORUS |
| 31 | 1:13 | VERSE 2 |
| 47 | 1:50 | CHORUS 2 |
| 55 | 2:08 | BRIDGE |
| 63 | 2:26 | FREEZE |
| 71 | 2:44 | FINAL |
| 79 | 3:02 | OUTRO |

---

## COLOR CODING

| Color | Track Type |
|-------|------------|
| Green | Guitars |
| Blue | Vocals |
| Purple | Ambient/Pads |
| Orange | Drums/Percussion (if any) |
| Gray | Buses |

---

## IMPORTING MIDI

1. Import `Kelly_Combined_Guitar.mid`
2. Track 1 = Fingerpicking
3. Track 2 = Strumming (starts at bar 23)
4. Set tempo to 82 BPM

---

## QUICK SETUP STEPS

1. **New Project** → Empty Project
2. **Set Tempo:** 82 BPM (Option+Click on tempo)
3. **Add Software Instrument** → Steel String Acoustic
4. **Duplicate Track** for strumming
5. **Add Audio Track** for vocals
6. **Create Buses** (Cmd+Option+N)
7. **Import MIDI** (Cmd+I)
8. **Add Plugins** per chain above
9. **Set Markers** (Option+')
10. **Save as Template** (File > Save as Template)

---

## LO-FI CHARACTER

To get the bedroom recording feel:
- Don't over-compress
- Leave dynamics intact
- Slight tape saturation (MSaturator at 5-10%)
- Roll off highs above 10kHz
- Add subtle room reverb
- Don't quantize - keep human timing
- Let notes ring and bleed

---

*Template for "When I Found You Sleeping"*
*82 BPM | D minor | Lo-Fi Bedroom Emo*
