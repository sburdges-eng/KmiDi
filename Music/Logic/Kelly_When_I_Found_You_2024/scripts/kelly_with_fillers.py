#!/usr/bin/env python3
"""
MITZVAH STYLE + FILLER TONES
- Original fingerpicking pattern (1-5-6-4-3-2)
- Added: Sustained bass drones
- Added: Ambient chord pad tones
- Added: Harmonic ghost notes
"""

from midiutil import MIDIFile
import random

random.seed(2024)

TEMPO = 76

# Standard tuning MIDI notes
STRINGS = {
    6: 40,  # E2
    5: 45,  # A2
    4: 50,  # D3
    3: 55,  # G3
    2: 59,  # B3
    1: 64,  # E4
}

# Chord voicings
CHORDS = {
    'Dm':    {4: 0, 3: 2, 2: 3, 1: 1},
    'Dm7':   {4: 0, 3: 2, 2: 1, 1: 1},
    'Am':    {5: 0, 4: 2, 3: 2, 2: 1, 1: 0},
    'Am7':   {5: 0, 4: 2, 3: 0, 2: 1, 1: 0},
    'Fmaj7': {4: 3, 3: 2, 2: 1, 1: 0},
    'E':     {6: 0, 5: 2, 4: 2, 3: 1, 2: 0, 1: 0},
    'E7':    {6: 0, 5: 2, 4: 0, 3: 1, 2: 0, 1: 0},
}

# Bass roots for filler drones
BASS_ROOTS = {
    'Dm': 38,   # D2
    'Dm7': 38,
    'Am': 33,   # A1
    'Am7': 33,
    'Fmaj7': 41, # F2
    'E': 40,    # E2
    'E7': 40,
}

# Chord tones for pad fills (mid-range)
PAD_TONES = {
    'Dm': [50, 53, 57],    # D3, F3, A3
    'Dm7': [50, 53, 57, 60], # + C4
    'Am': [45, 48, 52],    # A2, C3, E3
    'Am7': [45, 48, 52, 55], # + G3
    'Fmaj7': [41, 45, 48, 52], # F2, A2, C3, E3
    'E': [40, 44, 47, 52],  # E2, G#2, B2, E3
    'E7': [40, 44, 47, 50], # E2, G#2, B2, D3
}

def get_note(chord, string):
    fret = CHORDS[chord].get(string)
    if fret is None:
        return None
    return STRINGS[string] + fret

def mitzvah_style(chord, bar, variation='normal'):
    """Main fingerpicking pattern"""
    notes = []
    beat = bar * 4

    if variation == 'full':
        pattern = [(1, 0), (5, 0.33), (6, 0.66), (4, 1.0), (3, 1.33), (2, 1.66)]
        for string, offset in pattern:
            note = get_note(chord, string)
            if note:
                t = beat + offset + random.gauss(0, 0.02)
                vel = random.randint(75, 88) if offset == 0 else random.randint(45, 58)
                dur = random.uniform(2.0, 3.5)
                notes.append((note, t, dur, vel))

    elif variation == 'sparse':
        note = get_note(chord, random.choice([1, 4, 5]))
        if note:
            t = beat + random.gauss(0, 0.015)
            vel = random.randint(72, 85)
            dur = random.uniform(3.0, 4.5)
            notes.append((note, t, dur, vel))

        if random.random() > 0.3:
            note = get_note(chord, random.choice([2, 3]))
            if note:
                t = beat + random.uniform(2.0, 3.0)
                vel = random.randint(35, 48)
                dur = random.uniform(2.5, 4.0)
                notes.append((note, t, dur, vel))

    elif variation == 'arpeggio':
        available = [s for s in [6,5,4,3,2,1] if get_note(chord, s)]
        for i, string in enumerate(available[:4]):
            note = get_note(chord, string)
            t = beat + (i * 0.5) + random.gauss(0, 0.02)
            vel = 75 - (i * 8)
            dur = random.uniform(2.5, 4.0)
            notes.append((note, t, dur, max(35, vel)))

    elif variation == 'dying':
        note = get_note(chord, random.choice([4, 5, 1]))
        if note:
            notes.append((note, beat, random.uniform(5.0, 8.0), random.randint(25, 38)))

    return notes

def bass_drone(chord, bar, duration_bars=1):
    """Sustained bass drone filler"""
    notes = []
    beat = bar * 4
    bass = BASS_ROOTS.get(chord)
    if bass:
        # Main drone - very soft, long sustain
        t = beat + random.uniform(-0.1, 0.1)
        dur = duration_bars * 4 - 0.5
        vel = random.randint(28, 38)  # Very soft
        notes.append((bass, t, dur, vel))

        # Octave above, even softer
        if random.random() > 0.5:
            notes.append((bass + 12, t + 0.1, dur - 0.3, vel - 8))
    return notes

def pad_fill(chord, bar):
    """Ambient pad chord tones"""
    notes = []
    beat = bar * 4
    tones = PAD_TONES.get(chord, [])

    # Pick 1-2 random chord tones
    selected = random.sample(tones, min(2, len(tones)))

    for tone in selected:
        # Stagger entry slightly
        t = beat + random.uniform(0.5, 1.5)
        dur = random.uniform(2.5, 3.5)
        vel = random.randint(22, 32)  # Ghost-like, barely there
        notes.append((tone, t, dur, vel))

    return notes

def harmonic_ghost(chord, bar):
    """Occasional harmonic ghost notes"""
    notes = []
    if random.random() > 0.7:  # Only 30% of the time
        beat = bar * 4
        # Natural harmonics at 12th fret (octave up)
        base_notes = [52, 57, 62, 64]  # E3, A3, D4, E4 harmonics
        harm = random.choice(base_notes)
        t = beat + random.uniform(2.5, 3.5)
        dur = random.uniform(1.5, 2.5)
        vel = random.randint(18, 26)  # Whisper soft
        notes.append((harm, t, dur, vel))
    return notes

# Create MIDI - 2 tracks: Guitar + Filler
midi = MIDIFile(2, deinterleave=False)
midi.addTempo(0, 0, TEMPO)
midi.addTempo(1, 0, TEMPO)

# Track 0: Main guitar (Program 25 - Acoustic Steel)
midi.addProgramChange(0, 0, 0, 25)
# Track 1: Filler tones (Program 25 - same guitar, or 24 for nylon)
midi.addProgramChange(1, 1, 0, 24)  # Nylon guitar for softer fill

guitar_notes = []
filler_notes = []
bar = 0

print("Creating MITZVAH STYLE + FILLER TONES...")
print("="*50)

# INTRO (4 bars)
print("Intro: sparse arpeggios + bass drones...")
for chord in ['Dm', 'Dm', 'Am', 'Am']:
    guitar_notes.extend(mitzvah_style(chord, bar, 'arpeggio'))
    filler_notes.extend(bass_drone(chord, bar))
    bar += 1

# VERSE 1 (8 bars)
print("Verse 1: full pattern + pad fills...")
v1 = ['Dm', 'Dm', 'Am', 'Am', 'Fmaj7', 'Fmaj7', 'E', 'E7']
for i, chord in enumerate(v1):
    var = 'full' if i % 2 == 0 else 'sparse'
    guitar_notes.extend(mitzvah_style(chord, bar, var))
    filler_notes.extend(bass_drone(chord, bar))
    if i % 2 == 1:  # Add pad on alternate bars
        filler_notes.extend(pad_fill(chord, bar))
    filler_notes.extend(harmonic_ghost(chord, bar))
    bar += 1

# VERSE 2 (8 bars)
print("Verse 2...")
v2 = ['Dm', 'Dm', 'Am', 'Am', 'Fmaj7', 'Dm', 'E', 'Am']
for i, chord in enumerate(v2):
    var = 'sparse' if i in [4, 5] else 'full'
    guitar_notes.extend(mitzvah_style(chord, bar, var))
    filler_notes.extend(bass_drone(chord, bar))
    filler_notes.extend(pad_fill(chord, bar))
    bar += 1

# CHORUS (8 bars)
print("Chorus: full energy + harmonics...")
chorus = ['Dm', 'Dm', 'Am', 'Am', 'E', 'E7', 'Fmaj7', 'Am']
for i, chord in enumerate(chorus):
    guitar_notes.extend(mitzvah_style(chord, bar, 'full'))
    filler_notes.extend(bass_drone(chord, bar))
    filler_notes.extend(pad_fill(chord, bar))
    filler_notes.extend(harmonic_ghost(chord, bar))
    bar += 1

# INSTRUMENTAL (8 bars)
print("Instrumental: arpeggios + sustained drones...")
for chord in ['Dm', 'Dm7', 'Am', 'Am7', 'Fmaj7', 'Dm', 'E', 'Am']:
    guitar_notes.extend(mitzvah_style(chord, bar, 'arpeggio'))
    filler_notes.extend(bass_drone(chord, bar, 2))  # Longer drones
    bar += 1

# VERSE 3 (8 bars)
print("Verse 3: sparse + ambient...")
for chord in ['Dm', 'Dm', 'Am', 'Am', 'Fmaj7', 'Am', 'E', 'Am']:
    guitar_notes.extend(mitzvah_style(chord, bar, 'sparse'))
    filler_notes.extend(bass_drone(chord, bar))
    filler_notes.extend(pad_fill(chord, bar))
    filler_notes.extend(harmonic_ghost(chord, bar))
    bar += 1

# CHORUS 2 (8 bars)
print("Chorus 2...")
for chord in chorus:
    var = 'full' if bar % 2 == 0 else 'sparse'
    guitar_notes.extend(mitzvah_style(chord, bar, var))
    filler_notes.extend(bass_drone(chord, bar))
    filler_notes.extend(pad_fill(chord, bar))
    bar += 1

# BRIDGE (8 bars)
print("Bridge: minimal fills...")
for chord in ['Dm', 'Dm7', 'Am', 'Am7', 'E', 'E7', 'Dm', 'Am']:
    guitar_notes.extend(mitzvah_style(chord, bar, 'sparse'))
    filler_notes.extend(bass_drone(chord, bar))
    bar += 1

# FINAL (8 bars)
print("Final: dying notes + sustained drones...")
for chord in ['Dm', 'Dm', 'Am', 'Am', 'E', 'E', 'Dm', 'Am']:
    guitar_notes.extend(mitzvah_style(chord, bar, 'dying'))
    filler_notes.extend(bass_drone(chord, bar, 2))
    bar += 1

# OUTRO (4 bars)
print("Outro: fade...")
for chord in ['Dm', 'Am']:
    guitar_notes.extend(mitzvah_style(chord, bar, 'dying'))
    # Long drone fade
    bass = BASS_ROOTS.get(chord)
    if bass:
        filler_notes.append((bass, bar * 4, 12.0, 20))
    bar += 2

# Final hold
midi.addNote(0, 0, 50, bar * 4, 10.0, 20)
midi.addNote(1, 1, 38, bar * 4, 12.0, 15)  # D2 bass drone

# Write notes
print("="*50)
duration = (bar + 2) * 4 * 60 / TEMPO
print(f"Bars: {bar}")
print(f"Guitar notes: {len(guitar_notes)}")
print(f"Filler notes: {len(filler_notes)}")
print(f"Total: {len(guitar_notes) + len(filler_notes)}")
print(f"Duration: {duration/60:.1f} min")

for note, time, dur, vel in guitar_notes:
    midi.addNote(0, 0, note, max(0, time), dur, vel)

for note, time, dur, vel in filler_notes:
    midi.addNote(1, 1, note, max(0, time), dur, vel)

output = "/Users/seanburdges/Desktop/Kelly_Mitzvah_Fillers.mid"
with open(output, 'wb') as f:
    midi.writeFile(f)

print(f"\nSaved: {output}")
print("""
FILLER TONES ADDED:
- Bass drones (D2, A1, F2, E2) - vel 28-38
- Pad chord tones (soft sustained) - vel 22-32
- Harmonic ghosts (12th fret naturals) - vel 18-26
- Nylon guitar timbre on filler track
""")
