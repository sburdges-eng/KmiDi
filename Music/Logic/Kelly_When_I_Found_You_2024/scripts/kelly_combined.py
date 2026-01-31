#!/usr/bin/env python3
"""
COMBINED: Human Fingerpicking + Front Porch Step Strumming
Two tracks in one MIDI file
"""

from midiutil import MIDIFile
import random

TEMPO = 82

STRINGS = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}

CHORDS = {
    'Dm':    {4: 0, 3: 2, 2: 3, 1: 1},
    'Dm7':   {4: 0, 3: 2, 2: 1, 1: 1},
    'Am':    {5: 0, 4: 2, 3: 2, 2: 1, 1: 0},
    'Am7':   {5: 0, 4: 2, 3: 0, 2: 1, 1: 0},
    'Fmaj7': {4: 3, 3: 2, 2: 1, 1: 0},
    'F':     {4: 3, 3: 2, 2: 1, 1: 1},
    'C':     {5: 3, 4: 2, 3: 0, 2: 1, 1: 0},
    'E':     {6: 0, 5: 2, 4: 2, 3: 1, 2: 0, 1: 0},
    'E7':    {6: 0, 5: 2, 4: 0, 3: 1, 2: 0, 1: 0},
}

def get_note(chord, string):
    fret = CHORDS[chord].get(string)
    if fret is None:
        return None
    return STRINGS[string] + fret

# ============ FINGERPICKING ============
def human_pick(chord, bar, energy='medium'):
    random.seed(2025 + bar)
    notes = []
    beat = bar * 4

    if energy == 'whisper':
        vel_base, vel_range = 28, 18
        note_chance = 0.7
        timing_drift = 0.06
        dur_base = 2.5
    elif energy == 'gentle':
        vel_base, vel_range = 42, 22
        note_chance = 0.85
        timing_drift = 0.045
        dur_base = 2.0
    elif energy == 'medium':
        vel_base, vel_range = 55, 25
        note_chance = 0.95
        timing_drift = 0.035
        dur_base = 1.5
    elif energy == 'strong':
        vel_base, vel_range = 70, 22
        note_chance = 1.0
        timing_drift = 0.025
        dur_base = 1.2
    elif energy == 'intense':
        vel_base, vel_range = 82, 15
        note_chance = 1.0
        timing_drift = 0.02
        dur_base = 1.0
    else:
        vel_base, vel_range = 50, 25
        note_chance = 0.9
        timing_drift = 0.04
        dur_base = 1.8

    pick_gap = random.uniform(0.12, 0.22)
    pattern = [1, 5, 6, 4, 3, 2]
    current_time = 0

    for i, string in enumerate(pattern):
        note = get_note(chord, string)
        if note is None or random.random() > note_chance:
            continue

        drift = random.gauss(-0.02 if string in [5,6] else 0.015, timing_drift)
        t = beat + current_time + drift

        vel = vel_base + random.randint(5, 15) if i == 0 else vel_base - i*5 + random.randint(-8, 8)
        vel = max(22, min(95, vel))

        dur = random.uniform(dur_base * 0.6, dur_base) if string in [5,6] else random.uniform(dur_base, dur_base + 1.2)
        notes.append((note, max(0, t), dur, vel))
        current_time += pick_gap + random.uniform(-0.03, 0.05)

    if energy in ['medium', 'strong', 'intense'] and random.random() > 0.3:
        for string in random.sample([1, 4, 3, 2], random.randint(2, 3)):
            note = get_note(chord, string)
            if note:
                t = beat + 2.0 + random.uniform(0, 0.4) + random.gauss(0, timing_drift)
                vel = vel_base - 10 + random.randint(-10, 5)
                notes.append((note, max(0, t), random.uniform(dur_base * 0.8, dur_base + 0.7), max(25, min(85, vel))))

    return notes

def bass_thumb(chord, bar):
    random.seed(3000 + bar)
    notes = []
    bass_map = {'Dm': 38, 'Dm7': 38, 'Am': 33, 'Am7': 33, 'F': 41, 'Fmaj7': 41, 'C': 36, 'E': 40, 'E7': 40}
    bass = bass_map.get(chord)
    if bass and random.random() > 0.4:
        notes.append((bass, max(0, bar * 4 + random.gauss(0, 0.03)), random.uniform(1.5, 2.5), random.randint(35, 52)))
    return notes

# ============ STRUMMING ============
def front_porch_strum(chord, bar, intensity='full'):
    random.seed(4000 + bar)
    notes = []
    beat = bar * 4

    if intensity == 'full':
        pattern = [(0.0, 'down', 1.0), (1.0, 'down', 0.85), (1.5, 'up', 0.7), (2.0, 'up', 0.75), (2.5, 'down', 0.9), (3.5, 'up', 0.65)]
    elif intensity == 'half':
        pattern = [(0.0, 'down', 1.0), (1.0, 'down', 0.8), (1.5, 'up', 0.65)]
    else:
        pattern = [(0.0, 'down', 1.0), (2.5, 'down', 0.95)]

    for beat_offset, direction, vel_mod in pattern:
        notes.extend(rake_strum(chord, beat + beat_offset, direction, vel_mod))
    return notes

def rake_strum(chord, start_time, direction, vel_mod):
    notes = []
    strings = [6, 5, 4, 3, 2, 1] if direction == 'down' else [1, 2, 3, 4]
    base_vel = int(78 * vel_mod) if direction == 'down' else int(65 * vel_mod)

    available = [(s, get_note(chord, s)) for s in strings if get_note(chord, s)]
    strum_time = 0

    for i, (string, note) in enumerate(available):
        gap = random.uniform(0.015, 0.030)
        t = start_time + strum_time + random.gauss(0, 0.008)

        if direction == 'down':
            vel = base_vel - i * random.randint(4, 8) + random.randint(-6, 6)
        else:
            vel = base_vel + i * random.randint(2, 5) + random.randint(-5, 5)

        notes.append((note, max(0, t), random.uniform(0.4, 0.7), max(35, min(95, vel))))
        strum_time += gap
    return notes

# ============ CREATE COMBINED MIDI ============
midi = MIDIFile(2, deinterleave=False)
midi.addTempo(0, 0, TEMPO)
midi.addTempo(1, 0, TEMPO)
midi.addProgramChange(0, 0, 0, 25)  # Track 0: Fingerpicking (Steel)
midi.addProgramChange(1, 1, 0, 25)  # Track 1: Strumming (Steel)

pick_notes = []
strum_notes = []

print(f"Creating COMBINED guitar tracks at {TEMPO} BPM...")
print("Track 1: Human fingerpicking (1-5-6-4-3-2)")
print("Track 2: Front Porch Step strumming (D-D-U-U-D-U)")
print("="*55)

# Song structure with bar counter
bar = 0

# INTRO (6 bars) - Fingerpicking only
print("Intro: whisper fingerpicking...")
for chord in ['Dm', 'Dm', 'Am', 'Am', 'Dm', 'Am']:
    pick_notes.extend(human_pick(chord, bar, 'whisper'))
    pick_notes.extend(bass_thumb(chord, bar))
    bar += 1

# VERSE 1 (8 bars) - Fingerpicking only
print("Verse 1: gentle fingerpicking...")
for chord in ['Dm', 'Dm', 'Am', 'Am', 'Fmaj7', 'F', 'Dm', 'Am']:
    pick_notes.extend(human_pick(chord, bar, 'gentle'))
    pick_notes.extend(bass_thumb(chord, bar))
    bar += 1

# BUILD (4 bars)
print("Building...")
for chord in ['Dm', 'Am', 'F', 'E']:
    pick_notes.extend(human_pick(chord, bar, 'medium'))
    bar += 1

# PRE-CHORUS (4 bars)
print("Pre-chorus: strong...")
for chord in ['Dm', 'Am', 'F', 'E7']:
    pick_notes.extend(human_pick(chord, bar, 'strong'))
    bar += 1

# CHORUS 1 (8 bars) - BOTH TRACKS
print(">>> CHORUS 1: Fingerpicking + Strumming...")
for chord in ['Dm', 'Dm', 'Am', 'Am', 'F', 'C', 'Dm', 'Am']:
    pick_notes.extend(human_pick(chord, bar, 'intense'))
    strum_notes.extend(front_porch_strum(chord, bar, 'full'))
    bar += 1

# VERSE 2 (8 bars) - Fingerpicking only
print("Verse 2: gentle fingerpicking...")
for chord in ['Dm', 'Dm', 'Am', 'Am', 'F', 'Fmaj7', 'Dm', 'Am']:
    pick_notes.extend(human_pick(chord, bar, 'gentle'))
    pick_notes.extend(bass_thumb(chord, bar))
    bar += 1

# BUILD 2 (4 bars)
print("Building again...")
for chord in ['Am', 'F', 'Dm', 'E']:
    pick_notes.extend(human_pick(chord, bar, 'medium'))
    bar += 1

# CHORUS 2 (8 bars) - BOTH TRACKS
print(">>> CHORUS 2: Fingerpicking + Strumming...")
for chord in ['Dm', 'Am', 'F', 'C', 'Dm', 'Am', 'E7', 'Dm']:
    pick_notes.extend(human_pick(chord, bar, 'strong'))
    strum_notes.extend(front_porch_strum(chord, bar, 'full'))
    bar += 1

# BRIDGE (8 bars) - Both but softer strum
print("Bridge: medium + half strum...")
for i, chord in enumerate(['Dm', 'Dm7', 'Am', 'Am7', 'F', 'Fmaj7', 'E', 'E7']):
    pick_notes.extend(human_pick(chord, bar, 'medium'))
    strum_notes.extend(front_porch_strum(chord, bar, 'half' if i % 2 == 0 else 'accent'))
    bar += 1

# FINAL VERSE (8 bars) - Fingerpicking only, whisper
print("Final verse: whisper...")
for chord in ['Dm', 'Am', 'F', 'Am', 'Dm', 'F', 'Am', 'Dm']:
    pick_notes.extend(human_pick(chord, bar, 'whisper'))
    bar += 1

# OUTRO (6 bars) - Fingerpicking fading
print("Outro: fading...")
for chord in ['Dm', 'Am', 'Dm', 'Am', 'Dm', 'Dm']:
    pick_notes.extend(human_pick(chord, bar, 'whisper'))
    bar += 1

# Final notes
pick_notes.append((50, bar * 4, 10.0, 25))

# Write to MIDI
print("="*55)
duration = (bar + 2) * 4 * 60 / TEMPO
print(f"Bars: {bar} | Duration: {duration/60:.1f} min")
print(f"Fingerpicking notes: {len(pick_notes)}")
print(f"Strumming notes: {len(strum_notes)}")
print(f"Total: {len(pick_notes) + len(strum_notes)}")

for note, time, dur, vel in pick_notes:
    midi.addNote(0, 0, note, time, dur, vel)

for note, time, dur, vel in strum_notes:
    midi.addNote(1, 1, note, time, dur, vel)

output = "/Users/seanburdges/Desktop/Kelly_Combined_Guitar.mid"
with open(output, 'wb') as f:
    midi.writeFile(f)

print(f"\nSaved: {output}")
print("""
COMBINED TRACKS:
- Track 1: Fingerpicking (throughout)
- Track 2: Strumming (chorus/bridge only)
- Both at 82 BPM, synced
- Ready to open in Logic Pro
""")
