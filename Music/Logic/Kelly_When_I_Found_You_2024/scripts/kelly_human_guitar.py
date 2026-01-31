#!/usr/bin/env python3
"""
HUMAN GUITAR - Not a track pad
- Real fingerpicking feel (1-5-6-4-3-2)
- Mitzvah warmth + bedroom imperfection
- Wide velocity but MUSICAL dynamics
- Notes don't stack - they sequence like real picking
- Each note has attack/decay character
"""

from midiutil import MIDIFile
import random

random.seed(2025)

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

def human_pick(chord, bar, energy='medium'):
    notes = []
    beat = bar * 4

    if energy == 'whisper':
        vel_base, vel_range = 28, 18
        note_chance = 0.7
        timing_drift = 0.06
        dur_base, dur_range = 2.5, 2.0
    elif energy == 'gentle':
        vel_base, vel_range = 42, 22
        note_chance = 0.85
        timing_drift = 0.045
        dur_base, dur_range = 2.0, 1.5
    elif energy == 'medium':
        vel_base, vel_range = 55, 25
        note_chance = 0.95
        timing_drift = 0.035
        dur_base, dur_range = 1.5, 1.2
    elif energy == 'strong':
        vel_base, vel_range = 70, 22
        note_chance = 1.0
        timing_drift = 0.025
        dur_base, dur_range = 1.2, 1.0
    elif energy == 'intense':
        vel_base, vel_range = 82, 15
        note_chance = 1.0
        timing_drift = 0.02
        dur_base, dur_range = 1.0, 0.8
    else:
        vel_base, vel_range = 50, 25
        note_chance = 0.9
        timing_drift = 0.04
        dur_base, dur_range = 1.8, 1.4

    pick_gap = random.uniform(0.12, 0.22)
    pattern = [1, 5, 6, 4, 3, 2]
    current_time = 0

    for i, string in enumerate(pattern):
        note = get_note(chord, string)
        if note is None:
            continue
        if random.random() > note_chance:
            continue

        if string in [5, 6]:
            drift = random.gauss(-0.02, timing_drift)
        else:
            drift = random.gauss(0.015, timing_drift)

        t = beat + current_time + drift

        if i == 0:
            vel = vel_base + random.randint(5, 15)
        else:
            decay = i * random.randint(3, 7)
            vel = vel_base - decay + random.randint(-8, 8)
        vel = max(22, min(95, vel))

        if string in [5, 6]:
            dur = random.uniform(dur_base * 0.6, dur_base * 0.9)
        else:
            dur = random.uniform(dur_base, dur_base + dur_range)

        notes.append((note, max(0, t), dur, vel))
        current_time += pick_gap + random.uniform(-0.03, 0.05)

    if energy in ['medium', 'strong', 'intense'] and random.random() > 0.3:
        second_pattern = random.sample([1, 4, 3, 2], random.randint(2, 3))
        current_time = 2.0 + random.uniform(0, 0.4)

        for string in second_pattern:
            note = get_note(chord, string)
            if note:
                drift = random.gauss(0, timing_drift)
                t = beat + current_time + drift
                vel = vel_base - 10 + random.randint(-10, 5)
                vel = max(25, min(85, vel))
                dur = random.uniform(dur_base * 0.8, dur_base + dur_range * 0.7)
                notes.append((note, max(0, t), dur, vel))
                current_time += pick_gap + random.uniform(-0.02, 0.04)

    return notes

def bass_thumb(chord, bar):
    notes = []
    bass_map = {'Dm': 38, 'Dm7': 38, 'Am': 33, 'Am7': 33,
                'F': 41, 'Fmaj7': 41, 'C': 36, 'E': 40, 'E7': 40}
    bass = bass_map.get(chord)
    if bass and random.random() > 0.4:
        t = bar * 4 + random.gauss(0, 0.03)
        vel = random.randint(35, 52)
        dur = random.uniform(1.5, 2.5)
        notes.append((bass, max(0, t), dur, vel))
    return notes

# Create MIDI
midi = MIDIFile(1, deinterleave=False)
midi.addTempo(0, 0, TEMPO)
midi.addProgramChange(0, 0, 0, 25)

all_notes = []
bar = 0

print(f"Creating HUMAN GUITAR at {TEMPO} BPM...")
print("="*55)

# INTRO (6 bars)
print("Intro: whisper...")
for chord in ['Dm', 'Dm', 'Am', 'Am', 'Dm', 'Am']:
    all_notes.extend(human_pick(chord, bar, 'whisper'))
    all_notes.extend(bass_thumb(chord, bar))
    bar += 1

# VERSE 1 (8 bars)
print("Verse 1: gentle...")
for chord in ['Dm', 'Dm', 'Am', 'Am', 'Fmaj7', 'F', 'Dm', 'Am']:
    all_notes.extend(human_pick(chord, bar, 'gentle'))
    all_notes.extend(bass_thumb(chord, bar))
    bar += 1

# BUILD (4 bars)
print("Building...")
for chord in ['Dm', 'Am', 'F', 'E']:
    all_notes.extend(human_pick(chord, bar, 'medium'))
    bar += 1

# PRE-CHORUS (4 bars)
print("Pre-chorus: strong...")
for chord in ['Dm', 'Am', 'F', 'E7']:
    all_notes.extend(human_pick(chord, bar, 'strong'))
    bar += 1

# CHORUS (8 bars) - fingerpicking continues under strum
print("Chorus: intense picking...")
for chord in ['Dm', 'Dm', 'Am', 'Am', 'F', 'C', 'Dm', 'Am']:
    all_notes.extend(human_pick(chord, bar, 'intense'))
    bar += 1

# VERSE 2 (8 bars)
print("Verse 2: gentle...")
for chord in ['Dm', 'Dm', 'Am', 'Am', 'F', 'Fmaj7', 'Dm', 'Am']:
    all_notes.extend(human_pick(chord, bar, 'gentle'))
    all_notes.extend(bass_thumb(chord, bar))
    bar += 1

# BUILD 2 (4 bars)
print("Building again...")
for chord in ['Am', 'F', 'Dm', 'E']:
    all_notes.extend(human_pick(chord, bar, 'medium'))
    bar += 1

# CHORUS 2 (8 bars)
print("Chorus 2...")
for chord in ['Dm', 'Am', 'F', 'C', 'Dm', 'Am', 'E7', 'Dm']:
    all_notes.extend(human_pick(chord, bar, 'strong'))
    bar += 1

# BRIDGE (8 bars)
print("Bridge: medium...")
for chord in ['Dm', 'Dm7', 'Am', 'Am7', 'F', 'Fmaj7', 'E', 'E7']:
    all_notes.extend(human_pick(chord, bar, 'medium'))
    bar += 1

# FINAL VERSE (8 bars)
print("Final verse: whisper...")
for chord in ['Dm', 'Am', 'F', 'Am', 'Dm', 'F', 'Am', 'Dm']:
    all_notes.extend(human_pick(chord, bar, 'whisper'))
    bar += 1

# OUTRO (6 bars)
print("Outro: fading...")
for chord in ['Dm', 'Am', 'Dm', 'Am', 'Dm', 'Dm']:
    all_notes.extend(human_pick(chord, bar, 'whisper'))
    bar += 1

all_notes.append((50, bar * 4, 10.0, 25))

print("="*55)
duration = (bar + 2) * 4 * 60 / TEMPO
print(f"Bars: {bar} | Notes: {len(all_notes)} | Duration: {duration/60:.1f} min")

for note, time, dur, vel in all_notes:
    midi.addNote(0, 0, note, time, dur, vel)

output = "/Users/seanburdges/Desktop/Kelly_Human_Guitar.mid"
with open(output, 'wb') as f:
    midi.writeFile(f)

print(f"\nSaved: {output}")
