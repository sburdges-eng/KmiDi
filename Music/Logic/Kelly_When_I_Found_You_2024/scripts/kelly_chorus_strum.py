#!/usr/bin/env python3
"""
CHORUS STRUMMING - Front Porch Step "Island of a Misfit Boy" style
- Raw folk-emo strum pattern
- D D U U D U pattern (down down up up down up)
- ~100 BPM feel but matched to song tempo
- Aggressive but human strumming
- Rake across strings, not synth pad
"""

from midiutil import MIDIFile
import random

random.seed(2026)

TEMPO = 82  # Match main guitar

STRINGS = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}

CHORDS = {
    'Dm':    {4: 0, 3: 2, 2: 3, 1: 1},
    'Am':    {5: 0, 4: 2, 3: 2, 2: 1, 1: 0},
    'F':     {4: 3, 3: 2, 2: 1, 1: 1},
    'C':     {5: 3, 4: 2, 3: 0, 2: 1, 1: 0},
    'E7':    {6: 0, 5: 2, 4: 0, 3: 1, 2: 0, 1: 0},
}

def get_note(chord, string):
    fret = CHORDS[chord].get(string)
    if fret is None:
        return None
    return STRINGS[string] + fret

def front_porch_strum(chord, bar, intensity='full'):
    """
    Front Porch Step style strum:
    D D U U D U pattern
    |1 + 2 + 3 + 4 +|
    |D . D U U D . U|

    Beat positions: 1, 2, 2.5, 3, 3.5, 4.5
    Down strums: low to high (bass first)
    Up strums: high to low (treble first)
    """
    notes = []
    beat = bar * 4

    # Pattern: (beat_offset, direction, velocity_mod)
    # D D U U D U
    if intensity == 'full':
        pattern = [
            (0.0, 'down', 1.0),      # Beat 1 - strong down
            (1.0, 'down', 0.85),     # Beat 2 - down
            (1.5, 'up', 0.7),        # Beat 2+ - up
            (2.0, 'up', 0.75),       # Beat 3 - up
            (2.5, 'down', 0.9),      # Beat 3+ - accent down
            (3.5, 'up', 0.65),       # Beat 4+ - up
        ]
    elif intensity == 'half':
        # Just first half of bar
        pattern = [
            (0.0, 'down', 1.0),
            (1.0, 'down', 0.8),
            (1.5, 'up', 0.65),
        ]
    elif intensity == 'accent':
        # Strong accents only
        pattern = [
            (0.0, 'down', 1.0),
            (2.5, 'down', 0.95),
        ]
    else:
        pattern = [
            (0.0, 'down', 1.0),
            (1.0, 'down', 0.85),
            (1.5, 'up', 0.7),
            (2.0, 'up', 0.75),
            (2.5, 'down', 0.9),
            (3.5, 'up', 0.65),
        ]

    for beat_offset, direction, vel_mod in pattern:
        strum_notes = rake_strum(chord, beat + beat_offset, direction, vel_mod)
        notes.extend(strum_notes)

    return notes

def rake_strum(chord, start_time, direction, vel_mod):
    """
    Rake across strings with human timing
    Down: 6->1 (bass to treble)
    Up: 1->6 (treble to bass, usually partial)
    """
    notes = []

    if direction == 'down':
        strings = [6, 5, 4, 3, 2, 1]
        base_vel = int(78 * vel_mod)
    else:
        # Up strums usually only hit top 4 strings
        strings = [1, 2, 3, 4]
        base_vel = int(65 * vel_mod)

    # Get available notes for this chord
    available = []
    for s in strings:
        note = get_note(chord, s)
        if note:
            available.append((s, note))

    # Rake timing: 12-25ms between strings (human variation)
    strum_time = 0
    for i, (string, note) in enumerate(available):
        # Time between string hits
        gap = random.uniform(0.015, 0.030)

        # Human timing drift
        drift = random.gauss(0, 0.008)
        t = start_time + strum_time + drift

        # Velocity decay through strum
        if direction == 'down':
            # Down: strong on bass, decay to treble
            vel = base_vel - (i * random.randint(4, 8)) + random.randint(-6, 6)
        else:
            # Up: medium on treble, slight build
            vel = base_vel + (i * random.randint(2, 5)) + random.randint(-5, 5)

        vel = max(35, min(95, vel))

        # Duration: let ring but not too long (strumming mutes previous)
        dur = random.uniform(0.4, 0.7)

        notes.append((note, max(0, t), dur, vel))
        strum_time += gap

    return notes

def muted_chunk(chord, bar):
    """Percussive muted strum - palm mute effect"""
    notes = []
    beat = bar * 4

    # Just bass strings, very short, percussive
    for offset in [0, 1.0, 2.5]:
        for string in [5, 4]:
            note = get_note(chord, string)
            if note:
                t = beat + offset + random.gauss(0, 0.015)
                vel = random.randint(55, 72)
                dur = 0.1  # Very short - muted
                notes.append((note, max(0, t), dur, vel))

    return notes

# Create MIDI
midi = MIDIFile(1, deinterleave=False)
midi.addTempo(0, 0, TEMPO)
midi.addProgramChange(0, 0, 0, 25)  # Steel acoustic

all_notes = []
bar = 0

print(f"Creating FRONT PORCH STEP style strumming at {TEMPO} BPM...")
print("D D U U D U pattern - raw folk-emo")
print("="*55)

# This track only plays during CHORUS sections
# Rest of song is silent (fingerpicking track covers it)

# SILENCE for intro/verse/build (22 bars)
print("Bars 0-21: Silent (fingerpicking only)...")
bar = 22

# CHORUS 1 (8 bars) - Full strum
print("Chorus 1: Full D-D-U-U-D-U strum...")
for chord in ['Dm', 'Dm', 'Am', 'Am', 'F', 'C', 'Dm', 'Am']:
    all_notes.extend(front_porch_strum(chord, bar, 'full'))
    bar += 1

# SILENCE for verse 2 / build (12 bars)
print("Bars 30-41: Silent...")
bar = 42

# CHORUS 2 (8 bars) - Even more intense
print("Chorus 2: Intense strumming...")
for chord in ['Dm', 'Am', 'F', 'C', 'Dm', 'Am', 'E7', 'Dm']:
    all_notes.extend(front_porch_strum(chord, bar, 'full'))
    bar += 1

# BRIDGE (8 bars) - Half strum, pulling back
print("Bridge: Half strum...")
for chord in ['Dm', 'Dm', 'Am', 'Am', 'F', 'F', 'E7', 'E7']:
    if bar % 2 == 0:
        all_notes.extend(front_porch_strum(chord, bar, 'half'))
    else:
        all_notes.extend(front_porch_strum(chord, bar, 'accent'))
    bar += 1

# FINAL - just accents
print("Final: Accent strums only...")
for chord in ['Dm', 'Am', 'F', 'Am']:
    all_notes.extend(front_porch_strum(chord, bar, 'accent'))
    bar += 1

# Last hit
available = [get_note('Dm', s) for s in [4,3,2,1] if get_note('Dm', s)]
for i, note in enumerate(available):
    t = bar * 4 + (i * 0.02)
    all_notes.append((note, t, 4.0, 70 - i*5))

print("="*55)
duration = (bar + 1) * 4 * 60 / TEMPO
print(f"Bars with strumming: 16 chorus + 8 bridge + 4 final = 28")
print(f"Total notes: {len(all_notes)}")
print(f"Pattern: D D U U D U (|1 . 2 + 3 + . 4|)")

for note, time, dur, vel in all_notes:
    midi.addNote(0, 0, note, time, dur, vel)

output = "/Users/seanburdges/Desktop/Kelly_Chorus_Strum.mid"
with open(output, 'wb') as f:
    midi.writeFile(f)

print(f"\nSaved: {output}")
print("""
FRONT PORCH STEP STYLE:
- D D U U D U strum pattern
- Rake across strings (12-25ms between)
- Down strums: bass → treble, strong attack
- Up strums: treble only (strings 1-4)
- Velocity decay through each strum
- Plays only on chorus/bridge sections
- Layer with fingerpicking track in Logic
""")
