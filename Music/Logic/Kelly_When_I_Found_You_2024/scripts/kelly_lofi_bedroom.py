#!/usr/bin/env python3
"""
LO-FI BEDROOM EMO VERSION
Based on genre research:
- Aggressive timing drift (±30-50ms)
- Wide velocity swings (some notes barely there)
- Strings ring and bleed together
- Structural dynamics for fracture points
- "Sloppy" = human, not mistakes
- 76 BPM warm Mitzvah feel + bedroom imperfection
"""

from midiutil import MIDIFile
import random

random.seed(2025)  # New seed for different variation

TEMPO = 76

# Standard tuning
STRINGS = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}

# Chord voicings
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
    'Bb':    {5: 1, 4: 3, 3: 3, 2: 3, 1: 1},
}

def get_note(chord, string):
    fret = CHORDS[chord].get(string)
    if fret is None:
        return None
    return STRINGS[string] + fret

def ms_to_beats(ms):
    """Convert milliseconds to beats at current tempo"""
    return (ms / 1000) * (TEMPO / 60)

def bedroom_pick(chord, bar, intensity='normal'):
    """
    Lo-fi bedroom fingerpicking:
    - Aggressive timing drift (±30-50ms)
    - Wide velocity swings
    - Notes that almost disappear
    - Strings bleeding together
    """
    notes = []
    beat = bar * 4

    # Base pattern: 1-5-6-4-3-2 but with human slop
    if intensity == 'sparse_intro':
        # Barely there - single notes, long gaps
        strings = [random.choice([1, 4, 5])]
        for i, s in enumerate(strings):
            note = get_note(chord, s)
            if note:
                # Heavy timing drift
                drift = random.gauss(0, 0.05)  # ±50ms worth
                t = beat + drift
                vel = random.randint(25, 40)  # Almost disappearing
                dur = random.uniform(4.0, 6.0)  # Ring forever
                notes.append((note, max(0, t), dur, vel))

    elif intensity == 'verse_gentle':
        # Soft fingerpicking, loose timing
        pattern = [(1, 0), (5, 0.4), (4, 1.0), (3, 1.8), (2, 2.5)]
        for string, base_time in pattern:
            note = get_note(chord, string)
            if note and random.random() > 0.15:  # Sometimes skip notes
                # Bedroom slop: ±30-50ms drift
                drift = random.gauss(0, 0.04)
                t = beat + base_time + drift
                # Wide velocity range - some notes barely there
                if random.random() > 0.7:
                    vel = random.randint(20, 32)  # Ghost notes
                else:
                    vel = random.randint(45, 62)
                dur = random.uniform(2.0, 4.0)
                notes.append((note, max(0, t), dur, vel))

    elif intensity == 'building':
        # More notes, still loose but gaining energy
        pattern = [(1, 0), (5, 0.33), (6, 0.66), (4, 1.0), (3, 1.5), (2, 2.0), (1, 2.5), (4, 3.0)]
        for string, base_time in pattern:
            note = get_note(chord, string)
            if note:
                drift = random.gauss(0, 0.035)
                t = beat + base_time + drift
                vel = random.randint(50, 72)
                dur = random.uniform(1.5, 3.0)
                notes.append((note, max(0, t), dur, vel))

    elif intensity == 'fracture':
        # THE BREAK - harder attack, still human
        # "From fucking dusk fucking dawn"
        pattern = [(6, 0), (5, 0.15), (4, 0.3), (3, 0.45), (2, 0.6), (1, 0.75)]
        for i, (string, base_time) in enumerate(pattern):
            note = get_note(chord, string)
            if note:
                drift = random.gauss(0, 0.02)  # Tighter but not perfect
                t = beat + base_time + drift
                # Strong attack that decays
                vel = 88 - (i * 8) + random.randint(-5, 5)
                dur = random.uniform(2.5, 4.0)
                notes.append((note, max(0, t), dur, max(40, vel)))

        # Second strum at beat 2
        for i, (string, base_time) in enumerate(pattern):
            note = get_note(chord, string)
            if note:
                t = beat + 2.0 + base_time + random.gauss(0, 0.025)
                vel = 75 - (i * 6) + random.randint(-5, 5)
                dur = random.uniform(2.0, 3.5)
                notes.append((note, max(0, t), dur, max(35, vel)))

    elif intensity == 'post_fracture':
        # Coming down - sparse, shaky
        note = get_note(chord, random.choice([1, 2, 3]))
        if note:
            t = beat + random.uniform(0.5, 1.5)
            vel = random.randint(30, 45)
            dur = random.uniform(3.0, 5.0)
            notes.append((note, t, dur, vel))

    elif intensity == 'weight':
        # "And now it's 2024" - single chord, let it ring
        available = [s for s in [6,5,4,3,2,1] if get_note(chord, s)]
        for i, string in enumerate(available[:4]):
            note = get_note(chord, string)
            if note:
                t = beat + (i * 0.08) + random.gauss(0, 0.015)
                vel = 55 - (i * 5)
                dur = random.uniform(6.0, 10.0)  # Ring out
                notes.append((note, max(0, t), dur, vel))

    elif intensity == 'silence':
        # The incomplete sentence - almost nothing
        if random.random() > 0.6:
            note = get_note(chord, 1)
            if note:
                t = beat + random.uniform(1.0, 2.0)
                vel = random.randint(18, 28)
                dur = random.uniform(2.0, 3.0)
                notes.append((note, t, dur, vel))

    else:  # 'normal'
        pattern = [(1, 0), (5, 0.5), (4, 1.0), (3, 1.5), (2, 2.0), (1, 2.8)]
        for string, base_time in pattern:
            note = get_note(chord, string)
            if note:
                drift = random.gauss(0, 0.04)
                t = beat + base_time + drift
                vel = random.randint(40, 65)
                dur = random.uniform(2.0, 3.5)
                notes.append((note, max(0, t), dur, vel))

    return notes

def drone_undertone(chord, bar, duration_bars=2):
    """Subtle bass drone underneath"""
    notes = []
    bass_notes = {'Dm': 38, 'Dm7': 38, 'Am': 33, 'Am7': 33,
                  'F': 41, 'Fmaj7': 41, 'C': 36, 'E': 40, 'E7': 40, 'Bb': 34}
    bass = bass_notes.get(chord)
    if bass:
        t = bar * 4 + random.uniform(-0.1, 0.2)
        dur = duration_bars * 4 - 0.5
        vel = random.randint(22, 32)
        notes.append((bass, max(0, t), dur, vel))
    return notes

# Create MIDI
midi = MIDIFile(2, deinterleave=False)
midi.addTempo(0, 0, TEMPO)
midi.addTempo(1, 0, TEMPO)
midi.addProgramChange(0, 0, 0, 25)  # Steel acoustic
midi.addProgramChange(1, 1, 0, 24)  # Nylon for drone

guitar = []
drone = []
bar = 0

print("Creating LO-FI BEDROOM EMO version...")
print("Aggressive timing drift, wide velocity, fracture structure")
print("="*55)

# ============================================
# SONG STRUCTURE WITH FRACTURE POINTS
# ============================================

# INTRO (8 bars) - Barely there, establishing intimacy
print("Intro: sparse, intimate...")
for chord in ['Dm', 'Dm', 'Am', 'Am', 'F', 'F', 'Dm', 'Dm']:
    guitar.extend(bedroom_pick(chord, bar, 'sparse_intro'))
    if bar % 2 == 0:
        drone.extend(drone_undertone(chord, bar, 2))
    bar += 1

# VERSE 1 (8 bars) - Gentle fingerpicking, loose timing
print("Verse 1: gentle, loose...")
for chord in ['Dm', 'Dm', 'Am', 'Am', 'F', 'C', 'Dm', 'Am']:
    guitar.extend(bedroom_pick(chord, bar, 'verse_gentle'))
    drone.extend(drone_undertone(chord, bar))
    bar += 1

# PRE-FRACTURE (4 bars) - Building tension
print("Building tension...")
for chord in ['Dm', 'Am', 'F', 'E']:
    guitar.extend(bedroom_pick(chord, bar, 'building'))
    drone.extend(drone_undertone(chord, bar))
    bar += 1

# FRACTURE 1 (2 bars) - "From fucking dusk fucking dawn"
print(">>> FRACTURE: 'From fucking dusk fucking dawn'")
guitar.extend(bedroom_pick('Dm', bar, 'fracture'))
bar += 1
guitar.extend(bedroom_pick('Am', bar, 'fracture'))
bar += 1

# POST-FRACTURE (4 bars) - Coming down, shaky
print("Post-fracture: shaky, sparse...")
for chord in ['F', 'F', 'Dm', 'Dm']:
    guitar.extend(bedroom_pick(chord, bar, 'post_fracture'))
    bar += 1

# VERSE 2 (8 bars) - Return to control
print("Verse 2: controlled again...")
for chord in ['Dm', 'Dm', 'Am', 'Am', 'F', 'C', 'Am', 'Dm']:
    guitar.extend(bedroom_pick(chord, bar, 'verse_gentle'))
    drone.extend(drone_undertone(chord, bar))
    bar += 1

# BUILDING AGAIN (4 bars)
print("Building again...")
for chord in ['Am', 'F', 'Dm', 'E7']:
    guitar.extend(bedroom_pick(chord, bar, 'building'))
    bar += 1

# FRACTURE 2 (2 bars) - "My body said help her / My body said—"
print(">>> FRACTURE 2: 'My body said—' (silence)")
guitar.extend(bedroom_pick('Am', bar, 'fracture'))
bar += 1
# THE SILENCE - almost nothing
guitar.extend(bedroom_pick('Dm', bar, 'silence'))
bar += 1

# SUSPENDED (4 bars) - Hanging in the silence
print("Suspended in silence...")
for chord in ['Dm', 'Dm', 'Am', 'Am']:
    guitar.extend(bedroom_pick(chord, bar, 'silence'))
    bar += 1

# VERSE 3 (8 bars) - Quiet return
print("Verse 3: quiet return...")
for chord in ['Dm', 'Am', 'F', 'C', 'Dm', 'Am', 'F', 'Dm']:
    guitar.extend(bedroom_pick(chord, bar, 'verse_gentle'))
    if bar % 2 == 0:
        drone.extend(drone_undertone(chord, bar))
    bar += 1

# FINAL WEIGHT (4 bars) - "And now it's 2024"
print(">>> WEIGHT: 'And now it's 2024'")
guitar.extend(bedroom_pick('Dm', bar, 'weight'))
bar += 2
guitar.extend(bedroom_pick('Am', bar, 'weight'))
bar += 2

# OUTRO (8 bars) - Fading, barely there
print("Outro: fading...")
for chord in ['Dm', 'Dm', 'Am', 'Am', 'F', 'Dm', 'Am', 'Dm']:
    guitar.extend(bedroom_pick(chord, bar, 'sparse_intro'))
    bar += 1

# Final held note
midi.addNote(0, 0, 50, bar * 4, 12.0, 18)  # D3 ring out
midi.addNote(1, 1, 38, bar * 4, 15.0, 12)  # D2 drone fade

# Write
print("="*55)
duration = (bar + 3) * 4 * 60 / TEMPO
print(f"Bars: {bar}")
print(f"Guitar notes: {len(guitar)}")
print(f"Drone notes: {len(drone)}")
print(f"Duration: {duration/60:.1f} min")

for note, time, dur, vel in guitar:
    midi.addNote(0, 0, note, time, dur, vel)

for note, time, dur, vel in drone:
    midi.addNote(1, 1, note, time, dur, vel)

output = "/Users/seanburdges/Desktop/Kelly_LoFi_Bedroom.mid"
with open(output, 'wb') as f:
    midi.writeFile(f)

print(f"\nSaved: {output}")
print("""
LO-FI BEDROOM EMO APPLIED:
- Timing drift: ±30-50ms (human slop)
- Velocity: 18-88 (some notes barely there)
- Fracture points built into structure
- Sparse intro/outro (intimacy)
- Building → Fracture → Silence → Weight
- Strings ring and bleed together
""")
