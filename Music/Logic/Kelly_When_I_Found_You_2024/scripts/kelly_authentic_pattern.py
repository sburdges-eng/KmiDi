#!/usr/bin/env python3
"""
AUTHENTIC FINGERPICKING: 1-5-6-4-3-2 string order
Based on real guitarist picking sequence
"""

from midiutil import MIDIFile
import random

random.seed(2024)

TEMPO = 76
TICKS = 960

def ms_to_ticks(ms):
    return int((ms / 1000) * (TEMPO / 60) * TICKS)

def ticks_to_beats(ticks):
    return ticks / TICKS

# Guitar strings
STRINGS = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}

# Chord shapes: (string, fret)
CHORDS = {
    'Dm': {1: 1, 2: 3, 3: 2, 4: 0, 5: None, 6: None},
    'Dm7': {1: 1, 2: 1, 3: 2, 4: 0, 5: None, 6: None},
    'Am': {1: 0, 2: 1, 3: 2, 4: 2, 5: 0, 6: None},
    'Am7': {1: 0, 2: 1, 3: 0, 4: 2, 5: 0, 6: None},
    'Fmaj7': {1: 0, 2: 1, 3: 2, 4: 3, 5: None, 6: None},
    'F': {1: 1, 2: 1, 3: 2, 4: 3, 5: None, 6: None},
    'E': {1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 0},
    'E7': {1: 0, 2: 0, 3: 1, 4: 0, 5: 2, 6: 0},
    'Gm': {1: 3, 2: 3, 3: 3, 4: 5, 5: 5, 6: 3},
    'Bb': {1: 1, 2: 3, 3: 3, 4: 3, 5: 1, 6: None},
    'C': {1: 0, 2: 1, 3: 0, 4: 2, 5: 3, 6: None},
}

def get_note(chord, string):
    """Get MIDI note for a string in a chord"""
    fret = CHORDS[chord].get(string)
    if fret is None:
        return None
    return STRINGS[string] + fret

def authentic_pattern(chord, start_beat, variation='full'):
    """
    Authentic picking: 1-5-6-4-3-2
    With human timing and velocity
    """
    notes = []
    
    # The authentic order: 1, 5, 6, 4, 3, 2
    if variation == 'full':
        pattern = [1, 5, 6, 4, 3, 2]
        timing = [0, 0.5, 1.0, 1.5, 2.0, 2.5]  # Even 8th notes base
    elif variation == 'half':
        pattern = [1, 5, 6, 4]
        timing = [0, 0.5, 1.0, 1.5]
    elif variation == 'sparse':
        pattern = [1, 5, 6]
        timing = [0, 1.0, 2.0]
    elif variation == 'intro':
        pattern = [6, 4, 3, 2, 1]  # Low to high for intro
        timing = [0, 0.4, 0.8, 1.2, 1.6]
    else:
        pattern = [1, 5, 6, 4, 3, 2]
        timing = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
    
    for i, (string, base_time) in enumerate(zip(pattern, timing)):
        midi_note = get_note(chord, string)
        if midi_note is None:
            continue
        
        # Human timing variation
        if string in [5, 6]:  # Bass strings - slightly early
            offset_ms = random.gauss(-10, 8)
        else:  # Treble - laid back
            offset_ms = random.gauss(12, 10)
        
        time = start_beat + base_time + ticks_to_beats(ms_to_ticks(offset_ms))
        
        # Velocity based on string role
        if string == 1:  # Melody string - medium
            vel = random.randint(55, 68)
        elif string in [5, 6]:  # Bass - stronger
            vel = random.randint(65, 80)
        elif string == 2:  # High accent
            vel = random.randint(50, 62)
        else:  # Mid strings - softer
            vel = random.randint(45, 58)
        
        # Duration varies
        if string in [5, 6]:
            dur = random.uniform(0.4, 0.6)  # Bass shorter (palm mute)
        else:
            dur = random.uniform(0.6, 1.0)  # Treble rings
        
        notes.append((midi_note, time, dur, vel))
    
    # Add ghost note occasionally
    if random.random() > 0.6:
        ghost_string = random.choice([3, 4])
        ghost_note = get_note(chord, ghost_string)
        if ghost_note:
            t = start_beat + random.uniform(2.8, 3.5)
            notes.append((ghost_note, t, 0.3, random.randint(22, 32)))
    
    return notes

def slow_arpeggio(chord, start_beat):
    """Slow arpeggio for intros/outros"""
    notes = []
    pattern = [6, 5, 4, 3, 2, 1]  # Low to high
    
    for i, string in enumerate(pattern):
        midi_note = get_note(chord, string)
        if midi_note is None:
            continue
        
        # Slow roll: 80-120ms between notes
        offset_ms = i * random.uniform(80, 120)
        time = start_beat + ticks_to_beats(ms_to_ticks(offset_ms))
        vel = random.randint(45, 58) - (i * 2)  # Slight fade
        dur = random.uniform(1.5, 2.5)  # Let ring
        
        notes.append((midi_note, time, dur, max(35, vel)))
    
    return notes

# ============================================
# CREATE SONG
# ============================================

midi = MIDIFile(1, deinterleave=False)
midi.addTempo(0, 0, TEMPO)
midi.addProgramChange(0, 0, 0, 25)

all_notes = []
bar = 0

print("Creating AUTHENTIC 1-5-6-4-3-2 pattern...")
print("="*50)

# INTRO (4 bars)
print("Intro...")
all_notes.extend(slow_arpeggio('Dm', bar * 4))
bar += 1
all_notes.extend(authentic_pattern('Dm', bar * 4, 'sparse'))
bar += 1
all_notes.extend(slow_arpeggio('Am', bar * 4))
bar += 1
all_notes.extend(authentic_pattern('Am7', bar * 4, 'sparse'))
bar += 1

# VERSE 1 (8 bars)
print("Verse 1: 1-5-6-4-3-2 pattern...")
v1 = ['Dm', 'Dm', 'Am', 'Am', 'Fmaj7', 'Fmaj7', 'E', 'E7']
for chord in v1:
    all_notes.extend(authentic_pattern(chord, bar * 4, 'full'))
    bar += 1

# VERSE 2 (8 bars)
print("Verse 2...")
v2 = ['Dm', 'Dm7', 'Am', 'Am', 'Fmaj7', 'Dm', 'E', 'Am']
for i, chord in enumerate(v2):
    var = 'sparse' if i == 5 else 'full'
    all_notes.extend(authentic_pattern(chord, bar * 4, var))
    bar += 1

# CHORUS (8 bars)
print("Chorus...")
chorus = ['Dm', 'Dm', 'Am', 'Am', 'E', 'E7', 'Fmaj7', 'Am']
for chord in chorus:
    all_notes.extend(authentic_pattern(chord, bar * 4, 'full'))
    bar += 1

# INSTRUMENTAL (8 bars)
print("Instrumental...")
inst = ['Dm', 'Gm', 'Bb', 'Am', 'Dm', 'F', 'E', 'Am']
for i, chord in enumerate(inst):
    if i % 2 == 0:
        all_notes.extend(slow_arpeggio(chord, bar * 4))
    else:
        all_notes.extend(authentic_pattern(chord, bar * 4, 'half'))
    bar += 1

# VERSE 3 (8 bars)
print("Verse 3: sparse...")
v3 = ['Dm', 'Dm', 'Am', 'Am', 'Fmaj7', 'Am', 'E', 'Am']
for chord in v3:
    all_notes.extend(authentic_pattern(chord, bar * 4, 'sparse'))
    bar += 1

# CHORUS 2 (8 bars)
print("Chorus 2...")
for chord in chorus:
    all_notes.extend(authentic_pattern(chord, bar * 4, 'full'))
    bar += 1

# BRIDGE (8 bars)
print("Bridge...")
bridge = ['Dm', 'Dm7', 'Am', 'Am7', 'E', 'E7', 'Dm', 'Am']
for i, chord in enumerate(bridge):
    var = 'half' if i % 2 == 0 else 'sparse'
    all_notes.extend(authentic_pattern(chord, bar * 4, var))
    bar += 1

# FINAL (8 bars)
print("Final...")
final = ['Dm', 'Dm', 'Am', 'Am', 'E', 'E', 'Dm', 'Am']
for i, chord in enumerate(final):
    if i < 6:
        all_notes.extend(authentic_pattern(chord, bar * 4, 'sparse'))
    else:
        # Just bass note fading
        note = get_note(chord, 5) or get_note(chord, 4)
        if note:
            all_notes.append((note, bar * 4, 4.0, 30 - i*2))
    bar += 1

# OUTRO (8 bars)
print("Outro...")
for chord in ['Dm', 'Am', 'Dm', 'Am']:
    note = get_note(chord, 1)  # High string only
    if note:
        all_notes.append((note, bar * 4, 5.0, random.randint(20, 28)))
    bar += 2

# Final
midi.addNote(0, 0, 50, bar * 4, 10.0, 18)

# Write
print("="*50)
duration = (bar + 2) * 4 * 60 / TEMPO
print(f"Bars: {bar} | Notes: {len(all_notes)} | Duration: {duration/60:.1f} min")

for note, time, dur, vel in all_notes:
    midi.addNote(0, 0, note, time, dur, vel)

output = "/Users/seanburdges/Desktop/Kelly_Guitar_156432.mid"
with open(output, 'wb') as f:
    midi.writeFile(f)

print(f"Saved: {output}")
print("\nPattern: 1-5-6-4-3-2 (high E, A, low E, D, G, B)")
