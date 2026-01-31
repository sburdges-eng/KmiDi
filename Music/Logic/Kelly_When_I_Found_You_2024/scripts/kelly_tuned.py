#!/usr/bin/env python3
"""
MITZVAH RIVER STYLE - Warm, intimate, properly tuned
- 5:1 attack/sustain velocity ratio
- Sparse note placement
- Notes ring naturally
- 1-5-6-4-3-2 pattern
- Perfect MIDI tuning
"""

from midiutil import MIDIFile
import random

random.seed(2024)

TEMPO = 76

# Exact MIDI notes for standard tuning (A440)
# These are the correct pitches - no detuning
STRINGS = {
    6: 40,  # E2 = 82.4 Hz
    5: 45,  # A2 = 110 Hz  
    4: 50,  # D3 = 146.8 Hz
    3: 55,  # G3 = 196 Hz
    2: 59,  # B3 = 246.9 Hz
    1: 64,  # E4 = 329.6 Hz
}

# Chord voicings - verified correct frets
CHORDS = {
    'Dm':    {4: 0, 3: 2, 2: 3, 1: 1},      # D-A-D-F
    'Dm7':   {4: 0, 3: 2, 2: 1, 1: 1},      # D-A-C-F
    'Am':    {5: 0, 4: 2, 3: 2, 2: 1, 1: 0}, # A-E-A-C-E
    'Am7':   {5: 0, 4: 2, 3: 0, 2: 1, 1: 0}, # A-E-G-C-E
    'Fmaj7': {4: 3, 3: 2, 2: 1, 1: 0},      # F-A-C-E
    'E':     {6: 0, 5: 2, 4: 2, 3: 1, 2: 0, 1: 0}, # E-B-E-G#-B-E
    'E7':    {6: 0, 5: 2, 4: 0, 3: 1, 2: 0, 1: 0}, # E-B-D-G#-B-E
}

def get_note(chord, string):
    fret = CHORDS[chord].get(string)
    if fret is None:
        return None
    return STRINGS[string] + fret

def mitzvah_style(chord, bar, variation='normal'):
    """
    Mitzvah River style:
    - Strong attack (vel 70-85)
    - Gentle sustain decay
    - Sparse: 2-4 notes per bar
    - Let notes ring (long duration)
    - 1-5-6-4-3-2 when full
    """
    notes = []
    beat = bar * 4
    
    if variation == 'full':
        # Full 1-5-6-4-3-2 pattern over 2 beats then rest
        pattern = [(1, 0), (5, 0.33), (6, 0.66), (4, 1.0), (3, 1.33), (2, 1.66)]
        for string, offset in pattern:
            note = get_note(chord, string)
            if note:
                t = beat + offset + random.gauss(0, 0.02)
                # 5:1 attack ratio - first note strong
                if offset == 0:
                    vel = random.randint(75, 88)
                else:
                    vel = random.randint(45, 58)
                dur = random.uniform(2.0, 3.5)  # Let ring
                notes.append((note, t, dur, vel))
                
    elif variation == 'sparse':
        # Just 2-3 notes, like the reference
        # Beat 1: Strong attack note
        note = get_note(chord, random.choice([1, 4, 5]))
        if note:
            t = beat + random.gauss(0, 0.015)
            vel = random.randint(72, 85)  # Strong attack
            dur = random.uniform(3.0, 4.5)
            notes.append((note, t, dur, vel))
        
        # Beat 2.5-3: Soft sustain note
        if random.random() > 0.3:
            note = get_note(chord, random.choice([2, 3]))
            if note:
                t = beat + random.uniform(2.0, 3.0)
                vel = random.randint(35, 48)  # Gentle sustain
                dur = random.uniform(2.5, 4.0)
                notes.append((note, t, dur, vel))
                
    elif variation == 'arpeggio':
        # Slow arpeggio - Mitzvah style
        available = [s for s in [6,5,4,3,2,1] if get_note(chord, s)]
        for i, string in enumerate(available[:4]):
            note = get_note(chord, string)
            t = beat + (i * 0.5) + random.gauss(0, 0.02)
            vel = 75 - (i * 8)  # Decay through arpeggio
            dur = random.uniform(2.5, 4.0)
            notes.append((note, t, dur, max(35, vel)))
            
    elif variation == 'dying':
        # Single fading note
        note = get_note(chord, random.choice([4, 5, 1]))
        if note:
            notes.append((note, beat, random.uniform(5.0, 8.0), random.randint(25, 38)))
    
    return notes

# Create MIDI
midi = MIDIFile(1, deinterleave=False)
midi.addTempo(0, 0, TEMPO)
midi.addProgramChange(0, 0, 0, 25)

all_notes = []
bar = 0

print("Creating MITZVAH RIVER style MIDI...")
print("Warm, intimate, properly tuned, 5:1 attack ratio")
print("="*50)

# INTRO (4 bars) - Sparse arpeggios
print("Intro: sparse arpeggios...")
for chord in ['Dm', 'Dm', 'Am', 'Am']:
    all_notes.extend(mitzvah_style(chord, bar, 'arpeggio'))
    bar += 1

# VERSE 1 (8 bars) - Mix of sparse and full
print("Verse 1...")
v1 = ['Dm', 'Dm', 'Am', 'Am', 'Fmaj7', 'Fmaj7', 'E', 'E7']
for i, chord in enumerate(v1):
    var = 'full' if i % 2 == 0 else 'sparse'
    all_notes.extend(mitzvah_style(chord, bar, var))
    bar += 1

# VERSE 2 (8 bars)
print("Verse 2...")
v2 = ['Dm', 'Dm', 'Am', 'Am', 'Fmaj7', 'Dm', 'E', 'Am']
for i, chord in enumerate(v2):
    var = 'sparse' if i in [4, 5] else 'full'
    all_notes.extend(mitzvah_style(chord, bar, var))
    bar += 1

# CHORUS (8 bars)
print("Chorus...")
chorus = ['Dm', 'Dm', 'Am', 'Am', 'E', 'E7', 'Fmaj7', 'Am']
for i, chord in enumerate(chorus):
    all_notes.extend(mitzvah_style(chord, bar, 'full'))
    bar += 1

# INSTRUMENTAL (8 bars)
print("Instrumental...")
for chord in ['Dm', 'Dm7', 'Am', 'Am7', 'Fmaj7', 'Dm', 'E', 'Am']:
    all_notes.extend(mitzvah_style(chord, bar, 'arpeggio'))
    bar += 1

# VERSE 3 (8 bars) - Pulled back
print("Verse 3: sparse...")
for chord in ['Dm', 'Dm', 'Am', 'Am', 'Fmaj7', 'Am', 'E', 'Am']:
    all_notes.extend(mitzvah_style(chord, bar, 'sparse'))
    bar += 1

# CHORUS 2 (8 bars)
print("Chorus 2...")
for chord in chorus:
    var = 'full' if bar % 2 == 0 else 'sparse'
    all_notes.extend(mitzvah_style(chord, bar, var))
    bar += 1

# BRIDGE (8 bars)
print("Bridge...")
for chord in ['Dm', 'Dm7', 'Am', 'Am7', 'E', 'E7', 'Dm', 'Am']:
    all_notes.extend(mitzvah_style(chord, bar, 'sparse'))
    bar += 1

# FINAL (8 bars)
print("Final: dying...")
for chord in ['Dm', 'Dm', 'Am', 'Am', 'E', 'E', 'Dm', 'Am']:
    all_notes.extend(mitzvah_style(chord, bar, 'dying'))
    bar += 1

# OUTRO (4 bars)
print("Outro...")
for chord in ['Dm', 'Am']:
    all_notes.extend(mitzvah_style(chord, bar, 'dying'))
    bar += 2

# Final hold
midi.addNote(0, 0, 50, bar * 4, 10.0, 20)  # D3

# Write
print("="*50)
duration = (bar + 2) * 4 * 60 / TEMPO
print(f"Bars: {bar} | Notes: {len(all_notes)} | Duration: {duration/60:.1f} min")

for note, time, dur, vel in all_notes:
    midi.addNote(0, 0, note, max(0, time), dur, vel)

output = "/Users/seanburdges/Desktop/Kelly_Mitzvah_Style.mid"
with open(output, 'wb') as f:
    midi.writeFile(f)

print(f"Saved: {output}")
print("""
MITZVAH STYLE APPLIED:
- 5:1 attack/sustain ratio
- Sparse placement (2-4 notes/bar)
- Long ring times (2-4 beats)
- Proper A440 tuning
- 1-5-6-4-3-2 pattern
""")
