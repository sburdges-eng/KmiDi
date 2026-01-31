#!/usr/bin/env python3
"""
ORGANIC GUITAR MIDI - Sounds like a real guitarist
Key techniques:
1. Strum rake: notes 15-40ms apart, not simultaneous
2. Finger weight variation: thumb heavier, fingers lighter
3. String resonance: some notes ring into others
4. Natural groove: slight push/pull against the beat
5. Breath points: natural pauses between phrases
6. Ghost strings: very quiet sympathetic notes
"""

from midiutil import MIDIFile
import random
import math

random.seed(42)

TEMPO = 76
TICKS = 960

def ms_to_ticks(ms):
    return int((ms / 1000) * (TEMPO / 60) * TICKS)

def ticks_to_beats(ticks):
    return ticks / TICKS

# Guitar string MIDI notes (standard tuning)
STRINGS = {
    6: 40,  # E2
    5: 45,  # A2
    4: 50,  # D3
    3: 55,  # G3
    2: 59,  # B3
    1: 64,  # E4
}

# Chord fingerings as (string, fret) - realistic guitar voicings
CHORD_SHAPES = {
    'Dm': [
        (4, 0),   # D open
        (3, 2),   # A
        (2, 3),   # D
        (1, 1),   # F
    ],
    'Dm7': [
        (4, 0),   # D open
        (3, 2),   # A  
        (2, 1),   # C
        (1, 1),   # F
    ],
    'Am': [
        (5, 0),   # A open
        (4, 2),   # E
        (3, 2),   # A
        (2, 1),   # C
        (1, 0),   # E open
    ],
    'Am7': [
        (5, 0),   # A open
        (4, 2),   # E
        (3, 0),   # G open
        (2, 1),   # C
        (1, 0),   # E open
    ],
    'Fmaj7': [
        (4, 3),   # F
        (3, 2),   # A
        (2, 1),   # C
        (1, 0),   # E open
    ],
    'F': [
        (4, 3),   # F
        (3, 2),   # A
        (2, 1),   # C
        (1, 1),   # F
    ],
    'E': [
        (6, 0),   # E open
        (5, 2),   # B
        (4, 2),   # E
        (3, 1),   # G#
        (2, 0),   # B open
        (1, 0),   # E open
    ],
    'E7': [
        (6, 0),   # E open
        (5, 2),   # B
        (4, 0),   # D open
        (3, 1),   # G#
        (2, 0),   # B open
        (1, 0),   # E open
    ],
    'Gm': [
        (6, 3),   # G
        (5, 5),   # D
        (4, 5),   # G
        (3, 3),   # Bb
    ],
    'Bb': [
        (5, 1),   # Bb
        (4, 3),   # F
        (3, 3),   # Bb
        (2, 3),   # D
    ],
    'C': [
        (5, 3),   # C
        (4, 2),   # E
        (3, 0),   # G
        (2, 1),   # C
        (1, 0),   # E
    ],
}

def get_note(string, fret):
    """Get MIDI note from string and fret"""
    return STRINGS[string] + fret

def strum_chord(chord_name, start_beat, direction='down', velocity_base=65, 
                strum_speed='normal', let_ring=True):
    """
    Strum a chord with realistic timing
    direction: 'down' (low to high) or 'up' (high to low)
    strum_speed: 'fast', 'normal', 'slow', 'arpeggio'
    """
    notes = []
    shape = CHORD_SHAPES[chord_name]
    
    # Strum timing based on speed
    if strum_speed == 'fast':
        inter_note_ms = random.uniform(12, 22)
    elif strum_speed == 'slow':
        inter_note_ms = random.uniform(35, 55)
    elif strum_speed == 'arpeggio':
        inter_note_ms = random.uniform(80, 150)
    else:  # normal
        inter_note_ms = random.uniform(18, 35)
    
    # Order strings by direction
    if direction == 'down':
        ordered = sorted(shape, key=lambda x: -x[0])  # Low strings first
    else:
        ordered = sorted(shape, key=lambda x: x[0])   # High strings first
    
    # Generate notes with strum rake
    for i, (string, fret) in enumerate(ordered):
        midi_note = get_note(string, fret)
        
        # Time offset for strum rake
        offset_ms = i * inter_note_ms
        # Add human variation
        offset_ms += random.gauss(0, 5)
        
        time = start_beat + ticks_to_beats(ms_to_ticks(offset_ms))
        
        # Velocity: first note strongest, decay through strum
        if direction == 'down':
            vel = velocity_base + random.randint(5, 12) - (i * 3)
        else:
            vel = velocity_base - 5 + (i * 2)  # Up strums build
        
        vel = max(35, min(100, vel + random.randint(-5, 5)))
        
        # Duration: let ring or muted
        if let_ring:
            dur = random.uniform(2.0, 4.0)
        else:
            dur = random.uniform(0.3, 0.6)
        
        notes.append((midi_note, time, dur, vel))
    
    return notes

def fingerpick_pattern(chord_name, start_beat, pattern='travis', phrase_position=0):
    """
    Fingerpick with realistic timing
    pattern: 'travis', 'arpeggio', 'sparse', 'rolling'
    """
    notes = []
    shape = CHORD_SHAPES[chord_name]
    
    # Separate bass and treble strings
    bass_notes = [(s, f) for s, f in shape if s >= 4]
    treble_notes = [(s, f) for s, f in shape if s <= 3]
    
    if not bass_notes:
        bass_notes = shape[:1]
    if not treble_notes:
        treble_notes = shape[-2:]
    
    # Human timing function
    def human_offset(base_ms=0, is_bass=False):
        if is_bass:
            # Bass slightly early (driving)
            return base_ms + random.gauss(-8, 6)
        else:
            # Treble laid back
            return base_ms + random.gauss(15, 10)
    
    if pattern == 'travis':
        # Beat 1: Bass (thumb)
        s, f = random.choice(bass_notes)
        t = start_beat + ticks_to_beats(ms_to_ticks(human_offset(0, True)))
        v = random.randint(68, 82)
        notes.append((get_note(s, f), t, random.uniform(0.4, 0.6), v))
        
        # Beat 1.5: High treble (index)
        if treble_notes:
            s, f = treble_notes[-1] if len(treble_notes) > 0 else treble_notes[0]
            t = start_beat + 0.5 + ticks_to_beats(ms_to_ticks(human_offset(0, False)))
            v = random.randint(48, 62)
            notes.append((get_note(s, f), t, random.uniform(0.5, 0.8), v))
        
        # Beat 2: Ghost/rest or soft bass
        if random.random() > 0.4:
            s, f = random.choice(bass_notes)
            t = start_beat + 1.0 + ticks_to_beats(ms_to_ticks(human_offset(0, True)))
            v = random.randint(30, 45)  # Ghost note
            notes.append((get_note(s, f), t, random.uniform(0.2, 0.35), v))
        
        # Beat 2.5: Mid treble (middle finger)
        if len(treble_notes) > 1:
            s, f = treble_notes[0]
            t = start_beat + 1.5 + ticks_to_beats(ms_to_ticks(human_offset(0, False)))
            v = random.randint(45, 58)
            notes.append((get_note(s, f), t, random.uniform(0.5, 0.7), v))
        
        # Beat 3: Alternating bass
        s, f = bass_notes[-1] if len(bass_notes) > 1 else bass_notes[0]
        t = start_beat + 2.0 + ticks_to_beats(ms_to_ticks(human_offset(0, True)))
        v = random.randint(62, 75)
        notes.append((get_note(s, f), t, random.uniform(0.4, 0.55), v))
        
        # Beat 3.5: High treble
        if treble_notes:
            s, f = treble_notes[-1]
            t = start_beat + 2.5 + ticks_to_beats(ms_to_ticks(human_offset(0, False)))
            v = random.randint(45, 58)
            notes.append((get_note(s, f), t, random.uniform(0.5, 0.8), v))
        
        # Beat 4: Optional ghost
        if random.random() > 0.5:
            s, f = random.choice(bass_notes)
            t = start_beat + 3.0 + ticks_to_beats(ms_to_ticks(human_offset(0, True)))
            v = random.randint(28, 40)
            notes.append((get_note(s, f), t, random.uniform(0.2, 0.3), v))
        
        # Beat 4.5: Mid treble
        if random.random() > 0.3 and len(treble_notes) > 1:
            s, f = treble_notes[0]
            t = start_beat + 3.5 + ticks_to_beats(ms_to_ticks(human_offset(0, False)))
            v = random.randint(42, 55)
            notes.append((get_note(s, f), t, random.uniform(0.5, 0.7), v))
    
    elif pattern == 'sparse':
        # Just bass + one treble
        s, f = random.choice(bass_notes)
        t = start_beat + ticks_to_beats(ms_to_ticks(human_offset(0, True)))
        v = random.randint(55, 70)
        notes.append((get_note(s, f), t, random.uniform(0.8, 1.2), v))
        
        if random.random() > 0.3:
            s, f = random.choice(treble_notes)
            t = start_beat + random.uniform(1.5, 2.5) + ticks_to_beats(ms_to_ticks(human_offset(0, False)))
            v = random.randint(40, 55)
            notes.append((get_note(s, f), t, random.uniform(0.8, 1.5), v))
    
    elif pattern == 'rolling':
        # Continuous roll through strings
        all_notes = sorted(shape, key=lambda x: -x[0])
        roll_gap = random.uniform(0.2, 0.35)
        for i, (s, f) in enumerate(all_notes):
            t = start_beat + (i * roll_gap) + ticks_to_beats(ms_to_ticks(random.gauss(0, 8)))
            v = random.randint(45, 60) + random.randint(-3, 3)
            notes.append((get_note(s, f), t, random.uniform(0.6, 1.0), v))
    
    # Add sympathetic string resonance (very quiet ghost notes)
    if random.random() > 0.7 and len(notes) > 0:
        # Add a quiet open string
        open_strings = [40, 45, 55, 59, 64]  # E, A, G, B, E
        ghost = random.choice(open_strings)
        t = notes[0][1] + random.uniform(0.1, 0.3)
        notes.append((ghost, t, random.uniform(1.0, 2.0), random.randint(15, 25)))
    
    return notes

def add_phrase_breath(notes, breath_beat, breath_duration=0.25):
    """Add a natural pause/breath before a phrase"""
    # Slightly shorten notes that would overlap the breath
    adjusted = []
    for note, time, dur, vel in notes:
        if time + dur > breath_beat and time < breath_beat:
            dur = max(0.1, breath_beat - time - 0.1)
        adjusted.append((note, time, dur, vel))
    return adjusted

# ============================================
# CREATE THE SONG
# ============================================

midi = MIDIFile(1, deinterleave=False)
midi.addTempo(0, 0, TEMPO)
midi.addProgramChange(0, 0, 0, 25)  # Steel string acoustic

all_notes = []
bar = 0

print("Creating ORGANIC guitar MIDI...")
print("="*50)

# INTRO (4 bars) - Gentle arpeggios
print("Intro: Dm/Am arpeggios with strum rake...")
# Bar 1: Dm arpeggio (slow, let ring)
all_notes.extend(strum_chord('Dm', bar * 4, 'down', 55, 'arpeggio', True))
bar += 1
# Bar 2: Rest with gentle bass
all_notes.extend(fingerpick_pattern('Dm', bar * 4, 'sparse'))
bar += 1
# Bar 3: Am arpeggio
all_notes.extend(strum_chord('Am', bar * 4, 'up', 50, 'arpeggio', True))
bar += 1
# Bar 4: Transition
all_notes.extend(fingerpick_pattern('Am7', bar * 4, 'sparse'))
bar += 1

# VERSE 1 (8 bars)
print("Verse 1: Travis picking Dm → Am → Fmaj7 → E...")
v1 = ['Dm', 'Dm', 'Am', 'Am', 'Fmaj7', 'Fmaj7', 'E', 'E7']
for i, chord in enumerate(v1):
    all_notes.extend(fingerpick_pattern(chord, bar * 4, 'travis', i/8))
    bar += 1

# VERSE 2 (8 bars)
print("Verse 2: Building with variation...")
v2 = ['Dm', 'Dm7', 'Am', 'Am', 'Fmaj7', 'Dm', 'E', 'Am']
for i, chord in enumerate(v2):
    if i == 5:  # "Shea butter" - sparse pause
        all_notes.extend(fingerpick_pattern(chord, bar * 4, 'sparse'))
    else:
        all_notes.extend(fingerpick_pattern(chord, bar * 4, 'travis'))
    bar += 1

# CHORUS (8 bars)
print("Chorus: Fuller with strums...")
chorus = ['Dm', 'Dm', 'Am', 'Am', 'E', 'E7', 'Fmaj7', 'Am']
for i, chord in enumerate(chorus):
    if i in [0, 4]:  # Accent beats with gentle strum
        all_notes.extend(strum_chord(chord, bar * 4, 'down', 60, 'normal', True))
        all_notes.extend(fingerpick_pattern(chord, bar * 4 + 2, 'sparse'))
    else:
        all_notes.extend(fingerpick_pattern(chord, bar * 4, 'travis'))
    bar += 1

# INSTRUMENTAL (8 bars)
print("Instrumental: Rolling arpeggios...")
inst = ['Dm', 'Gm', 'Bb', 'Am', 'Dm', 'F', 'E', 'Am']
for i, chord in enumerate(inst):
    if i % 2 == 0:
        all_notes.extend(strum_chord(chord, bar * 4, 'down', 50, 'arpeggio', True))
    else:
        all_notes.extend(fingerpick_pattern(chord, bar * 4, 'rolling'))
    bar += 1

# VERSE 3 (8 bars) - Pulled back
print("Verse 3: Sparse, breathing room...")
v3 = ['Dm', 'Dm', 'Am', 'Am', 'Fmaj7', 'Am', 'E', 'Am']
for i, chord in enumerate(v3):
    all_notes.extend(fingerpick_pattern(chord, bar * 4, 'sparse'))
    bar += 1

# CHORUS 2 (8 bars)
print("Chorus 2: Return...")
for i, chord in enumerate(chorus):
    all_notes.extend(fingerpick_pattern(chord, bar * 4, 'travis'))
    bar += 1

# BRIDGE (8 bars)
print("Bridge: Sustained, emotional...")
bridge = ['Dm', 'Dm7', 'Am', 'Am7', 'E', 'E7', 'Dm', 'Am']
for i, chord in enumerate(bridge):
    if i % 2 == 0:
        all_notes.extend(strum_chord(chord, bar * 4, 'down', 45, 'slow', True))
    else:
        all_notes.extend(fingerpick_pattern(chord, bar * 4, 'sparse'))
    bar += 1

# FINAL (8 bars) - Dying away
print("Final: Fading to silence...")
final = ['Dm', 'Dm', 'Am', 'Am', 'E', 'E', 'Dm', 'Am']
for i, chord in enumerate(final):
    vel = 50 - (i * 4)  # Fade out
    if i < 6:
        all_notes.extend(fingerpick_pattern(chord, bar * 4, 'sparse'))
    else:
        # Very quiet single notes
        shape = CHORD_SHAPES[chord]
        s, f = shape[0]
        all_notes.append((get_note(s, f), bar * 4, 3.0, max(25, vel)))
    bar += 1

# OUTRO (8 bars) - Nearly silent
print("Outro: Almost nothing...")
outro = ['Dm', 'Am', 'Dm', 'Am']
for chord in outro:
    shape = CHORD_SHAPES[chord]
    s, f = shape[0]
    all_notes.append((get_note(s, f), bar * 4, 4.0, random.randint(20, 30)))
    bar += 2  # Long held notes

# Final sustain
midi.addNote(0, 0, 50, bar * 4, 12.0, 22)  # D3 fading

# Write all notes
print("="*50)
duration = (bar + 3) * 4 * 60 / TEMPO
print(f"Total bars: {bar}")
print(f"Total notes: {len(all_notes)}")
print(f"Duration: {duration:.0f}s ({duration/60:.1f} min)")

for note, time, dur, vel in all_notes:
    midi.addNote(0, 0, note, time, dur, vel)

output = "/Users/seanburdges/Desktop/Kelly_Guitar_Organic.mid"
with open(output, 'wb') as f:
    midi.writeFile(f)

print(f"\nSaved: {output}")
print("""
ORGANIC TECHNIQUES:
━━━━━━━━━━━━━━━━━━━
• Strum rake: 15-40ms between strings
• Bass notes early, treble laid-back  
• Velocity decay through strums
• Ghost notes for resonance
• Variable note durations
• Natural phrase breathing
• Sympathetic string simulation
""")
