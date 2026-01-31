#!/usr/bin/env python3
"""
WHEN I FOUND YOU - Final Guitar MIDI
Based on comprehensive music theory study and fingerstyle analysis

76 BPM | 4/4 | ~4 minutes (~76 bars)
Dm-focused progression with voicing techniques
"""

from midiutil import MIDIFile
import random
import math

random.seed(2024)

TEMPO = 76
BEATS_PER_BAR = 4
MS_PER_BEAT = 60000 / TEMPO  # ~789ms

def ms_to_beats(ms):
    return ms / MS_PER_BEAT

# ============================================
# CHORD VOICINGS (from theory study)
# Standard tuning: E2=40, A2=45, D3=50, G3=55, B3=59, E4=64
# ============================================

CHORDS = {
    # Root position voicings
    'Dm': {
        'name': 'D minor',
        'root': 50,      # D3
        'bass_alt': 45,  # A2 (5th)
        'notes': [50, 53, 57, 62, 65],  # D3, F3, A3, D4, F4
        'voicing_high': [62, 65, 69],   # D4, F4, A4 (high voicing for flair)
    },
    'Dm7': {
        'name': 'D minor 7',
        'root': 50,
        'bass_alt': 45,
        'notes': [50, 53, 57, 60, 65],  # D3, F3, A3, C4, F4
        'voicing_high': [60, 65, 69],
    },
    'Am': {
        'name': 'A minor', 
        'root': 45,      # A2
        'bass_alt': 40,  # E2 (5th)
        'notes': [45, 52, 57, 60, 64],  # A2, E3, A3, C4, E4
        'voicing_high': [60, 64, 69],   # C4, E4, A4
    },
    'Am7': {
        'name': 'A minor 7',
        'root': 45,
        'bass_alt': 40,
        'notes': [45, 52, 55, 60, 64],  # A2, E3, G3, C4, E4
        'voicing_high': [55, 60, 64],
    },
    'Fmaj7': {
        'name': 'F major 7',
        'root': 41,      # F2
        'bass_alt': 48,  # C3 (5th)
        'notes': [41, 48, 53, 57, 60, 64],  # F2, C3, F3, A3, C4, E4
        'voicing_high': [60, 64, 65],   # C4, E4, F4
    },
    'F': {
        'name': 'F major',
        'root': 41,
        'bass_alt': 48,
        'notes': [41, 48, 53, 57, 60],  # F2, C3, F3, A3, C4
        'voicing_high': [57, 60, 65],
    },
    'E': {
        'name': 'E major',
        'root': 40,      # E2
        'bass_alt': 47,  # B2 (5th)
        'notes': [40, 47, 52, 56, 59, 64],  # E2, B2, E3, G#3, B3, E4
        'voicing_high': [56, 59, 64],
    },
    'E7': {
        'name': 'E dominant 7',
        'root': 40,
        'bass_alt': 47,
        'notes': [40, 47, 50, 56, 59, 62],  # E2, B2, D3, G#3, B3, D4
        'voicing_high': [56, 59, 62],
    },
    'Bb': {
        'name': 'Bb major',
        'root': 46,      # Bb2
        'bass_alt': 53,  # F3
        'notes': [46, 53, 58, 62, 65],  # Bb2, F3, Bb3, D4, F4
        'voicing_high': [58, 62, 65],
    },
    'Gm': {
        'name': 'G minor',
        'root': 43,      # G2
        'bass_alt': 50,  # D3
        'notes': [43, 50, 55, 58, 62],  # G2, D3, G3, Bb3, D4
        'voicing_high': [55, 58, 62],
    },
    'C': {
        'name': 'C major',
        'root': 48,      # C3
        'bass_alt': 43,  # G2
        'notes': [48, 52, 55, 60, 64],  # C3, E3, G3, C4, E4
        'voicing_high': [60, 64, 67],
    },
}

# ============================================
# HUMANIZATION (from timing research)
# ============================================

def human_time(beat_pos, note_type='melody', phrase_prog=0, is_flair=False):
    """
    Apply human micro-timing:
    - Bass: slightly early (-5 to +5ms) for drive
    - Melody: laid-back (+10 to +25ms)
    - Flair notes: can be more varied
    """
    if note_type == 'bass':
        offset_ms = random.gauss(0, 6)
    elif note_type == 'ghost':
        offset_ms = random.gauss(-3, 4)
    elif is_flair:
        # Voicing/flair can be more rubato
        offset_ms = random.gauss(15, 12)
    else:
        # Melody notes laid-back
        offset_ms = random.gauss(15, 8)
    
    # Phrase-end rubato
    if phrase_prog > 0.75:
        offset_ms += random.uniform(5, 15) * phrase_prog
    
    return beat_pos + ms_to_beats(max(-20, min(40, offset_ms)))

def human_velocity(base, beat_in_bar, note_type='melody', phrase_prog=0):
    """Dynamic velocity with beat accents and phrase shaping"""
    if note_type == 'bass':
        vel = random.randint(65, 78)
    elif note_type == 'ghost':
        vel = random.randint(25, 35)
    elif note_type == 'flair':
        vel = random.randint(50, 68)
    else:
        vel = random.randint(48, 65)
    
    # Beat 1 accent
    if beat_in_bar == 0:
        vel += random.randint(6, 12)
    elif beat_in_bar == 2:
        vel += random.randint(2, 6)
    
    # Phrase dynamics (sine curve)
    phrase_curve = math.sin(phrase_prog * math.pi) * 0.6 + 0.4
    vel = int(vel * phrase_curve)
    
    # Human variance
    vel += random.randint(-4, 4)
    
    return max(25, min(100, vel))

def human_duration(base_dur, note_type='melody', let_ring=False):
    """Natural duration variation"""
    if note_type == 'bass':
        factor = random.uniform(0.4, 0.6)  # Palm muted
    elif note_type == 'ghost':
        factor = random.uniform(0.15, 0.25)
    elif let_ring:
        factor = random.uniform(0.9, 1.2)  # Ring longer
    else:
        factor = random.uniform(0.6, 0.85)
    
    return base_dur * factor

# ============================================
# PATTERN GENERATORS
# ============================================

def fingerpick_bar(chord_name, bar_start, phrase_prog=0, density='normal'):
    """
    Generate one bar of fingerpicking
    Travis-style: bass on 1 & 3, melody between
    """
    c = CHORDS[chord_name]
    notes = []
    
    if density == 'sparse':
        # Just bass + 1-2 melody notes
        # Beat 1: Bass
        t = human_time(bar_start, 'bass', phrase_prog)
        v = human_velocity(70, 0, 'bass', phrase_prog)
        d = human_duration(0.8, 'bass')
        notes.append((c['root'], t, d, v))
        
        # Beat 2.5: One melody note
        if random.random() > 0.3:
            t = human_time(bar_start + 1.5, 'melody', phrase_prog)
            v = human_velocity(55, 1, 'melody', phrase_prog)
            d = human_duration(0.6, 'melody', let_ring=True)
            notes.append((random.choice(c['notes'][2:]), t, d, v))
        
        # Beat 3: Alt bass
        t = human_time(bar_start + 2, 'bass', phrase_prog)
        v = human_velocity(65, 2, 'bass', phrase_prog)
        d = human_duration(0.7, 'bass')
        notes.append((c['bass_alt'], t, d, v))
        
    elif density == 'dying':
        # Just one or two notes, very quiet
        t = human_time(bar_start, 'bass', phrase_prog)
        v = random.randint(30, 45)
        notes.append((c['root'], t, 2.0, v))
        
        if random.random() > 0.5:
            t = human_time(bar_start + 2.5, 'melody', phrase_prog)
            v = random.randint(25, 38)
            notes.append((random.choice(c['notes'][2:]), t, 1.5, v))
    
    else:  # normal
        # Beat 1: Bass root
        t = human_time(bar_start, 'bass', phrase_prog)
        v = human_velocity(72, 0, 'bass', phrase_prog)
        d = human_duration(0.6, 'bass')
        notes.append((c['root'], t, d, v))
        
        # Beat 1.5 (and): High melody
        t = human_time(bar_start + 0.5, 'melody', phrase_prog)
        v = human_velocity(58, 0, 'melody', phrase_prog)
        d = human_duration(0.5, 'melody')
        notes.append((random.choice(c['notes'][-2:]), t, d, v))
        
        # Beat 2: Ghost or rest
        if random.random() > 0.4:
            t = human_time(bar_start + 1, 'ghost', phrase_prog)
            v = human_velocity(30, 1, 'ghost', phrase_prog)
            d = human_duration(0.2, 'ghost')
            notes.append((random.choice(c['notes'][1:3]), t, d, v))
        
        # Beat 2.5: Mid melody
        t = human_time(bar_start + 1.5, 'melody', phrase_prog)
        v = human_velocity(55, 1, 'melody', phrase_prog)
        d = human_duration(0.5, 'melody')
        notes.append((random.choice(c['notes'][2:4]), t, d, v))
        
        # Beat 3: Alt bass
        t = human_time(bar_start + 2, 'bass', phrase_prog)
        v = human_velocity(68, 2, 'bass', phrase_prog)
        d = human_duration(0.6, 'bass')
        notes.append((c['bass_alt'], t, d, v))
        
        # Beat 3.5: High melody
        t = human_time(bar_start + 2.5, 'melody', phrase_prog)
        v = human_velocity(52, 2, 'melody', phrase_prog)
        d = human_duration(0.5, 'melody')
        notes.append((random.choice(c['notes'][-2:]), t, d, v))
        
        # Beat 4: Ghost or skip
        if random.random() > 0.5:
            t = human_time(bar_start + 3, 'ghost', phrase_prog)
            v = human_velocity(28, 3, 'ghost', phrase_prog)
            d = human_duration(0.2, 'ghost')
            notes.append((random.choice(c['notes'][2:4]), t, d, v))
        
        # Beat 4.5: Mid melody
        if random.random() > 0.3:
            t = human_time(bar_start + 3.5, 'melody', phrase_prog)
            v = human_velocity(50, 3, 'melody', phrase_prog)
            d = human_duration(0.5, 'melody')
            notes.append((random.choice(c['notes'][2:4]), t, d, v))
    
    return notes

def voicing_arpeggio(chord_name, bar_start, direction='up'):
    """
    Arpeggiated chord voicing for flair moments
    Used at intro, verse endings, etc.
    """
    c = CHORDS[chord_name]
    notes = []
    
    # Use high voicing for brightness
    voicing_notes = c.get('voicing_high', c['notes'][-3:])
    
    if direction == 'up':
        note_order = sorted(voicing_notes)
    else:
        note_order = sorted(voicing_notes, reverse=True)
    
    # Strum-like timing: 20-40ms between notes
    base_time = bar_start
    for i, note in enumerate(note_order):
        offset_ms = i * random.uniform(25, 45)
        t = human_time(base_time, 'flair', 0, is_flair=True) + ms_to_beats(offset_ms)
        v = human_velocity(55 - i*3, 0, 'flair', 0)  # Slight decrescendo
        d = human_duration(1.5, 'melody', let_ring=True)
        notes.append((note, t, d, v))
    
    return notes

def hammer_on_phrase(chord_name, bar_start, phrase_prog=0):
    """
    Hammer-on ornament (ligado technique)
    """
    c = CHORDS[chord_name]
    notes = []
    
    # Bass note
    t = human_time(bar_start, 'bass', phrase_prog)
    v = human_velocity(70, 0, 'bass', phrase_prog)
    notes.append((c['root'], t, 0.6, v))
    
    # Hammer-on sequence: pick first note, hammer the rest
    melody_notes = sorted(c['notes'][2:5])
    
    base_time = bar_start + 0.5
    for i, note in enumerate(melody_notes):
        offset_ms = i * random.uniform(18, 35)
        t = human_time(base_time, 'melody', phrase_prog) + ms_to_beats(offset_ms)
        # Hammered notes are softer
        v = human_velocity(55 - i*6, 0, 'melody', phrase_prog)
        d = 0.4 + i * 0.1
        notes.append((note, t, d, max(30, v)))
    
    return notes

def pull_off_phrase(chord_name, bar_start, phrase_prog=0):
    """
    Pull-off ornament (descending ligado)
    """
    c = CHORDS[chord_name]
    notes = []
    
    # Start with highest note picked
    melody_notes = sorted(c['notes'][2:5], reverse=True)
    
    base_time = bar_start
    for i, note in enumerate(melody_notes):
        offset_ms = i * random.uniform(20, 40)
        t = human_time(base_time, 'melody', phrase_prog) + ms_to_beats(offset_ms)
        v = human_velocity(60 - i*5, 0, 'melody', phrase_prog)
        d = 0.5 + i * 0.1
        notes.append((note, t, d, max(32, v)))
    
    return notes

# ============================================
# SONG STRUCTURE (~76 bars for 4 minutes)
# ============================================

midi = MIDIFile(1, deinterleave=False)
midi.addTempo(0, 0, TEMPO)
midi.addProgramChange(0, 0, 0, 25)  # Steel acoustic guitar

all_notes = []
bar = 0

print("Creating guitar MIDI...")
print(f"Tempo: {TEMPO} BPM | Target: ~4 minutes")
print("="*50)

# ----- INTRO (4 bars) - Voicing arpeggios -----
print(f"Intro (bars 1-4): Dm voicing arpeggios...")
# Bar 1-2: Dm arpeggio up, let ring
all_notes.extend(voicing_arpeggio('Dm', bar * 4, 'up'))
bar += 1
all_notes.extend(fingerpick_bar('Dm', bar * 4, 0.5, 'sparse'))
bar += 1
# Bar 3-4: Am arpeggio, transition
all_notes.extend(voicing_arpeggio('Am', bar * 4, 'down'))
bar += 1
all_notes.extend(fingerpick_bar('Am7', bar * 4, 1.0, 'sparse'))
bar += 1

# ----- VERSE 1 (8 bars) -----
print(f"Verse 1 (bars 5-12): Dm → Am → Fmaj7 → E...")
v1_chords = ['Dm', 'Dm', 'Am', 'Am', 'Fmaj7', 'Fmaj7', 'E', 'E7']
for i, chord in enumerate(v1_chords):
    phrase_prog = i / 8
    all_notes.extend(fingerpick_bar(chord, bar * 4, phrase_prog, 'normal'))
    bar += 1
# Verse 1 ending flair
all_notes.extend(pull_off_phrase('Am', (bar-1) * 4 + 2, 1.0))

# ----- VERSE 2 (8 bars) -----
print(f"Verse 2 (bars 13-20): Building intensity...")
v2_chords = ['Dm', 'Dm7', 'Am', 'Am', 'Fmaj7', 'Dm', 'E', 'Am']
for i, chord in enumerate(v2_chords):
    phrase_prog = i / 8
    # "Shea butter" pause at bar 5 of verse
    density = 'sparse' if i == 5 else 'normal'
    all_notes.extend(fingerpick_bar(chord, bar * 4, phrase_prog, density))
    bar += 1

# ----- CHORUS (8 bars) -----
print(f"Chorus (bars 21-28): Dm → Am → E → Fmaj7...")
chorus_chords = ['Dm', 'Dm', 'Am', 'Am', 'E', 'E7', 'Fmaj7', 'Am']
for i, chord in enumerate(chorus_chords):
    phrase_prog = i / 8
    all_notes.extend(fingerpick_bar(chord, bar * 4, phrase_prog, 'normal'))
    # Add hammer-on at "I hit the floor"
    if i == 4:
        all_notes.extend(hammer_on_phrase(chord, bar * 4 + 2, phrase_prog))
    bar += 1

# ----- INSTRUMENTAL (8 bars) - Voicing exploration -----
print(f"Instrumental (bars 29-36): Voicing techniques...")
inst_chords = ['Dm', 'Gm', 'Bb', 'Am', 'Dm', 'F', 'E', 'Am']
for i, chord in enumerate(inst_chords):
    phrase_prog = i / 8
    if i % 2 == 0:
        # Even bars: arpeggio voicing
        direction = 'up' if i % 4 == 0 else 'down'
        all_notes.extend(voicing_arpeggio(chord, bar * 4, direction))
    else:
        # Odd bars: sparse picking
        all_notes.extend(fingerpick_bar(chord, bar * 4, phrase_prog, 'sparse'))
    bar += 1

# ----- VERSE 3 (8 bars) - Pulled back -----
print(f"Verse 3 (bars 37-44): Sparse, whispered...")
v3_chords = ['Dm', 'Dm', 'Am', 'Am', 'Fmaj7', 'Am', 'E', 'Am']
for i, chord in enumerate(v3_chords):
    phrase_prog = i / 8
    all_notes.extend(fingerpick_bar(chord, bar * 4, phrase_prog, 'sparse'))
    bar += 1

# ----- CHORUS 2 (8 bars) -----
print(f"Chorus 2 (bars 45-52): Return of theme...")
for i, chord in enumerate(chorus_chords):
    phrase_prog = i / 8
    all_notes.extend(fingerpick_bar(chord, bar * 4, phrase_prog, 'normal'))
    bar += 1

# ----- BRIDGE (8 bars) -----
print(f"Bridge (bars 53-60): 'Your silence carved a hole'...")
bridge_chords = ['Dm', 'Dm7', 'Am', 'Am7', 'E', 'E7', 'Dm', 'Am']
for i, chord in enumerate(bridge_chords):
    phrase_prog = i / 8
    all_notes.extend(fingerpick_bar(chord, bar * 4, phrase_prog, 'sparse'))
    bar += 1

# ----- FINAL REVEAL (8 bars) -----
print(f"Final (bars 61-68): 'You weren't resting...'")
final_chords = ['Dm', 'Dm', 'Am', 'Am', 'E', 'E', 'Dm', 'Am']
for i, chord in enumerate(final_chords):
    all_notes.extend(fingerpick_bar(chord, bar * 4, i/8, 'dying'))
    bar += 1

# ----- OUTRO (8 bars) - Fading voicings -----
print(f"Outro (bars 69-76): Dying away...")
outro_chords = ['Dm', 'Am', 'Dm', 'Am']
for i, chord in enumerate(outro_chords):
    # Very sparse, just voicing arpeggios dying
    if i < 2:
        all_notes.extend(voicing_arpeggio(chord, bar * 4, 'up'))
        bar += 1
        all_notes.extend(fingerpick_bar(chord, bar * 4, 1.0, 'dying'))
        bar += 1
    else:
        all_notes.extend(fingerpick_bar(chord, bar * 4, 1.0, 'dying'))
        bar += 1
        bar += 1  # Rest bar

# Final held note
print(f"Final chord...")
midi.addNote(0, 0, 50, bar * 4, 8.0, 28)  # D3, very soft, long ring

# ============================================
# WRITE TO FILE
# ============================================

duration_sec = (bar + 2) * 4 * 60 / TEMPO
duration_min = duration_sec / 60

print("="*50)
print(f"Total bars: {bar}")
print(f"Total notes: {len(all_notes)}")
print(f"Duration: {duration_sec:.0f}s ({duration_min:.1f} minutes)")

for note, time, dur, vel in all_notes:
    midi.addNote(0, 0, note, time, dur, vel)

output = "/Users/seanburdges/Desktop/Kelly_Guitar_Final.mid"
with open(output, 'wb') as f:
    midi.writeFile(f)

print(f"\nSaved: {output}")
print("""
TECHNIQUES APPLIED:
━━━━━━━━━━━━━━━━━━━
• Travis picking (bass on 1 & 3, melody between)
• Voicing arpeggios (intro, instrumental, outro)
• Hammer-ons at emotional peaks
• Pull-offs at verse endings
• Human micro-timing (bass early, melody laid-back)
• Dynamic phrase shaping (ppp → mp → ppp)
• Ghost notes for texture
• Palm-muted bass (shorter durations)
• Phrase-end rubato

STRUCTURE:
━━━━━━━━━━
Intro → Verse 1 → Verse 2 → Chorus →
Instrumental → Verse 3 → Chorus 2 → 
Bridge → Final → Outro
""")
