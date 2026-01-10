#!/usr/bin/env python3
"""
Generate Complete Kelly Song Arrangement
"When I Found You Sleeping" - Full MIDI arrangement with all parts
"""

import json
from pathlib import Path
from mido import MidiFile, MidiTrack, Message, MetaMessage

# Note mappings
NOTE_MAP = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 
            'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 
            'A#': 10, 'Bb': 10, 'B': 11}

# Chord voicing templates (intervals from root)
CHORD_VOICINGS = {
    'maj': [0, 4, 7],       # Major triad
    'min': [0, 3, 7],       # Minor triad  
    'm': [0, 3, 7],         # Minor (alternate)
    '5': [0, 7],            # Power chord
    '7': [0, 4, 7, 10],     # Dominant 7th
    'maj7': [0, 4, 7, 11],  # Major 7th
}

def parse_chord_name(chord_str):
    """Parse chord string into root note and quality."""
    # Find root note
    if len(chord_str) >= 2 and chord_str[1] in ['#', 'b']:
        root = chord_str[:2]
        quality = chord_str[2:] if len(chord_str) > 2 else 'maj'
    else:
        root = chord_str[0]
        quality = chord_str[1:] if len(chord_str) > 1 else 'maj'
    
    # Default to major if no quality specified
    if quality == '':
        quality = 'maj'
    
    return root, quality

def get_chord_notes(chord_str, octave=4):
    """Get MIDI notes for a chord."""
    root, quality = parse_chord_name(chord_str)
    
    # Get root MIDI note
    root_note = NOTE_MAP.get(root, 0) + (octave * 12)
    
    # Get voicing intervals
    intervals = CHORD_VOICINGS.get(quality, CHORD_VOICINGS['maj'])
    
    # Build chord notes
    return [root_note + interval for interval in intervals]

def load_intent(intent_path):
    """Load intent from JSON file."""
    with open(intent_path, 'r') as f:
        return json.load(f)

def create_chord_track(chords, key, bars_per_chord=4, tempo_bpm=82):
    """Create a MIDI track with chord voicings."""
    track = MidiTrack()
    track.append(MetaMessage('track_name', name='Chords', time=0))
    
    # Calculate ticks per bar (480 ticks per quarter note, 4 quarters per bar)
    ticks_per_beat = 480
    ticks_per_bar = ticks_per_beat * 4
    ticks_per_chord = ticks_per_bar * bars_per_chord
    
    time = 0
    for chord_name in chords:
        notes = get_chord_notes(chord_name, octave=4)
        
        # Note on
        for i, note in enumerate(notes):
            track.append(Message('note_on', note=note, velocity=70, time=time if i == 0 else 0))
        
        time += ticks_per_chord
        
        # Note off
        for i, note in enumerate(notes):
            track.append(Message('note_off', note=note, velocity=0, time=time if i == 0 else 0))
        time = 0
    
    return track

def create_arpeggio_track(chords, key, bars_per_chord=4, pattern='1-3-5-3'):
    """Create fingerpicking arpeggio pattern."""
    track = MidiTrack()
    track.append(MetaMessage('track_name', name='Arpeggio Guitar', time=0))
    
    ticks_per_beat = 480
    ticks_per_bar = ticks_per_beat * 4
    ticks_per_chord = ticks_per_bar * bars_per_chord
    ticks_per_note = ticks_per_beat  # Eighth notes
    
    pattern_indices = [int(p) - 1 for p in pattern.split('-')]
    
    time = 0
    for chord_name in chords:
        notes = get_chord_notes(chord_name, octave=3)
        
        # Play pattern for each bar
        for bar in range(bars_per_chord):
            for idx in pattern_indices:
                if idx < len(notes):
                    note = notes[idx]
                    # Slight velocity variation for human feel
                    velocity = 60 + (bar * 2) + (idx % 3) * 5
                    track.append(Message('note_on', note=note, velocity=velocity, time=time))
                    time = ticks_per_note // 2  # Note length
                    track.append(Message('note_off', note=note, velocity=0, time=time))
                    time = ticks_per_note // 2  # Gap to next note
        time = 0
    
    return track

def create_bass_track(chords, key, bars_per_chord=4):
    """Create simple bass line following chord roots."""
    track = MidiTrack()
    track.append(MetaMessage('track_name', name='Bass', time=0))
    
    ticks_per_beat = 480
    ticks_per_bar = ticks_per_beat * 4
    ticks_per_chord = ticks_per_bar * bars_per_chord
    
    time = 0
    for chord_name in chords:
        # Get root note of chord in bass octave
        root, _ = parse_chord_name(chord_name)
        root_note = NOTE_MAP.get(root, 0) + (2 * 12)  # Octave 2
        
        # Play root on beats 1 and 3 of each bar
        for bar in range(bars_per_chord):
            for beat in [0, 2]:  # Beats 1 and 3
                track.append(Message('note_on', note=root_note, velocity=80, time=time))
                time = ticks_per_beat
                track.append(Message('note_off', note=root_note, velocity=0, time=time))
                time = ticks_per_beat if beat == 0 else 0
        time = 0
    
    return track

def create_drums_track(bars, style='lofi'):
    """Create lo-fi drum pattern."""
    track = MidiTrack()
    track.append(MetaMessage('track_name', name='Drums', time=0))
    
    # GM Drum mapping
    kick = 36
    snare = 38
    hihat_closed = 42
    hihat_open = 46
    
    ticks_per_beat = 480
    ticks_per_bar = ticks_per_beat * 4
    
    time = 0
    for bar in range(bars):
        # Basic lo-fi beat: kick on 1 and 3, snare on 2 and 4
        # Kick on beat 1
        track.append(Message('note_on', note=kick, velocity=90, time=time, channel=9))
        track.append(Message('note_off', note=kick, velocity=0, time=10, channel=9))
        
        # Hi-hat on off-beats
        for beat in range(8):  # 8th notes
            time = ticks_per_beat // 2 if beat > 0 else ticks_per_beat - 10
            vel = 45 if beat % 2 == 0 else 35  # Vary velocity
            track.append(Message('note_on', note=hihat_closed, velocity=vel, time=time, channel=9))
            track.append(Message('note_off', note=hihat_closed, velocity=0, time=10, channel=9))
            
            # Snare on beat 2
            if beat == 2:
                track.append(Message('note_on', note=snare, velocity=70, time=0, channel=9))
                track.append(Message('note_off', note=snare, velocity=0, time=10, channel=9))
            # Kick on beat 3
            elif beat == 4:
                track.append(Message('note_on', note=kick, velocity=85, time=0, channel=9))
                track.append(Message('note_off', note=kick, velocity=0, time=10, channel=9))
            # Snare on beat 4
            elif beat == 6:
                track.append(Message('note_on', note=snare, velocity=75, time=0, channel=9))
                track.append(Message('note_off', note=snare, velocity=0, time=10, channel=9))
    
    return track

def generate_complete_arrangement(intent_path, output_dir):
    """Generate complete Kelly song arrangement."""
    print("=" * 70)
    print("GENERATING COMPLETE KELLY SONG ARRANGEMENT")
    print("=" * 70)
    
    # Load intent
    intent_data = load_intent(intent_path)
    key = intent_data['technical_constraints']['technical_key']
    mode = intent_data['technical_constraints']['technical_mode']
    print(f"\n✓ Loaded intent: {key} {mode}")
    
    # Song structure
    verse_chords = ['F', 'C', 'Am', 'Dm']  # The stuck, the cycle
    chorus_chords = ['F', 'C', 'Am', 'Bb']  # The drop, the reveal
    
    bars_per_chord = 2
    
    # Create MIDI file
    mid = MidiFile(ticks_per_beat=480)
    
    # Tempo track
    tempo_track = MidiTrack()
    tempo_bpm = 82
    tempo_track.append(MetaMessage('set_tempo', tempo=int(60000000 / tempo_bpm), time=0))
    tempo_track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    mid.tracks.append(tempo_track)
    
    print(f"✓ Tempo: {tempo_bpm} BPM")
    print(f"✓ Time signature: 4/4")
    
    # Full song structure: Intro - Verse 1 - Chorus - Verse 2 - Chorus - Bridge - Chorus - Outro
    full_progression = (
        verse_chords * 2 +       # Intro (8 bars)
        verse_chords * 2 +       # Verse 1 (8 bars)
        chorus_chords * 2 +      # Chorus (8 bars)
        verse_chords * 2 +       # Verse 2 (8 bars)
        chorus_chords * 2 +      # Chorus (8 bars)
        ['Dm', 'Bb', 'C', 'F'] + # Bridge (4 bars)
        chorus_chords * 2 +      # Final Chorus (8 bars)
        ['F', 'C', 'Am', 'F']    # Outro (4 bars)
    )
    
    total_bars = len(full_progression) * bars_per_chord
    print(f"✓ Structure: {len(full_progression)} chords, {total_bars} bars")
    print(f"✓ Sections: Intro → Verse → Chorus → Verse → Chorus → Bridge → Chorus → Outro")
    
    # Create tracks
    print("\n✓ Generating tracks...")
    mid.tracks.append(create_arpeggio_track(full_progression, key, bars_per_chord))
    print("  - Arpeggio guitar (fingerpicking)")
    
    mid.tracks.append(create_bass_track(full_progression, key, bars_per_chord))
    print("  - Bass (root notes)")
    
    mid.tracks.append(create_drums_track(total_bars, style='lofi'))
    print("  - Drums (lo-fi beat)")
    
    mid.tracks.append(create_chord_track(full_progression, key, bars_per_chord, tempo_bpm))
    print("  - Chord pads")
    
    # Save file
    output_path = Path(output_dir) / "kelly_song_complete_arrangement.mid"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(output_path))
    
    print(f"\n✅ Complete arrangement saved to: {output_path}")
    print(f"   Duration: ~{total_bars * 2} seconds at {tempo_bpm} BPM")
    print(f"   Tracks: {len(mid.tracks)}")
    print(f"   Emotional arc: {intent_data['song_intent']['mood_primary']} with {intent_data['song_intent']['narrative_arc']}")
    
    return output_path

def generate_rule_breaking_examples(output_dir):
    """Generate MIDI examples demonstrating various rule breaks."""
    print("\n" + "=" * 70)
    print("GENERATING RULE-BREAKING MIDI EXAMPLES")
    print("=" * 70)
    
    examples = [
        {
            'name': 'modal_interchange_example',
            'chords': ['F', 'C', 'Bbm', 'F'],  # Bbm borrowed from F minor
            'key': 'F',
            'description': 'Modal Interchange - Borrowed iv chord creates bittersweet feel'
        },
        {
            'name': 'parallel_fifths_power_chords',
            'chords': ['E5', 'A5', 'D5', 'A5'],  # Power chord progression
            'key': 'E',
            'description': 'Parallel Fifths - Power chords for unified, powerful sound'
        },
        {
            'name': 'avoided_resolution',
            'chords': ['C', 'G', 'Am', 'F'],  # Ends on IV instead of I
            'key': 'C',
            'description': 'Avoided Resolution - Ending on IV creates unresolved yearning'
        },
        {
            'name': 'deceptive_cadence',
            'chords': ['C', 'Am', 'F', 'G', 'Am'],  # V-vi instead of V-I
            'key': 'C',
            'description': 'Deceptive Cadence - False resolution creates surprise'
        }
    ]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for example in examples:
        mid = MidiFile(ticks_per_beat=480)
        
        # Tempo track
        tempo_track = MidiTrack()
        tempo_track.append(MetaMessage('set_tempo', tempo=int(60000000 / 90), time=0))
        mid.tracks.append(tempo_track)
        
        # Chord track
        mid.tracks.append(create_chord_track(example['chords'], example['key'], bars_per_chord=2, tempo_bpm=90))
        
        output_path = output_dir / f"{example['name']}.mid"
        mid.save(str(output_path))
        
        print(f"✓ {example['name']}")
        print(f"  {example['description']}")
        print(f"  Progression: {' → '.join(example['chords'])}")
        print(f"  Saved: {output_path}")
        print()
    
    return len(examples)

if __name__ == '__main__':
    import sys
    
    # Paths
    repo_root = Path(__file__).parent
    intent_path = repo_root / "examples_music-brain" / "intents" / "kelly_when_i_found_you_sleeping.json"
    output_dir = repo_root / "vault" / "Songs" / "when-i-found-you-sleeping" / "midi"
    examples_dir = repo_root / "vault" / "Songwriting_Guides" / "midi_examples"
    
    print("DAiW Music Brain - Complete Song Generator")
    print("=" * 70)
    print(f"Intent file: {intent_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Generate complete Kelly song
    try:
        kelly_output = generate_complete_arrangement(str(intent_path), str(output_dir))
    except Exception as e:
        print(f"\n❌ Error generating Kelly song: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Generate rule-breaking examples
    try:
        num_examples = generate_rule_breaking_examples(str(examples_dir))
        print(f"\n✅ Generated {num_examples} rule-breaking MIDI examples")
    except Exception as e:
        print(f"\n❌ Error generating examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("ALL MIDI GENERATION COMPLETE")
    print("=" * 70)
    print("\nFiles generated:")
    print(f"  1. Complete Kelly song: {kelly_output}")
    print(f"  2. Rule-breaking examples: {examples_dir}")
    print("\nNext steps:")
    print("  - Import MIDI into your DAW")
    print("  - Apply humanization and effects")
    print("  - Reference the intent file for emotional context")
    print("\n✨ 'Interrogate Before Generate' - Mission accomplished!")
