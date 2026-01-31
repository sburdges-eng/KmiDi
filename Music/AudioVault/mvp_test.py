#!/usr/bin/env python3
"""
MVP Test: "I feel broken" → MIDI → Kit → Playback
Full pipeline test of the therapy session system
"""

import sys
import json
from pathlib import Path
from midiutil import MIDIFile

# Add DAiW to path
sys.path.insert(0, str(Path.home() / "Downloads" / "DAiW-Music-Brain"))

try:
    from music_brain.session.interrogator import TherapySession
    from music_brain.session.generator import NoteEvent, render_plan_to_midi
    DAIW_AVAILABLE = True
except ImportError:
    DAIW_AVAILABLE = False
    print("⚠️  DAiW Music Brain not found - using simplified pipeline")

    # Define NoteEvent fallback
    class NoteEvent:
        def __init__(self, pitch, onset, duration, velocity):
            self.pitch = pitch
            self.onset = onset
            self.duration = duration
            self.velocity = velocity

VAULT_DIR = Path.home() / "Music" / "AudioVault"
KITS_DIR = VAULT_DIR / "kits"
OUTPUT_DIR = VAULT_DIR / "output"


def create_simple_beat():
    """Create a simple lo-fi beat for testing"""
    # Create NoteEvent list
    notes = []

    # 4-bar pattern at 82 BPM (Kelly song tempo)
    bpm = 82
    quarter_note = 480  # PPQ

    # Kick pattern (on beats 1 and 3)
    for bar in range(4):
        offset = bar * 4 * quarter_note
        notes.append(NoteEvent(36, offset, quarter_note, 80))  # Beat 1
        notes.append(NoteEvent(36, offset + 2 * quarter_note, quarter_note, 75))  # Beat 3

    # Snare pattern (on beats 2 and 4)
    for bar in range(4):
        offset = bar * 4 * quarter_note
        notes.append(NoteEvent(38, offset + quarter_note, quarter_note, 85))  # Beat 2
        notes.append(NoteEvent(38, offset + 3 * quarter_note, quarter_note, 82))  # Beat 4

    # Hi-hat pattern (8th notes)
    for bar in range(4):
        for beat in range(8):
            offset = bar * 4 * quarter_note + beat * (quarter_note // 2)
            velocity = 70 if beat % 2 == 0 else 60  # Accent on downbeats
            notes.append(NoteEvent(42, offset, quarter_note // 2, velocity))

    return notes, bpm


def therapy_to_beat(user_input):
    """Convert therapy session input to beat"""
    if not DAIW_AVAILABLE:
        print("📝 Using simplified beat (DAiW not available)")
        return create_simple_beat()

    try:
        # Run therapy session
        print(f"\n💬 Input: \"{user_input}\"")
        print("🧠 Processing with TherapySession...")

        session = TherapySession()
        response = session.process_feeling(user_input)

        print(f"✅ Response: {response[:100]}...")

        # For MVP, use simple beat
        # In full version, this would analyze emotional intensity
        # and generate appropriate rhythm
        return create_simple_beat()

    except Exception as e:
        print(f"⚠️  Therapy session error: {e}")
        print("   Falling back to simple beat")
        return create_simple_beat()


def save_midi(notes, bpm, output_path):
    """Save notes to MIDI file"""
    midi = MIDIFile(1)  # 1 track
    track = 0
    channel = 9  # Drum channel
    time = 0

    midi.addTrackName(track, time, "LoFi Drums")
    midi.addTempo(track, time, bpm)

    # Add all notes
    for note in notes:
        midi_time = note.onset / 480  # Convert to beats
        duration = note.duration / 480
        midi.addNote(track, channel, note.pitch, midi_time, duration, note.velocity)

    # Write to file
    with open(output_path, 'wb') as f:
        midi.writeFile(f)


def load_kit_info():
    """Load kit JSON"""
    kit_file = KITS_DIR / "LoFi_Bedroom_Kit.json"
    if not kit_file.exists():
        return None

    with open(kit_file) as f:
        return json.load(f)


def main():
    """Run MVP test"""
    print("=" * 60)
    print("MVP TEST: 'I feel broken' → MIDI → Kit → Playback")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Therapy session
    user_input = "I feel broken"
    notes, bpm = therapy_to_beat(user_input)

    print(f"\n✅ Generated {len(notes)} MIDI notes")
    print(f"   BPM: {bpm}")
    print(f"   Duration: {max(n.onset + n.duration for n in notes) / 480 / bpm * 60:.1f} seconds")

    # Step 2: Save MIDI
    midi_file = OUTPUT_DIR / "i_feel_broken.mid"
    save_midi(notes, bpm, midi_file)
    print(f"\n✅ MIDI saved: {midi_file}")

    # Step 3: Load kit
    kit_info = load_kit_info()
    if kit_info:
        print(f"\n✅ Kit loaded: {kit_info['kit_name']}")
        print(f"   Zones: {len(kit_info['zones'])}")
    else:
        print("\n⚠️  Kit not found")

    # Step 4: Playback instructions
    print("\n" + "=" * 60)
    print("✅ MVP TEST COMPLETE")
    print("=" * 60)
    print("\n📖 To play in Logic Pro:")
    print(f"   1. Open Logic Pro")
    print(f"   2. Create Software Instrument track")
    print(f"   3. Load Sampler/Quick Sampler")
    print(f"   4. Load kit: ~/Music/Audio Music Apps/Sampler Instruments/LoFi_Bedroom_Kit_GUIDE.txt")
    print(f"   5. Import MIDI: {midi_file}")
    print(f"   6. Press play ▶️")
    print(f"\n🎵 Or play directly:")
    print(f"   python3 -m pygame.examples.midi --input '{midi_file}'")
    print(f"\n📁 All files in: {VAULT_DIR}")
    print()


if __name__ == "__main__":
    main()
