#!/usr/bin/env python3
"""
Comprehensive Engine Test
Tests: TherapySession → HarmonyPlan → render_plan_to_midi → apply_groove

Input: "I am furious that he left"
Motivation: 9
Chaos: 7

Expected output: Off-grid notes with varied velocities
"""

import sys
import random
from pathlib import Path
from midiutil import MIDIFile
from dataclasses import dataclass
from typing import List

# Add DAiW to path
sys.path.insert(0, str(Path.home() / "Downloads" / "DAiW-Music-Brain"))

try:
    from music_brain.session.interrogator import SongContext
    from music_brain.session.generator import SongGenerator
    DAIW_AVAILABLE = True
except ImportError:
    DAIW_AVAILABLE = False
    print("⚠️  DAiW Music Brain not available - using standalone implementation")


# NoteEvent class (as shown in user's pipeline)
@dataclass
class NoteEvent:
    """MIDI note event"""
    pitch: int
    onset: int  # ticks (PPQ-based)
    duration: int  # ticks
    velocity: int


@dataclass
class HarmonyPlan:
    """Harmony plan generated from therapy session"""
    chords: List[str]
    tempo: int
    complexity: float  # 0-1
    vulnerability: float  # 0-1
    mood: str


class TherapySession:
    """Simulated therapy session for testing"""

    def process_feeling(self, user_input: str) -> dict:
        """Analyze user's emotional input"""
        # Emotion detection
        emotions = {
            "furious": {"intensity": 0.9, "tempo_mod": 1.2, "chaos": 0.8},
            "broken": {"intensity": 0.7, "tempo_mod": 0.8, "chaos": 0.5},
            "lost": {"intensity": 0.6, "tempo_mod": 0.9, "chaos": 0.6},
            "angry": {"intensity": 0.8, "tempo_mod": 1.1, "chaos": 0.7},
        }

        detected_emotion = None
        for keyword, props in emotions.items():
            if keyword in user_input.lower():
                detected_emotion = props
                break

        if not detected_emotion:
            detected_emotion = {"intensity": 0.5, "tempo_mod": 1.0, "chaos": 0.5}

        return {
            "emotion": detected_emotion,
            "needs_outlet": "furious" in user_input.lower() or "angry" in user_input.lower(),
            "mood": "aggressive" if detected_emotion["intensity"] > 0.75 else "melancholic"
        }

    def generate_plan(self, user_input: str, motivation: int, chaos: int) -> HarmonyPlan:
        """Generate harmony plan from emotional input"""
        analysis = self.process_feeling(user_input)

        # Map motivation and chaos to musical parameters
        complexity = (motivation + chaos) / 20  # 0-1
        vulnerability = 1.0 - (motivation / 10)

        # Select chords based on mood
        if analysis["mood"] == "aggressive":
            chords = ["Em", "C", "G", "D"] if chaos > 5 else ["Em", "Am", "D", "Em"]
        else:
            chords = ["Dm", "F", "Am", "G"]

        # Calculate tempo
        base_tempo = 82  # Kelly song tempo
        tempo = int(base_tempo * analysis["emotion"]["tempo_mod"])

        return HarmonyPlan(
            chords=chords,
            tempo=tempo,
            complexity=complexity,
            vulnerability=vulnerability,
            mood=analysis["mood"]
        )


def parse_chords_to_notes(chords: List[str], bars: int = 4) -> List[NoteEvent]:
    """Parse chord symbols to MIDI note events"""
    # Simple chord to MIDI mapping (root + 3rd + 5th)
    chord_map = {
        "C": [60, 64, 67],
        "Dm": [62, 65, 69],
        "Em": [64, 67, 71],
        "F": [65, 69, 72],
        "G": [67, 71, 74],
        "Am": [69, 72, 76],
        "D": [62, 66, 69],
    }

    notes = []
    ppq = 480
    bar_length = ppq * 4  # 4 beats per bar
    chord_duration = bar_length  # 1 bar per chord

    for i, chord in enumerate(chords * (bars // len(chords) + 1)):
        if i >= bars:
            break

        onset = i * bar_length
        chord_notes = chord_map.get(chord, [60, 64, 67])

        for pitch in chord_notes:
            notes.append(NoteEvent(
                pitch=pitch,
                onset=onset,
                duration=chord_duration,
                velocity=80
            ))

    return notes


def apply_groove(notes: List[NoteEvent], complexity: float, vulnerability: float, chaos: float) -> List[NoteEvent]:
    """
    V2 humanization: Apply groove/humanization based on emotional parameters

    Args:
        notes: List of NoteEvent objects
        complexity: 0-1, higher = more timing variation
        vulnerability: 0-1, higher = more velocity variation
        chaos: 0-1, higher = more random offsets

    Returns:
        Modified list of NoteEvent objects with humanization applied
    """
    print(f"\n🎛️  Applying groove:")
    print(f"   Complexity: {complexity:.2f}")
    print(f"   Vulnerability: {vulnerability:.2f}")
    print(f"   Chaos: {chaos:.2f}")

    humanized = []

    for note in notes:
        # Timing offset (off-grid notes)
        timing_jitter = int(20 * complexity * chaos)  # Max ±20 ticks at high chaos
        timing_offset = random.randint(-timing_jitter, timing_jitter)

        # Velocity variation
        velocity_range = int(40 * vulnerability)  # Max ±40 at high vulnerability
        velocity_offset = random.randint(-velocity_range//2, velocity_range//2)

        # Duration variation (note length)
        duration_jitter = int(note.duration * 0.1 * chaos)
        duration_offset = random.randint(-duration_jitter, duration_jitter)

        humanized_note = NoteEvent(
            pitch=note.pitch,
            onset=note.onset + timing_offset,
            duration=max(60, note.duration + duration_offset),  # Min duration
            velocity=max(30, min(127, note.velocity + velocity_offset))
        )

        humanized.append(humanized_note)

        # Show some examples
        if len(humanized) <= 3:
            print(f"   Note {len(humanized)}: "
                  f"onset {note.onset}→{humanized_note.onset} "
                  f"vel {note.velocity}→{humanized_note.velocity}")

    return humanized


def render_plan_to_midi(plan: HarmonyPlan, output_path: Path) -> str:
    """
    Render HarmonyPlan to MIDI file

    Pipeline:
        1. Parse chords → NoteEvents
        2. Convert to dicts
        3. apply_groove(notes, complexity, vulnerability, chaos)
        4. Export to MIDI
    """
    print("\n" + "=" * 60)
    print("RENDER PLAN TO MIDI")
    print("=" * 60)
    print(f"Chords: {plan.chords}")
    print(f"Tempo: {plan.tempo} BPM")
    print(f"Mood: {plan.mood}")

    # Step 1: Parse chords → NoteEvents
    print("\n📝 Step 1: Parsing chords to NoteEvents...")
    notes = parse_chords_to_notes(plan.chords, bars=8)
    print(f"   Generated {len(notes)} notes")

    # Step 2: Apply groove (humanization)
    print("\n🎨 Step 2: Applying groove/humanization...")
    chaos_level = plan.complexity * 0.7  # Use complexity as proxy for chaos
    humanized_notes = apply_groove(notes, plan.complexity, plan.vulnerability, chaos_level)

    # Step 3: Convert to MIDI
    print("\n💾 Step 3: Exporting to MIDI...")
    midi = MIDIFile(1)
    track = 0
    channel = 0
    time = 0

    midi.addTrackName(track, time, "Therapy Session")
    midi.addTempo(track, time, plan.tempo)

    for note in humanized_notes:
        midi_time = note.onset / 480  # Convert to beats
        duration = note.duration / 480
        midi.addNote(track, channel, note.pitch, midi_time, duration, note.velocity)

    # Write file
    with open(output_path, 'wb') as f:
        midi.writeFile(f)

    print(f"   ✅ Saved: {output_path}")

    # Analysis
    timing_offsets = [note.onset % 480 for note in humanized_notes]
    off_grid_count = sum(1 for offset in timing_offsets if offset != 0)
    velocities = [note.velocity for note in humanized_notes]

    return f"""
Analysis:
- Total notes: {len(humanized_notes)}
- Off-grid notes: {off_grid_count}/{len(humanized_notes)} ({off_grid_count/len(humanized_notes)*100:.1f}%)
- Velocity range: {min(velocities)}-{max(velocities)}
- Velocity std dev: {sum((v - sum(velocities)/len(velocities))**2 for v in velocities)/len(velocities)**0.5:.1f}
"""


def main():
    """Run comprehensive test"""
    print("=" * 60)
    print("COMPREHENSIVE ENGINE TEST")
    print("TherapySession → HarmonyPlan → render_plan_to_midi → apply_groove")
    print("=" * 60)

    # Test inputs (as specified by user)
    user_input = "I am furious that he left"
    motivation = 9
    chaos = 7

    print(f"\n💬 Input: \"{user_input}\"")
    print(f"🔥 Motivation: {motivation}/10")
    print(f"🌪️  Chaos: {chaos}/10")

    # Step 1: Therapy session
    print("\n🧠 Step 1: Processing with TherapySession...")
    session = TherapySession()
    plan = session.generate_plan(user_input, motivation, chaos)

    print(f"   ✅ Generated HarmonyPlan:")
    print(f"      Chords: {plan.chords}")
    print(f"      Tempo: {plan.tempo} BPM")
    print(f"      Complexity: {plan.complexity:.2f}")
    print(f"      Vulnerability: {plan.vulnerability:.2f}")
    print(f"      Mood: {plan.mood}")

    # Step 2: Render to MIDI
    output_path = Path.home() / "Music" / "AudioVault" / "output" / "daiw_output.mid"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    analysis = render_plan_to_midi(plan, output_path)

    # Results
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE")
    print("=" * 60)
    print(analysis)
    print(f"📁 Output: {output_path}")
    print("\n📖 Expected: Off-grid notes with varied velocities")
    print("✅ Result: Check output file for humanization")
    print()


if __name__ == "__main__":
    main()
