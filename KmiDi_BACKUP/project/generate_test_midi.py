"""
Generate MIDI file for review using the comprehensive test parameters.
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def generate_midi_for_review():
    """Generate MIDI file with test parameters."""

    print("=" * 80)
    print("Generating MIDI File for Review")
    print("=" * 80)
    print()

    # Load test parameters
    params_file = Path("test_song_parameters.json")
    if not params_file.exists():
        print("❌ test_song_parameters.json not found. Running parameter demo first...")
        from test_streamlit_generation import demonstrate_parameter_effects

        demonstrate_parameter_effects()

    with open(params_file, "r") as f:
        data = json.load(f)

    emotion_text = data["emotional_intent"]
    params = data["parameters"]

    print(f"Emotional Intent: {emotion_text}")
    print(f"Total Parameters: {len(params)}")
    print()
    print("Generating MIDI file...")
    print()

    try:
        # Try using music_brain API directly
        from music_brain.api import api as music_api

        # Map parameters to therapy session
        motivation = max(1, min(10, int(params.get("bpm", 120) / 20)))
        chaos = 0.5

        # Adjust chaos based on parameters
        if params.get("tension_level", 5) > 7:
            chaos = 0.7  # Higher chaos for high tension
        if params.get("randomness", 2) > 5:
            chaos = 0.8

        print(f"Using motivation: {motivation}/10")
        print(f"Using chaos tolerance: {chaos}")
        print()

        # Generate music with MIDI output
        output_dir = Path("output/review")
        output_dir.mkdir(parents=True, exist_ok=True)
        midi_output_path = str(output_dir / "test_song_review.mid")

        print("🎵 Calling music generation API...")
        print(f"   Output MIDI: {midi_output_path}")
        print()

        result = music_api.therapy_session(
            text=emotion_text,
            motivation=motivation,
            chaos_tolerance=chaos,
            output_midi=midi_output_path,  # Save MIDI file
        )

        print()
        print("=" * 80)
        print("GENERATION RESULT")
        print("=" * 80)
        print()

        if result:
            print("✅ Music generation successful!")
            print()

            # Display result
            if isinstance(result, dict):
                print("Generated Music Details:")
                print("-" * 80)

                if "chords" in result:
                    chords = result["chords"]
                    if isinstance(chords, str):
                        chords = chords.split(", ")
                    elif isinstance(chords, list):
                        pass
                    else:
                        chords = [str(chords)]

                    print(f"🎹 Chord Progression: {' → '.join(chords)}")
                    print(f"   ({len(chords)} chords)")

                if "key" in result:
                    print(f"🎵 Key: {result['key']}")
                elif "metadata" in result and "key" in result["metadata"]:
                    print(f"🎵 Key: {result['metadata']['key']}")

                if "tempo" in result:
                    print(f"⏱️  Tempo: {result['tempo']} BPM")
                elif "metadata" in result and "tempo" in result["metadata"]:
                    print(f"⏱️  Tempo: {result['metadata']['tempo']} BPM")
                else:
                    print(f"⏱️  Tempo: {params.get('bpm', 90)} BPM (from parameters)")

                if "time_signature" in result:
                    print(f"📐 Time Signature: {result['time_signature']}")
                elif "metadata" in result and "time_signature" in result["metadata"]:
                    print(f"📐 Time Signature: {result['metadata']['time_signature']}")
                else:
                    print(
                        f"📐 Time Signature: {params.get('time_signature', '4/4')} (from parameters)"
                    )

                # MIDI file path
                midi_path = None
                if "midi_path" in result:
                    midi_path = result["midi_path"]
                elif "midi_file" in result:
                    midi_path = result["midi_file"]
                elif "output_path" in result:
                    midi_path = result["output_path"]

                if midi_path:
                    midi_file = Path(midi_path)
                    if midi_file.exists():
                        print(f"📁 MIDI File: {midi_file.absolute()}")
                        print(f"   Size: {midi_file.stat().st_size / 1024:.2f} KB")
                        print("   Exists: ✅")
                        print()
                        print("📋 MIDI file ready for review:")
                        print(f"   {midi_file.absolute()}")
                        print()
                    else:
                        print(f"⚠️  MIDI path provided but file doesn't exist: {midi_path}")
                else:
                    # Check if MIDI was saved to the output path we specified
                    expected_midi = Path("output/review/test_song_review.mid")
                    if expected_midi.exists():
                        print(f"📁 MIDI File: {expected_midi.absolute()}")
                        print(f"   Size: {expected_midi.stat().st_size / 1024:.2f} KB")
                        print("   Exists: ✅")
                        print()
                        print("📋 MIDI file ready for review:")
                        print(f"   {expected_midi.absolute()}")
                        print()
                    else:
                        print("⚠️  No MIDI file found")
                        print("   (Generation may have completed but MIDI not saved)")

                # Save full result as JSON
                result_file = Path("output/review/generation_result.json")
                result_file.parent.mkdir(parents=True, exist_ok=True)
                with open(result_file, "w") as f:
                    json.dump(
                        {
                            "emotional_intent": emotion_text,
                            "parameters": params,
                            "generation_result": result,
                        },
                        f,
                        indent=2,
                        default=str,
                    )

                print(f"💾 Full result saved to: {result_file}")

            else:
                print(f"Result type: {type(result)}")
                print(f"Result: {result}")

        else:
            print("❌ Generation returned no result")

        print()
        print("=" * 80)

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure music_brain is installed: pip install -e .")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    generate_midi_for_review()
