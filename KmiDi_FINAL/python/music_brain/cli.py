#!/usr/bin/env python3
"""
DAiW-Music-Brain Command Line Interface (CLI)

CLI for emotion-driven music generation, analysis, and interactive teaching.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Import core music_brain modules
from music_brain.structure.comprehensive_engine import TherapySession
from music_brain.data.emotional_mapping import EMOTIONAL_PRESETS
from music_brain.session.intent_schema import RuleBreakingCategory # Assuming this exists

# --- Command Handlers ---

def cmd_extract(args):
    print(f"Extracting groove from MIDI file: {args.midi_file}")
    # Placeholder for actual groove extraction logic
    output_path = args.output or Path(args.midi_file).with_suffix("_groove.json")
    output_path.write_text(json.dumps({"swing_factor": 0.6, "tempo_bpm": 120}))
    print(f"Groove extracted and saved to {output_path}")
    return 0

def cmd_apply(args):
    print(f"Applying genre '{args.genre}' to MIDI file: {args.midi_file}")
    # Placeholder for actual genre application logic
    output_path = args.output or Path(args.midi_file).with_name(f"{Path(args.midi_file).stem}_{args.genre}.mid")
    output_path.write_bytes(b"midi_with_genre_applied")
    print(f"Genre applied and saved to {output_path}")
    return 0

def cmd_analyze(args):
    print(f"Analyzing MIDI file: {args.midi_file}")
    # Placeholder for actual analysis logic
    result = {"chords": ["Cmaj", "Gmaj", "Am", "Fmaj"], "scales": ["C Major"]}
    print(json.dumps(result, indent=2))
    return 0

def cmd_diagnose(args):
    print(f"Diagnosing harmonic issues for progression: {args.progression}")
    # Placeholder for actual diagnosis logic
    print(f"No obvious issues for {args.progression}. It's a common progression.")
    return 0

def cmd_intent(args):
    if args.subcommand == 'new':
        print(f"Creating new intent template with title: {args.title}")
        # Placeholder for intent creation
        template = {"title": args.title, "core_wound": "", "emotional_intent": "", "technical": {}}
        output_path = Path(f"{args.title.replace(' ', '_').lower()}_intent.json")
        output_path.write_text(json.dumps(template, indent=2))
        print(f"Intent template created at {output_path}")
    elif args.subcommand == 'suggest':
        print(f"Suggesting rule-breaks for topic: {args.topic}")
        # Placeholder for suggestion logic
        if args.topic == "grief":
            print("Suggested rule-breaks: HARMONY_AvoidTonicResolution, ARRANGEMENT_BuriedVocals")
        else:
            print("No specific suggestions for this topic.")
    else:
        print("Unknown intent subcommand.")
    return 0

def cmd_teach(args):
    print(f"Starting interactive teaching mode for topic: {args.topic}")
    # Placeholder for interactive teaching logic
    print("Interactive session started. (Not fully implemented in this stub)")
    return 0

def cmd_generate(args):
    print(f"Generating music from emotional intent: '{args.emotion_text}'")
    try:
        session = TherapySession()
        affect = session.process_core_input(args.emotion_text)
        session.set_scales(motivation=7, chaos=0.5) # Default values
        plan = session.generate_plan()

        output = {
            "status": "success",
            "result": {
                "affect": {
                    "primary": affect,
                    "secondary": session.state.affect_result.secondary if session.state.affect_result else None,
                    "intensity": session.state.affect_result.intensity if session.state.affect_result else 0.0,
                },
                "plan": {
                    "root_note": plan.root_note,
                    "mode": plan.mode,
                    "tempo_bpm": plan.tempo_bpm,
                    "length_bars": plan.length_bars,
                    "chord_symbols": plan.chord_symbols,
                    "complexity": plan.complexity,
                },
            }
        }
        print(json.dumps(output, indent=2))
    except Exception as e:
        print(f"Error generating music: {e}", file=sys.stderr)
        return 1
    return 0

def main():
    parser = argparse.ArgumentParser(
        prog="daiw",
        description="DAiW-Music-Brain Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  daiw extract drums.mid            # Extract groove from MIDI
  daiw apply --genre funk track.mid # Apply genre groove template
  daiw analyze --chords song.mid    # Analyze chord progression
  daiw diagnose "F-C-Am-Dm"         # Diagnose harmonic issues
  daiw intent new --title "My Song" # Create intent template
  daiw intent suggest grief         # Suggest rules to break
  daiw teach rulebreaking           # Interactive teaching mode
  daiw generate "I feel peaceful"   # Generate music from emotion
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract groove from MIDI")
    extract_parser.add_argument("midi_file", type=str, help="Path to MIDI file")
    extract_parser.add_argument("--output", type=str, help="Output JSON file for groove data")
    extract_parser.set_defaults(func=cmd_extract)

    # Apply command
    apply_parser = subparsers.add_parser("apply", help="Apply genre groove template to MIDI")
    apply_parser.add_argument("midi_file", type=str, help="Path to input MIDI file")
    apply_parser.add_argument("--genre", type=str, required=True, help="Genre to apply")
    apply_parser.add_argument("--output", type=str, help="Output MIDI file")
    apply_parser.set_defaults(func=cmd_apply)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze MIDI file for chords/progressions")
    analyze_parser.add_argument("midi_file", type=str, help="Path to MIDI file")
    analyze_parser.add_argument("--chords", action="store_true", help="Analyze chord progression")
    analyze_parser.set_defaults(func=cmd_analyze)

    # Diagnose command
    diagnose_parser = subparsers.add_parser("diagnose", help="Diagnose harmonic issues in a progression")
    diagnose_parser.add_argument("progression", type=str, help="Chord progression (e.g., C-G-Am-F)")
    diagnose_parser.set_defaults(func=cmd_diagnose)

    # Intent command (new/suggest)
    intent_parser = subparsers.add_parser("intent", help="Manage emotional intents")
    intent_subparsers = intent_parser.add_subparsers(dest="subcommand")

    new_intent_parser = intent_subparsers.add_parser("new", help="Create a new intent template")
    new_intent_parser.add_argument("--title", type=str, required=True, help="Title for the new intent")
    new_intent_parser.set_defaults(func=cmd_intent)

    suggest_intent_parser = intent_subparsers.add_parser("suggest", help="Suggest rule-breaks for a topic")
    suggest_intent_parser.add_argument("topic", type=str, help="Topic for rule-break suggestions (e.g., grief)")
    suggest_intent_parser.set_defaults(func=cmd_intent)

    # Teach command
    teach_parser = subparsers.add_parser("teach", help="Start interactive teaching mode")
    teach_parser.add_argument("topic", type=str, help="Teaching topic (e.g., rulebreaking, voice_leading)")
    teach_parser.set_defaults(func=cmd_teach)

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate music from emotional intent")
    generate_parser.add_argument("emotion_text", type=str, help="Emotional intent as text")
    generate_parser.set_defaults(func=cmd_generate)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    sys.exit(main())
