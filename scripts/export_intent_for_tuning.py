#!/usr/bin/env python3
"""
Export KmiDi song intent data for fine-tuning a Gemini model in Google AI Studio.

Google AI Studio Tuning Format (JSONL):
{"contents": [{"role": "user", "parts": [{"text": "prompt"}]}, \
{"role": "model", "parts": [{"text": "target"}]}]}

Usage:
    python scripts/export_intent_for_tuning.py --input-dir data/intents \\
        --output data/gemini_tuning_data.jsonl
"""

import argparse
import json
from pathlib import Path


def format_as_gemini_jsonl(user_prompt, model_response):
    """Format a single example into Google AI Studio's JSONL tuning format."""
    return json.dumps(
        {
            "contents": [
                {"role": "user", "parts": [{"text": user_prompt}]},
                {"role": "model", "parts": [{"text": json.dumps(model_response, indent=2)}]},
            ]
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Export intent data for Google AI Studio")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/intents",
        help="Directory containing intent JSON files",
    )
    parser.add_argument(
        "--output", type=str, default="data/gemini_tuning_data.jsonl", help="Output JSONL file"
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []

    # If the input directory exists, process real data
    if input_path.exists():
        for file in input_path.glob("*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)

                # Assume core_desire is the user prompt
                prompt = data.get("core_desire", "Generate a song intent.")
                examples.append(format_as_gemini_jsonl(prompt, data))
            except Exception as e:
                print(f"Skipping {file}: {e}")
    else:
        print(
            f"Input directory {args.input_dir} not found. Generating synthetic examples for "
            f"demonstration."
        )

        # Generate a few synthetic examples based on the schema
        synthetic_examples = [
            {
                "prompt": "Create an intimate and powerful pop ballad with a focus on piano "
                "and emotional build-up.",
                "response": {
                    "core_desire": "intimate and powerful pop ballad",
                    "mood_primary": "intimate",
                    "genre": "pop",
                    "tempo": 72,
                    "key_mode": "C major",
                    "structure": [
                        {"name": "intro", "bars": 4},
                        {"name": "verse", "bars": 8},
                        {"name": "chorus", "bars": 8},
                        {"name": "build", "bars": 4},
                        {"name": "chorus", "bars": 8},
                        {"name": "outro", "bars": 4},
                    ],
                    "instruments": [
                        {"instrument": "piano", "techniques": ["legato", "arpeggio"]},
                        {"instrument": "cello", "techniques": ["sustained"]},
                        {"instrument": "vocals", "techniques": ["vibrato"]},
                    ],
                    "groove_feel": "Rubato/Fluid",
                    "narrative_arc": "Climb-to-Climax",
                },
            },
            {
                "prompt": "Write a driving techno track at 128 BPM in A minor with a minimal "
                "structure and aggressive synths.",
                "response": {
                    "core_desire": "driving techno track with aggressive synths",
                    "mood_primary": "aggressive",
                    "genre": "techno",
                    "tempo": 128,
                    "key_mode": "A minor",
                    "structure": [
                        {"name": "intro", "bars": 16},
                        {"name": "drop", "bars": 32},
                        {"name": "verse", "bars": 16},
                        {"name": "drop", "bars": 32},
                        {"name": "outro", "bars": 16},
                    ],
                    "instruments": [
                        {"instrument": "drum_machine", "techniques": ["sidechain"]},
                        {"instrument": "acid_synth", "techniques": ["filter_cutoff"]},
                        {"instrument": "sub_bass", "techniques": ["mono"]},
                    ],
                    "groove_feel": "Straight/Driving",
                    "narrative_arc": "Constant-Energy",
                },
            },
        ]

        for ex in synthetic_examples:
            examples.append(format_as_gemini_jsonl(ex["prompt"], ex["response"]))

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(ex + "\n")

    print(f"Exported {len(examples)} examples to {output_path}")
    print("\nNext steps:")
    print("1. Go to Google AI Studio (https://aistudio.google.com/)")
    print("2. Select 'New Tuned Model' from the sidebar.")
    print(f"3. Upload {output_path} as the training data.")
    print("4. Select a base model (e.g., Gemini 1.5 Flash).")
    print("5. Click 'Tune' to start the process.")


if __name__ == "__main__":
    main()
