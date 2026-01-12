#!/usr/bin/env python3
"""
Preference Bridge - Command-line interface for C++ to call Python preference model

This script allows C++ code to record preferences by calling this script as a subprocess.
Usage:
    python preference_bridge.py record_parameter_adjustment '{"parameter_name": "valence", "old_value": 0.5, "new_value": 0.7}'
"""

import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path to import music_brain
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from music_brain.learning.user_preferences import UserPreferenceModel


def record_parameter_adjustment(model, args):
    """Record a parameter adjustment."""
    model.record_parameter_adjustment(
        parameter_name=args['parameter_name'],
        old_value=args['old_value'],
        new_value=args['new_value'],
        context=args.get('context', {})
    )


def record_emotion_selection(model, args):
    """Record an emotion selection."""
    model.record_emotion_selection(
        emotion_name=args['emotion_name'],
        valence=args['valence'],
        arousal=args['arousal'],
        context=args.get('context', {})
    )


def record_midi_generation(model, args):
    """Record a MIDI generation event."""
    return model.record_midi_generation(
        generation_id=args['generation_id'],
        intent_text=args['intent_text'],
        parameters=args['parameters'],
        emotion=args.get('emotion'),
        rule_breaks=args.get('rule_breaks', [])
    )


def record_midi_feedback(model, args):
    """Record MIDI feedback."""
    model.record_midi_feedback(
        generation_id=args['generation_id'],
        accepted=args['accepted']
    )


def record_midi_modification(model, args):
    """Record a MIDI modification."""
    model.record_midi_modification(
        generation_id=args['generation_id'],
        parameter_name=args['parameter_name'],
        old_value=args['old_value'],
        new_value=args['new_value']
    )


def record_rule_break_modification(model, args):
    """Record a rule-break modification."""
    model.record_rule_break_modification(
        rule_break=args['rule_break'],
        action=args['action'],
        context=args.get('context', {})
    )


METHODS = {
    'record_parameter_adjustment': record_parameter_adjustment,
    'record_emotion_selection': record_emotion_selection,
    'record_midi_generation': record_midi_generation,
    'record_midi_feedback': record_midi_feedback,
    'record_midi_modification': record_midi_modification,
    'record_rule_break_modification': record_rule_break_modification,
}


def main():
    parser = argparse.ArgumentParser(description='Preference tracking bridge for C++')
    parser.add_argument('method', choices=list(METHODS.keys()), help='Method to call')
    parser.add_argument('args_json', help='JSON string with method arguments')
    parser.add_argument('--user-id', default='default', help='User ID')

    args = parser.parse_args()

    # Parse JSON arguments
    try:
        method_args = json.loads(args.args_json)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Create preference model
    model = UserPreferenceModel(user_id=args.user_id)

    # Call method
    try:
        method_func = METHODS[args.method]
        result = method_func(model, method_args)

        # Print result if any (for generation ID, etc.)
        if result:
            print(json.dumps({'result': result}))

        sys.exit(0)
    except Exception as e:
        print(f"Error calling {args.method}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
