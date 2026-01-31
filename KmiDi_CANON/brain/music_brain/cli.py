"""
DAiW CLI - Command line interface for Music Brain toolkit

Usage:
    daiw extract <midi_file>                    Extract groove from MIDI
    daiw apply --genre <genre> <midi_file>      Apply groove template
    daiw humanize <midi_file> [options]         Apply drum humanization (Drunken Drummer)
    daiw analyze --chords <midi_file>           Analyze chord progression
    daiw diagnose <progression>                 Diagnose harmonic issues
    daiw reharm <progression> [--style <style>] Generate reharmonizations
    daiw teach <topic>                          Interactive teaching mode

    daiw intent new [--title <title>]           Create new intent template
    daiw intent process <file>                  Generate elements from intent
    daiw intent suggest <emotion>               Suggest rules to break
    daiw intent list                            List all rule-breaking options
    daiw intent validate <file>                 Validate intent file

Humanization Styles:
    tight   - Minimal drift, confident (complexity=0.1, vulnerability=0.2)
    natural - Human feel, balanced (complexity=0.4, vulnerability=0.5)
    loose   - Relaxed, laid back (complexity=0.6, vulnerability=0.6)
    drunk   - Maximum chaos, fragile (complexity=0.9, vulnerability=0.8)
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports to speed up CLI startup
def get_groove_module():
    from music_brain.groove import extract_groove, apply_groove
    return extract_groove, apply_groove


def get_humanize_module():
    from music_brain.groove import (
        humanize_midi_file, GrooveSettings, settings_from_intent, quick_humanize,
        list_presets, settings_from_preset, get_preset
    )
    return (humanize_midi_file, GrooveSettings, settings_from_intent, quick_humanize,
            list_presets, settings_from_preset, get_preset)

def get_structure_module():
    from music_brain.structure import analyze_chords, detect_sections
    return analyze_chords, detect_sections

def get_session_module():
    from music_brain.session import RuleBreakingTeacher
    return RuleBreakingTeacher


def get_intent_module():
    from music_brain.session.intent_schema import (
        CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints, 
        SystemDirective, suggest_rule_break, validate_intent, list_all_rules
    )
    from music_brain.session.intent_processor import IntentProcessor, process_intent
    return (CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints,
            SystemDirective, suggest_rule_break, validate_intent, list_all_rules,
            IntentProcessor, process_intent)


def cmd_extract(args):
    """Extract groove from MIDI file."""
    extract_groove, _ = get_groove_module()
    
    midi_path = Path(args.midi_file)
    if not midi_path.exists():
        logger.error("Error: File not found: %s", midi_path)
        return 1
    
    logger.info(f"Extracting groove from: {midi_path}")
    groove = extract_groove(str(midi_path))
    
    output_path = midi_path.stem + "_groove.json"
    if args.output:
        output_path = args.output
    
    with open(output_path, 'w') as f:
        json.dump(groove.to_dict(), f, indent=2)
    
    logger.info(f"Groove saved to: {output_path}")
    logger.info(f"  Timing deviation: {groove.timing_stats['mean_deviation_ms']:.1f}ms avg")
    logger.info(f"  Velocity range: {groove.velocity_stats['min']}-{groove.velocity_stats['max']}")
    logger.info(f"  Swing factor: {groove.swing_factor:.2f}")
    return 0


def cmd_apply(args):
    """Apply groove template to MIDI file."""
    _, apply_groove = get_groove_module()

    midi_path = Path(args.midi_file)
    if not midi_path.exists():
        logger.error("Error: File not found: %s", midi_path)
        return 1

    logger.info(f"Applying {args.genre} groove to: {midi_path}")

    output_path = args.output or f"{midi_path.stem}_grooved.mid"
    apply_groove(str(midi_path), genre=args.genre, output=output_path, intensity=args.intensity)

    logger.info(f"Output saved to: {output_path}")
    return 0


def cmd_humanize(args):
    """Apply drum humanization to MIDI file."""
    (humanize_midi_file, GrooveSettings, settings_from_intent, quick_humanize,
     list_presets, settings_from_preset, get_preset) = get_humanize_module()

    # Handle list-presets subcommand
    if hasattr(args, 'list_presets') and args.list_presets:
        presets = list_presets()
        logger.info("\n=== Available Humanization Presets ===\n")
        for preset_name in sorted(presets):
            preset_data = get_preset(preset_name)
            desc = preset_data.get("description", "No description")
            groove = preset_data.get("groove_settings", {})
            c = groove.get("complexity", 0.5)
            v = groove.get("vulnerability", 0.5)
            logger.info(f"  {preset_name}")
            logger.info(f"    {desc}")
            logger.info(f"    complexity={c:.2f}, vulnerability={v:.2f}")
            logger.info()
        return 0

    midi_path = Path(args.midi_file)
    if not midi_path.exists():
        logger.error("Error: File not found: %s", midi_path)
        return 1

    # Determine humanization approach
    if args.preset:
        # Load from preset
        try:
            settings = settings_from_preset(args.preset)
            preset_data = get_preset(args.preset)
            complexity = settings.complexity
            vulnerability = settings.vulnerability
            logger.info(f"Humanizing drums with '{args.preset}' preset: {midi_path}")
            logger.info(f"  ({preset_data.get('description', '')})")
        except ValueError as e:
            logger.error("Error: %s", e)
            return 1
    elif args.style:
        # Quick style preset
        logger.info(f"Humanizing drums with '{args.style}' style: {midi_path}")
        complexity_map = {
            "tight": (0.1, 0.2),
            "natural": (0.4, 0.5),
            "loose": (0.6, 0.6),
            "drunk": (0.9, 0.8),
        }
        complexity, vulnerability = complexity_map.get(args.style, (0.4, 0.5))
        settings = GrooveSettings(complexity=complexity, vulnerability=vulnerability)
    else:
        # Manual complexity/vulnerability
        complexity = args.complexity
        vulnerability = args.vulnerability
        logger.info(f"Humanizing drums (complexity={complexity:.2f}, vulnerability={vulnerability:.2f}): {midi_path}")
        settings = GrooveSettings(complexity=complexity, vulnerability=vulnerability)

    # Apply optional overrides
    if args.no_ghost_notes:
        settings.enable_ghost_notes = False

    output_path = args.output or f"{midi_path.stem}_humanized.mid"

    result_path = humanize_midi_file(
        input_path=str(midi_path),
        output_path=output_path,
        complexity=complexity,
        vulnerability=vulnerability,
        drum_channel=args.channel,
        settings=settings,
        seed=args.seed,
    )

    logger.info(f"\n=== Drum Humanization Applied ===")
    logger.info(f"  Complexity:    {complexity:.2f} (timing chaos)")
    logger.info(f"  Vulnerability: {vulnerability:.2f} (dynamic fragility)")
    logger.info(f"  Ghost notes:   {'enabled' if settings.enable_ghost_notes else 'disabled'}")
    logger.info(f"  Output:        {result_path}")

    return 0


def cmd_analyze(args):
    """Analyze chord progression in MIDI file."""
    analyze_chords, detect_sections = get_structure_module()
    
    midi_path = Path(args.midi_file)
    if not midi_path.exists():
        logger.error("Error: File not found: %s", midi_path)
        return 1
    
    if args.chords:
        logger.info(f"Analyzing chords in: {midi_path}")
        progression = analyze_chords(str(midi_path))
        
        logger.info("\n=== Chord Analysis ===")
        logger.info(f"Key: {progression.key}")
        logger.info(f"Progression: {' - '.join(progression.chords)}")
        logger.info(f"Roman numerals: {' - '.join(progression.roman_numerals)}")
        
        if progression.borrowed_chords:
            logger.info(f"\nBorrowed chords detected:")
            for chord, source in progression.borrowed_chords.items():
                logger.info(f"  {chord} ← borrowed from {source}")
    
    if args.sections:
        logger.info(f"\nDetecting sections in: {midi_path}")
        sections = detect_sections(str(midi_path))
        
        logger.info("\n=== Section Analysis ===")
        for section in sections:
            logger.info(f"  {section.name}: bars {section.start_bar}-{section.end_bar} (energy: {section.energy:.2f})")
    
    return 0


def cmd_diagnose(args):
    """Diagnose issues in a chord progression string."""
    from music_brain.structure.progression import diagnose_progression
    
    progression = args.progression
    logger.info(f"Diagnosing: {progression}")
    
    diagnosis = diagnose_progression(progression)
    
    logger.info("\n=== Harmonic Diagnosis ===")
    logger.info(f"Key estimate: {diagnosis['key']}")
    logger.info(f"Mode: {diagnosis['mode']}")
    
    if diagnosis['issues']:
        logger.info("\nPotential issues:")
        for issue in diagnosis['issues']:
            logger.info(f"  ⚠ {issue}")
    else:
        logger.info("\n✓ No obvious issues detected")
    
    if diagnosis['suggestions']:
        logger.info("\nSuggestions:")
        for suggestion in diagnosis['suggestions']:
            logger.info(f"  → {suggestion}")
    
    return 0


def cmd_reharm(args):
    """Generate reharmonization suggestions."""
    from music_brain.structure.progression import generate_reharmonizations
    
    progression = args.progression
    style = args.style or "jazz"
    
    logger.info(f"Reharmonizing: {progression}")
    logger.info(f"Style: {style}")
    
    suggestions = generate_reharmonizations(progression, style=style, count=args.count)
    
    logger.info("\n=== Reharmonization Suggestions ===")
    for i, suggestion in enumerate(suggestions, 1):
        logger.info(f"\n{i}. {' - '.join(suggestion['chords'])}")
        logger.info(f"   Technique: {suggestion['technique']}")
        logger.info(f"   Mood shift: {suggestion['mood']}")
    
    return 0


def cmd_teach(args):
    """Interactive teaching mode."""
    RuleBreakingTeacher = get_session_module()
    
    topic = args.topic.lower().replace("-", "_").replace(" ", "_")
    
    valid_topics = [
        "rulebreaking", "rule_breaking", "borrowed", "borrowed_chords",
        "modal_mixture", "substitutions", "rhythm", "production"
    ]
    
    if topic not in valid_topics:
        logger.info(f"Unknown topic: {args.topic}")
        logger.info(f"Available topics: {', '.join(valid_topics)}")
        return 1
    
    teacher = RuleBreakingTeacher()
    
    if args.quick:
        # Quick single lesson
        teacher.quick_lesson(topic)
    else:
        # Interactive mode
        teacher.interactive_session(topic)
    
    return 0


def cmd_intent(args):
    """Handle intent-based song generation."""
    (CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints,
     SystemDirective, suggest_rule_break, validate_intent, list_all_rules,
     IntentProcessor, process_intent) = get_intent_module()
    
    if args.subcommand == 'new':
        # Create new intent from template
        logger.info("Creating new song intent...")
        
        intent = CompleteSongIntent(
            title=args.title or "Untitled Song",
            song_root=SongRoot(
                core_event="[What happened?]",
                core_resistance="[What holds you back?]",
                core_longing="[What do you want to feel?]",
                core_stakes="Personal",
                core_transformation="[How should you feel at the end?]",
            ),
            song_intent=SongIntent(
                mood_primary="[Primary emotion]",
                mood_secondary_tension=0.5,
                imagery_texture="[Visual/tactile quality]",
                vulnerability_scale="Medium",
                narrative_arc="Climb-to-Climax",
            ),
            technical_constraints=TechnicalConstraints(
                technical_genre="[Genre]",
                technical_tempo_range=(80, 120),
                technical_key="F",
                technical_mode="major",
                technical_groove_feel="Organic/Breathing",
                technical_rule_to_break="",
                rule_breaking_justification="",
            ),
            system_directive=SystemDirective(
                output_target="Chord progression",
                output_feedback_loop="Harmony",
            ),
        )
        
        output = args.output or "song_intent.json"
        intent.save(output)
        logger.info(f"Template saved to: {output}")
        logger.info("\nEdit the file to fill in your intent, then run:")
        logger.info(f"  daiw intent process {output}")
    
    elif args.subcommand == 'process':
        # Process intent to generate elements
        if not args.file:
            logger.info("Error: Please specify an intent file")
            return 1
        
        intent_path = Path(args.file)
        if not intent_path.exists():
            logger.info(f"Error: File not found: {intent_path}")
            return 1
        
        logger.info(f"Processing intent: {intent_path}")
        intent = CompleteSongIntent.load(str(intent_path))
        
        # Validate first
        issues = validate_intent(intent)
        if issues and not args.force:
            logger.info("\n⚠️  Intent validation issues:")
            for issue in issues:
                logger.info(f"  - {issue}")
            logger.info("\nFix issues or use --force to proceed anyway")
            return 1
        
        # Process
        result = process_intent(intent)
        
        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("🎵 GENERATED ELEMENTS")
        logger.info("=" * 60)
        
        # Harmony
        harmony = result['harmony']
        logger.info(f"\n📌 HARMONY ({harmony.rule_broken})")
        logger.info(f"   Progression: {' - '.join(harmony.chords)}")
        logger.info(f"   Roman: {' - '.join(harmony.roman_numerals)}")
        logger.info(f"   Effect: {harmony.rule_effect}")
        
        # Groove
        groove = result['groove']
        logger.info(f"\n📌 GROOVE ({groove.rule_broken})")
        logger.info(f"   Pattern: {groove.pattern_name}")
        logger.info(f"   Tempo: {groove.tempo_bpm} BPM")
        logger.info(f"   Effect: {groove.rule_effect}")
        
        # Arrangement
        arr = result['arrangement']
        logger.info(f"\n📌 ARRANGEMENT ({arr.rule_broken})")
        for section in arr.sections:
            logger.info(f"   {section['name']}: {section['bars']} bars @ {section['energy']:.0%} energy")
        
        # Production
        prod = result['production']
        logger.info(f"\n📌 PRODUCTION ({prod.rule_broken})")
        logger.info(f"   Vocal: {prod.vocal_treatment}")
        for note in prod.eq_notes[:2]:
            logger.info(f"   EQ: {note}")
        
        logger.info("\n" + "=" * 60)
        
        # Save output if requested
        if args.output:
            import json
            output_data = {
                "intent_summary": result['intent_summary'],
                "harmony": {
                    "chords": harmony.chords,
                    "roman_numerals": harmony.roman_numerals,
                    "rule_broken": harmony.rule_broken,
                    "effect": harmony.rule_effect,
                },
                "groove": {
                    "pattern": groove.pattern_name,
                    "tempo": groove.tempo_bpm,
                    "swing": groove.swing_factor,
                },
                "arrangement": {
                    "sections": arr.sections,
                    "dynamic_arc": arr.dynamic_arc,
                },
                "production": {
                    "vocal_treatment": prod.vocal_treatment,
                    "eq_notes": prod.eq_notes,
                    "dynamics_notes": prod.dynamics_notes,
                },
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"\nOutput saved to: {args.output}")
    
    elif args.subcommand == 'suggest':
        # Suggest rules to break based on emotion
        emotion = args.emotion
        suggestions = suggest_rule_break(emotion)
        
        logger.info(f"\n🎯 Suggested rules to break for '{emotion}':\n")
        
        if not suggestions:
            logger.info(f"  No specific suggestions for '{emotion}'")
            logger.info("  Try: grief, anger, nostalgia, defiance, dissociation")
        else:
            for i, sug in enumerate(suggestions, 1):
                logger.info(f"{i}. {sug['rule']}")
                logger.info(f"   What: {sug['description']}")
                logger.info(f"   Effect: {sug['effect']}")
                logger.info(f"   Use when: {sug['use_when']}")
                logger.info()
    
    elif args.subcommand == 'list':
        # List all available rules
        rules = list_all_rules()
        
        logger.info("\n📋 Available Rule-Breaking Options:\n")
        for category, rule_list in rules.items():
            logger.info(f"  {category}:")
            for rule in rule_list:
                logger.info(f"    - {rule}")
            logger.info()
    
    elif args.subcommand == 'validate':
        # Validate an intent file
        if not args.file:
            logger.info("Error: Please specify an intent file")
            return 1
        
        intent_path = Path(args.file)
        if not intent_path.exists():
            logger.info(f"Error: File not found: {intent_path}")
            return 1
        
        intent = CompleteSongIntent.load(str(intent_path))
        issues = validate_intent(intent)
        
        if issues:
            logger.info("\n⚠️  Validation issues found:")
            for issue in issues:
                logger.info(f"  - {issue}")
            return 1
        else:
            logger.info("✅ Intent is valid!")
            return 0
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog='daiw',
        description='DAiW - Digital Audio intelligent Workstation CLI'
    )
    parser.add_argument('--version', action='version', version='%(prog)s 0.2.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract groove from MIDI')
    extract_parser.add_argument('midi_file', help='MIDI file to extract from')
    extract_parser.add_argument('-o', '--output', help='Output JSON file')
    
    # Apply command
    apply_parser = subparsers.add_parser('apply', help='Apply groove template')
    apply_parser.add_argument('midi_file', help='MIDI file to process')
    apply_parser.add_argument('-g', '--genre', default='funk',
                              choices=['funk', 'jazz', 'rock', 'hiphop', 'edm', 'latin'],
                              help='Genre groove template')
    apply_parser.add_argument('-o', '--output', help='Output MIDI file')
    apply_parser.add_argument('-i', '--intensity', type=float, default=0.5,
                              help='Groove intensity 0.0-1.0')

    # Humanize command (Drunken Drummer)
    humanize_parser = subparsers.add_parser('humanize', help='Apply drum humanization (Drunken Drummer)')
    humanize_parser.add_argument('midi_file', nargs='?', help='MIDI file to humanize')
    humanize_parser.add_argument('-o', '--output', help='Output MIDI file')
    humanize_parser.add_argument('-p', '--preset',
                                 help='Emotional preset (use --list-presets to see options)')
    humanize_parser.add_argument('-s', '--style',
                                 choices=['tight', 'natural', 'loose', 'drunk'],
                                 help='Quick style preset')
    humanize_parser.add_argument('-c', '--complexity', type=float, default=0.5,
                                 help='Timing chaos 0.0-1.0 (ignored if --style/--preset used)')
    humanize_parser.add_argument('-v', '--vulnerability', type=float, default=0.5,
                                 help='Dynamic fragility 0.0-1.0 (ignored if --style/--preset used)')
    humanize_parser.add_argument('--channel', type=int, default=9,
                                 help='MIDI drum channel (default: 9, i.e. channel 10)')
    humanize_parser.add_argument('--no-ghost-notes', action='store_true',
                                 help='Disable ghost note generation')
    humanize_parser.add_argument('--seed', type=int,
                                 help='Random seed for reproducibility')
    humanize_parser.add_argument('--list-presets', action='store_true',
                                 help='List available emotional presets')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze MIDI file')
    analyze_parser.add_argument('midi_file', help='MIDI file to analyze')
    analyze_parser.add_argument('-c', '--chords', action='store_true', help='Analyze chords')
    analyze_parser.add_argument('-s', '--sections', action='store_true', help='Detect sections')
    
    # Diagnose command
    diagnose_parser = subparsers.add_parser('diagnose', help='Diagnose chord progression')
    diagnose_parser.add_argument('progression', help='Chord progression (e.g., "F-C-Am-Dm")')
    
    # Reharm command
    reharm_parser = subparsers.add_parser('reharm', help='Generate reharmonizations')
    reharm_parser.add_argument('progression', help='Chord progression to reharmonize')
    reharm_parser.add_argument('-s', '--style', default='jazz',
                               choices=['jazz', 'pop', 'rnb', 'classical', 'experimental'],
                               help='Reharmonization style')
    reharm_parser.add_argument('-n', '--count', type=int, default=3,
                               help='Number of suggestions')
    
    # Teach command
    teach_parser = subparsers.add_parser('teach', help='Interactive teaching mode')
    teach_parser.add_argument('topic', help='Topic to learn (rulebreaking, borrowed, etc.)')
    teach_parser.add_argument('-q', '--quick', action='store_true', help='Quick single lesson')
    
    # Intent command with subcommands
    intent_parser = subparsers.add_parser('intent', help='Intent-based song generation')
    intent_subparsers = intent_parser.add_subparsers(dest='subcommand', help='Intent commands')
    
    # intent new
    intent_new = intent_subparsers.add_parser('new', help='Create new intent template')
    intent_new.add_argument('-t', '--title', help='Song title')
    intent_new.add_argument('-o', '--output', help='Output file (default: song_intent.json)')
    
    # intent process
    intent_process = intent_subparsers.add_parser('process', help='Process intent to generate elements')
    intent_process.add_argument('file', help='Intent JSON file')
    intent_process.add_argument('-o', '--output', help='Save output to JSON')
    intent_process.add_argument('-f', '--force', action='store_true', help='Proceed despite validation issues')
    
    # intent suggest
    intent_suggest = intent_subparsers.add_parser('suggest', help='Suggest rules to break')
    intent_suggest.add_argument('emotion', help='Target emotion (grief, anger, nostalgia, etc.)')
    
    # intent list
    intent_subparsers.add_parser('list', help='List all rule-breaking options')
    
    # intent validate
    intent_validate = intent_subparsers.add_parser('validate', help='Validate intent file')
    intent_validate.add_argument('file', help='Intent JSON file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    commands = {
        'extract': cmd_extract,
        'apply': cmd_apply,
        'humanize': cmd_humanize,
        'analyze': cmd_analyze,
        'diagnose': cmd_diagnose,
        'reharm': cmd_reharm,
        'teach': cmd_teach,
        'intent': cmd_intent,
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
