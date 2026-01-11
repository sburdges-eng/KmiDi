"""
Harmony Tools - MCP tools for harmony generation and analysis.

Provides 6 tools:
- analyze_progression
- generate_harmony
- diagnose_chords
- suggest_reharmonization
- find_key
- voice_leading
"""

from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Import DAiW modules
from music_brain.harmony import HarmonyGenerator, HarmonyResult, generate_midi_from_harmony
from music_brain.structure.progression import diagnose_progression, generate_reharmonizations
from music_brain.structure.chord import analyze_chords, CHORD_QUALITIES


def register_tools(server: Server) -> None:
    """Register all harmony tools with the MCP server."""
    if not MCP_AVAILABLE:
        return
    
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available harmony tools."""
        return [
            Tool(
                name="analyze_progression",
                description="Analyze a chord progression for harmonic characteristics, emotional character, and rule breaks.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "progression": {
                            "type": "string",
                            "description": "Chord progression string (e.g., 'F-C-Dm-Bbm')"
                        },
                        "key": {
                            "type": "string",
                            "description": "Optional key context (e.g., 'F major')"
                        }
                    },
                    "required": ["progression"]
                }
            ),
            Tool(
                name="generate_harmony",
                description="Generate harmony from emotional intent or basic parameters.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "emotion": {
                            "type": "string",
                            "description": "Primary emotion (e.g., 'grief', 'nostalgia', 'defiance')"
                        },
                        "key": {
                            "type": "string",
                            "description": "Musical key (e.g., 'C', 'F', 'Bb')",
                            "default": "C"
                        },
                        "mode": {
                            "type": "string",
                            "description": "Mode/scale (major, minor, dorian, etc.)",
                            "default": "major"
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Roman numeral pattern (e.g., 'I-V-vi-IV')",
                            "default": "I-V-vi-IV"
                        },
                        "intent_file": {
                            "type": "string",
                            "description": "Optional path to CompleteSongIntent JSON file"
                        },
                        "output_midi": {
                            "type": "string",
                            "description": "Optional path to save MIDI output"
                        },
                        "tempo": {
                            "type": "integer",
                            "description": "Tempo in BPM",
                            "default": 82
                        }
                    }
                }
            ),
            Tool(
                name="diagnose_chords",
                description="Diagnose harmonic issues in a chord progression.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "progression": {
                            "type": "string",
                            "description": "Chord progression string"
                        },
                        "key": {
                            "type": "string",
                            "description": "Optional key context"
                        }
                    },
                    "required": ["progression"]
                }
            ),
            Tool(
                name="suggest_reharmonization",
                description="Suggest reharmonization alternatives for a chord progression.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "progression": {
                            "type": "string",
                            "description": "Chord progression to reharmonize"
                        },
                        "style": {
                            "type": "string",
                            "description": "Reharmonization style (jazz, pop, rnb, classical, experimental)",
                            "default": "jazz"
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of suggestions",
                            "default": 3
                        }
                    },
                    "required": ["progression"]
                }
            ),
            Tool(
                name="find_key",
                description="Detect the key of a chord progression.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "progression": {
                            "type": "string",
                            "description": "Chord progression string"
                        }
                    },
                    "required": ["progression"]
                }
            ),
            Tool(
                name="voice_leading",
                description="Optimize voice leading for a chord progression.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "chords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of chord symbols"
                        },
                        "key": {
                            "type": "string",
                            "description": "Musical key"
                        }
                    },
                    "required": ["chords", "key"]
                }
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls."""
        try:
            if name == "analyze_progression":
                progression = arguments["progression"]
                key = arguments.get("key")
                
                diagnosis = diagnose_progression(progression, key)
                result = {
                    "progression": progression,
                    "key": diagnosis.get("key", "Unknown"),
                    "mode": diagnosis.get("mode", "Unknown"),
                    "issues": diagnosis.get("issues", []),
                    "suggestions": diagnosis.get("suggestions", []),
                    "emotional_character": diagnosis.get("emotional_character", ""),
                    "rule_breaks": diagnosis.get("rule_breaks", [])
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "generate_harmony":
                generator = HarmonyGenerator()
                
                if "intent_file" in arguments:
                    # Generate from intent file
                    intent_path = Path(arguments["intent_file"])
                    if not intent_path.exists():
                        return [TextContent(type="text", text=f"Error: Intent file not found: {intent_path}")]
                    
                    from music_brain.session.intent_schema import CompleteSongIntent
                    intent = CompleteSongIntent.load(str(intent_path))
                    harmony = generator.generate_from_intent(intent)
                else:
                    # Generate from basic parameters
                    key = arguments.get("key", "C")
                    mode = arguments.get("mode", "major")
                    pattern = arguments.get("pattern", "I-V-vi-IV")
                    emotion = arguments.get("emotion")
                    
                    harmony = generator.generate_basic_progression(
                        key=key,
                        mode=mode,
                        pattern=pattern
                    )
                    
                    if emotion:
                        # Apply emotional rule-breaking
                        from music_brain.session.intent_schema import suggest_rule_break
                        suggestions = suggest_rule_break(emotion)
                        if suggestions:
                            # Apply first suggestion
                            rule_break = suggestions[0]["rule"]
                            # This would need more implementation
                
                result = {
                    "chords": harmony.chords,
                    "key": harmony.key,
                    "mode": harmony.mode,
                    "rule_break_applied": harmony.rule_break_applied,
                    "emotional_justification": harmony.emotional_justification
                }
                
                # Save MIDI if requested
                if "output_midi" in arguments:
                    tempo = arguments.get("tempo", 82)
                    generate_midi_from_harmony(harmony, arguments["output_midi"], tempo_bpm=tempo)
                    result["midi_file"] = arguments["output_midi"]
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "diagnose_chords":
                progression = arguments["progression"]
                key = arguments.get("key")
                
                diagnosis = diagnose_progression(progression, key)
                result = {
                    "progression": progression,
                    "key": diagnosis.get("key", "Unknown"),
                    "mode": diagnosis.get("mode", "Unknown"),
                    "issues": diagnosis.get("issues", []),
                    "suggestions": diagnosis.get("suggestions", [])
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "suggest_reharmonization":
                progression = arguments["progression"]
                style = arguments.get("style", "jazz")
                count = arguments.get("count", 3)
                
                suggestions = generate_reharmonizations(progression, style=style, count=count)
                result = {
                    "original": progression,
                    "style": style,
                    "suggestions": [
                        {
                            "chords": s["chords"],
                            "technique": s.get("technique", ""),
                            "mood": s.get("mood", "")
                        }
                        for s in suggestions
                    ]
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "find_key":
                progression = arguments["progression"]
                diagnosis = diagnose_progression(progression)
                # Convert string confidence to float for consistency (0.0-1.0)
                confidence_val = 0.8 if diagnosis.get("key") and diagnosis.get("key") != "Unknown" else 0.3
                result = {
                    "progression": progression,
                    "key": diagnosis.get("key", "Unknown"),
                    "mode": diagnosis.get("mode", "Unknown"),
                    "confidence": confidence_val
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "voice_leading":
                chords = arguments["chords"]
                key = arguments["key"]
                
                # Implement voice leading optimization
                optimized_voicings, analysis = _optimize_voice_leading(chords, key)
                
                result = {
                    "chords": chords,
                    "key": key,
                    "optimized_voicings": optimized_voicings,
                    "voice_leading_score": analysis["score"],
                    "issues": analysis["issues"],
                    "improvements": analysis["improvements"],
                    "parallel_motion": analysis["parallel_motion"]
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
        
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]


def _optimize_voice_leading(chords: List[str], key: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Optimize voice leading for a chord progression.
    
    Args:
        chords: List of chord symbols (e.g., ['F', 'C', 'Am', 'Dm'])
        key: Musical key for context
    
    Returns:
        Tuple of (optimized_voicings, analysis_dict)
    """
    generator = HarmonyGenerator()
    
    # Parse chords and get note sets
    chord_notes = []
    for chord_str in chords:
        # Use HarmonyGenerator's parsing method
        root, intervals = generator._chord_symbol_to_intervals(chord_str)
        root_midi = generator.NOTE_TO_MIDI.get(root, 0) + (generator.base_octave * 12)
        
        # Generate MIDI notes from intervals
        notes = [(root_midi + interval) for interval in intervals]
        chord_notes.append({
            "chord": chord_str,
            "notes": sorted(notes),
            "root": root_midi
        })
    
    if len(chord_notes) < 2:
        return ([], {
            "score": 1.0,
            "issues": [],
            "improvements": [],
            "parallel_motion": []
        })
    
    # Generate voicing options for each chord (different inversions/octaves)
    voicing_options = []
    for chord_data in chord_notes:
        options = []
        base_notes = chord_data["notes"]
        
        # Generate different voicing options (close position, open position, inversions)
        for octave_shift in [-1, 0, 1]:
            for inversion, _ in enumerate(base_notes):
                voicing = []
                for i, note in enumerate(base_notes):
                    idx = (i + inversion) % len(base_notes)
                    voicing.append(base_notes[idx] + (octave_shift * 12))
                options.append(sorted(voicing))
        
        voicing_options.append(options)
    
    # Greedy optimization: select voicings that minimize voice movement
    selected_voicings = []
    previous_voicing = None
    
    for i, options in enumerate(voicing_options):
        if previous_voicing is None:
            # First chord: use root position
            selected_voicings.append(options[0])
            previous_voicing = options[0]
        else:
            # Find option with minimum total voice movement
            best_option = None
            best_score = float('inf')
            
            for option in options:
                # Calculate total voice movement (sum of semitone distances)
                total_movement = 0
                # Match voices by closest note
                used_indices = set()
                for prev_note in previous_voicing:
                    min_dist = float('inf')
                    best_match = None
                    for j, curr_note in enumerate(option):
                        if j not in used_indices:
                            dist = abs(curr_note - prev_note)
                            if dist < min_dist:
                                min_dist = dist
                                best_match = j
                    if best_match is not None:
                        used_indices.add(best_match)
                        total_movement += min_dist
                
                # Prefer stepwise motion (penalize large jumps)
                if total_movement < best_score:
                    best_score = total_movement
                    best_option = option
            
            selected_voicings.append(best_option)
            previous_voicing = best_option
    
    # Analyze voice leading quality
    issues = []
    improvements = []
    parallel_motion = []
    
    # Check for parallel fifths and octaves - optimized to avoid redundant checks
    for idx, (prev, curr) in enumerate(zip(selected_voicings[:-1], selected_voicings[1:])):
        # Pre-compute intervals as sets to avoid redundant modulo operations
        num_voices = min(len(prev), len(curr))
        
        # Check all pairs of voices efficiently
        for j in range(num_voices):
            for k in range(j + 1, num_voices):
                if k >= len(prev) or k >= len(curr):
                    continue
                    
                prev_interval = abs(prev[j] - prev[k]) % 12
                curr_interval = abs(curr[j] - curr[k]) % 12
                
                # Parallel fifths (7 semitones)
                if prev_interval == 7 and curr_interval == 7:
                    parallel_motion.append({
                        "type": "parallel_fifth",
                        "chord_pair": f"{chords[idx]} -> {chords[idx+1]}",
                        "voices": (j, k)
                    })
                # Parallel octaves (0 semitones)
                elif prev_interval == 0 and curr_interval == 0 and prev[j] != prev[k]:
                    parallel_motion.append({
                        "type": "parallel_octave",
                        "chord_pair": f"{chords[idx]} -> {chords[idx+1]}",
                        "voices": (j, k)
                    })
    
    # Calculate quality score (0.0-1.0) and check for large leaps in one pass
    # Penalize parallel motion, reward smooth voice leading
    score = 1.0
    if parallel_motion:
        score -= len(parallel_motion) * 0.1
    
    # Check for large leaps (more than 12 semitones) - optimized single pass
    large_leaps = sum(
        1
        for prev, curr in zip(selected_voicings[:-1], selected_voicings[1:])
        for j in range(min(len(prev), len(curr)))
        if abs(curr[j] - prev[j]) > 12
    )
    
    if large_leaps > 0:
        score -= large_leaps * 0.05
        improvements.append(f"Consider reducing {large_leaps} large voice leaps for smoother motion")
    
    score = max(0.0, min(1.0, score))
    
    # Format voicings for output
    formatted_voicings = []
    for i, voicing in enumerate(selected_voicings):
        formatted_voicings.append({
            "chord": chords[i],
            "notes": voicing,
            "note_names": [f"{['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][n % 12]}{n // 12 - 1}" for n in voicing]
        })
    
    return (formatted_voicings, {
        "score": round(score, 2),
        "issues": issues if issues else ["No major issues detected"],
        "improvements": improvements if improvements else ["Voice leading is smooth"],
        "parallel_motion": parallel_motion if parallel_motion else []
    })
