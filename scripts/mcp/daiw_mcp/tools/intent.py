"""
Intent Tools - MCP tools for intent-based song generation.

Provides 4 tools:
- create_intent
- process_intent
- validate_intent
- suggest_rulebreaks
"""

from typing import Any, Dict, List
import json
from pathlib import Path

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Import DAiW modules
from music_brain.session.intent_schema import (
    CompleteSongIntent,
    SongRoot,
    SongIntent,
    TechnicalConstraints,
    SystemDirective,
    suggest_rule_break,
    validate_intent,
    list_all_rules
)
from music_brain.session.intent_processor import process_intent


def register_tools(server: Server) -> None:
    """Register all intent tools with the MCP server."""
    if not MCP_AVAILABLE:
        return
    
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available intent tools."""
        return [
            Tool(
                name="create_intent",
                description="Create a new song intent template with three-phase schema (Phase 0: Core Wound, Phase 1: Emotional Intent, Phase 2: Technical Constraints).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Song title"
                        },
                        "core_event": {
                            "type": "string",
                            "description": "Phase 0: What happened?"
                        },
                        "core_longing": {
                            "type": "string",
                            "description": "Phase 0: What do you want to feel?"
                        },
                        "mood_primary": {
                            "type": "string",
                            "description": "Phase 1: Primary emotion"
                        },
                        "key": {
                            "type": "string",
                            "description": "Phase 2: Musical key",
                            "default": "F"
                        },
                        "mode": {
                            "type": "string",
                            "description": "Phase 2: Mode (major, minor, etc.)",
                            "default": "major"
                        },
                        "output_file": {
                            "type": "string",
                            "description": "Path to save intent JSON file"
                        }
                    },
                    "required": ["title"]
                }
            ),
            Tool(
                name="process_intent",
                description="Process a CompleteSongIntent to generate musical elements (harmony, groove, arrangement, production).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "intent_file": {
                            "type": "string",
                            "description": "Path to CompleteSongIntent JSON file"
                        },
                        "output_midi": {
                            "type": "string",
                            "description": "Optional path to save generated MIDI"
                        }
                    },
                    "required": ["intent_file"]
                }
            ),
            Tool(
                name="validate_intent",
                description="Validate a CompleteSongIntent file against the schema.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "intent_file": {
                            "type": "string",
                            "description": "Path to CompleteSongIntent JSON file"
                        }
                    },
                    "required": ["intent_file"]
                }
            ),
            Tool(
                name="suggest_rulebreaks",
                description="Suggest intentional rule-breaks based on an emotion or mood.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "emotion": {
                            "type": "string",
                            "description": "Target emotion (e.g., 'grief', 'anger', 'nostalgia', 'defiance')"
                        }
                    },
                    "required": ["emotion"]
                }
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls."""
        try:
            if name == "create_intent":
                title = arguments["title"]
                output_file = arguments.get("output_file", "song_intent.json")
                
                intent = CompleteSongIntent(
                    title=title,
                    song_root=SongRoot(
                        core_event=arguments.get("core_event", "[What happened?]"),
                        core_resistance=arguments.get("core_resistance", "[What holds you back?]"),
                        core_longing=arguments.get("core_longing", "[What do you want to feel?]"),
                        core_stakes=arguments.get("core_stakes", "Personal"),
                        core_transformation=arguments.get("core_transformation", "[How should you feel at the end?]"),
                    ),
                    song_intent=SongIntent(
                        mood_primary=arguments.get("mood_primary", "[Primary emotion]"),
                        mood_secondary_tension=arguments.get("mood_secondary_tension", 0.5),
                        imagery_texture=arguments.get("imagery_texture", "[Visual/tactile quality]"),
                        vulnerability_scale=arguments.get("vulnerability_scale", "Medium"),
                        narrative_arc=arguments.get("narrative_arc", "Climb-to-Climax"),
                    ),
                    technical_constraints=TechnicalConstraints(
                        technical_genre=arguments.get("technical_genre", "[Genre]"),
                        technical_tempo_range=arguments.get("technical_tempo_range", (80, 120)),
                        technical_key=arguments.get("key", "F"),
                        technical_mode=arguments.get("mode", "major"),
                        technical_groove_feel=arguments.get("technical_groove_feel", "Organic/Breathing"),
                        technical_rule_to_break=arguments.get("technical_rule_to_break", ""),
                        rule_breaking_justification=arguments.get("rule_breaking_justification", ""),
                    ),
                    system_directive=SystemDirective(
                        output_target=arguments.get("output_target", "Chord progression"),
                        output_feedback_loop=arguments.get("output_feedback_loop", "Harmony"),
                    ),
                )
                
                intent.save(output_file)
                result = {
                    "status": "success",
                    "intent_file": output_file,
                    "title": title,
                    "message": "Intent template created. Edit the file to complete all phases."
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "process_intent":
                intent_file = arguments["intent_file"]
                if not Path(intent_file).exists():
                    return [TextContent(type="text", text=f"Error: Intent file not found: {intent_file}")]
                
                intent = CompleteSongIntent.load(intent_file)
                
                # Validate first
                issues = validate_intent(intent)
                if issues:
                    result = {
                        "status": "validation_errors",
                        "issues": issues,
                        "message": "Fix validation issues before processing"
                    }
                    return [TextContent(type="text", text=json.dumps(result, indent=2))]
                
                # Process intent
                result_dict = process_intent(intent)
                
                # Save MIDI if requested
                if "output_midi" in arguments:
                    from music_brain.harmony import generate_midi_from_harmony
                    harmony = result_dict["harmony"]
                    tempo = getattr(intent.technical_constraints, "technical_tempo_range", (82, 82))[0]
                    generate_midi_from_harmony(
                        harmony,
                        arguments["output_midi"],
                        tempo_bpm=tempo
                    )
                    result_dict["midi_file"] = arguments["output_midi"]
                
                result = {
                    "status": "success",
                    "intent_file": intent_file,
                    "generated_elements": {
                        "harmony": {
                            "chords": result_dict["harmony"].chords,
                            "rule_broken": result_dict["harmony"].rule_broken,
                            "effect": result_dict["harmony"].rule_effect
                        },
                        "groove": {
                            "pattern": result_dict["groove"].pattern_name,
                            "tempo": result_dict["groove"].tempo_bpm
                        }
                    }
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "validate_intent":
                intent_file = arguments["intent_file"]
                if not Path(intent_file).exists():
                    return [TextContent(type="text", text=f"Error: Intent file not found: {intent_file}")]
                
                intent = CompleteSongIntent.load(intent_file)
                issues = validate_intent(intent)
                
                result = {
                    "intent_file": intent_file,
                    "valid": len(issues) == 0,
                    "issues": issues
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "suggest_rulebreaks":
                emotion = arguments["emotion"]
                suggestions = suggest_rule_break(emotion)
                
                if not suggestions:
                    result = {
                        "emotion": emotion,
                        "suggestions": [],
                        "message": f"No specific suggestions for '{emotion}'. Try: grief, anger, nostalgia, defiance, dissociation"
                    }
                else:
                    result = {
                        "emotion": emotion,
                        "suggestions": [
                            {
                                "rule": s["rule"],
                                "description": s["description"],
                                "effect": s["effect"],
                                "use_when": s["use_when"]
                            }
                            for s in suggestions
                        ]
                    }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
        
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

