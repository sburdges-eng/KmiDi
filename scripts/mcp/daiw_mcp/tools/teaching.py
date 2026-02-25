"""
Teaching Tools - MCP tools for music theory education.

Provides 3 tools:
- explain_rulebreak
- get_progression_info
- emotion_to_music
"""

from typing import Any, Dict, List
import json

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Import DAiW modules
from music_brain.session.teaching import RuleBreakingTeacher
from music_brain.structure.progression import diagnose_progression
from music_brain.session.intent_schema import suggest_rule_break


def register_tools(server: Server) -> None:
    """Register all teaching tools with the MCP server."""
    if not MCP_AVAILABLE:
        return
    
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available teaching tools."""
        return [
            Tool(
                name="explain_rulebreak",
                description="Explain a rule-breaking technique with examples and emotional justification.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "rule_name": {
                            "type": "string",
                            "description": "Rule-breaking technique name (e.g., 'HARMONY_ModalInterchange', 'HARMONY_AvoidTonicResolution')"
                        }
                    },
                    "required": ["rule_name"]
                }
            ),
            Tool(
                name="get_progression_info",
                description="Get detailed information about a chord progression including theory, emotional character, and examples.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "progression": {
                            "type": "string",
                            "description": "Chord progression string (e.g., 'F-C-Dm-Bbm')"
                        }
                    },
                    "required": ["progression"]
                }
            ),
            Tool(
                name="emotion_to_music",
                description="Map an emotion to musical parameters (key, mode, rule-breaks, tempo).",
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
            if name == "explain_rulebreak":
                rule_name = arguments["rule_name"]
                teacher = RuleBreakingTeacher()
                
                # Get explanation from teaching module
                explanation = teacher.explain_rule(rule_name)
                
                result = {
                    "rule_name": rule_name,
                    "explanation": explanation if explanation else f"Explanation for {rule_name}",
                    "note": "Use daiw teach rulebreaking for interactive lessons"
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "get_progression_info":
                progression = arguments["progression"]
                diagnosis = diagnose_progression(progression)
                
                result = {
                    "progression": progression,
                    "key": diagnosis.get("key", "Unknown"),
                    "mode": diagnosis.get("mode", "Unknown"),
                    "emotional_character": diagnosis.get("emotional_character", ""),
                    "rule_breaks": diagnosis.get("rule_breaks", []),
                    "issues": diagnosis.get("issues", []),
                    "suggestions": diagnosis.get("suggestions", [])
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "emotion_to_music":
                emotion = arguments["emotion"]
                
                # Get rule-break suggestions
                suggestions = suggest_rule_break(emotion)
                
                # Map emotion to typical musical parameters
                emotion_mapping = {
                    "grief": {"key": "minor", "tempo": "slow", "rule_breaks": ["HARMONY_AvoidTonicResolution"]},
                    "anger": {"key": "minor", "tempo": "fast", "rule_breaks": ["HARMONY_UnresolvedDissonance"]},
                    "nostalgia": {"key": "major", "tempo": "moderate", "rule_breaks": ["HARMONY_ModalInterchange"]},
                    "defiance": {"key": "minor", "tempo": "fast", "rule_breaks": ["HARMONY_ParallelMotion"]},
                }
                
                base_mapping = emotion_mapping.get(emotion.lower(), {
                    "key": "major",
                    "tempo": "moderate",
                    "rule_breaks": []
                })
                
                result = {
                    "emotion": emotion,
                    "musical_parameters": {
                        "suggested_key": base_mapping["key"],
                        "suggested_tempo": base_mapping["tempo"],
                        "rule_break_suggestions": [
                            {
                                "rule": s["rule"],
                                "effect": s["effect"]
                            }
                            for s in suggestions
                        ] if suggestions else base_mapping["rule_breaks"]
                    },
                    "note": "These are suggestions. Emotional intent should drive technical choices."
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
        
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

