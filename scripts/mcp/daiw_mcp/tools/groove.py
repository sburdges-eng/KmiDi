"""
Groove Tools - MCP tools for groove extraction and application.

Provides 5 tools:
- extract_groove
- apply_groove
- analyze_pocket
- humanize_midi
- quantize_smart
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
from music_brain.groove import extract_groove, apply_groove
from music_brain.groove_engine import apply_groove as apply_groove_events


def register_tools(server: Server) -> None:
    """Register all groove tools with the MCP server."""
    if not MCP_AVAILABLE:
        return
    
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available groove tools."""
        return [
            Tool(
                name="extract_groove",
                description="Extract groove characteristics (timing, velocity, swing) from a MIDI file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "midi_file": {
                            "type": "string",
                            "description": "Path to MIDI file"
                        }
                    },
                    "required": ["midi_file"]
                }
            ),
            Tool(
                name="apply_groove",
                description="Apply a genre groove template to a MIDI file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "midi_file": {
                            "type": "string",
                            "description": "Path to input MIDI file"
                        },
                        "genre": {
                            "type": "string",
                            "description": "Genre template (funk, jazz, rock, hiphop, edm, latin)",
                            "default": "funk"
                        },
                        "output_file": {
                            "type": "string",
                            "description": "Path to output MIDI file"
                        },
                        "intensity": {
                            "type": "number",
                            "description": "Groove intensity (0.0-1.0)",
                            "default": 0.5
                        }
                    },
                    "required": ["midi_file"]
                }
            ),
            Tool(
                name="analyze_pocket",
                description="Analyze the timing pocket (groove feel) of a MIDI file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "midi_file": {
                            "type": "string",
                            "description": "Path to MIDI file"
                        }
                    },
                    "required": ["midi_file"]
                }
            ),
            Tool(
                name="humanize_midi",
                description="Add human feel to MIDI with complexity and vulnerability parameters.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "midi_file": {
                            "type": "string",
                            "description": "Path to input MIDI file"
                        },
                        "output_file": {
                            "type": "string",
                            "description": "Path to output MIDI file"
                        },
                        "complexity": {
                            "type": "number",
                            "description": "Timing chaos (0.0-1.0)",
                            "default": 0.5
                        },
                        "vulnerability": {
                            "type": "number",
                            "description": "Dynamic fragility (0.0-1.0)",
                            "default": 0.5
                        },
                        "preset": {
                            "type": "string",
                            "description": "Emotional preset name (optional)"
                        }
                    },
                    "required": ["midi_file"]
                }
            ),
            Tool(
                name="quantize_smart",
                description="Smart quantization that preserves musical feel.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "midi_file": {
                            "type": "string",
                            "description": "Path to input MIDI file"
                        },
                        "output_file": {
                            "type": "string",
                            "description": "Path to output MIDI file"
                        },
                        "strength": {
                            "type": "number",
                            "description": "Quantization strength (0.0-1.0)",
                            "default": 0.7
                        }
                    },
                    "required": ["midi_file"]
                }
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls."""
        try:
            if name == "extract_groove":
                midi_file = arguments["midi_file"]
                if not Path(midi_file).exists():
                    return [TextContent(type="text", text=f"Error: MIDI file not found: {midi_file}")]
                
                groove = extract_groove(midi_file)
                result = {
                    "midi_file": midi_file,
                    "timing_stats": {
                        "mean_deviation_ms": groove.timing_stats.get("mean_deviation_ms", 0),
                        "std_deviation_ms": groove.timing_stats.get("std_deviation_ms", 0)
                    },
                    "velocity_stats": {
                        "min": groove.velocity_stats.get("min", 0),
                        "max": groove.velocity_stats.get("max", 127),
                        "mean": groove.velocity_stats.get("mean", 64)
                    },
                    "swing_factor": groove.swing_factor,
                    "note_count": len(groove.notes)
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "apply_groove":
                midi_file = arguments["midi_file"]
                genre = arguments.get("genre", "funk")
                output_file = arguments.get("output_file", f"{Path(midi_file).stem}_grooved.mid")
                intensity = arguments.get("intensity", 0.5)
                
                if not Path(midi_file).exists():
                    return [TextContent(type="text", text=f"Error: MIDI file not found: {midi_file}")]
                
                apply_groove(midi_file, genre=genre, output=output_file, intensity=intensity)
                result = {
                    "input_file": midi_file,
                    "output_file": output_file,
                    "genre": genre,
                    "intensity": intensity,
                    "status": "success"
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "analyze_pocket":
                midi_file = arguments["midi_file"]
                if not Path(midi_file).exists():
                    return [TextContent(type="text", text=f"Error: MIDI file not found: {midi_file}")]
                
                groove = extract_groove(midi_file)
                result = {
                    "midi_file": midi_file,
                    "pocket_analysis": {
                        "feel": "ahead" if groove.timing_stats.get("mean_deviation_ms", 0) > 0 else "behind",
                        "tightness": "tight" if groove.timing_stats.get("std_deviation_ms", 0) < 10 else "loose",
                        "swing": "swung" if groove.swing_factor > 0.1 else "straight"
                    },
                    "timing_deviation_ms": groove.timing_stats.get("mean_deviation_ms", 0),
                    "swing_factor": groove.swing_factor
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "humanize_midi":
                midi_file = arguments["midi_file"]
                output_file = arguments.get("output_file", f"{Path(midi_file).stem}_humanized.mid")
                complexity = arguments.get("complexity", 0.5)
                vulnerability = arguments.get("vulnerability", 0.5)
                preset = arguments.get("preset")
                
                if not Path(midi_file).exists():
                    return [TextContent(type="text", text=f"Error: MIDI file not found: {midi_file}")]
                
                # Use groove engine for humanization
                from music_brain.groove import humanize_midi_file, GrooveSettings, settings_from_preset
                
                if preset:
                    settings = settings_from_preset(preset)
                else:
                    settings = GrooveSettings(complexity=complexity, vulnerability=vulnerability)
                
                result_path = humanize_midi_file(
                    input_path=midi_file,
                    output_path=output_file,
                    complexity=complexity,
                    vulnerability=vulnerability,
                    settings=settings
                )
                
                result = {
                    "input_file": midi_file,
                    "output_file": result_path,
                    "complexity": complexity,
                    "vulnerability": vulnerability,
                    "status": "success"
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "quantize_smart":
                midi_file = arguments["midi_file"]
                output_file = arguments.get("output_file", f"{Path(midi_file).stem}_quantized.mid")
                strength = arguments.get("strength", 0.7)
                
                if not Path(midi_file).exists():
                    return [TextContent(type="text", text=f"Error: MIDI file not found: {midi_file}")]
                
                # Implement smart quantization
                try:
                    import mido
                    from music_brain.utils.ppq import quantize_ticks
                    
                    mid = mido.MidiFile(midi_file)
                    output_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
                    
                    notes_quantized = 0
                    notes_preserved = 0
                    
                    for track in mid.tracks:
                        new_track = mido.MidiTrack()
                        output_mid.tracks.append(new_track)
                        
                        current_tick = 0
                        
                        for msg in track:
                            if msg.type in ['note_on', 'note_off']:
                                current_tick += msg.time
                                
                                # Quantize the tick position
                                quantized_tick = quantize_ticks(
                                    current_tick,
                                    ppq=mid.ticks_per_beat,
                                    resolution=16  # 16th note grid
                                )
                                
                                # Blend original and quantized timing based on strength
                                blended_tick = int(
                                    current_tick * (1.0 - strength) + 
                                    quantized_tick * strength
                                )
                                
                                # Calculate new time delta
                                if new_track:
                                    prev_tick = sum(m.time for m in new_track if m.type in ['note_on', 'note_off'])
                                    new_time = max(0, blended_tick - prev_tick)
                                else:
                                    new_time = blended_tick
                                
                                new_msg = msg.copy(time=new_time)
                                new_track.append(new_msg)
                                
                                # Track statistics
                                if abs(current_tick - quantized_tick) > mid.ticks_per_beat // 32:  # Significant deviation
                                    notes_quantized += 1
                                else:
                                    notes_preserved += 1
                            else:
                                # Copy non-note messages as-is
                                new_track.append(msg)
                    
                    output_mid.save(output_file)
                    
                    result = {
                        "input_file": midi_file,
                        "output_file": output_file,
                        "strength": strength,
                        "status": "success",
                        "notes_quantized": notes_quantized,
                        "notes_preserved": notes_preserved,
                        "note": f"Smart quantization applied with {strength*100:.0f}% strength. Feel preserved through partial quantization."
                    }
                except Exception as e:
                    result = {
                        "input_file": midi_file,
                        "error": str(e),
                        "status": "error"
                    }
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
        
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

