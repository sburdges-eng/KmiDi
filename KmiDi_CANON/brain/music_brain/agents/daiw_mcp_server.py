"""
DAiW MCP Server - Model Context Protocol Server for Voice Synthesis

This MCP server exposes the DAiW voice synthesis capabilities to AI agents
like Claude, enabling natural language control of:
- Voice synthesis and cloning
- Formant manipulation
- Text-to-speech with learned voices
- Real-time vocal effects
- DAW integration (Ableton, REAPER, Logic)

Based on the Model Context Protocol specification:
https://github.com/modelcontextprotocol/servers

Usage:
    # Start the MCP server
    python -m music_brain.agents.daiw_mcp_server

    # Or use with Claude Desktop config:
    {
        "mcpServers": {
            "daiw-voice": {
                "command": "python",
                "args": ["-m", "music_brain.agents.daiw_mcp_server"]
            }
        }
    }
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass
from enum import Enum
import logging

# MCP imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool, TextContent, ImageContent, EmbeddedResource,
        CallToolResult, ListToolsResult
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP not available. Install with: pip install mcp", file=sys.stderr)

# DAiW voice imports
try:
    from music_brain.vocal.parrot import ParrotVocalSynthesizer, ParrotConfig, VoiceModel
    from music_brain.voice.neural_voice import UnifiedNeuralVoice, NeuralVoiceConfig
    from music_brain.voice.cpp_bridge import VoiceCppBridge, VoiceSynthesisPipeline
    from music_brain.audio.framework_integrations import UnifiedAudioProcessor, EffectPreset
    DAIW_AVAILABLE = True
except ImportError:
    DAIW_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("daiw-mcp")


class DAiWVoiceTools(Enum):
    """Available tools in the DAiW MCP server"""
    # Voice Training
    TRAIN_VOICE = "train_voice"
    TRAIN_VOICE_BATCH = "train_voice_batch"
    LIST_VOICES = "list_voices"
    GET_VOICE_INFO = "get_voice_info"
    BLEND_VOICES = "blend_voices"

    # Synthesis
    SYNTHESIZE_SPEECH = "synthesize_speech"
    SYNTHESIZE_NEURAL = "synthesize_neural"
    SET_VOWEL = "set_vowel"
    SET_PITCH = "set_pitch"

    # Real-time Control
    NOTE_ON = "note_on"
    NOTE_OFF = "note_off"
    SET_FORMANT_SHIFT = "set_formant_shift"
    SET_BREATHINESS = "set_breathiness"
    SET_VIBRATO = "set_vibrato"

    # Effects
    APPLY_EFFECT_PRESET = "apply_effect_preset"
    LIST_EFFECT_PRESETS = "list_effect_presets"

    # DAW Integration
    CONNECT_CPP = "connect_cpp"
    DISCONNECT_CPP = "disconnect_cpp"
    SEND_TO_DAW = "send_to_daw"

    # Voice Cloning
    CLONE_VOICE = "clone_voice"
    QUICK_SPEAK = "quick_speak"


# Tool definitions for MCP
TOOL_DEFINITIONS = [
    {
        "name": "train_voice",
        "description": "Train a voice model from an audio file. The model learns formant characteristics, pitch, vibrato, and timbre.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_file": {
                    "type": "string",
                    "description": "Path to the audio file (WAV, MP3, etc.)"
                },
                "voice_name": {
                    "type": "string",
                    "description": "Name for the voice model"
                }
            },
            "required": ["audio_file", "voice_name"]
        }
    },
    {
        "name": "train_voice_batch",
        "description": "Train a voice model from multiple audio files for better accuracy.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of audio file paths"
                },
                "voice_name": {
                    "type": "string",
                    "description": "Name for the voice model"
                }
            },
            "required": ["audio_files", "voice_name"]
        }
    },
    {
        "name": "list_voices",
        "description": "List all trained voice models.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_voice_info",
        "description": "Get detailed information about a trained voice model.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "voice_name": {
                    "type": "string",
                    "description": "Name of the voice model"
                }
            },
            "required": ["voice_name"]
        }
    },
    {
        "name": "blend_voices",
        "description": "Blend two voice models together to create a new hybrid voice.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "voice1": {
                    "type": "string",
                    "description": "First voice model name"
                },
                "voice2": {
                    "type": "string",
                    "description": "Second voice model name"
                },
                "ratio": {
                    "type": "number",
                    "description": "Blend ratio (0.0 = voice1, 1.0 = voice2, 0.5 = equal)",
                    "default": 0.5
                },
                "output_name": {
                    "type": "string",
                    "description": "Name for the blended voice"
                }
            },
            "required": ["voice1", "voice2"]
        }
    },
    {
        "name": "synthesize_speech",
        "description": "Synthesize speech from text using formant synthesis with a trained voice.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to synthesize"
                },
                "voice_name": {
                    "type": "string",
                    "description": "Voice model to use (optional, uses current if not specified)"
                },
                "output_file": {
                    "type": "string",
                    "description": "Output file path (optional)"
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "synthesize_neural",
        "description": "Synthesize speech using neural TTS (Coqui, Bark, or OpenVoice).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to synthesize"
                },
                "backend": {
                    "type": "string",
                    "enum": ["auto", "coqui", "bark", "openvoice"],
                    "description": "Neural TTS backend to use",
                    "default": "auto"
                },
                "output_file": {
                    "type": "string",
                    "description": "Output file path (optional)"
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "set_vowel",
        "description": "Set the current vowel for real-time synthesis (A, E, I, O, U).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "vowel": {
                    "type": "string",
                    "enum": ["a", "e", "i", "o", "u"],
                    "description": "Vowel to set"
                }
            },
            "required": ["vowel"]
        }
    },
    {
        "name": "set_pitch",
        "description": "Set the pitch for synthesis in Hz.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pitch_hz": {
                    "type": "number",
                    "description": "Pitch in Hz (typical range: 80-400)"
                }
            },
            "required": ["pitch_hz"]
        }
    },
    {
        "name": "note_on",
        "description": "Trigger a note on for real-time synthesis.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "midi_note": {
                    "type": "integer",
                    "description": "MIDI note number (60 = middle C)",
                    "default": 60
                },
                "velocity": {
                    "type": "number",
                    "description": "Velocity (0.0 to 1.0)",
                    "default": 0.8
                }
            }
        }
    },
    {
        "name": "note_off",
        "description": "Release the current note.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "set_formant_shift",
        "description": "Set formant shift for voice character modification.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "shift": {
                    "type": "number",
                    "description": "Formant shift (1.0 = normal, <1.0 = darker, >1.0 = brighter)"
                }
            },
            "required": ["shift"]
        }
    },
    {
        "name": "set_breathiness",
        "description": "Set breathiness amount for the voice.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "Breathiness (0.0 to 1.0)"
                }
            },
            "required": ["amount"]
        }
    },
    {
        "name": "set_vibrato",
        "description": "Set vibrato intensity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "Vibrato intensity (0.0 to 1.0)"
                }
            },
            "required": ["amount"]
        }
    },
    {
        "name": "apply_effect_preset",
        "description": "Apply a voice effects preset.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "preset": {
                    "type": "string",
                    "enum": ["clean", "warm", "robotic", "ethereal"],
                    "description": "Effect preset to apply"
                }
            },
            "required": ["preset"]
        }
    },
    {
        "name": "list_effect_presets",
        "description": "List available effect presets.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "connect_cpp",
        "description": "Connect to the C++ VoiceProcessor for real-time synthesis.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host address",
                    "default": "127.0.0.1"
                },
                "port": {
                    "type": "integer",
                    "description": "Port number",
                    "default": 9000
                }
            }
        }
    },
    {
        "name": "disconnect_cpp",
        "description": "Disconnect from the C++ VoiceProcessor.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "clone_voice",
        "description": "Clone a voice from a reference audio file for neural TTS.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "reference_audio": {
                    "type": "string",
                    "description": "Path to reference audio file"
                }
            },
            "required": ["reference_audio"]
        }
    },
    {
        "name": "quick_speak",
        "description": "Quick function to speak text with optional voice cloning.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to speak"
                },
                "voice_file": {
                    "type": "string",
                    "description": "Optional voice file to clone"
                },
                "output_file": {
                    "type": "string",
                    "description": "Optional output file path"
                }
            },
            "required": ["text"]
        }
    }
]


class DAiWMCPServer:
    """
    MCP Server for DAiW Voice Synthesis

    Exposes voice synthesis capabilities to AI agents via
    the Model Context Protocol.
    """

    def __init__(self):
        self.parrot: Optional[ParrotVocalSynthesizer] = None
        self.neural_voice: Optional[UnifiedNeuralVoice] = None
        self.cpp_bridge: Optional[VoiceCppBridge] = None
        self.audio_processor: Optional[UnifiedAudioProcessor] = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize DAiW components"""
        if not DAIW_AVAILABLE:
            logger.warning("DAiW components not available")
            return

        try:
            self.parrot = ParrotVocalSynthesizer(ParrotConfig())
            logger.info("Parrot synthesizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Parrot: {e}")

        try:
            self.audio_processor = UnifiedAudioProcessor()
            logger.info("Audio processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize audio processor: {e}")

    async def handle_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Handle a tool call"""
        logger.info(f"Tool called: {name} with args: {arguments}")

        try:
            if name == "train_voice":
                return await self._train_voice(arguments)
            elif name == "train_voice_batch":
                return await self._train_voice_batch(arguments)
            elif name == "list_voices":
                return await self._list_voices()
            elif name == "get_voice_info":
                return await self._get_voice_info(arguments)
            elif name == "blend_voices":
                return await self._blend_voices(arguments)
            elif name == "synthesize_speech":
                return await self._synthesize_speech(arguments)
            elif name == "synthesize_neural":
                return await self._synthesize_neural(arguments)
            elif name == "set_vowel":
                return await self._set_vowel(arguments)
            elif name == "set_pitch":
                return await self._set_pitch(arguments)
            elif name == "note_on":
                return await self._note_on(arguments)
            elif name == "note_off":
                return await self._note_off()
            elif name == "set_formant_shift":
                return await self._set_formant_shift(arguments)
            elif name == "set_breathiness":
                return await self._set_breathiness(arguments)
            elif name == "set_vibrato":
                return await self._set_vibrato(arguments)
            elif name == "apply_effect_preset":
                return await self._apply_effect_preset(arguments)
            elif name == "list_effect_presets":
                return await self._list_effect_presets()
            elif name == "connect_cpp":
                return await self._connect_cpp(arguments)
            elif name == "disconnect_cpp":
                return await self._disconnect_cpp()
            elif name == "clone_voice":
                return await self._clone_voice(arguments)
            elif name == "quick_speak":
                return await self._quick_speak(arguments)
            else:
                return json.dumps({"error": f"Unknown tool: {name}"})

        except Exception as e:
            logger.error(f"Tool error: {e}")
            return json.dumps({"error": str(e)})

    # Tool implementations
    async def _train_voice(self, args: Dict) -> str:
        if not self.parrot:
            return json.dumps({"error": "Parrot not initialized"})

        model = self.parrot.train_parrot(
            args["audio_file"],
            args["voice_name"]
        )

        return json.dumps({
            "success": True,
            "voice_name": model.name,
            "exposure_time": model.characteristics.exposure_time,
            "confidence": model.characteristics.confidence,
            "average_pitch": model.characteristics.average_pitch
        })

    async def _train_voice_batch(self, args: Dict) -> str:
        if not self.parrot:
            return json.dumps({"error": "Parrot not initialized"})

        model = self.parrot.train_parrot_batch(
            args["audio_files"],
            args["voice_name"]
        )

        return json.dumps({
            "success": True,
            "voice_name": model.name,
            "files_processed": len(args["audio_files"]),
            "exposure_time": model.characteristics.exposure_time,
            "confidence": model.characteristics.confidence
        })

    async def _list_voices(self) -> str:
        if not self.parrot:
            return json.dumps({"voices": []})

        return json.dumps({"voices": self.parrot.list_voices()})

    async def _get_voice_info(self, args: Dict) -> str:
        if not self.parrot:
            return json.dumps({"error": "Parrot not initialized"})

        info = self.parrot.get_voice_info(args["voice_name"])
        return json.dumps(info)

    async def _blend_voices(self, args: Dict) -> str:
        if not self.parrot:
            return json.dumps({"error": "Parrot not initialized"})

        model = self.parrot.blend_voices(
            args["voice1"],
            args["voice2"],
            args.get("ratio", 0.5),
            args.get("output_name")
        )

        return json.dumps({
            "success": True,
            "blended_voice": model.name
        })

    async def _synthesize_speech(self, args: Dict) -> str:
        if not self.parrot:
            return json.dumps({"error": "Parrot not initialized"})

        audio = self.parrot.synthesize_vocal(
            args["text"],
            args.get("voice_name"),
            args.get("output_file")
        )

        return json.dumps({
            "success": True,
            "samples": len(audio),
            "output_file": args.get("output_file")
        })

    async def _synthesize_neural(self, args: Dict) -> str:
        if not self.neural_voice:
            from music_brain.voice.neural_voice import UnifiedNeuralVoice, NeuralVoiceConfig, NeuralVoiceBackend

            backend_map = {
                "auto": NeuralVoiceBackend.AUTO,
                "coqui": NeuralVoiceBackend.COQUI,
                "bark": NeuralVoiceBackend.BARK,
                "openvoice": NeuralVoiceBackend.OPENVOICE
            }

            config = NeuralVoiceConfig(
                backend=backend_map.get(args.get("backend", "auto"), NeuralVoiceBackend.AUTO)
            )
            self.neural_voice = UnifiedNeuralVoice(config)

        audio = self.neural_voice.synthesize(args["text"])

        if args.get("output_file"):
            self.neural_voice.save_audio(audio, args["output_file"])

        return json.dumps({
            "success": True,
            "samples": len(audio),
            "sample_rate": self.neural_voice.get_sample_rate(),
            "output_file": args.get("output_file")
        })

    async def _set_vowel(self, args: Dict) -> str:
        if self.cpp_bridge and self.cpp_bridge.connected:
            from music_brain.vocal.parrot import VowelType
            vowel_map = {
                'a': VowelType.A,
                'e': VowelType.E,
                'i': VowelType.I,
                'o': VowelType.O,
                'u': VowelType.U
            }
            self.cpp_bridge.set_vowel(vowel_map.get(args["vowel"].lower(), VowelType.A))
            return json.dumps({"success": True, "vowel": args["vowel"]})

        return json.dumps({"error": "Not connected to C++ processor"})

    async def _set_pitch(self, args: Dict) -> str:
        if self.cpp_bridge and self.cpp_bridge.connected:
            self.cpp_bridge.set_pitch(args["pitch_hz"])
            return json.dumps({"success": True, "pitch_hz": args["pitch_hz"]})

        return json.dumps({"error": "Not connected to C++ processor"})

    async def _note_on(self, args: Dict) -> str:
        if self.cpp_bridge and self.cpp_bridge.connected:
            self.cpp_bridge.note_on(
                args.get("midi_note", 60),
                args.get("velocity", 0.8)
            )
            return json.dumps({"success": True})

        return json.dumps({"error": "Not connected to C++ processor"})

    async def _note_off(self) -> str:
        if self.cpp_bridge and self.cpp_bridge.connected:
            self.cpp_bridge.note_off()
            return json.dumps({"success": True})

        return json.dumps({"error": "Not connected to C++ processor"})

    async def _set_formant_shift(self, args: Dict) -> str:
        if self.cpp_bridge and self.cpp_bridge.connected:
            self.cpp_bridge.set_formant_shift(args["shift"])
            return json.dumps({"success": True, "shift": args["shift"]})

        return json.dumps({"error": "Not connected to C++ processor"})

    async def _set_breathiness(self, args: Dict) -> str:
        if self.cpp_bridge and self.cpp_bridge.connected:
            self.cpp_bridge.set_breathiness(args["amount"])
            return json.dumps({"success": True, "breathiness": args["amount"]})

        return json.dumps({"error": "Not connected to C++ processor"})

    async def _set_vibrato(self, args: Dict) -> str:
        if self.cpp_bridge and self.cpp_bridge.connected:
            self.cpp_bridge.set_vibrato(args["amount"])
            return json.dumps({"success": True, "vibrato": args["amount"]})

        return json.dumps({"error": "Not connected to C++ processor"})

    async def _apply_effect_preset(self, args: Dict) -> str:
        if not self.audio_processor:
            return json.dumps({"error": "Audio processor not initialized"})

        preset_map = {
            "clean": EffectPreset.VOICE_CLEAN,
            "warm": EffectPreset.VOICE_WARM,
            "robotic": EffectPreset.VOICE_ROBOTIC,
            "ethereal": EffectPreset.VOICE_ETHEREAL
        }

        preset = preset_map.get(args["preset"])
        if preset:
            self.audio_processor.apply_preset(preset)
            return json.dumps({"success": True, "preset": args["preset"]})

        return json.dumps({"error": f"Unknown preset: {args['preset']}"})

    async def _list_effect_presets(self) -> str:
        return json.dumps({
            "presets": ["clean", "warm", "robotic", "ethereal"]
        })

    async def _connect_cpp(self, args: Dict) -> str:
        from music_brain.voice.cpp_bridge import VoiceCppBridge, CppBridgeConfig

        config = CppBridgeConfig(
            cpp_host=args.get("host", "127.0.0.1"),
            cpp_port=args.get("port", 9000),
            auto_connect=False
        )

        self.cpp_bridge = VoiceCppBridge(config)
        success = self.cpp_bridge.connect()

        return json.dumps({
            "success": success,
            "connected": self.cpp_bridge.connected
        })

    async def _disconnect_cpp(self) -> str:
        if self.cpp_bridge:
            self.cpp_bridge.disconnect()
            return json.dumps({"success": True})

        return json.dumps({"error": "Not connected"})

    async def _clone_voice(self, args: Dict) -> str:
        if not self.neural_voice:
            from music_brain.voice.neural_voice import UnifiedNeuralVoice
            self.neural_voice = UnifiedNeuralVoice()

        success = self.neural_voice.clone_voice(args["reference_audio"])
        return json.dumps({"success": success})

    async def _quick_speak(self, args: Dict) -> str:
        from music_brain.voice.neural_voice import quick_voice_clone, quick_neural_speak

        if args.get("voice_file"):
            audio = quick_voice_clone(
                args["voice_file"],
                args["text"],
                args.get("output_file")
            )
        else:
            audio = quick_neural_speak(
                args["text"],
                args.get("output_file")
            )

        return json.dumps({
            "success": True,
            "samples": len(audio)
        })


async def main():
    """Run the MCP server"""
    if not MCP_AVAILABLE:
        print("MCP library not available. Install with: pip install mcp")
        sys.exit(1)

    server = Server("daiw-voice")
    daiw = DAiWMCPServer()

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        return [
            Tool(
                name=tool["name"],
                description=tool["description"],
                inputSchema=tool["inputSchema"]
            )
            for tool in TOOL_DEFINITIONS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
        result = await daiw.handle_tool(name, arguments)
        return [TextContent(type="text", text=result)]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
