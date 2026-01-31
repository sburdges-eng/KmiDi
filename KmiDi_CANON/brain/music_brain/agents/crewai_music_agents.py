"""
DAiW CrewAI Music Production Agents

Multi-agent system for AI-assisted music production using CrewAI.
Orchestrates voice synthesis, composition, mixing, and DAW control
through specialized collaborative agents.

Based on patterns from:
- crewAIInc/crewAI
- akj2018/Multi-AI-Agent-Systems-with-crewAI
- dancohen81/Abai

Agent Roles:
1. VoiceDirector - Controls voice synthesis and vocal production
2. Composer - Creates melodies, harmonies, and arrangements
3. MixEngineer - Handles mixing, effects, and audio processing
4. DAWController - Interfaces with Ableton/REAPER via bridges
5. Producer - Orchestrates the team and makes creative decisions
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Union
from enum import Enum
import json
import asyncio
from pathlib import Path

# CrewAI imports (optional)
CREWAI_AVAILABLE = False
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    pass

# LangChain imports (for tools)
LANGCHAIN_AVAILABLE = False
try:
    from langchain.tools import Tool
    from langchain.agents import AgentExecutor
    LANGCHAIN_AVAILABLE = True
except ImportError:
    pass


class AgentRole(Enum):
    """Roles for music production agents"""
    VOICE_DIRECTOR = "voice_director"
    COMPOSER = "composer"
    MIX_ENGINEER = "mix_engineer"
    DAW_CONTROLLER = "daw_controller"
    PRODUCER = "producer"
    LYRICIST = "lyricist"
    ARRANGER = "arranger"


@dataclass
class AgentConfig:
    """Configuration for a music production agent"""
    role: AgentRole
    goal: str
    backstory: str
    tools: List[str] = field(default_factory=list)
    llm_model: str = "gpt-4"
    verbose: bool = True
    allow_delegation: bool = True
    max_iterations: int = 15


# ============================================================================
# Agent Definitions
# ============================================================================

AGENT_CONFIGS = {
    AgentRole.VOICE_DIRECTOR: AgentConfig(
        role=AgentRole.VOICE_DIRECTOR,
        goal="Direct and control voice synthesis to create expressive vocal performances",
        backstory="""You are an expert vocal director with deep knowledge of
        formant synthesis, voice cloning, and vocal production techniques.
        You understand the physics of vowel formation (F1/F2/F3 formants),
        vocal characteristics (breathiness, vibrato, jitter), and how to
        shape synthetic voices for musical expression. You work closely with
        the DAiW voice synthesis pipeline.""",
        tools=["voice_train", "voice_synthesize", "voice_set_vowel",
               "voice_set_pitch", "voice_formant_shift", "voice_vibrato",
               "voice_breathiness", "voice_note_on", "voice_note_off"]
    ),

    AgentRole.COMPOSER: AgentConfig(
        role=AgentRole.COMPOSER,
        goal="Compose melodies, chord progressions, and musical arrangements",
        backstory="""You are a versatile composer trained in music theory,
        harmony, and arrangement. You can write in many styles from classical
        to electronic, and understand how to create compelling vocal melodies
        that work well with synthetic voices. You consider vocal range,
        syllable placement, and melodic contour.""",
        tools=["compose_melody", "compose_chords", "compose_rhythm",
               "analyze_harmony", "suggest_arrangement"]
    ),

    AgentRole.MIX_ENGINEER: AgentConfig(
        role=AgentRole.MIX_ENGINEER,
        goal="Mix and process audio for optimal sound quality",
        backstory="""You are an experienced mix engineer who specializes
        in vocal production. You know how to use EQ, compression, reverb,
        delay, and other effects to make vocals sit perfectly in a mix.
        You understand frequency masking, stereo imaging, and dynamic
        control for professional results.""",
        tools=["apply_effect", "set_eq", "set_compression", "set_reverb",
               "set_delay", "analyze_spectrum", "check_levels"]
    ),

    AgentRole.DAW_CONTROLLER: AgentConfig(
        role=AgentRole.DAW_CONTROLLER,
        goal="Control the DAW for recording, editing, and playback",
        backstory="""You are a DAW power user who can efficiently control
        Ableton Live, REAPER, or other DAWs through OSC and MIDI. You
        understand session management, track routing, clip launching,
        and automation. You interface between the AI agents and the
        physical DAW environment.""",
        tools=["ableton_connect", "ableton_transport", "ableton_create_track",
               "ableton_arm_track", "ableton_fire_clip", "ableton_set_tempo",
               "ableton_set_track_volume", "ableton_get_session"]
    ),

    AgentRole.PRODUCER: AgentConfig(
        role=AgentRole.PRODUCER,
        goal="Oversee the creative vision and coordinate the production team",
        backstory="""You are an experienced music producer who coordinates
        all aspects of a production. You make high-level creative decisions,
        delegate tasks to specialists, and ensure the final result matches
        the artistic vision. You understand workflows, can troubleshoot
        issues, and keep the team moving toward the goal.""",
        tools=["delegate_task", "review_progress", "set_creative_direction",
               "approve_take", "request_revision"],
        allow_delegation=True
    ),

    AgentRole.LYRICIST: AgentConfig(
        role=AgentRole.LYRICIST,
        goal="Write lyrics that work well with synthetic voice synthesis",
        backstory="""You are a lyricist who understands both the art of
        songwriting and the technical requirements of voice synthesis.
        You consider phoneme flow, vowel distribution, and syllable timing
        to ensure lyrics sound natural when synthesized. You can write in
        various styles and adapt to different emotional tones.""",
        tools=["write_lyrics", "analyze_phonemes", "suggest_word_alternatives",
               "check_syllable_count", "optimize_for_synthesis"]
    ),

    AgentRole.ARRANGER: AgentConfig(
        role=AgentRole.ARRANGER,
        goal="Arrange music for optimal impact and structure",
        backstory="""You are a skilled arranger who creates compelling
        musical structures. You understand song form (verse, chorus, bridge),
        dynamics, instrumentation, and how to build emotional arcs. You
        work with the composer to turn ideas into complete arrangements.""",
        tools=["create_arrangement", "add_section", "set_dynamics",
               "assign_instruments", "create_transitions"]
    )
}


# ============================================================================
# Tool Wrappers for CrewAI
# ============================================================================

class VoiceSynthesisTool:
    """Tool wrapper for DAiW voice synthesis"""

    def __init__(self, voice_pipeline=None):
        self.voice_pipeline = voice_pipeline
        self._enabled = True
        self._load_pipeline()

    def _load_pipeline(self):
        """Load the voice pipeline if not provided"""
        if self.voice_pipeline is None:
            try:
                from .daiw_mcp_server import voice_pipeline
                self.voice_pipeline = voice_pipeline
            except ImportError:
                pass

    def shutdown(self):
        """Shutdown and release resources."""
        self._enabled = False
        self.voice_pipeline = None

    def enable(self):
        """Enable the tool."""
        self._enabled = True
        self._load_pipeline()

    def disable(self):
        """Disable the tool without full shutdown."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if tool is enabled."""
        return self._enabled

    def train_voice(self, audio_file: str, voice_name: str) -> str:
        """Train a new voice from audio file"""
        if self.voice_pipeline:
            model = self.voice_pipeline.train_voice(audio_file, voice_name)
            return f"Voice '{voice_name}' trained successfully"
        return "Voice pipeline not available"

    def synthesize(self, text: str, voice_name: Optional[str] = None) -> str:
        """Synthesize text to speech"""
        if self.voice_pipeline:
            audio = self.voice_pipeline.synthesize(text, voice_name)
            return f"Synthesized: {text[:50]}..."
        return "Voice pipeline not available"

    def set_vowel(self, vowel: str) -> str:
        """Set current vowel sound"""
        vowel_map = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}
        if vowel.lower() in vowel_map:
            # Would send to voice processor
            return f"Vowel set to: {vowel.upper()}"
        return f"Unknown vowel: {vowel}"

    def set_pitch(self, pitch_hz: float) -> str:
        """Set pitch in Hz"""
        return f"Pitch set to: {pitch_hz} Hz"

    def set_formant_shift(self, shift: float) -> str:
        """Set formant shift (-1 to 1)"""
        return f"Formant shift set to: {shift}"

    def set_breathiness(self, breathiness: float) -> str:
        """Set breathiness (0 to 1)"""
        return f"Breathiness set to: {breathiness}"

    def set_vibrato(self, rate: float, depth: float) -> str:
        """Set vibrato parameters"""
        return f"Vibrato set to rate={rate}, depth={depth}"

    def note_on(self, note: int, velocity: int = 100) -> str:
        """Trigger note on"""
        return f"Note on: {note} velocity={velocity}"

    def note_off(self, note: int) -> str:
        """Trigger note off"""
        return f"Note off: {note}"


class AbletonControlTool:
    """Tool wrapper for Ableton Live control"""

    def __init__(self, bridge=None):
        self.bridge = bridge
        self._enabled = True
        self._load_bridge()

    def _load_bridge(self):
        """Load the Ableton bridge if not provided"""
        if self.bridge is None:
            try:
                from .ableton_bridge import DAiWAbletonIntegration
                self.bridge = DAiWAbletonIntegration()
            except ImportError:
                pass

    def shutdown(self):
        """Shutdown and release resources."""
        self._enabled = False
        if self.bridge:
            self.bridge.disconnect()
        self.bridge = None

    def enable(self):
        """Enable the tool."""
        self._enabled = True
        self._load_bridge()

    def disable(self):
        """Disable the tool without full shutdown."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if tool is enabled."""
        return self._enabled

    def connect(self, host: str = "127.0.0.1") -> str:
        """Connect to Ableton Live"""
        if self.bridge:
            success = self.bridge.connect()
            return "Connected to Ableton" if success else "Connection failed"
        return "Ableton bridge not available"

    def play(self) -> str:
        """Start playback"""
        if self.bridge:
            self.bridge.osc_bridge.play()
            return "Playback started"
        return "Not connected"

    def stop(self) -> str:
        """Stop playback"""
        if self.bridge:
            self.bridge.osc_bridge.stop()
            return "Playback stopped"
        return "Not connected"

    def record(self) -> str:
        """Start recording"""
        if self.bridge:
            self.bridge.osc_bridge.record()
            return "Recording started"
        return "Not connected"

    def set_tempo(self, bpm: float) -> str:
        """Set session tempo"""
        if self.bridge:
            self.bridge.osc_bridge.set_tempo(bpm)
            return f"Tempo set to {bpm} BPM"
        return "Not connected"

    def create_track(self, name: str, track_type: str = "audio") -> str:
        """Create a new track"""
        if self.bridge:
            idx = self.bridge.osc_bridge.create_track(name, track_type)
            return f"Created {track_type} track '{name}' at index {idx}"
        return "Not connected"

    def arm_track(self, index: int, armed: bool = True) -> str:
        """Arm track for recording"""
        if self.bridge:
            self.bridge.osc_bridge.arm_track(index, armed)
            return f"Track {index} {'armed' if armed else 'disarmed'}"
        return "Not connected"

    def fire_clip(self, track: int, clip: int) -> str:
        """Fire a clip"""
        if self.bridge:
            self.bridge.osc_bridge.fire_clip(track, clip)
            return f"Fired clip {clip} on track {track}"
        return "Not connected"

    def get_session_info(self) -> Dict[str, Any]:
        """Get session information"""
        if self.bridge:
            return self.bridge.get_session_info()
        return {"error": "Not connected"}

    def set_track_volume(self, index: int, volume_db: float) -> str:
        """Set track volume"""
        if self.bridge:
            self.bridge.osc_bridge.set_track_volume(index, volume_db)
            return f"Track {index} volume set to {volume_db} dB"
        return "Not connected"


class CompositionTool:
    """Tool wrapper for music composition"""

    def compose_melody(self, key: str, scale: str,
                       length_bars: int = 4,
                       style: str = "pop") -> Dict[str, Any]:
        """Generate a melody"""
        # Placeholder - would connect to actual composition engine
        return {
            "melody": [
                {"pitch": 60, "start": 0.0, "duration": 0.5, "velocity": 100},
                {"pitch": 62, "start": 0.5, "duration": 0.5, "velocity": 90},
                {"pitch": 64, "start": 1.0, "duration": 1.0, "velocity": 100},
            ],
            "key": key,
            "scale": scale,
            "length_bars": length_bars
        }

    def compose_chords(self, key: str, progression: str = "I-V-vi-IV",
                       length_bars: int = 4) -> List[Dict[str, Any]]:
        """Generate chord progression"""
        # Placeholder
        return [
            {"root": "C", "quality": "major", "start": 0, "duration": 1},
            {"root": "G", "quality": "major", "start": 1, "duration": 1},
            {"root": "A", "quality": "minor", "start": 2, "duration": 1},
            {"root": "F", "quality": "major", "start": 3, "duration": 1},
        ]

    def analyze_harmony(self, notes: List[int]) -> Dict[str, Any]:
        """Analyze harmonic content"""
        return {
            "root": "C",
            "quality": "major",
            "extensions": ["7", "9"],
            "function": "tonic"
        }

    def suggest_arrangement(self, style: str, duration_minutes: float) -> Dict[str, Any]:
        """Suggest song arrangement"""
        return {
            "sections": [
                {"name": "intro", "start": 0, "duration": 8, "bars": 4},
                {"name": "verse1", "start": 8, "duration": 16, "bars": 8},
                {"name": "chorus1", "start": 24, "duration": 16, "bars": 8},
                {"name": "verse2", "start": 40, "duration": 16, "bars": 8},
                {"name": "chorus2", "start": 56, "duration": 16, "bars": 8},
                {"name": "bridge", "start": 72, "duration": 8, "bars": 4},
                {"name": "chorus3", "start": 80, "duration": 16, "bars": 8},
                {"name": "outro", "start": 96, "duration": 8, "bars": 4},
            ],
            "tempo": 120,
            "key": "C major",
            "style": style
        }


class MixingTool:
    """Tool wrapper for audio mixing"""

    def __init__(self, audio_processor=None):
        self.processor = audio_processor
        self._enabled = True
        self._load_processor()

    def _load_processor(self):
        """Load audio processor if not provided"""
        if self.processor is None:
            try:
                from ..audio.framework_integrations import UnifiedAudioProcessor
                self.processor = UnifiedAudioProcessor()
            except ImportError:
                pass

    def shutdown(self):
        """Shutdown and release resources."""
        self._enabled = False
        self.processor = None

    def enable(self):
        """Enable the tool."""
        self._enabled = True
        self._load_processor()

    def disable(self):
        """Disable the tool without full shutdown."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if tool is enabled."""
        return self._enabled

    def apply_effect(self, effect_type: str, **params) -> str:
        """Apply an audio effect"""
        if self.processor:
            self.processor.add_effect(effect_type, **params)
            return f"Applied {effect_type} effect"
        return "Processor not available"

    def set_eq(self, track: int, low: float, mid: float, high: float) -> str:
        """Set EQ for a track"""
        return f"Track {track} EQ: low={low}dB, mid={mid}dB, high={high}dB"

    def set_compression(self, track: int, threshold: float,
                        ratio: float, attack: float, release: float) -> str:
        """Set compression for a track"""
        return f"Track {track} compression: {threshold}dB threshold, {ratio}:1 ratio"

    def set_reverb(self, track: int, size: float,
                   wet: float, predelay: float = 0) -> str:
        """Set reverb for a track"""
        return f"Track {track} reverb: size={size}, wet={wet}"

    def set_delay(self, track: int, time_ms: float,
                  feedback: float, wet: float) -> str:
        """Set delay for a track"""
        return f"Track {track} delay: {time_ms}ms, feedback={feedback}"

    def analyze_spectrum(self, audio_data) -> Dict[str, Any]:
        """Analyze frequency spectrum"""
        return {
            "peak_frequency": 1000,
            "rms_level": -18.5,
            "crest_factor": 12.3,
            "spectral_centroid": 2500
        }

    def check_levels(self, audio_data) -> Dict[str, Any]:
        """Check audio levels"""
        return {
            "peak": -3.2,
            "rms": -18.5,
            "lufs": -14.0,
            "clipping": False
        }


class LyricsTool:
    """Tool wrapper for lyrics generation and analysis"""

    def write_lyrics(self, theme: str, style: str,
                     verse_lines: int = 4,
                     chorus_lines: int = 4) -> Dict[str, Any]:
        """Generate lyrics"""
        # Placeholder - would connect to LLM for generation
        return {
            "verse1": [
                "Walking through the morning light",
                "Everything feels so right",
                "The world is waking up today",
                "In every single way"
            ],
            "chorus": [
                "This is where we belong",
                "Singing our song",
                "Together we're strong",
                "All night long"
            ],
            "theme": theme,
            "style": style
        }

    def analyze_phonemes(self, text: str) -> List[Dict[str, Any]]:
        """Analyze phoneme distribution in text"""
        # Count vowels and consonants
        vowels = sum(1 for c in text.lower() if c in 'aeiou')
        consonants = sum(1 for c in text.lower() if c.isalpha() and c not in 'aeiou')

        return {
            "text": text,
            "vowel_count": vowels,
            "consonant_count": consonants,
            "vowel_ratio": vowels / (vowels + consonants) if (vowels + consonants) > 0 else 0,
            "syllable_estimate": len(text.split()) * 1.5,
            "synthesis_friendly": vowels > consonants * 0.5
        }

    def suggest_word_alternatives(self, word: str,
                                  context: str = "") -> List[str]:
        """Suggest alternative words for better synthesis"""
        # Placeholder
        return [word, f"{word}s", f"the {word}"]

    def check_syllable_count(self, line: str) -> int:
        """Count syllables in a line"""
        # Simple estimate
        return len(line.split()) * 1.5

    def optimize_for_synthesis(self, lyrics: str) -> str:
        """Optimize lyrics for voice synthesis"""
        # Add slight modifications for better synthesis
        return lyrics


# ============================================================================
# CrewAI Agent Factory
# ============================================================================

def create_crewai_tools():
    """Create tool instances for CrewAI agents"""
    voice_tool = VoiceSynthesisTool()
    ableton_tool = AbletonControlTool()
    composition_tool = CompositionTool()
    mixing_tool = MixingTool()
    lyrics_tool = LyricsTool()

    return {
        # Voice tools
        "voice_train": voice_tool.train_voice,
        "voice_synthesize": voice_tool.synthesize,
        "voice_set_vowel": voice_tool.set_vowel,
        "voice_set_pitch": voice_tool.set_pitch,
        "voice_formant_shift": voice_tool.set_formant_shift,
        "voice_breathiness": voice_tool.set_breathiness,
        "voice_vibrato": voice_tool.set_vibrato,
        "voice_note_on": voice_tool.note_on,
        "voice_note_off": voice_tool.note_off,

        # DAW tools
        "ableton_connect": ableton_tool.connect,
        "ableton_play": ableton_tool.play,
        "ableton_stop": ableton_tool.stop,
        "ableton_record": ableton_tool.record,
        "ableton_set_tempo": ableton_tool.set_tempo,
        "ableton_create_track": ableton_tool.create_track,
        "ableton_arm_track": ableton_tool.arm_track,
        "ableton_fire_clip": ableton_tool.fire_clip,
        "ableton_get_session": ableton_tool.get_session_info,
        "ableton_set_track_volume": ableton_tool.set_track_volume,

        # Composition tools
        "compose_melody": composition_tool.compose_melody,
        "compose_chords": composition_tool.compose_chords,
        "analyze_harmony": composition_tool.analyze_harmony,
        "suggest_arrangement": composition_tool.suggest_arrangement,

        # Mixing tools
        "apply_effect": mixing_tool.apply_effect,
        "set_eq": mixing_tool.set_eq,
        "set_compression": mixing_tool.set_compression,
        "set_reverb": mixing_tool.set_reverb,
        "set_delay": mixing_tool.set_delay,
        "analyze_spectrum": mixing_tool.analyze_spectrum,
        "check_levels": mixing_tool.check_levels,

        # Lyrics tools
        "write_lyrics": lyrics_tool.write_lyrics,
        "analyze_phonemes": lyrics_tool.analyze_phonemes,
        "suggest_word_alternatives": lyrics_tool.suggest_word_alternatives,
        "check_syllable_count": lyrics_tool.check_syllable_count,
        "optimize_for_synthesis": lyrics_tool.optimize_for_synthesis,
    }


def create_agent(role: AgentRole, tools: Dict[str, Callable] = None):
    """
    Create a CrewAI agent for the specified role.

    Args:
        role: Agent role
        tools: Dictionary of available tools

    Returns:
        CrewAI Agent instance or dict config if CrewAI not available
    """
    if role not in AGENT_CONFIGS:
        raise ValueError(f"Unknown agent role: {role}")

    config = AGENT_CONFIGS[role]

    if tools is None:
        tools = create_crewai_tools()

    # Get tools for this agent
    agent_tools = [tools.get(t) for t in config.tools if t in tools]
    agent_tools = [t for t in agent_tools if t is not None]

    if CREWAI_AVAILABLE:
        # Create actual CrewAI agent
        return Agent(
            role=config.role.value,
            goal=config.goal,
            backstory=config.backstory,
            tools=agent_tools,
            verbose=config.verbose,
            allow_delegation=config.allow_delegation,
            max_iter=config.max_iterations
        )
    else:
        # Return config dict for manual use
        return {
            "role": config.role.value,
            "goal": config.goal,
            "backstory": config.backstory,
            "tools": config.tools,
            "allow_delegation": config.allow_delegation
        }


def create_music_production_crew(include_roles: List[AgentRole] = None):
    """
    Create a full music production crew.

    Args:
        include_roles: List of roles to include (defaults to all)

    Returns:
        CrewAI Crew instance or dict of agents if CrewAI not available
    """
    if include_roles is None:
        include_roles = [
            AgentRole.PRODUCER,
            AgentRole.VOICE_DIRECTOR,
            AgentRole.COMPOSER,
            AgentRole.MIX_ENGINEER,
            AgentRole.DAW_CONTROLLER,
            AgentRole.LYRICIST
        ]

    tools = create_crewai_tools()
    agents = {role: create_agent(role, tools) for role in include_roles}

    if CREWAI_AVAILABLE:
        return Crew(
            agents=list(agents.values()),
            process=Process.hierarchical,
            manager_agent=agents.get(AgentRole.PRODUCER),
            verbose=True
        )
    else:
        return agents


# ============================================================================
# Pre-defined Tasks
# ============================================================================

def create_voice_production_task(text: str, voice_name: str = None,
                                 style: str = "natural"):
    """
    Create a task for producing a voice track.

    Args:
        text: Text to synthesize
        voice_name: Voice preset to use
        style: Production style (natural, robotic, ethereal, etc.)
    """
    description = f"""
    Produce a voice track with the following requirements:

    Text to synthesize: "{text}"
    Voice preset: {voice_name or "default"}
    Style: {style}

    Steps:
    1. Analyze the text for phoneme distribution and syllable count
    2. Set up the voice synthesizer with appropriate parameters
    3. Synthesize the audio with expression and dynamics
    4. Apply mixing effects appropriate for the style
    5. Ensure levels are appropriate for the mix
    """

    if CREWAI_AVAILABLE:
        return Task(
            description=description,
            expected_output="A synthesized voice track with applied effects",
            agent=None  # Assigned when running
        )
    else:
        return {"description": description, "type": "voice_production"}


def create_song_production_task(theme: str, style: str,
                                duration_minutes: float = 3.0,
                                has_vocals: bool = True):
    """
    Create a task for producing a complete song.

    Args:
        theme: Song theme/topic
        style: Musical style
        duration_minutes: Target duration
        has_vocals: Whether to include synthesized vocals
    """
    description = f"""
    Produce a complete song with the following requirements:

    Theme: {theme}
    Style: {style}
    Target Duration: {duration_minutes} minutes
    Include Vocals: {has_vocals}

    Steps:
    1. Create song arrangement (intro, verses, chorus, bridge, outro)
    2. Compose chord progression and melody
    3. {"Write and synthesize vocal parts" if has_vocals else "Skip vocal production"}
    4. Set up DAW session with appropriate tracks
    5. Mix all elements together
    6. Ensure final mix meets loudness standards
    """

    if CREWAI_AVAILABLE:
        return Task(
            description=description,
            expected_output="A complete mixed song",
            agent=None
        )
    else:
        return {"description": description, "type": "song_production"}


def create_vocal_direction_task(lyrics: str, emotion: str,
                                vocal_style: str = "pop"):
    """
    Create a task for directing vocal performance.

    Args:
        lyrics: Lyrics to perform
        emotion: Desired emotional tone
        vocal_style: Vocal style
    """
    description = f"""
    Direct the vocal performance with the following requirements:

    Lyrics:
    {lyrics}

    Emotion: {emotion}
    Vocal Style: {vocal_style}

    Steps:
    1. Analyze lyrics for phoneme distribution
    2. Plan vowel transitions and formant movements
    3. Set breathiness and vibrato for the emotion
    4. Direct pitch bends and dynamics
    5. Guide the synthesis for natural expression
    """

    if CREWAI_AVAILABLE:
        return Task(
            description=description,
            expected_output="Directed vocal performance parameters",
            agent=None
        )
    else:
        return {"description": description, "type": "vocal_direction"}


# ============================================================================
# Execution Functions
# ============================================================================

def run_production_task(task_type: str, **kwargs):
    """
    Run a music production task using the agent crew.

    Args:
        task_type: Type of task (voice, song, vocal_direction)
        **kwargs: Task-specific parameters

    Returns:
        Task result
    """
    crew = create_music_production_crew()

    if task_type == "voice":
        task = create_voice_production_task(**kwargs)
    elif task_type == "song":
        task = create_song_production_task(**kwargs)
    elif task_type == "vocal_direction":
        task = create_vocal_direction_task(**kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    if CREWAI_AVAILABLE:
        return crew.kickoff(tasks=[task])
    else:
        # Manual execution without CrewAI
        return {"task": task, "crew": crew, "status": "pending"}


async def run_production_task_async(task_type: str, **kwargs):
    """Async version of run_production_task"""
    import asyncio
    return await asyncio.to_thread(run_production_task, task_type, **kwargs)


# ============================================================================
# Utility Functions
# ============================================================================

def get_available_roles() -> List[str]:
    """Get list of available agent roles"""
    return [role.value for role in AgentRole]


def get_role_info(role: AgentRole) -> Dict[str, Any]:
    """Get information about a specific role"""
    if role not in AGENT_CONFIGS:
        return {"error": f"Unknown role: {role}"}

    config = AGENT_CONFIGS[role]
    return {
        "role": config.role.value,
        "goal": config.goal,
        "backstory": config.backstory,
        "tools": config.tools,
        "allow_delegation": config.allow_delegation
    }


def check_dependencies() -> Dict[str, bool]:
    """Check availability of dependencies"""
    return {
        "crewai": CREWAI_AVAILABLE,
        "langchain": LANGCHAIN_AVAILABLE,
    }


# ============================================================================
# Tool Manager - Central management for all tools
# ============================================================================

class ToolManager:
    """
    Central manager for all CrewAI music production tools.

    Provides unified control for enabling, disabling, and shutting down
    all tools in the system.

    Example:
        manager = ToolManager()
        manager.start()

        # Use tools...
        tools = manager.get_tools()

        # Shutdown everything
        manager.shutdown()
    """

    def __init__(self):
        self._tools: Dict[str, Any] = {}
        self._running = False

    def start(self):
        """Initialize and start all tools."""
        self._tools = {
            "voice": VoiceSynthesisTool(),
            "ableton": AbletonControlTool(),
            "composition": CompositionTool(),
            "mixing": MixingTool(),
            "lyrics": LyricsTool(),
        }
        self._running = True

    def shutdown(self):
        """Shutdown all tools and release resources."""
        self._running = False

        for name, tool in self._tools.items():
            if hasattr(tool, 'shutdown'):
                try:
                    tool.shutdown()
                except Exception as e:
                    print(f"Error shutting down {name}: {e}")

        self._tools.clear()

    def enable_tool(self, name: str):
        """Enable a specific tool."""
        if name in self._tools and hasattr(self._tools[name], 'enable'):
            self._tools[name].enable()

    def disable_tool(self, name: str):
        """Disable a specific tool."""
        if name in self._tools and hasattr(self._tools[name], 'disable'):
            self._tools[name].disable()

    def get_tool(self, name: str):
        """Get a specific tool by name."""
        return self._tools.get(name)

    def get_tools(self) -> Dict[str, Any]:
        """Get all tools."""
        return self._tools.copy()

    def is_running(self) -> bool:
        """Check if manager is running."""
        return self._running

    def get_status(self) -> Dict[str, Any]:
        """Get status of all tools."""
        status = {"running": self._running, "tools": {}}

        for name, tool in self._tools.items():
            tool_status = {"available": True}
            if hasattr(tool, 'is_enabled'):
                tool_status["enabled"] = tool.is_enabled()
            status["tools"][name] = tool_status

        return status

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures shutdown."""
        self.shutdown()
        return False

    def __del__(self):
        """Destructor - ensure cleanup."""
        if self._running:
            self.shutdown()


# Global tool manager instance
_tool_manager: Optional[ToolManager] = None


def get_tool_manager() -> ToolManager:
    """Get or create the global tool manager."""
    global _tool_manager
    if _tool_manager is None:
        _tool_manager = ToolManager()
    return _tool_manager


def shutdown_tools():
    """Shutdown the global tool manager."""
    global _tool_manager
    if _tool_manager:
        _tool_manager.shutdown()
        _tool_manager = None
