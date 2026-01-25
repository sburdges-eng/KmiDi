"""
DAiW Comprehensive Engine
=========================
Integrates the Therapist (Phase 0/1), Constraints (Phase 2), and
Direct MIDI Generation (Phase 3) into a single production pipeline.

Logic Flow:
1. TherapySession processes text -> AffectResult
2. TherapySession generates HarmonyPlan (with mode/tempo/chords)
3. render_plan_to_midi() converts Plan -> MIDI using music_brain.daw.logic

Philosophy: "Interrogate Before Generate" - The tool shouldn't finish art
for people; it should make them braver.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

MIDO_AVAILABLE = False

# ==============================================================================
# 1. AFFECT ANALYZER (Scored & Ranked)
# ==============================================================================


@dataclass
class AffectResult:
    """Result of emotional content analysis."""

    primary: str
    secondary: Optional[str]
    scores: Dict[str, float]
    intensity: float  # 0.0 to 1.0


class AffectAnalyzer:
    """
    Analyzes text for emotional content using weighted keywords.
    Exposes raw scores for tie-breaking and nuance.
    """

    KEYWORDS = {
        "grief": {
            "loss",
            "gone",
            "miss",
            "dead",
            "died",
            "funeral",
            "mourning",
            "never again",
            "empty",
        },
        "rage": {
            "angry",
            "furious",
            "hate",
            "betrayed",
            "unfair",
            "revenge",
            "burn",
            "fight",
            "destroy",
        },
        "awe": {"wonder", "beautiful", "infinite", "god", "universe", "transcend", "light", "vast"},
        "nostalgia": {
            "remember",
            "used to",
            "childhood",
            "back when",
            "old days",
            "memory",
            "home",
        },
        "fear": {"scared", "terrified", "panic", "can't breathe", "trapped", "anxious", "dread"},
        "dissociation": {
            "numb",
            "nothing",
            "floating",
            "unreal",
            "detached",
            "fog",
            "grey",
            "wall",
        },
        "defiance": {"won't", "refuse", "stand", "strong", "break", "free", "my own", "no more"},
        "tenderness": {"soft", "gentle", "hold", "love", "kind", "care", "fragile", "warm"},
        "confusion": {"why", "lost", "don't know", "spinning", "chaos", "strange", "question"},
    }

    def analyze(self, text: str) -> AffectResult:
        """
        Analyze text for emotional content.

        Args:
            text: Raw user input describing their emotional state

        Returns:
            AffectResult with primary/secondary affects, scores, and intensity
        """
        if not text:
            return AffectResult("neutral", None, {}, 0.0)

        text = text.lower()
        scores = dict.fromkeys(self.KEYWORDS, 0.0)

        for affect, words in self.KEYWORDS.items():
            for word in words:
                if word in text:
                    scores[affect] += 1.0

        # Sort affects by score descending
        sorted_affects = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        primary = sorted_affects[0][0] if sorted_affects[0][1] > 0 else "neutral"
        secondary = (
            sorted_affects[1][0] if len(sorted_affects) > 1 and sorted_affects[1][1] > 0 else None
        )

        # Calculate intensity (simple saturation at 3 keywords)
        intensity = min(1.0, sorted_affects[0][1] / 3.0) if sorted_affects[0][1] > 0 else 0.0

        return AffectResult(
            primary=primary, secondary=secondary, scores=scores, intensity=intensity
        )


# ==============================================================================
# 2. DATA MODELS (Source of Truth)
# ==============================================================================


@dataclass
class TherapyState:
    """
    Single Source of Truth for the session state.
    Replaces the deprecated CoreWoundModel with a unified schema.
    """

    # Narrative
    core_wound_name: str = ""
    narrative_entity_name: str = ""  # For externalization: "The Shadow", "Mr. Fog"

    # Quantifiable (from Motivational Interviewing techniques)
    motivation_scale: int = 5  # 1-10: "How much do you need this song to exist?"
    chaos_tolerance: float = 0.3  # 0.0 to 1.0: "How much control do you need?"

    # Inferred from analysis
    affect_result: Optional[AffectResult] = None
    suggested_mode: str = "ionian"


@dataclass
class HarmonyPlan:
    """
    Complete blueprint for generation.
    Can be passed to music_brain.structure.progression functions
    and rendered to MIDI via music_brain.daw.logic.
    """

    root_note: str  # "C", "F#"
    mode: str  # "minor", "dorian", "phrygian", etc.
    tempo_bpm: int
    time_signature: str  # "4/4", "6/8"
    length_bars: int  # Derived from motivation_scale
    chord_symbols: List[str]  # ["Cm7", "Fm9"]
    harmonic_rhythm: str  # "1_chord_per_bar", "syncopated"
    mood_profile: str  # "rage", "grief", etc.
    complexity: float  # 0.0 - 1.0, influences generation chaos
    vulnerability: float = 0.0  # Optional vulnerability scale (0-1) for therapy contexts
    structure: Optional[List[Dict[str, Any]]] = None  # Song sections with bars and chords
    instruments: Optional[List[Dict[str, Any]]] = (
        None  # Instrument definitions with channels and types
    )


# ==============================================================================
# 3. OBLIQUE STRATEGIES (Tiered by Chaos Tolerance)
# ==============================================================================

STRATEGIES_MILD = [
    "Remove specifics and convert to ambiguities.",
    "Work at a different speed.",
    "Use fewer notes.",
    "Repetition is a form of change.",
    "What would your closest friend do?",
]

STRATEGIES_WILD = [
    "Honor thy error as a hidden intention.",
    "Use an unacceptable color.",
    "Make a sudden, destructive unpredictable action.",
    "Turn it upside down.",
    "Disconnect from desire.",
    "Abandon normal instruments.",
]


def get_strategy(tolerance: float) -> str:
    """
    Select an Oblique Strategy based on chaos tolerance.

    Low tolerance gets safe affirmations.
    High tolerance gets Brian Eno's wilder cards.

    Args:
        tolerance: 0.0 (need control) to 1.0 (let it break)

    Returns:
        A strategy prompt string
    """
    if tolerance < 0.3:
        return "Trust in the you of now."  # Safety strategy
    elif tolerance < 0.7:
        return random.choice(STRATEGIES_MILD)
    else:
        # High tolerance accesses full deck, weighted towards Wild
        deck = STRATEGIES_MILD + (STRATEGIES_WILD * 2)
        return random.choice(deck)


# ==============================================================================
# 4. THERAPY SESSION (Pure Logic Layer - No I/O)
# ==============================================================================


class TherapySession:
    """
    Core logic for the therapy/interrogation workflow.

    This class handles state management and transformation logic.
    It contains NO print statements - decoupled from UI for reuse in
    CLI, GUI, or Web API contexts.
    """

    def __init__(self):
        self.state = TherapyState()
        self.analyzer = AffectAnalyzer()

        # Affect-to-Mode mapping (music theory meets psychology)
        self.AFFECT_TO_MODE = {
            "awe": "lydian",  # Bright, floaty
            "nostalgia": "dorian",  # Sentimental minor
            "rage": "phrygian",  # Aggressive minor (flamenco)
            "fear": "phrygian",  # Tension
            "dissociation": "locrian",  # Unstable, diminished
            "grief": "aeolian",  # Sad natural minor
            "defiance": "mixolydian",  # Major with flat 7 (rock/rebellion)
            "tenderness": "ionian",  # Gentle major
            "confusion": "locrian",  # Disoriented
            "neutral": "ionian",
        }

    def process_core_input(self, text: str) -> str:
        """
        Step 1: Ingest the wound, analyze affect.

        Args:
            text: Raw user input describing what's hurting them

        Returns:
            String name of the detected primary affect
        """
        if not text.strip():
            # Edge case handling: Empty input returns neutral state.
            # "silence" is returned to caller to indicate lack of text,
            # but internal state is safely set to Neutral/Ionian.
            self.state.affect_result = AffectResult("neutral", None, {}, 0.0)
            self.state.suggested_mode = "ionian"
            return "silence"

        self.state.core_wound_name = text
        self.state.affect_result = self.analyzer.analyze(text)

        primary = self.state.affect_result.primary
        self.state.suggested_mode = self.AFFECT_TO_MODE.get(primary, "ionian")

        return primary

    def set_scales(self, motivation: int, chaos: float):
        """
        Step 2: Set numerical parameters from user input.

        Args:
            motivation: 1-10 scale ("How much do you need this song?")
            chaos: 0.0-1.0 ("How much control do you need?")
        """
        self.state.motivation_scale = max(1, min(10, motivation))
        self.state.chaos_tolerance = max(0.0, min(1.0, chaos))

    def generate_plan(self) -> HarmonyPlan:
        """
        Step 3: Factory that builds the HarmonyPlan based on State.

        Uses motivation_scale, chaos_tolerance, and affect_result to
        determine tempo, length, complexity, and chord selection.

        Returns:
            HarmonyPlan ready for MIDI rendering
        """
        # Safety Guard
        if self.state.affect_result is None:
            self.state.affect_result = AffectResult("neutral", None, {}, 0.0)

        # 1. Tempo Logic (Affect + Chaos)
        base_tempo = 100
        primary = self.state.affect_result.primary

        if primary in ["rage", "fear", "defiance"]:
            base_tempo = 130
        elif primary in ["grief", "dissociation"]:
            base_tempo = 70
        elif primary in ["awe"]:
            base_tempo = 90

        # Chaos modulates tempo (+/- 20bpm based on tolerance)
        final_tempo = base_tempo + int((self.state.chaos_tolerance * 40) - 20)

        # 2. Length Logic (Derived from Motivation)
        # Low motivation (1-3) -> 16 bars (Quick sketch)
        # Mid motivation (4-7) -> 32 bars (Standard idea)
        # High motivation (8-10) -> 64 bars (Full structure)
        if self.state.motivation_scale <= 3:
            length = 16
        elif self.state.motivation_scale <= 7:
            length = 32
        else:
            length = 64

        # 3. Complexity Nudge
        # If motivation is high, user can handle slightly more complex structures
        eff_complexity = self.state.chaos_tolerance
        if self.state.motivation_scale > 8:
            eff_complexity = min(1.0, eff_complexity + 0.1)

        # 4. Chord Selection Logic (Mode-based progressions)
        root = "C"
        mode = self.state.suggested_mode

        if mode == "locrian":
            chords = ["Cdim", "DbMaj7", "Ebm", "Cdim"]
        elif mode == "phrygian":
            chords = ["Cm", "Db", "Bbm", "Cm"]
        elif mode == "lydian":
            chords = ["C", "D", "Bm", "C"]
        elif mode == "mixolydian":
            chords = ["C", "Bb", "F", "C"]
        elif mode == "aeolian":
            chords = ["Cm", "Ab", "Fm", "Cm"]
        elif mode == "dorian":
            chords = ["Cm", "F", "Gm", "Cm"]
        else:  # Ionian/Neutral
            chords = ["C", "Am", "F", "G"]

        return HarmonyPlan(
            root_note=root,
            mode=mode,
            tempo_bpm=final_tempo,
            time_signature="4/4",
            length_bars=length,
            chord_symbols=chords,
            harmonic_rhythm="1_chord_per_bar",
            mood_profile=primary,
            complexity=eff_complexity,
        )


# ==============================================================================
# 5. INSTRUMENT PATTERN GENERATORS
# ==============================================================================


def generate_bass_pattern(
    chords: List[Any], bars: int, tempo: int, ppq: int, beats_per_bar: int
) -> List[Dict[str, Any]]:
    """
    Generate bass pattern (root notes) from chord progression.

    Args:
        chords: List of parsed chord objects
        bars: Number of bars
        tempo: Tempo in BPM
        ppq: Pulses per quarter note
        beats_per_bar: Beats per bar

    Returns:
        List of note dicts for bass track
    """
    notes = []
    bar_ticks = beats_per_bar * ppq
    start_tick = 0
    current_bar = 0

    while current_bar < bars:
        for parsed in chords:
            if current_bar >= bars:
                break

            root_midi = 36 + parsed.root_num  # C2 as base for bass (one octave below C3)
            duration_ticks = bar_ticks

            # Simple root note on beat 1, optional 8th note on beat 3
            notes.append(
                {
                    "pitch": root_midi,
                    "velocity": 100,
                    "start_tick": start_tick,
                    "duration_ticks": duration_ticks // 2,
                }
            )

            # Optional 8th note on beat 3
            if current_bar % 2 == 0:  # Every other bar
                notes.append(
                    {
                        "pitch": root_midi,
                        "velocity": 80,
                        "start_tick": start_tick + (bar_ticks // 2),
                        "duration_ticks": duration_ticks // 4,
                    }
                )

            start_tick += duration_ticks
            current_bar += 1

    return notes


def generate_drum_pattern(
    bars: int, tempo: int, ppq: int, beats_per_bar: int, style: str = "pop"
) -> List[Dict[str, Any]]:
    """
    Generate drum pattern (kick, snare, hi-hat).

    Args:
        bars: Number of bars
        tempo: Tempo in BPM
        ppq: Pulses per quarter note
        beats_per_bar: Beats per bar
        style: Drum style ("pop", "rock", "jazz")

    Returns:
        List of note dicts for drum track
    """
    notes = []
    bar_ticks = beats_per_bar * ppq
    sixteenth_ticks = ppq // 4

    # GM drum notes
    KICK = 36  # C1
    SNARE = 38  # D1
    HIHAT = 42  # F#1

    for bar in range(bars):
        bar_start = bar * bar_ticks

        if style == "pop" or style == "rock":
            # Kick on 1 and 3
            notes.append(
                {
                    "pitch": KICK,
                    "velocity": 100,
                    "start_tick": bar_start,
                    "duration_ticks": sixteenth_ticks * 2,
                }
            )
            notes.append(
                {
                    "pitch": KICK,
                    "velocity": 100,
                    "start_tick": bar_start + (bar_ticks // 2),
                    "duration_ticks": sixteenth_ticks * 2,
                }
            )

            # Snare on 2 and 4
            notes.append(
                {
                    "pitch": SNARE,
                    "velocity": 100,
                    "start_tick": bar_start + (bar_ticks // 4),
                    "duration_ticks": sixteenth_ticks * 2,
                }
            )
            notes.append(
                {
                    "pitch": SNARE,
                    "velocity": 100,
                    "start_tick": bar_start + (3 * bar_ticks // 4),
                    "duration_ticks": sixteenth_ticks * 2,
                }
            )

            # Hi-hat on 8th notes
            for i in range(8):
                notes.append(
                    {
                        "pitch": HIHAT,
                        "velocity": 70,
                        "start_tick": bar_start + (i * sixteenth_ticks * 2),
                        "duration_ticks": sixteenth_ticks,
                    }
                )
        else:  # jazz or other
            # Simpler pattern
            notes.append(
                {
                    "pitch": KICK,
                    "velocity": 90,
                    "start_tick": bar_start,
                    "duration_ticks": sixteenth_ticks * 2,
                }
            )
            notes.append(
                {
                    "pitch": SNARE,
                    "velocity": 90,
                    "start_tick": bar_start + (bar_ticks // 2),
                    "duration_ticks": sixteenth_ticks * 2,
                }
            )

    return notes


def generate_arpeggio_pattern(
    chords: List[Any], bars: int, tempo: int, ppq: int, beats_per_bar: int
) -> List[Dict[str, Any]]:
    """
    Generate arpeggiated chord pattern.

    Args:
        chords: List of parsed chord objects
        bars: Number of bars
        tempo: Tempo in BPM
        ppq: Pulses per quarter note
        beats_per_bar: Beats per bar

    Returns:
        List of note dicts for arpeggio track
    """
    notes = []
    bar_ticks = beats_per_bar * ppq
    start_tick = 0
    current_bar = 0

    try:
        from music_brain.structure.chord import CHORD_QUALITIES
    except ImportError:
        CHORD_QUALITIES = {"maj": (0, 4, 7), "min": (0, 3, 7)}

    while current_bar < bars:
        for parsed in chords:
            if current_bar >= bars:
                break

            quality = parsed.quality
            intervals = CHORD_QUALITIES.get(quality)
            if intervals is None:
                base_quality = "min" if "m" in quality else "maj"
                intervals = CHORD_QUALITIES.get(base_quality, (0, 4, 7))

            root_midi = 48 + parsed.root_num  # C3 as base
            eighth_ticks = ppq // 2

            # Arpeggiate chord notes in sequence
            for i, interval in enumerate(intervals):
                notes.append(
                    {
                        "pitch": root_midi + interval,
                        "velocity": 80,
                        "start_tick": start_tick + (i * eighth_ticks),
                        "duration_ticks": eighth_ticks,
                    }
                )

            start_tick += bar_ticks
            current_bar += 1

    return notes


def generate_melody_pattern(
    chords: List[Any], bars: int, tempo: int, ppq: int, beats_per_bar: int, mode: str = "major"
) -> List[Dict[str, Any]]:
    """
    Generate simple melody pattern from chord progression.

    Args:
        chords: List of parsed chord objects
        bars: Number of bars
        tempo: Tempo in BPM
        ppq: Pulses per quarter note
        beats_per_bar: Beats per bar
        mode: Musical mode

    Returns:
        List of note dicts for melody track
    """
    notes = []
    bar_ticks = beats_per_bar * ppq
    start_tick = 0
    current_bar = 0

    try:
        from music_brain.structure.chord import CHORD_QUALITIES
    except ImportError:
        CHORD_QUALITIES = {"maj": (0, 4, 7), "min": (0, 3, 7)}

    while current_bar < bars:
        for parsed in chords:
            if current_bar >= bars:
                break

            quality = parsed.quality
            intervals = CHORD_QUALITIES.get(quality)
            if intervals is None:
                base_quality = "min" if "m" in quality else "maj"
                intervals = CHORD_QUALITIES.get(base_quality, (0, 4, 7))

            root_midi = 60 + parsed.root_num  # C4 as base for melody
            quarter_ticks = ppq

            # Play root, third, or fifth based on bar position
            note_index = current_bar % len(intervals)
            pitch = root_midi + intervals[note_index]

            notes.append(
                {
                    "pitch": pitch,
                    "velocity": 90,
                    "start_tick": start_tick,
                    "duration_ticks": quarter_ticks * 2,  # Half note
                }
            )

            start_tick += bar_ticks
            current_bar += 1

    return notes


# ==============================================================================
# 6. HARMONY -> MIDI BRIDGE (REAL INTEGRATION)
# ==============================================================================


def render_plan_to_midi(
    plan: HarmonyPlan, output_path: str, include_guide_tones: bool = True
) -> str:
    """
    Render a HarmonyPlan to a MIDI file using existing music_brain components:
    - music_brain.structure.progression.parse_progression_string
    - music_brain.structure.chord.CHORD_QUALITIES
    - music_brain.daw.logic.LogicProject

    The progression is looped to fill the entire length_bars specified
    in the plan.

    Args:
        plan: The HarmonyPlan containing all generation parameters
        output_path: Where to write the MIDI file

    Returns:
        Path to the generated MIDI file
    """
    try:
        from music_brain.daw.logic import LOGIC_CHANNELS, LogicProject
        from music_brain.structure.chord import CHORD_QUALITIES
        from music_brain.structure.progression import parse_progression_string
    except ImportError as exc:
        print(f"[SYSTEM]: MIDI bridge unavailable: {exc}")
        print(f"          Chords would have been: {plan.chord_symbols}")
        return output_path

    # 1. Build project
    ts_nums = plan.time_signature.split("/")
    if len(ts_nums) != 2:
        time_sig = (4, 4)
    else:
        try:
            time_sig = (int(ts_nums[0]), int(ts_nums[1]))
        except ValueError:
            time_sig = (4, 4)

    project = LogicProject(
        name="DAiW_Session",
        tempo_bpm=plan.tempo_bpm,
        time_signature=time_sig,
    )
    project.key = f"{plan.root_note} {plan.mode}"

    # 2. Setup
    ppq = getattr(project, "ppq", 480)
    beats_per_bar = time_sig[0]
    bar_ticks = int(beats_per_bar * ppq)

    # 3. Handle structure if provided, otherwise use simple looping
    if plan.structure:
        # Process structure sections
        sections_data = []
        total_bars = 0

        for section in plan.structure:
            section_name = section.get("name", "section")
            section_bars = section.get("bars", 4)
            repetitions = section.get("repetitions", 1)
            section_chords = section.get("chords")

            # Use section-specific chords if provided, otherwise use plan chords
            if section_chords:
                progression_str = "-".join(section_chords)
            else:
                progression_str = "-".join(plan.chord_symbols)

            parsed_chords = parse_progression_string(progression_str)

            for rep in range(repetitions):
                sections_data.append(
                    {
                        "name": f"{section_name}_{rep + 1}" if repetitions > 1 else section_name,
                        "bars": section_bars,
                        "chords": parsed_chords,
                        "start_bar": total_bars,
                    }
                )
                total_bars += section_bars
    else:
        # Fall back to simple looping (backward compatibility)
        progression_str = "-".join(plan.chord_symbols)
        parsed_chords = parse_progression_string(progression_str)
        total_bars = plan.length_bars
        sections_data = [
            {
                "name": "main",
                "bars": total_bars,
                "chords": parsed_chords,
                "start_bar": 0,
            }
        ]

    # 4. Generate tracks based on instruments or default
    if plan.instruments:
        # Multi-track generation with different instruments
        for inst_def in plan.instruments:
            inst_name = inst_def.get("name", "instrument")
            inst_type = inst_def.get("type", "chord")
            inst_channel = inst_def.get("channel")
            inst_technique = inst_def.get("technique") or inst_def.get("pattern", "")
            inst_style = inst_def.get("style", "pop")

            # Determine MIDI channel
            if inst_channel is not None:
                channel = inst_channel
            elif inst_type == "drums":
                channel = LOGIC_CHANNELS.get("drums", 10)
            elif inst_type == "bass":
                channel = LOGIC_CHANNELS.get("bass", 1)
            elif inst_type == "guitar":
                channel = LOGIC_CHANNELS.get("guitar", 3)
            else:
                channel = LOGIC_CHANNELS.get("keys", 2)

            # Generate notes based on instrument type
            all_notes = []
            current_tick = 0

            for section in sections_data:
                section_chords = section["chords"]
                section_bars = section["bars"]

                if inst_type == "bass":
                    section_notes = generate_bass_pattern(
                        section_chords, section_bars, plan.tempo_bpm, ppq, beats_per_bar
                    )
                elif inst_type == "drums":
                    section_notes = generate_drum_pattern(
                        section_bars, plan.tempo_bpm, ppq, beats_per_bar, inst_style
                    )
                elif inst_type == "arpeggio" or (
                    inst_type == "chord" and inst_technique == "arpeggio"
                ):
                    section_notes = generate_arpeggio_pattern(
                        section_chords, section_bars, plan.tempo_bpm, ppq, beats_per_bar
                    )
                elif inst_type == "melody":
                    section_notes = generate_melody_pattern(
                        section_chords, section_bars, plan.tempo_bpm, ppq, beats_per_bar, plan.mode
                    )
                else:  # Default: chord voicings
                    section_notes = []
                    section_start_tick = current_tick
                    section_current_bar = 0

                    while section_current_bar < section_bars:
                        for parsed in section_chords:
                            if section_current_bar >= section_bars:
                                break

                            quality = parsed.quality
                            intervals = CHORD_QUALITIES.get(quality)
                            if intervals is None:
                                base_quality = "min" if "m" in quality else "maj"
                                intervals = CHORD_QUALITIES.get(base_quality, (0, 4, 7))

                            root_midi = 48 + parsed.root_num  # C3 as base
                            duration_ticks = bar_ticks

                            for interval in intervals:
                                section_notes.append(
                                    {
                                        "pitch": root_midi + interval,
                                        "velocity": 80,
                                        "start_tick": section_start_tick,
                                        "duration_ticks": duration_ticks,
                                    }
                                )

                            section_start_tick += duration_ticks
                            section_current_bar += 1

                # Adjust tick offsets for section position
                for note in section_notes:
                    note["start_tick"] += current_tick

                all_notes.extend(section_notes)
                current_tick += section_bars * bar_ticks

            # Add track
            project.add_track(
                name=inst_name,
                channel=channel,
                instrument=None,
                notes=all_notes,
            )
    else:
        # Default: single harmony track (backward compatibility)
        all_notes = []
        current_tick = 0

        for section in sections_data:
            section_chords = section["chords"]
            section_bars = section["bars"]
            section_start_tick = current_tick
            section_current_bar = 0

            while section_current_bar < section_bars:
                for parsed in section_chords:
                    if section_current_bar >= section_bars:
                        break

                    quality = parsed.quality
                    intervals = CHORD_QUALITIES.get(quality)
                    if intervals is None:
                        base_quality = "min" if "m" in quality else "maj"
                        intervals = CHORD_QUALITIES.get(base_quality, (0, 4, 7))

                    root_midi = 48 + parsed.root_num  # C3 as base
                    duration_ticks = bar_ticks

                    for interval in intervals:
                        all_notes.append(
                            {
                                "pitch": root_midi + interval,
                                "velocity": 80,
                                "start_tick": section_start_tick,
                                "duration_ticks": duration_ticks,
                            }
                        )

                    section_start_tick += duration_ticks
                    section_current_bar += 1

            current_tick += section_bars * bar_ticks

        channel = LOGIC_CHANNELS.get("keys", 2)
        project.add_track(
            name="Harmony",
            channel=channel,
            instrument=None,
            notes=all_notes,
        )

        if include_guide_tones:
            project.add_track(
                name="Guide Tones",
                channel=channel,
                instrument=None,
                notes=all_notes[:1] if all_notes else [],
            )

    try:
        midi_path = project.export_midi(output_path)
    except ImportError:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"")
        midi_path = output_path
    print(f"[SYSTEM]: MIDI written to {midi_path}")
    return midi_path


# ==============================================================================
# 6. CLI HANDLER (The "View" Layer)
# ==============================================================================


def run_cli():
    """
    Interactive command-line interface for the Therapy Engine.

    Follows the DAiW philosophy: "Interrogate Before Generate"

    Flow:
    1. Ask what's hurting
    2. Analyze and reflect back
    3. Get scaling parameters (motivation, chaos)
    4. Inject strategy if chaos is high
    5. Generate and export plan
    """
    session = TherapySession()
    print("--- DAiW THERAPY TERMINAL ---")

    # 1. Input Loop
    while True:
        text = input("[THERAPIST]: What is hurting you? >> ").strip()
        if text:
            break
        print("[THERAPIST]: Silence is an answer, but I need words to build structure.")

    # 2. Process
    affect = session.process_core_input(text)

    # 3. Reflect (Mirroring)
    if session.state.affect_result:
        print(
            f"\n[ANALYSIS]: Detected affect '{affect}' with intensity {session.state.affect_result.intensity:.2f}"
        )
        if session.state.affect_result.secondary:
            print(f"[ANALYSIS]: Underlying undertone: '{session.state.affect_result.secondary}'")

    # 4. Scaling
    try:
        mot = int(input("\n[THERAPIST]: Motivation (1-10)? >> "))
        chaos_in = int(input("[THERAPIST]: Tolerance for Chaos (1-10)? >> "))
        session.set_scales(mot, chaos_in / 10.0)
    except ValueError:
        print("[SYSTEM]: Invalid input. Defaulting to safe values.")
        session.set_scales(5, 0.3)

    # 5. Strategy Injection
    if session.state.chaos_tolerance > 0.6:
        strat = get_strategy(session.state.chaos_tolerance)
        print(f"\n[OBLIQUE STRATEGY]: {strat}")

    # 6. Plan Generation
    plan = session.generate_plan()

    # 7. Summary
    print("\n" + "=" * 40)
    print("GENERATION DIRECTIVE")
    print("=" * 40)
    print(f"Target Mode: {plan.root_note} {plan.mode}")
    print(f"Tempo: {plan.tempo_bpm} BPM")
    print(f"Length: {plan.length_bars} bars")
    print(f"Progression: {' - '.join(plan.chord_symbols)}")
    print(f"Complexity: {plan.complexity:.2f}")

    # 8. MIDI Export
    output_path = "daiw_therapy_session.mid"
    render_plan_to_midi(plan, output_path)


if __name__ == "__main__":
    run_cli()
