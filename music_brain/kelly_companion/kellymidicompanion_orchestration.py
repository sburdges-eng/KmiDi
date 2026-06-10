"""
Multi-Track Orchestration System

Coordinates multiple Kelly instances in the same DAW project:
- Tracks complement each other (no frequency/role collisions)
- Handles same-channel scenarios (layering vs. replacing)
- Time-interval annotations for coordination
- Instrument role assignment with awareness
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum, auto
import json
import uuid
from datetime import datetime

try:
    import mido

    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================


class TrackRole(Enum):
    """Musical role a track plays in arrangement."""

    LEAD_MELODY = auto()  # Primary melodic line
    COUNTER_MELODY = auto()  # Secondary melodic line
    HARMONY_PAD = auto()  # Sustained harmonic support
    CHORD_RHYTHM = auto()  # Rhythmic chord pattern
    BASS = auto()  # Low-frequency foundation
    DRUMS = auto()  # Percussion (Channel 9)
    TEXTURE = auto()  # Ambient/fill elements
    ACCENT = auto()  # Punctuation hits
    ARPEGGIATED = auto()  # Broken chord patterns
    DRONE = auto()  # Sustained single note


class FrequencyRange(Enum):
    """Frequency bands for collision avoidance."""

    SUB_BASS = (20, 60)  # 20-60 Hz
    BASS = (60, 250)  # 60-250 Hz
    LOW_MID = (250, 500)  # 250-500 Hz
    MID = (500, 2000)  # 500-2000 Hz
    HIGH_MID = (2000, 4000)  # 2-4 kHz
    PRESENCE = (4000, 8000)  # 4-8 kHz
    BRILLIANCE = (8000, 20000)  # 8-20 kHz


class ChannelMode(Enum):
    """How to handle same-channel conflicts."""

    LAYER = auto()  # Both tracks play simultaneously
    REPLACE = auto()  # New track replaces old in overlap
    ALTERNATE = auto()  # Tracks take turns
    DUCK = auto()  # One ducks under the other
    MERGE = auto()  # Combine into single track


# Role-to-frequency mappings
ROLE_FREQUENCIES = {
    TrackRole.BASS: [FrequencyRange.SUB_BASS, FrequencyRange.BASS],
    TrackRole.DRUMS: [FrequencyRange.SUB_BASS, FrequencyRange.BASS, FrequencyRange.HIGH_MID],
    TrackRole.LEAD_MELODY: [FrequencyRange.MID, FrequencyRange.HIGH_MID],
    TrackRole.COUNTER_MELODY: [FrequencyRange.LOW_MID, FrequencyRange.MID],
    TrackRole.HARMONY_PAD: [FrequencyRange.LOW_MID, FrequencyRange.MID],
    TrackRole.CHORD_RHYTHM: [FrequencyRange.LOW_MID, FrequencyRange.MID],
    TrackRole.TEXTURE: [FrequencyRange.HIGH_MID, FrequencyRange.PRESENCE],
    TrackRole.ARPEGGIATED: [FrequencyRange.MID, FrequencyRange.HIGH_MID],
    TrackRole.ACCENT: [FrequencyRange.MID, FrequencyRange.PRESENCE],
    TrackRole.DRONE: [FrequencyRange.BASS, FrequencyRange.LOW_MID],
}

# Complementary role pairs
COMPLEMENTARY_ROLES = {
    TrackRole.LEAD_MELODY: [TrackRole.HARMONY_PAD, TrackRole.BASS, TrackRole.ARPEGGIATED],
    TrackRole.BASS: [TrackRole.LEAD_MELODY, TrackRole.CHORD_RHYTHM, TrackRole.DRUMS],
    TrackRole.CHORD_RHYTHM: [TrackRole.BASS, TrackRole.LEAD_MELODY, TrackRole.DRUMS],
    TrackRole.DRUMS: [TrackRole.BASS, TrackRole.CHORD_RHYTHM, TrackRole.LEAD_MELODY],
    TrackRole.HARMONY_PAD: [TrackRole.LEAD_MELODY, TrackRole.ARPEGGIATED],
    TrackRole.TEXTURE: [TrackRole.LEAD_MELODY, TrackRole.HARMONY_PAD],
    TrackRole.ARPEGGIATED: [TrackRole.BASS, TrackRole.HARMONY_PAD, TrackRole.DRUMS],
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class TimeInterval:
    """Represents a time range in the arrangement."""

    start_beat: float  # Start position in beats
    end_beat: float  # End position in beats
    start_bar: int = 0  # Bar number (derived)
    end_bar: int = 0  # Bar number (derived)
    beats_per_bar: int = 4  # Time signature

    def __post_init__(self):
        self.start_bar = int(self.start_beat // self.beats_per_bar) + 1
        self.end_bar = int(self.end_beat // self.beats_per_bar) + 1

    def overlaps(self, other: "TimeInterval") -> bool:
        """Check if intervals overlap."""
        return not (self.end_beat <= other.start_beat or other.end_beat <= self.start_beat)

    def duration_beats(self) -> float:
        return self.end_beat - self.start_beat

    def duration_bars(self) -> float:
        return self.duration_beats() / self.beats_per_bar


@dataclass
class TrackAnnotation:
    """Annotation for a specific time interval on a track."""

    interval: TimeInterval
    annotation_type: str  # "section", "dynamic", "technique", "emotion"
    value: str  # e.g., "verse", "crescendo", "palm_mute", "grief"
    intensity: float = 0.5  # 0.0-1.0 intensity modifier
    notes: str = ""  # Additional notes


@dataclass
class KellyTrack:
    """A single track managed by Kelly."""

    track_id: str
    name: str
    midi_channel: int
    role: TrackRole
    instrument_program: int  # MIDI program 0-127
    emotion: str
    vulnerability: float

    # Time-based content
    intervals: List[TimeInterval] = field(default_factory=list)
    annotations: List[TrackAnnotation] = field(default_factory=list)

    # Coordination
    frequency_ranges: List[FrequencyRange] = field(default_factory=list)
    velocity_range: Tuple[int, int] = (60, 100)
    octave_range: Tuple[int, int] = (3, 5)

    # Flags
    is_active: bool = True
    is_leader: bool = False  # Leader track others follow

    def __post_init__(self):
        if not self.track_id:
            self.track_id = str(uuid.uuid4())[:8]
        if not self.frequency_ranges:
            self.frequency_ranges = ROLE_FREQUENCIES.get(self.role, [FrequencyRange.MID])

    def get_annotations_at(self, beat: float) -> List[TrackAnnotation]:
        """Get all annotations active at a specific beat."""
        return [a for a in self.annotations if a.interval.start_beat <= beat < a.interval.end_beat]

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "name": self.name,
            "midi_channel": self.midi_channel,
            "role": self.role.name,
            "instrument_program": self.instrument_program,
            "emotion": self.emotion,
            "vulnerability": self.vulnerability,
            "intervals": [(i.start_beat, i.end_beat) for i in self.intervals],
            "is_active": self.is_active,
            "is_leader": self.is_leader,
        }


@dataclass
class OrchestrationSession:
    """A session containing multiple coordinated Kelly tracks."""

    session_id: str
    name: str
    tempo_bpm: int
    time_signature: Tuple[int, int]  # (numerator, denominator)
    key: str
    mode: str

    tracks: Dict[str, KellyTrack] = field(default_factory=dict)
    global_emotion: str = "neutral"

    # Coordination state
    assigned_channels: Set[int] = field(default_factory=set)
    assigned_roles: Dict[TrackRole, str] = field(default_factory=dict)  # role -> track_id

    created_at: str = ""

    def __post_init__(self):
        if not self.session_id:
            self.session_id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        # Channel 9 always reserved for drums
        self.assigned_channels.add(9)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "tempo_bpm": self.tempo_bpm,
            "time_signature": list(self.time_signature),
            "key": self.key,
            "mode": self.mode,
            "global_emotion": self.global_emotion,
            "tracks": {tid: t.to_dict() for tid, t in self.tracks.items()},
            "assigned_channels": list(self.assigned_channels),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "OrchestrationSession":
        session = cls(
            session_id=data["session_id"],
            name=data["name"],
            tempo_bpm=data["tempo_bpm"],
            time_signature=tuple(data["time_signature"]),
            key=data["key"],
            mode=data["mode"],
        )
        session.global_emotion = data.get("global_emotion", "neutral")
        session.assigned_channels = set(data.get("assigned_channels", [9]))
        session.created_at = data.get("created_at", "")
        return session


# =============================================================================
# ORCHESTRATION ENGINE
# =============================================================================


class OrchestrationEngine:
    """
    Coordinates multiple Kelly tracks in a single project.

    Responsibilities:
    - Assign non-colliding MIDI channels
    - Ensure complementary instrument roles
    - Coordinate frequency ranges
    - Handle same-channel conflicts
    - Manage time-based annotations
    """

    def __init__(self, session: Optional[OrchestrationSession] = None):
        self.session = session or OrchestrationSession(
            session_id="",
            name="Untitled Session",
            tempo_bpm=120,
            time_signature=(4, 4),
            key="C",
            mode="major",
        )
        self._collision_warnings: List[str] = []

    # -------------------------------------------------------------------------
    # Channel Management
    # -------------------------------------------------------------------------

    def get_next_available_channel(self, prefer_channel: Optional[int] = None) -> int:
        """Get next available MIDI channel (0-15, excluding 9 for drums)."""
        if prefer_channel is not None and prefer_channel not in self.session.assigned_channels:
            if 0 <= prefer_channel <= 15 and prefer_channel != 9:
                return prefer_channel

        for ch in range(16):
            if ch not in self.session.assigned_channels and ch != 9:
                return ch

        # All channels full - return 0 for layering
        return 0

    def assign_channel(self, track_id: str, channel: int) -> bool:
        """Assign a channel to a track, returns True if successful."""
        if channel == 9:
            # Only drums can use channel 9
            track = self.session.tracks.get(track_id)
            if track and track.role != TrackRole.DRUMS:
                return False

        if channel in self.session.assigned_channels:
            track = self.session.tracks.get(track_id)
            if track:
                track.midi_channel = channel
                return True  # Allow layering

        self.session.assigned_channels.add(channel)
        track = self.session.tracks.get(track_id)
        if track:
            track.midi_channel = channel
        return True

    # -------------------------------------------------------------------------
    # Track Registration
    # -------------------------------------------------------------------------

    def register_track(
        self,
        name: str,
        role: TrackRole,
        emotion: str,
        instrument_program: int,
        vulnerability: float = 0.5,
        preferred_channel: Optional[int] = None,
        intervals: Optional[List[Tuple[float, float]]] = None,
    ) -> KellyTrack:
        """
        Register a new Kelly track in the session.

        Returns the created track with assigned channel and coordination metadata.
        """
        # Check for role conflicts
        role_suggestion = self._suggest_role_if_conflict(role)
        if role_suggestion != role:
            self._collision_warnings.append(
                f"Role {role.name} already assigned. Suggested: {role_suggestion.name}"
            )

        # Get channel
        channel = self.get_next_available_channel(preferred_channel)

        # Handle drums specially
        if role == TrackRole.DRUMS:
            channel = 9

        # Create track
        track = KellyTrack(
            track_id="",
            name=name,
            midi_channel=channel,
            role=role,
            instrument_program=instrument_program,
            emotion=emotion,
            vulnerability=vulnerability,
        )

        # Add intervals
        if intervals:
            track.intervals = [TimeInterval(s, e) for s, e in intervals]

        # Register
        self.session.tracks[track.track_id] = track
        self.session.assigned_channels.add(channel)
        self.session.assigned_roles[role] = track.track_id

        # Check complementary arrangement
        self._ensure_complementary(track)

        return track

    def _suggest_role_if_conflict(self, role: TrackRole) -> TrackRole:
        """Suggest alternative role if requested role is taken."""
        if role not in self.session.assigned_roles:
            return role

        # Find complementary alternative
        complements = COMPLEMENTARY_ROLES.get(role, [])
        for alt_role in complements:
            if alt_role not in self.session.assigned_roles:
                return alt_role

        # Fall back to texture/accent
        for fallback in [TrackRole.TEXTURE, TrackRole.ACCENT, TrackRole.COUNTER_MELODY]:
            if fallback not in self.session.assigned_roles:
                return fallback

        return role  # Allow duplicate if no alternatives

    def _ensure_complementary(self, new_track: KellyTrack):
        """Check that new track complements existing tracks."""
        new_freqs = set(new_track.frequency_ranges)

        for tid, existing in self.session.tracks.items():
            if tid == new_track.track_id:
                continue

            existing_freqs = set(existing.frequency_ranges)
            overlap = new_freqs & existing_freqs

            if overlap and existing.role != TrackRole.DRUMS:
                # Frequency collision - suggest adjustment
                self._collision_warnings.append(
                    f"Track '{new_track.name}' overlaps frequency with '{existing.name}' "
                    f"in {[f.name for f in overlap]}. Consider EQ separation."
                )

    # -------------------------------------------------------------------------
    # Same-Channel Handling
    # -------------------------------------------------------------------------

    def handle_channel_conflict(
        self,
        track_a_id: str,
        track_b_id: str,
        mode: ChannelMode = ChannelMode.LAYER,
    ) -> Dict:
        """
        Handle when two tracks share the same MIDI channel.

        Returns instructions for how to resolve.
        """
        track_a = self.session.tracks.get(track_a_id)
        track_b = self.session.tracks.get(track_b_id)

        if not track_a or not track_b:
            return {"error": "Track not found"}

        if track_a.midi_channel != track_b.midi_channel:
            return {
                "status": "no_conflict",
                "channels": [track_a.midi_channel, track_b.midi_channel],
            }  # noqa: E501

        result = {
            "mode": mode.name,
            "channel": track_a.midi_channel,
            "tracks": [track_a.name, track_b.name],
            "instructions": [],
        }

        if mode == ChannelMode.LAYER:
            result["instructions"] = [
                "Both tracks play on same channel",
                "Adjust velocities: track A at 70%, track B at 60%",
                "Consider panning: track A left, track B right",
                f"Octave separation: {track_a.name} stays, {track_b.name} +/- 1 octave",
            ]
            # Adjust velocity ranges
            track_a.velocity_range = (50, 90)
            track_b.velocity_range = (40, 80)

        elif mode == ChannelMode.REPLACE:
            result["instructions"] = [
                f"Track '{track_b.name}' replaces '{track_a.name}' during overlaps",
                "Non-overlapping sections play independently",
            ]

        elif mode == ChannelMode.ALTERNATE:
            result["instructions"] = [
                "Tracks alternate: A plays odd bars, B plays even bars",
                "Creates call-and-response pattern",
            ]

        elif mode == ChannelMode.DUCK:
            result["instructions"] = [
                f"'{track_b.name}' ducks under '{track_a.name}'",
                "Reduce velocity by 30% during overlap",
                "Consider sidechain compression in DAW",
            ]
            track_b.velocity_range = (30, 70)

        elif mode == ChannelMode.MERGE:
            result["instructions"] = [
                "Combine both tracks into single MIDI track",
                "Preserve all notes, remove duplicates at same time",
                "Apply unified groove template",
            ]

        return result

    # -------------------------------------------------------------------------
    # Interval & Annotation Management
    # -------------------------------------------------------------------------

    def add_interval(
        self,
        track_id: str,
        start_beat: float,
        end_beat: float,
        annotation: Optional[TrackAnnotation] = None,
    ) -> bool:
        """Add a time interval to a track with optional annotation."""
        track = self.session.tracks.get(track_id)
        if not track:
            return False

        interval = TimeInterval(start_beat, end_beat, beats_per_bar=self.session.time_signature[0])
        track.intervals.append(interval)

        if annotation:
            annotation.interval = interval
            track.annotations.append(annotation)

        return True

    def annotate_interval(
        self,
        track_id: str,
        start_beat: float,
        end_beat: float,
        annotation_type: str,
        value: str,
        intensity: float = 0.5,
        notes: str = "",
    ) -> bool:
        """Add annotation to a specific interval."""
        track = self.session.tracks.get(track_id)
        if not track:
            return False

        interval = TimeInterval(start_beat, end_beat, beats_per_bar=self.session.time_signature[0])
        annotation = TrackAnnotation(
            interval=interval,
            annotation_type=annotation_type,
            value=value,
            intensity=intensity,
            notes=notes,
        )
        track.annotations.append(annotation)
        return True

    def get_section_annotations(self, beat: float) -> Dict[str, List[TrackAnnotation]]:
        """Get all annotations active at a specific beat, organized by track."""
        result = {}
        for tid, track in self.session.tracks.items():
            annotations = track.get_annotations_at(beat)
            if annotations:
                result[track.name] = annotations
        return result

    # -------------------------------------------------------------------------
    # Complementary Analysis
    # -------------------------------------------------------------------------

    def analyze_arrangement(self) -> Dict:
        """Analyze current arrangement for balance and suggestions."""
        analysis = {
            "track_count": len(self.session.tracks),
            "roles_filled": [r.name for r in self.session.assigned_roles.keys()],
            "roles_missing": [],
            "frequency_balance": {},
            "channel_usage": {},
            "suggestions": [],
            "warnings": self._collision_warnings.copy(),
        }

        # Check missing essential roles
        essential_roles = [TrackRole.LEAD_MELODY, TrackRole.BASS, TrackRole.CHORD_RHYTHM]
        for role in essential_roles:
            if role not in self.session.assigned_roles:
                analysis["roles_missing"].append(role.name)
                analysis["suggestions"].append(f"Consider adding a {role.name} track")

        # Analyze frequency distribution
        freq_counts = {fr: 0 for fr in FrequencyRange}
        for track in self.session.tracks.values():
            for freq in track.frequency_ranges:
                freq_counts[freq] += 1

        analysis["frequency_balance"] = {fr.name: count for fr, count in freq_counts.items()}

        # Check for frequency crowding
        for freq, count in freq_counts.items():
            if count > 2 and freq not in [FrequencyRange.SUB_BASS]:
                analysis["suggestions"].append(
                    f"Frequency range {freq.name} has {count} tracks - consider EQ separation"
                )

        # Channel usage
        channel_tracks = {}
        for track in self.session.tracks.values():
            ch = track.midi_channel
            if ch not in channel_tracks:
                channel_tracks[ch] = []
            channel_tracks[ch].append(track.name)

        analysis["channel_usage"] = channel_tracks

        for ch, tracks in channel_tracks.items():
            if len(tracks) > 1:
                analysis["warnings"].append(
                    f"Channel {ch} has multiple tracks: {tracks}. Use handle_channel_conflict()"
                )

        return analysis

    def suggest_complementary_instrument(
        self,
        existing_emotion: str,
        existing_program: int,
        target_role: TrackRole,
    ) -> Dict:
        """Suggest instrument that complements existing tracks."""
        # Instrument groups that complement each other
        COMPLEMENTARY_INSTRUMENTS = {
            # Piano leads
            0: [40, 48, 32, 73],  # Piano -> Violin, Strings, Acoustic Bass, Flute
            # Guitar leads
            24: [40, 48, 33, 73],  # Nylon Guitar -> Violin, Strings, Electric Bass, Flute
            25: [33, 48, 38],  # Steel Guitar -> Electric Bass, Strings, Synth Bass
            # String leads
            40: [0, 24, 32],  # Violin -> Piano, Guitar, Acoustic Bass
            # Flute leads
            73: [0, 40, 48],  # Flute -> Piano, Violin, Strings
        }

        suggestions = COMPLEMENTARY_INSTRUMENTS.get(existing_program, [0, 48, 33])

        # Filter by role
        role_preferred = {
            TrackRole.BASS: [32, 33, 38, 39],  # Acoustic/Electric/Synth Bass
            TrackRole.HARMONY_PAD: [48, 49, 89],  # Strings, Pad
            TrackRole.LEAD_MELODY: [73, 40, 68],  # Flute, Violin, Oboe
            TrackRole.CHORD_RHYTHM: [24, 25, 4],  # Guitars, E.Piano
            TrackRole.TEXTURE: [89, 91, 95],  # Pads, Atmosphere
        }

        preferred = role_preferred.get(target_role, suggestions)

        return {
            "suggested_programs": preferred,
            "target_role": target_role.name,
            "reason": f"Complements program {existing_program} for {target_role.name}",
        }

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def save_session(self, filepath: str):
        """Save session to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.session.to_dict(), f, indent=2)

    def load_session(self, filepath: str):
        """Load session from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.session = OrchestrationSession.from_dict(data)

        # Rebuild assigned_roles
        self.session.assigned_roles = {}
        for tid, track_data in data.get("tracks", {}).items():
            role = TrackRole[track_data["role"]]
            self.session.assigned_roles[role] = tid

    def get_warnings(self) -> List[str]:
        """Get accumulated collision warnings."""
        return self._collision_warnings.copy()

    def clear_warnings(self):
        """Clear accumulated warnings."""
        self._collision_warnings = []


# =============================================================================
# MIDI OUTPUT COORDINATION
# =============================================================================


class MultiTrackMIDIWriter:
    """Write coordinated multi-track MIDI output."""

    def __init__(self, engine: OrchestrationEngine):
        if not MIDO_AVAILABLE:
            raise ImportError("mido package required")
        self.engine = engine

    def create_coordinated_midi(
        self,
        # track_id -> [(note, velocity, start_tick, duration_tick)]
        track_data: Dict[str, List[Tuple[int, int, int, int]]],
        output_path: str,
        ticks_per_beat: int = 480,
    ) -> str:
        """
        Create a single MIDI file with all coordinated tracks.

        Args:
            track_data: Dict mapping track_id to list of (note, velocity, start_tick, duration_tick)
            output_path: Output file path
            ticks_per_beat: MIDI resolution
        """
        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        session = self.engine.session

        # Track 0: Tempo and time signature
        meta_track = mido.MidiTrack()
        mid.tracks.append(meta_track)

        tempo = mido.bpm2tempo(session.tempo_bpm)
        meta_track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
        meta_track.append(
            mido.MetaMessage(
                "time_signature",
                numerator=session.time_signature[0],
                denominator=session.time_signature[1],
                time=0,
            )
        )

        # Create track for each registered Kelly track
        for track_id, notes in track_data.items():
            kelly_track = session.tracks.get(track_id)
            if not kelly_track:
                continue

            midi_track = mido.MidiTrack()
            mid.tracks.append(midi_track)

            # Track name
            midi_track.append(mido.MetaMessage("track_name", name=kelly_track.name, time=0))

            # Program change
            midi_track.append(
                mido.Message(
                    "program_change",
                    channel=kelly_track.midi_channel,
                    program=kelly_track.instrument_program,
                    time=0,
                )
            )

            # Build note events
            events = []
            for note, velocity, start_tick, duration in notes:
                # Clamp velocity to track's range
                v_min, v_max = kelly_track.velocity_range
                velocity = max(v_min, min(v_max, velocity))

                events.append((start_tick, "note_on", note, velocity))
                events.append((start_tick + duration, "note_off", note, 0))

            # Sort by time
            events.sort(key=lambda x: x[0])

            # Convert to delta time
            current_time = 0
            for abs_time, msg_type, note, vel in events:
                delta = abs_time - current_time
                midi_track.append(
                    mido.Message(
                        msg_type,
                        channel=kelly_track.midi_channel,
                        note=note,
                        velocity=vel,
                        time=delta,
                    )
                )
                current_time = abs_time

        mid.save(output_path)
        return output_path


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_session(
    name: str,
    tempo: int = 120,
    key: str = "C",
    mode: str = "major",
    time_sig: Tuple[int, int] = (4, 4),
) -> OrchestrationEngine:
    """Create a new orchestration session."""
    session = OrchestrationSession(
        session_id="",
        name=name,
        tempo_bpm=tempo,
        time_signature=time_sig,
        key=key,
        mode=mode,
    )
    return OrchestrationEngine(session)


def quick_register_tracks(
    engine: OrchestrationEngine,
    track_configs: List[Dict],
) -> List[KellyTrack]:
    """
    Quickly register multiple tracks.

    track_configs: List of dicts with keys:
        - name: str
        - role: str (TrackRole name)
        - emotion: str
        - program: int (MIDI program)
        - intervals: Optional[List[Tuple[float, float]]]
    """
    tracks = []
    for cfg in track_configs:
        role = TrackRole[cfg["role"]]
        track = engine.register_track(
            name=cfg["name"],
            role=role,
            emotion=cfg.get("emotion", "neutral"),
            instrument_program=cfg.get("program", 0),
            vulnerability=cfg.get("vulnerability", 0.5),
            intervals=cfg.get("intervals"),
        )
        tracks.append(track)
    return tracks


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Create session for "When I Found You Sleeping"
    engine = create_session(
        name="When I Found You Sleeping",
        tempo=82,
        key="F",
        mode="major",
        time_sig=(4, 4),
    )

    # Register multiple tracks
    tracks = quick_register_tracks(
        engine,
        [
            {
                "name": "Piano (Grief)",
                "role": "LEAD_MELODY",
                "emotion": "grief",
                "program": 0,  # Acoustic Grand Piano
                "vulnerability": 0.9,
                "intervals": [(0, 64)],  # 16 bars
            },
            {
                "name": "Strings (Swell)",
                "role": "HARMONY_PAD",
                "emotion": "grief",
                "program": 48,  # String Ensemble
                "vulnerability": 0.85,
                "intervals": [(16, 64)],  # Enter at bar 5
            },
            {
                "name": "Bass (Foundation)",
                "role": "BASS",
                "emotion": "grief",
                "program": 33,  # Electric Bass Finger
                "vulnerability": 0.6,
                "intervals": [(0, 64)],
            },
        ],
    )

    # Add annotations
    engine.annotate_interval(
        tracks[0].track_id,
        start_beat=0,
        end_beat=16,
        annotation_type="section",
        value="verse_1",
        intensity=0.5,
        notes="Misdirection setup - sounds like falling in love",
    )

    engine.annotate_interval(
        tracks[0].track_id,
        start_beat=48,
        end_beat=64,
        annotation_type="dynamic",
        value="crescendo",
        intensity=0.9,
        notes="Build to reveal",
    )

    # Analyze
    analysis = engine.analyze_arrangement()
    print("\n=== Arrangement Analysis ===")
    print(f"Tracks: {analysis['track_count']}")
    print(f"Roles filled: {analysis['roles_filled']}")
    print(f"Missing: {analysis['roles_missing']}")
    print(f"Warnings: {analysis['warnings']}")
    print(f"Suggestions: {analysis['suggestions']}")

    # Test same-channel handling
    # Force both piano and strings to channel 0 for demo
    tracks[0].midi_channel = 0
    tracks[1].midi_channel = 0

    conflict = engine.handle_channel_conflict(
        tracks[0].track_id,
        tracks[1].track_id,
        mode=ChannelMode.LAYER,
    )
    print(f"\nChannel conflict resolution: {conflict['mode']}")
    for instruction in conflict["instructions"]:
        print(f"  - {instruction}")
