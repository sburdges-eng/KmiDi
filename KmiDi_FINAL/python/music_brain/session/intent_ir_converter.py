"""
Intent IR Converter - Convert CompleteSongIntent to IntentFrame

Maps high-level songwriting intent to canonical Intent IR v1 format.
"""

from typing import Optional
from .intent_ir import (
    IntentFrame,
    IntentMeta,
    EmotionState,
    MusicalIntent,
    TimeScope,
    IntentConstraints,
    IntentProvenance,
    IntentSource,
)

# Import CompleteSongIntent from the project location
# Try to import from both possible locations
try:
    from music_brain.session.intent_schema import CompleteSongIntent, AFFECT_MAPPINGS
except ImportError:
    try:
        import sys
        import os

        # Add project path if needed
        project_path = os.path.join(
            os.path.dirname(__file__), "../../..", "KmiDi_PROJECT", "source", "python"
        )
        if os.path.exists(project_path):
            sys.path.insert(0, project_path)
        from music_brain.session.intent_schema import CompleteSongIntent, AFFECT_MAPPINGS
    except ImportError:
        # Fallback: define minimal types if import fails
        CompleteSongIntent = None
        AFFECT_MAPPINGS = {}


def mood_to_vad(mood_primary: str) -> tuple[float, float, float]:
    """
    Map mood_primary string to VAD (Valence, Arousal, Dominance) coordinates.

    Uses heuristics based on emotion research and the AFFECT_MAPPINGS.
    """
    mood_lower = mood_primary.lower() if mood_primary else ""

    # Valence mapping (negative to positive)
    negative_valence = [
        "grief",
        "sadness",
        "melancholy",
        "desperation",
        "rage",
        "anger",
        "nervousness",
        "fear",
        "confusion",
        "dissociation",
    ]
    positive_valence = [
        "joy",
        "euphoria",
        "triumphant hope",
        "serenity",
        "liberation",
        "acceptance",
        "determination",
    ]

    if any(m in mood_lower for m in negative_valence):
        valence = -0.5
    elif any(m in mood_lower for m in positive_valence):
        valence = 0.5
    else:
        valence = 0.0  # neutral

    # Arousal mapping (low to high)
    high_arousal = [
        "rage",
        "anger",
        "euphoria",
        "triumphant hope",
        "defiance",
        "determination",
        "nervousness",
        "desperation",
    ]
    low_arousal = ["grief", "sadness", "melancholy", "serenity", "acceptance", "dissociation"]

    if any(m in mood_lower for m in high_arousal):
        arousal = 0.8
    elif any(m in mood_lower for m in low_arousal):
        arousal = 0.2
    else:
        arousal = 0.5  # moderate

    # Dominance mapping (submissive to dominant)
    high_dominance = ["defiance", "determination", "triumphant hope", "rage", "anger"]
    low_dominance = ["grief", "sadness", "nervousness", "fear", "dissociation", "confusion"]

    if any(m in mood_lower for m in high_dominance):
        dominance = 0.8
    elif any(m in mood_lower for m in low_dominance):
        dominance = 0.2
    else:
        dominance = 0.5  # moderate

    return (valence, arousal, dominance)


def tempo_range_to_bias(tempo_range: tuple[int, int]) -> float:
    """
    Convert tempo range to tempo_bias (-1.0 to +1.0).

    Assumes typical tempo range is 60-180 BPM.
    """
    if not tempo_range or len(tempo_range) < 2:
        return 0.0

    min_tempo, max_tempo = tempo_range
    avg_tempo = (min_tempo + max_tempo) / 2.0

    # Normalize: 60 BPM = -1.0, 180 BPM = +1.0
    # Center at 120 BPM = 0.0
    normalized = (avg_tempo - 120.0) / 60.0
    return max(-1.0, min(1.0, normalized))


def mode_to_preference(mode: str) -> int:
    """
    Convert mode string to mode_preference (-1, 0, +1).

    -1 = minor, 0 = neutral, +1 = major
    """
    mode_lower = mode.lower() if mode else ""

    minor_modes = ["minor", "aeolian", "dorian", "phrygian", "locrian"]
    major_modes = ["major", "ionian", "lydian", "mixolydian"]

    if any(m in mode_lower for m in minor_modes):
        return -1
    elif any(m in mode_lower for m in major_modes):
        return 1
    else:
        return 0


def groove_feel_to_density(groove_feel: str) -> float:
    """
    Convert groove_feel to rhythmic_density and groove_strength.
    Returns (rhythmic_density, groove_strength).
    """
    groove_lower = groove_feel.lower() if groove_feel else ""

    # Rhythmic density mapping
    dense_feels = ["straight/driving", "syncopated", "mechanical", "push-pull"]
    sparse_feels = ["laid back", "rubato/free", "organic/breathing"]

    if any(f in groove_lower for f in dense_feels):
        density = 0.8
    elif any(f in groove_lower for f in sparse_feels):
        density = 0.2
    else:
        density = 0.5

    # Groove strength mapping
    tight_feels = ["straight/driving", "mechanical", "swung"]
    loose_feels = ["rubato/free", "organic/breathing", "laid back"]

    if any(f in groove_lower for f in tight_feels):
        strength = 0.8
    elif any(f in groove_lower for f in loose_feels):
        strength = 0.2
    else:
        strength = 0.5

    return (density, strength)


def vulnerability_to_confidence(vulnerability_scale: str) -> float:
    """
    Map vulnerability scale to confidence.
    Higher vulnerability = lower confidence (more uncertainty).
    """
    vuln_lower = vulnerability_scale.lower() if vulnerability_scale else "medium"

    if "high" in vuln_lower:
        return 0.6  # Lower confidence for high vulnerability
    elif "low" in vuln_lower:
        return 0.9  # Higher confidence for low vulnerability
    else:
        return 0.75  # Medium confidence


def convert_complete_song_intent_to_ir(
    complete_intent: "CompleteSongIntent",
    session_id: int = 0,
    source: IntentSource = IntentSource.ML_TEXT,
) -> IntentFrame:
    """
    Convert CompleteSongIntent to IntentFrame.

    Args:
        complete_intent: The CompleteSongIntent to convert
        session_id: Session identifier
        source: Intent source (default ML_TEXT)

    Returns:
        IntentFrame with converted values
    """
    if complete_intent is None:
        # Return default frame
        frame = IntentFrame()
        frame.meta.session_id = session_id
        frame.provenance.source = source
        return frame

    # Extract emotion from mood_primary
    mood_primary = complete_intent.song_intent.mood_primary if complete_intent.song_intent else ""
    valence, arousal, dominance = mood_to_vad(mood_primary)

    # Get confidence from vulnerability
    vulnerability = (
        complete_intent.song_intent.vulnerability_scale if complete_intent.song_intent else "Medium"
    )
    confidence = vulnerability_to_confidence(vulnerability)

    # Extract tension for intensity
    tension = (
        complete_intent.song_intent.mood_secondary_tension if complete_intent.song_intent else 0.5
    )

    # Create EmotionState
    emotion = EmotionState(
        valence=valence,
        arousal=arousal,
        dominance=dominance,
        discrete_id=-1,  # Not mapped from CompleteSongIntent
        intensity=tension,
        confidence=confidence,
    )

    # Extract musical intent from technical constraints
    tech = complete_intent.technical_constraints if complete_intent.technical_constraints else None

    # Tempo bias
    tempo_range = tech.technical_tempo_range if tech else (80, 120)
    tempo_bias = tempo_range_to_bias(tempo_range)

    # Mode preference
    mode = tech.technical_mode if tech else ""
    mode_preference = mode_to_preference(mode)

    # Groove parameters
    groove_feel = tech.technical_groove_feel if tech else ""
    rhythmic_density, groove_strength = groove_feel_to_density(groove_feel)

    # Map narrative arc to harmonic motion
    narrative_arc = complete_intent.song_intent.narrative_arc if complete_intent.song_intent else ""
    arc_lower = narrative_arc.lower() if narrative_arc else ""

    static_arcs = ["static reflection", "repetitive despair"]
    active_arcs = ["climb-to-climax", "sudden shift", "rise and fall", "spiral"]

    if any(a in arc_lower for a in static_arcs):
        harmonic_motion = 0.2
    elif any(a in arc_lower for a in active_arcs):
        harmonic_motion = 0.8
    else:
        harmonic_motion = 0.5

    # Map imagery texture to texture density
    imagery_texture = (
        complete_intent.song_intent.imagery_texture if complete_intent.song_intent else ""
    )
    texture_lower = imagery_texture.lower() if imagery_texture else ""

    dense_textures = ["muddy/thick", "chaotic", "claustrophobic", "deep shadow"]
    sparse_textures = ["sparse/empty", "open/vast", "hazy/dreamy"]

    if any(t in texture_lower for t in dense_textures):
        texture_density = 0.8
    elif any(t in texture_lower for t in sparse_textures):
        texture_density = 0.2
    else:
        texture_density = 0.5

    # Harmonic tension from mood tension
    harmonic_tension = tension

    # Melodic activity from arousal
    melodic_activity = arousal

    # Contour variance from narrative arc
    if "climb" in arc_lower or "rise" in arc_lower:
        contour_variance = 0.8
    elif "descent" in arc_lower or "static" in arc_lower:
        contour_variance = 0.2
    else:
        contour_variance = 0.5

    # Dynamic range from vulnerability and tension
    dynamic_range = max(tension, 1.0 - confidence)

    # Create MusicalIntent
    music = MusicalIntent(
        tempo_bias=tempo_bias,
        rhythmic_density=rhythmic_density,
        groove_strength=groove_strength,
        harmonic_tension=harmonic_tension,
        harmonic_motion=harmonic_motion,
        mode_preference=mode_preference,
        melodic_activity=melodic_activity,
        contour_variance=contour_variance,
        dynamic_range=dynamic_range,
        texture_density=texture_density,
    )

    # Time scope - default to immediate
    time = TimeScope(
        start_bar=-1,  # immediate
        end_bar=-1,  # open-ended
        fade_in_beats=0.0,
        fade_out_beats=0.0,
    )

    # Constraints - default to all engines allowed
    constraints = IntentConstraints(
        allowed_engines_mask=0xFFFFFFFF,
        forbidden_engines_mask=0,
        max_cpu_cost=1.0,
        max_event_rate=1000.0,
    )

    # Provenance
    provenance = IntentProvenance(
        source=source,
        user_override_weight=0.5,  # Default balance
    )

    # Create IntentFrame
    frame = IntentFrame(
        meta=IntentMeta(
            ir_version=1,
            intent_id=0,  # Will be generated
            session_id=session_id,
        ),
        emotion=emotion,
        music=music,
        time=time,
        constraints=constraints,
        provenance=provenance,
    )

    # Generate intent_id
    frame.meta.intent_id = frame.generate_intent_id()

    return frame
