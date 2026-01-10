"""
Streamlit Demo Frontend for KmiDi Music Generation.

A beautiful, user-friendly web interface for emotion-driven music generation.
Deployable to Streamlit Cloud, Heroku, or any cloud platform.
"""

import streamlit as st
import sys
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configuration
API_URL = os.getenv("KMIDI_API_URL", "http://127.0.0.1:8000")
USE_API = os.getenv("KMIDI_USE_API", "false").lower() == "true"

# Page configuration
st.set_page_config(
    page_title="KmiDi - Music Generation",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #3a6ea5, #5a8fcf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #1e1e1e;
        border-left: 4px solid #7fc97f;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #1e1e1e;
        border-left: 4px solid #3a6ea5;
    }
    .stButton>button {
        width: 100%;
        background-color: #3a6ea5;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.75rem;
    }
    .stButton>button:hover {
        background-color: #5a8fcf;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #242424;
        border: 1px solid #2a2a2a;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "generation_history" not in st.session_state:
    st.session_state.generation_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def check_api_health() -> bool:
    """Check if API is available."""
    if not USE_API:
        return False

    try:
        import requests

        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def generate_via_api(emotion_text: str, technical_params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate music via FastAPI service."""
    import requests

    api_request = {
        "intent": {
            "emotional_intent": emotion_text,
            "technical": (
                {
                    "key": technical_params.get("key"),
                    "bpm": technical_params.get("bpm"),
                    "genre": technical_params.get("genre"),
                }
                if (
                    technical_params.get("key")
                    or technical_params.get("bpm")
                    or technical_params.get("genre")
                )
                else None
            ),
        },
        "output_format": "midi",
    }

    response = requests.post(f"{API_URL}/generate", json=api_request, timeout=60)
    response.raise_for_status()
    return response.json()


def generate_via_direct_api(emotion_text: str, technical_params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate music via direct music_brain API."""
    try:
        from music_brain.api import api as music_api

        # Map technical params to therapy session parameters
        motivation = 7
        if technical_params.get("bpm"):
            motivation = max(1, min(10, int(technical_params["bpm"] / 20)))

        chaos = 0.5

        result = music_api.therapy_session(
            text=emotion_text,
            motivation=motivation,
            chaos_tolerance=chaos,
            output_midi=None,
        )

        return {
            "status": "success",
            "result": result,
        }
    except ImportError:
        raise ImportError("Music brain API not available. Install dependencies: pip install -e .")


# Header
st.markdown('<div class="main-header">🎵 KmiDi Music Generation</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #a8a8a8; margin-bottom: 2rem;'>"
    "Transform your emotions into music. Describe what you're feeling, and let AI compose for you."
    "</p>",
    unsafe_allow_html=True,
)

# Sidebar - Settings and Parameters
with st.sidebar:
    st.header("⚙️ Settings")

    # API Connection Status
    if USE_API:
        api_online = check_api_health()
        status_color = "🟢" if api_online else "🔴"
        status_text = "Online" if api_online else "Offline"
        st.markdown(f"**API Status:** {status_color} {status_text}")
        if not api_online:
            st.info(f"⚠️ API unavailable at {API_URL}. Using direct API fallback.")
    else:
        st.info("Using direct API mode")

    st.markdown("---")

    # Emotion Presets
    st.subheader("🎭 Emotion Presets")
    try:
        from music_brain.data.emotional_mapping import EMOTIONAL_PRESETS

        if EMOTIONAL_PRESETS:
            available_emotions = ["Custom"] + sorted(EMOTIONAL_PRESETS.keys())
            selected_emotion_preset = st.selectbox(
                "Choose Emotion Preset",
                available_emotions,
                index=0,
                help="Select a preset emotion or choose Custom to enter your own",
            )

            if selected_emotion_preset != "Custom" and selected_emotion_preset in EMOTIONAL_PRESETS:
                preset_info = EMOTIONAL_PRESETS[selected_emotion_preset]
                if isinstance(preset_info, dict):
                    valence = preset_info.get("valence", 0.0)
                    arousal = preset_info.get("arousal", 0.5)
                    description = preset_info.get("description", "")

                    st.info(f"**{selected_emotion_preset}**\n\n{description}")
                    if st.session_state.get("use_preset_text", False):
                        st.session_state["preset_text"] = selected_emotion_preset
        else:
            selected_emotion_preset = "Custom"
            st.info("No emotion presets available")
    except ImportError:
        selected_emotion_preset = "Custom"
        st.warning("Emotion presets not available")

    st.markdown("---")

    # Basic Parameters (Always Visible)
    st.subheader("🎹 Basic Parameters")

    col_basic1, col_basic2 = st.columns(2)
    with col_basic1:
        key_options = ["Auto", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_mode = st.selectbox(
            "Key Mode",
            ["Auto", "Major", "Minor", "Dorian", "Mixolydian", "Phrygian", "Lydian", "Locrian"],
            index=0,
        )
        selected_key = st.selectbox("Musical Key", key_options, index=0)

    with col_basic2:
        bpm = st.slider(
            "Tempo (BPM)",
            40,
            200,
            120,
            help="Beats per minute - slower for calm, faster for energetic",
        )
        time_sig_num = st.selectbox("Time Signature (beats)", [2, 3, 4, 5, 6, 7, 8, 9, 12], index=2)
        time_sig_den = st.selectbox("Time Signature (note)", [2, 4, 8, 16], index=1)

    genre_options = [
        "Auto",
        "Ambient",
        "Blues",
        "Classical",
        "Electronic",
        "Folk",
        "Funk",
        "Jazz",
        "Pop",
        "Rock",
        "Soul",
        "Cinematic",
        "Experimental",
    ]
    selected_genre = st.selectbox("Genre", genre_options, index=0)

    st.markdown("---")

    # Advanced Parameters (Expandable Sections)
    with st.expander("🎼 Harmony & Chord Parameters", expanded=False):
        col_harm1, col_harm2 = st.columns(2)
        with col_harm1:
            chord_style = st.selectbox(
                "Chord Style",
                [
                    "Auto",
                    "Triads",
                    "Sevenths",
                    "Extended (9th, 11th, 13th)",
                    "Add Chords",
                    "Sus Chords",
                    "Power Chords",
                    "Cluster Chords",
                ],
                index=0,
            )
            chord_complexity = st.slider("Chord Complexity", 1, 10, 5, help="1=Simple, 10=Complex")
            chord_density = st.slider(
                "Chord Change Density", 1, 10, 5, help="How often chords change"
            )
            voice_leading = st.selectbox(
                "Voice Leading Style",
                ["Auto", "Smooth", "Angular", "Contrary Motion", "Parallel Motion", "Mixed"],
                index=0,
            )
        with col_harm2:
            tension_level = st.slider(
                "Tension Level", 0, 10, 5, help="0=Very Stable, 10=Very Tense"
            )
            dissonance = st.slider("Dissonance", 0, 10, 3, help="0=Consonant, 10=Highly Dissonant")
            resolution_style = st.selectbox(
                "Resolution Style",
                [
                    "Auto",
                    "Strong (Perfect)",
                    "Weak (Deceptive)",
                    "Delayed",
                    "No Resolution",
                    "Partial",
                ],
                index=0,
            )
            harmonic_rhythm = st.slider(
                "Harmonic Rhythm", 1, 10, 5, help="Speed of harmonic changes"
            )

    with st.expander("🥁 Rhythm & Groove Parameters", expanded=False):
        col_rhythm1, col_rhythm2 = st.columns(2)
        with col_rhythm1:
            groove_style = st.selectbox(
                "Groove Style",
                [
                    "Auto",
                    "Straight",
                    "Swing",
                    "Shuffle",
                    "Latin",
                    "African",
                    "Indian",
                    "Funk",
                    "Reggae",
                ],
                index=0,
            )
            swing_amount = st.slider(
                "Swing Amount (%)", 0, 100, 0, help="0=Straight, 100=Heavy swing"
            )
            syncopation = st.slider(
                "Syncopation Level", 0, 10, 5, help="0=On-beat, 10=Highly syncopated"
            )
            rhythmic_complexity = st.slider("Rhythmic Complexity", 1, 10, 5)
        with col_rhythm2:
            subdivision = st.selectbox(
                "Subdivision", ["Auto", "1/4", "1/8", "1/16", "1/32", "Triplets", "Mixed"], index=1
            )
            accent_pattern = st.selectbox(
                "Accent Pattern",
                ["Auto", "Strong-Weak", "Weak-Strong", "Backbeat", "Polyrhythmic", "Random"],
                index=0,
            )
            ghost_notes = st.slider("Ghost Notes", 0, 10, 2, help="Subtle, quiet notes")
            groove_pocket = st.slider("Groove Pocket", 0, 10, 5, help="Humanized timing feel")

    with st.expander("🎵 Melody Parameters", expanded=False):
        col_mel1, col_mel2 = st.columns(2)
        with col_mel1:
            melodic_range = st.selectbox(
                "Melodic Range",
                ["Auto", "Narrow (1 octave)", "Medium (2 octaves)", "Wide (3+ octaves)", "Extreme"],
                index=1,
            )
            melodic_complexity = st.slider(
                "Melodic Complexity", 1, 10, 6, help="1=Simple, 10=Ornate"
            )
            contour = st.selectbox(
                "Melodic Contour",
                ["Auto", "Ascending", "Descending", "Arch", "Wave", "Plateau", "Convex", "Concave"],
                index=0,
            )
            interval_style = st.selectbox(
                "Interval Style",
                ["Auto", "Stepwise", "Leaps", "Mixed", "Chromatic", "Wide Leaps"],
                index=2,
            )
        with col_mel2:
            phrasing = st.selectbox(
                "Phrasing Style",
                [
                    "Auto",
                    "Long Legato",
                    "Short Staccato",
                    "Mixed",
                    "Question-Answer",
                    "Call-Response",
                ],
                index=0,
            )
            ornamentation = st.slider("Ornamentation", 0, 10, 3, help="Trills, grace notes, etc.")
            repetition = st.slider(
                "Melodic Repetition", 0, 10, 5, help="0=No repetition, 10=Highly repetitive"
            )
            development = st.slider("Melodic Development", 0, 10, 6, help="Variation and evolution")

    with st.expander("📐 Structure & Form Parameters", expanded=False):
        col_struct1, col_struct2 = st.columns(2)
        with col_struct1:
            song_length = st.slider(
                "Song Length (bars)", 8, 128, 32, help="Total length in measures"
            )
            num_sections = st.slider("Number of Sections", 2, 8, 4)
            intro_style = st.selectbox(
                "Intro Style",
                ["Auto", "None", "Fade In", "Full Intro", "Vamp", "Solo Instrument"],
                index=2,
            )
            outro_style = st.selectbox(
                "Outro Style",
                ["Auto", "Fade Out", "Full Outro", "Vamp", "Decrescendo", "Abrupt Stop"],
                index=1,
            )
        with col_struct2:
            section_transitions = st.selectbox(
                "Section Transitions",
                ["Auto", "Smooth", "Sudden", "Build", "Drop", "Bridge", "Mixed"],
                index=0,
            )
            repetition_variation = st.slider(
                "Repetition vs Variation", 0, 10, 5, help="0=Repetitive, 10=Varied"
            )
            development_arc = st.selectbox(
                "Development Arc",
                ["Auto", "Linear Build", "Peak in Middle", "Plateau", "Wave", "Descending"],
                index=0,
            )
            bridge_probability = st.slider(
                "Bridge Section", 0, 100, 30, help="Chance of including bridge"
            )

    with st.expander("🎚️ Dynamics & Expression Parameters", expanded=False):
        col_dyn1, col_dyn2 = st.columns(2)
        with col_dyn1:
            overall_volume = st.slider("Overall Volume", 0, 100, 75, help="0=Silent, 100=Maximum")
            volume_envelope = st.selectbox(
                "Volume Envelope",
                ["Auto", "Constant", "Crescendo", "Decrescendo", "Wave", "Build-Drop", "Custom"],
                index=0,
            )
            accent_strength = st.slider("Accent Strength", 0, 10, 5)
            dynamic_range = st.slider("Dynamic Range", 1, 10, 7, help="1=Flat, 10=Very dynamic")
        with col_dyn2:
            vibrato = st.slider("Vibrato Amount", 0, 10, 3)
            attack_time = st.selectbox(
                "Attack Time",
                ["Auto", "Very Fast", "Fast", "Medium", "Slow", "Very Slow", "Mixed"],
                index=2,
            )
            articulation = st.selectbox(
                "Articulation",
                ["Auto", "Staccato", "Legato", "Tenuto", "Marcato", "Mixed"],
                index=1,
            )
            expression_intensity = st.slider("Expression Intensity", 0, 10, 6)

    with st.expander("🎹 Instrumentation & Texture Parameters", expanded=False):
        col_inst1, col_inst2 = st.columns(2)
        with col_inst1:
            num_voices = st.slider("Number of Voices/Instruments", 1, 16, 4)
            texture_density = st.slider("Texture Density", 1, 10, 6, help="1=Sparse, 10=Dense")
            layering_style = st.selectbox(
                "Layering Style",
                ["Auto", "Monophonic", "Homophonic", "Polyphonic", "Heterophonic", "Mixed"],
                index=3,
            )
            voice_roles = st.multiselect(
                "Voice Roles (select multiple)",
                [
                    "Melody",
                    "Harmony",
                    "Bass",
                    "Rhythm",
                    "Percussion",
                    "Textural",
                    "Lead",
                    "Support",
                ],
                default=["Melody", "Harmony", "Bass", "Rhythm"],
            )
        with col_inst2:
            instrument_balance = st.selectbox(
                "Instrument Balance",
                [
                    "Auto",
                    "Melody Prominent",
                    "Harmony Prominent",
                    "Rhythm Prominent",
                    "Balanced",
                    "Textural",
                ],
                index=4,
            )
            space_filling = st.slider("Space Filling", 1, 10, 6, help="1=Minimal, 10=Full")
            frequency_range = st.selectbox(
                "Frequency Range Coverage",
                ["Auto", "Low Focus", "Mid Focus", "High Focus", "Full Spectrum", "Sparse"],
                index=4,
            )
            stereo_width = st.slider("Stereo Width", 0, 10, 5, help="0=Mono, 10=Wide stereo")

    with st.expander("💫 Emotional & Psychological Parameters", expanded=False):
        col_emo1, col_emo2 = st.columns(2)
        with col_emo1:
            valence = st.slider(
                "Valence (Positivity)", -10, 10, 0, help="-10=Very Negative, 10=Very Positive"
            )
            arousal = st.slider("Arousal (Energy)", 0, 10, 5, help="0=Calm, 10=Intense")
            intensity = st.slider("Emotional Intensity", 0, 10, 6)
            energy_level = st.slider("Energy Level", 0, 10, 5)
        with col_emo2:
            tension_release = st.slider(
                "Tension-Release Balance", 0, 10, 5, help="0=Tension, 10=Release"
            )
            stability = st.slider("Stability", 0, 10, 5, help="0=Unstable, 10=Stable")
            predictability = st.slider(
                "Predictability", 0, 10, 5, help="0=Unpredictable, 10=Predictable"
            )
            emotional_complexity = st.slider(
                "Emotional Complexity", 1, 10, 5, help="Multiple emotions"
            )

    with st.expander("🎛️ Production & Sound Design Parameters", expanded=False):
        col_prod1, col_prod2 = st.columns(2)
        with col_prod1:
            reverb_amount = st.slider("Reverb Amount", 0, 10, 3, help="0=Dry, 10=Wet")
            reverb_type = st.selectbox(
                "Reverb Type",
                ["Auto", "Hall", "Room", "Plate", "Spring", "Chamber", "Cathedral"],
                index=0,
            )
            compression = st.slider("Compression", 0, 10, 4)
            eq_low = st.slider("EQ Low (Bass)", -10, 10, 0)
            eq_mid = st.slider("EQ Mid", -10, 10, 0)
        with col_prod2:
            eq_high = st.slider("EQ High (Treble)", -10, 10, 0)
            saturation = st.slider("Saturation/Warmth", 0, 10, 2)
            filter_cutoff = st.slider(
                "Filter Cutoff", 0, 100, 100, help="0=Full filter, 100=No filter"
            )
            delay_amount = st.slider("Delay Amount", 0, 10, 1)
            chorus_flange = st.slider("Chorus/Flange", 0, 10, 0)

    with st.expander("🎨 Style & Character Parameters", expanded=False):
        col_style1, col_style2 = st.columns(2)
        with col_style1:
            style_period = st.selectbox(
                "Style Period",
                [
                    "Auto",
                    "Medieval",
                    "Renaissance",
                    "Baroque",
                    "Classical",
                    "Romantic",
                    "20th Century",
                    "Modern",
                    "Contemporary",
                    "Futuristic",
                ],
                index=8,
            )
            cultural_influence = st.multiselect(
                "Cultural Influence",
                ["Western", "Eastern", "African", "Latin", "Indian", "Middle Eastern", "None"],
                default=["Western"],
            )
            performance_style = st.selectbox(
                "Performance Style",
                ["Auto", "Classical", "Jazz", "Rock", "Electronic", "Folk", "World", "Hybrid"],
                index=0,
            )
            humanization = st.slider("Humanization", 0, 10, 6, help="0=Machine-like, 10=Very human")
        with col_style2:
            imperfection = st.slider(
                "Imperfection/Character", 0, 10, 4, help="0=Perfect, 10=Characterful"
            )
            randomness = st.slider("Randomness", 0, 10, 2, help="0=Deterministic, 10=Random")
            innovation = st.slider(
                "Innovation/Uniqueness", 0, 10, 5, help="0=Traditional, 10=Innovative"
            )
            emotional_authenticity = st.slider(
                "Emotional Authenticity", 0, 10, 8, help="Genuine expression"
            )

    st.markdown("---")

    # Generation Button
    generate_button = st.button(
        "🎵 Generate Music",
        type="primary",
        use_container_width=True,
        help="Click to generate music based on your emotional intent and parameters",
    )

    st.markdown("---")

    # History
    if st.session_state.generation_history:
        st.subheader("📜 Recent Generations")
        for i, hist in enumerate(reversed(st.session_state.generation_history[-5:])):
            with st.expander(
                f"{hist.get('emotion', 'Unknown')[:30]}... ({hist.get('timestamp', '')[:10]})"
            ):
                st.write(f"**Emotion:** {hist.get('emotion', 'N/A')}")
                st.write(f"**Key:** {hist.get('key', 'Auto')} | **BPM:** {hist.get('bpm', 120)}")
                if st.button("Load", key=f"load_{i}"):
                    st.session_state["preset_text"] = hist.get("emotion", "")
                    st.rerun()


# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💭 Emotional Intent")

    # Emotion input
    emotion_text = st.text_area(
        "What is the mood you want to convey?",
        value=st.session_state.get("preset_text", ""),
        height=150,
        placeholder=(
            "Example: 'I'm feeling grief hidden as love' or "
            "'I want to express joyful anticipation with underlying tension'"
        ),
        help=(
            "Be specific about the emotion, mood, or feeling you want to capture in music. "
            "The more detail you provide, the better the AI can translate it into musical elements."
        ),
    )

    # Additional context (optional)
    with st.expander("➕ Additional Context (Optional)"):
        core_wound = st.text_input(
            "Core Wound / Event",
            placeholder="The underlying emotional wound or experience",
            help="Optional: Describe the core emotional experience or event",
        )
        core_desire = st.text_input(
            "Core Desire / Longing",
            placeholder="What you're longing for or hoping for",
            help="Optional: Describe what you desire or long for",
        )

with col2:
    st.subheader("📊 Quick Preview")

    if emotion_text:
        st.info(f"**Mood:** {emotion_text[:50]}{'...' if len(emotion_text) > 50 else ''}")
        if selected_key != "Auto" and key_mode != "Auto":
            key_display = f"{selected_key} {key_mode}"
        elif selected_key != "Auto":
            key_display = selected_key
        else:
            key_display = "Auto-detect"
        st.info(f"**Key:** {key_display}")
        st.info(f"**Tempo:** {bpm} BPM ({time_sig_num}/{time_sig_den})")
        st.info(f"**Genre:** {selected_genre if selected_genre != 'Auto' else 'Auto-detect'}")
        st.info(f"**Voices:** {num_voices} | **Complexity:** {chord_complexity}/10")
    else:
        st.info("Enter the mood you want to convey to see preview")


# Generate Music
if generate_button:
    if not emotion_text.strip():
        st.error("⚠️ Please enter your emotional intent before generating music.")
    else:
        with st.spinner("🎵 Generating your music... This may take a moment."):
            try:
                # Prepare comprehensive technical parameters
                technical_params = {
                    # Basic
                    "key": None if selected_key == "Auto" else selected_key,
                    "key_mode": None if key_mode == "Auto" else key_mode.lower(),
                    "bpm": bpm,
                    "time_signature": f"{time_sig_num}/{time_sig_den}",
                    "genre": None if selected_genre == "Auto" else selected_genre.lower(),
                    # Harmony
                    "chord_style": chord_style.lower() if chord_style != "Auto" else None,
                    "chord_complexity": chord_complexity,
                    "chord_density": chord_density,
                    "voice_leading": voice_leading.lower() if voice_leading != "Auto" else None,
                    "tension_level": tension_level,
                    "dissonance": dissonance,
                    "resolution_style": (
                        resolution_style.lower() if resolution_style != "Auto" else None
                    ),
                    "harmonic_rhythm": harmonic_rhythm,
                    # Rhythm
                    "groove_style": groove_style.lower() if groove_style != "Auto" else None,
                    "swing_amount": swing_amount,
                    "syncopation": syncopation,
                    "rhythmic_complexity": rhythmic_complexity,
                    "subdivision": subdivision.lower() if subdivision != "Auto" else None,
                    "accent_pattern": accent_pattern.lower() if accent_pattern != "Auto" else None,
                    "ghost_notes": ghost_notes,
                    "groove_pocket": groove_pocket,
                    # Melody
                    "melodic_range": melodic_range.lower() if melodic_range != "Auto" else None,
                    "melodic_complexity": melodic_complexity,
                    "contour": contour.lower() if contour != "Auto" else None,
                    "interval_style": interval_style.lower() if interval_style != "Auto" else None,
                    "phrasing": phrasing.lower() if phrasing != "Auto" else None,
                    "ornamentation": ornamentation,
                    "repetition": repetition,
                    "development": development,
                    # Structure
                    "song_length": song_length,
                    "num_sections": num_sections,
                    "intro_style": intro_style.lower() if intro_style != "Auto" else None,
                    "outro_style": outro_style.lower() if outro_style != "Auto" else None,
                    "section_transitions": (
                        section_transitions.lower() if section_transitions != "Auto" else None
                    ),
                    "repetition_variation": repetition_variation,
                    "development_arc": (
                        development_arc.lower() if development_arc != "Auto" else None
                    ),
                    "bridge_probability": bridge_probability,
                    # Dynamics
                    "overall_volume": overall_volume,
                    "volume_envelope": (
                        volume_envelope.lower() if volume_envelope != "Auto" else None
                    ),
                    "accent_strength": accent_strength,
                    "dynamic_range": dynamic_range,
                    "vibrato": vibrato,
                    "attack_time": attack_time.lower() if attack_time != "Auto" else None,
                    "articulation": articulation.lower() if articulation != "Auto" else None,
                    "expression_intensity": expression_intensity,
                    # Instrumentation
                    "num_voices": num_voices,
                    "texture_density": texture_density,
                    "layering_style": layering_style.lower() if layering_style != "Auto" else None,
                    "voice_roles": voice_roles,
                    "instrument_balance": (
                        instrument_balance.lower() if instrument_balance != "Auto" else None
                    ),
                    "space_filling": space_filling,
                    "frequency_range": (
                        frequency_range.lower() if frequency_range != "Auto" else None
                    ),
                    "stereo_width": stereo_width,
                    # Emotional
                    "valence": valence,
                    "arousal": arousal,
                    "intensity": intensity,
                    "energy_level": energy_level,
                    "tension_release": tension_release,
                    "stability": stability,
                    "predictability": predictability,
                    "emotional_complexity": emotional_complexity,
                    # Production
                    "reverb_amount": reverb_amount,
                    "reverb_type": reverb_type.lower() if reverb_type != "Auto" else None,
                    "compression": compression,
                    "eq_low": eq_low,
                    "eq_mid": eq_mid,
                    "eq_high": eq_high,
                    "saturation": saturation,
                    "filter_cutoff": filter_cutoff,
                    "delay_amount": delay_amount,
                    "chorus_flange": chorus_flange,
                    # Style
                    "style_period": style_period.lower() if style_period != "Auto" else None,
                    "cultural_influence": cultural_influence,
                    "performance_style": (
                        performance_style.lower() if performance_style != "Auto" else None
                    ),
                    "humanization": humanization,
                    "imperfection": imperfection,
                    "randomness": randomness,
                    "innovation": innovation,
                    "emotional_authenticity": emotional_authenticity,
                }

                # Generate music
                if USE_API and check_api_health():
                    result = generate_via_api(emotion_text, technical_params)
                else:
                    result = generate_via_direct_api(emotion_text, technical_params)

                # Store result
                st.session_state.last_result = result

                # Add to history
                st.session_state.generation_history.append(
                    {
                        "emotion": emotion_text,
                        "key": selected_key,
                        "bpm": bpm,
                        "genre": selected_genre,
                        "timestamp": datetime.now().isoformat(),
                        "result": result,
                    }
                )

                st.success("✅ Music generated successfully!")

            except Exception as e:
                st.error(f"❌ Generation failed: {str(e)}")
                st.exception(e)


# Display Results
if st.session_state.last_result:
    st.markdown("---")
    st.subheader("🎼 Generated Music")

    result = st.session_state.last_result

    # Extract result data
    if isinstance(result, dict):
        if result.get("status") == "success":
            music_data = result.get("result", {})

            # Create columns for results
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)

                # Chord Progression
                if isinstance(music_data, dict) and "chords" in music_data:
                    chords = music_data["chords"]
                    if isinstance(chords, str):
                        chords = chords.split(", ")
                    st.write("**🎹 Chord Progression:**")
                    chord_display = " → ".join(chords) if isinstance(chords, list) else chords
                    st.code(chord_display, language=None)

                # Key and Tempo
                if isinstance(music_data, dict):
                    metadata = music_data.get("metadata", {})
                    if metadata.get("key"):
                        st.write(f"**🎵 Key:** {metadata['key']}")
                    if metadata.get("tempo"):
                        st.write(f"**⏱️ Tempo:** {metadata['tempo']} BPM")
                    if metadata.get("time_signature"):
                        st.write(f"**📐 Time Signature:** {metadata['time_signature']}")

                st.markdown("</div>", unsafe_allow_html=True)

            with res_col2:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)

                # MIDI Download
                midi_path = None
                if isinstance(music_data, dict):
                    midi_path = music_data.get("midi_path") or music_data.get("midi_file")

                if midi_path and Path(midi_path).exists():
                    with open(midi_path, "rb") as f:
                        midi_data = f.read()
                    st.download_button(
                        label="📥 Download MIDI File",
                        data=midi_data,
                        file_name=f"kmidi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mid",
                        mime="audio/midi",
                        use_container_width=True,
                    )
                    st.info(f"MIDI file ready: {Path(midi_path).name}")
                else:
                    st.info("MIDI file not yet generated. Check full result for details.")

                st.markdown("</div>", unsafe_allow_html=True)

            # Full Result (expandable)
            with st.expander("🔍 View Full Result (JSON)"):
                st.json(result)

            # Regenerate option
            st.markdown("---")
            col_reg1, col_reg2, col_reg3 = st.columns(3)
            with col_reg2:
                if st.button("🔄 Regenerate with Same Settings", use_container_width=True):
                    st.rerun()
        else:
            st.error(f"Generation failed: {result.get('detail', 'Unknown error')}")
            if "error" in result:
                st.exception(result["error"])


# Instructions and Help
with st.expander("📖 How to Use KmiDi"):
    st.markdown(
        """
    ### Step-by-Step Guide

        1. **Enter Your Mood** 📝
           - Describe what mood you want to convey
           - Be specific: "grief hidden as love" is better than just "sad"
           - Or choose from emotion presets in the sidebar

    2. **Adjust Parameters** ⚙️ (Optional)
       - **Key**: Choose a musical key (or Auto for AI to decide)
       - **Tempo (BPM)**: Slower for calm/reflective, faster for energetic
       - **Genre**: Select a genre style (or Auto)

    3. **Generate Music** 🎵
       - Click the "Generate Music" button
       - Wait a few moments while AI creates your composition

    4. **Download & Enjoy** 📥
       - Download the generated MIDI file
       - Import into your favorite DAW or MIDI player
       - Experiment with different emotions and parameters!

    ### Tips for Best Results

    - **Be Specific**: "Joyful anticipation with underlying tension" works better than "happy"
    - **Experiment**: Try different combinations of emotions and parameters
    - **Use Presets**: Start with emotion presets to see how the system works
    - **Context Matters**: Add core wound/desire for more nuanced results

    ### About KmiDi

    KmiDi uses AI to translate emotional intent into musical composition.
    The system understands complex emotions and generates chord progressions,
    melodies, and structures that match your emotional state.
    """
    )

with st.expander("🔧 Technical Details"):
    st.markdown(
        f"""
    - **API Mode**: {'FastAPI Service' if USE_API else 'Direct API'}
    - **API URL**: {API_URL if USE_API else 'N/A'}
    - **Version**: 1.0.0
    - **Framework**: Streamlit
    """
    )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888888;'>"
    "🎵 KmiDi Music Generation System - Version 1.0.0 | "
    "Built with ❤️ for creative expression"
    "</p>",
    unsafe_allow_html=True,
)
