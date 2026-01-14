"""
KmiDi Streamlit Interface - Refined & Overhauled

A modern, intuitive interface for emotion-driven music generation with
integrated AI orchestration (LLM reasoning, image generation, audio generation).
"""

import streamlit as st
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import base64
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "source" / "python"))

# Configuration
API_URL = os.getenv("KMIDI_API_URL", "http://127.0.0.1:8000")
USE_API = os.getenv("KMIDI_USE_API", "false").lower() == "true"
ORCHESTRATOR_MODE = os.getenv("KMIDI_ORCHESTRATOR_MODE", "false").lower() == "true"
LLM_MODEL_PATH = os.getenv("KMIDI_LLM_MODEL_PATH", "")

# Page configuration
st.set_page_config(
    page_title="KmiDi - Intelligent Music Generation",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced Custom CSS
st.markdown(
    """
<style>
    /* Main Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }

    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    .info-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: #1e293b;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
    }

    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #334155;
    }

    /* Status indicators */
    .status-online {
        color: #10b981;
        font-weight: 600;
    }

    .status-offline {
        color: #ef4444;
        font-weight: 600;
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
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "generation_mode" not in st.session_state:
    st.session_state.generation_mode = "basic"  # "basic" or "orchestrator"


def check_api_health() -> bool:
    """Check if FastAPI service is available."""
    if not USE_API:
        return False
    try:
        import requests
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def initialize_orchestrator():
    """Initialize the orchestrator if model path is available."""
    if not LLM_MODEL_PATH or not Path(LLM_MODEL_PATH).exists():
        return None

    try:
        from mcp_workstation.orchestrator import Orchestrator
        if st.session_state.orchestrator is None:
            with st.spinner("Initializing AI Orchestrator..."):
                st.session_state.orchestrator = Orchestrator(
                    llm_model_path=LLM_MODEL_PATH,
                    output_dir=str(project_root / "orchestrator_outputs")
                )
        return st.session_state.orchestrator
    except Exception as e:
        st.error(f"Failed to initialize orchestrator: {e}")
        return None


def generate_via_orchestrator(
    user_prompt: str,
    enable_image: bool = True,
    enable_audio: bool = False
) -> Dict[str, Any]:
    """Generate music using the full orchestrator workflow."""
    orchestrator = initialize_orchestrator()
    if not orchestrator:
        raise RuntimeError("Orchestrator not available. Check LLM_MODEL_PATH.")

    result = orchestrator.execute_workflow(
        user_prompt,
        enable_image_gen=enable_image,
        enable_audio_gen=enable_audio
    )

    # Convert CompleteSongIntent to dict for display
    return {
        "status": "success",
        "mode": "orchestrator",
        "intent": {
            "mood_primary": result.song_intent.mood_primary,
            "technical_genre": result.technical_constraints.technical_genre,
            "technical_key": result.technical_constraints.technical_key,
            "explanation": result.explanation,
        },
        "midi": result.midi_plan,
        "image": result.generated_image_data,
        "audio": result.generated_audio_data,
        "image_prompt": result.image_prompt,
        "audio_prompt": result.audio_texture_prompt,
    }


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
            "mode": "direct",
            "result": result,
        }
    except ImportError:
        raise ImportError("Music brain API not available. Install dependencies: pip install -e .")


# ============================================================================
# HEADER
# ============================================================================
st.markdown('<div class="main-header">🎵 KmiDi</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Intelligent Music Generation • Emotion-Driven Composition • AI Orchestration</div>',
    unsafe_allow_html=True,
)

# ============================================================================
# SIDEBAR - Settings & Quick Controls
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    # Generation Mode Selection
    st.subheader("🎯 Generation Mode")
    mode_options = ["Basic (FastAPI/Direct)", "Orchestrator (Full AI Pipeline)"]
    selected_mode_idx = 0 if st.session_state.generation_mode == "basic" else 1
    mode_choice = st.radio(
        "Select generation mode",
        mode_options,
        index=selected_mode_idx,
        help="Orchestrator mode requires LLM model path configured"
    )
    st.session_state.generation_mode = "orchestrator" if mode_choice == mode_options[1] else "basic"

    # API Status
    st.markdown("---")
    st.subheader("🔌 Connection Status")
    if st.session_state.generation_mode == "orchestrator":
        orchestrator = initialize_orchestrator()
        if orchestrator:
            st.markdown('<p class="status-online">✅ Orchestrator Ready</p>', unsafe_allow_html=True)
            if LLM_MODEL_PATH:
                st.caption(f"Model: {Path(LLM_MODEL_PATH).name}")
        else:
            st.markdown('<p class="status-offline">❌ Orchestrator Unavailable</p>', unsafe_allow_html=True)
            st.info("Set KMIDI_LLM_MODEL_PATH environment variable")
    else:
        if USE_API:
            api_online = check_api_health()
            if api_online:
                st.markdown('<p class="status-online">🟢 API Online</p>', unsafe_allow_html=True)
                st.caption(f"URL: {API_URL}")
            else:
                st.markdown('<p class="status-offline">🔴 API Offline</p>', unsafe_allow_html=True)
                st.info("Using direct API fallback")
        else:
            st.info("Using direct API mode")

    st.markdown("---")

    # Quick Parameters (Simplified)
    st.subheader("🎹 Quick Parameters")

    col_key, col_mode = st.columns(2)
    with col_key:
        key_options = ["Auto", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        selected_key = st.selectbox("Key", key_options, index=0)
    with col_mode:
        key_mode = st.selectbox(
            "Mode",
            ["Auto", "Major", "Minor", "Dorian", "Mixolydian"],
            index=0,
        )

    bpm = st.slider("Tempo (BPM)", 40, 200, 120, help="Beats per minute")

    genre_options = [
        "Auto",
        "Ambient", "Blues", "Classical", "Electronic",
        "Folk", "Funk", "Jazz", "Pop", "Rock",
        "Soul", "Cinematic", "Experimental",
    ]
    selected_genre = st.selectbox("Genre", genre_options, index=0)

    # Orchestrator-specific options
    if st.session_state.generation_mode == "orchestrator":
        st.markdown("---")
        st.subheader("🎨 AI Features")
        enable_image = st.checkbox("Generate Image", value=True, help="Create visual representation")
        enable_audio = st.checkbox("Generate Audio Texture", value=False, help="Create audio texture layer")

    st.markdown("---")

    # History
    if st.session_state.generation_history:
        st.subheader("📜 Recent")
        for i, hist in enumerate(reversed(st.session_state.generation_history[-3:])):
            with st.expander(f"#{len(st.session_state.generation_history) - i}: {hist.get('emotion', 'Unknown')[:25]}..."):
                st.caption(f"Key: {hist.get('key', 'Auto')} | BPM: {hist.get('bpm', 120)}")
                if st.button("Load", key=f"load_{i}"):
                    st.session_state["preset_text"] = hist.get("emotion", "")
                    st.rerun()


# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["🎵 Generate", "📊 Results", "📖 Guide"])

with tab1:
    col_main, col_preview = st.columns([2, 1])

    with col_main:
        st.subheader("💭 Emotional Intent")

        emotion_text = st.text_area(
            "Describe the mood, emotion, or feeling you want to express:",
            value=st.session_state.get("preset_text", ""),
            height=200,
            placeholder=(
                "Example: 'I'm feeling grief hidden as love, with a longing for connection'\n"
                "or 'Joyful anticipation with underlying tension and hope'"
            ),
            help="Be specific and detailed for best results",
        )

        # Additional context (for orchestrator mode)
        if st.session_state.generation_mode == "orchestrator":
            with st.expander("➕ Deep Context (Optional)"):
                core_wound = st.text_input(
                    "Core Wound / Event",
                    placeholder="The underlying emotional experience",
                )
                core_desire = st.text_input(
                    "Core Desire / Longing",
                    placeholder="What you're longing for",
                )
                vulnerability = st.select_slider(
                    "Vulnerability Level",
                    options=["Low", "Medium", "High"],
                    value="Medium",
                )

        # Generate Button
        st.markdown("---")
        generate_button = st.button(
            "🎵 Generate Music",
            type="primary",
            use_container_width=True,
            help="Click to start generation",
        )

    with col_preview:
        st.subheader("📊 Preview")

        if emotion_text:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.write(f"**Mood:** {emotion_text[:60]}{'...' if len(emotion_text) > 60 else ''}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            key_display = f"{selected_key} {key_mode}" if selected_key != "Auto" and key_mode != "Auto" else "Auto-detect"
            st.write(f"**Key:** {key_display}")
            st.write(f"**Tempo:** {bpm} BPM")
            st.write(f"**Genre:** {selected_genre if selected_genre != 'Auto' else 'Auto-detect'}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Enter your emotional intent to see preview")

    # Generation Logic
    if generate_button:
        if not emotion_text.strip():
            st.error("⚠️ Please enter your emotional intent before generating.")
        else:
            with st.spinner("🎵 Generating your music... This may take a moment."):
                try:
                    if st.session_state.generation_mode == "orchestrator":
                        # Use orchestrator
                        result = generate_via_orchestrator(
                            emotion_text,
                            enable_image=enable_image if st.session_state.generation_mode == "orchestrator" else False,
                            enable_audio=enable_audio if st.session_state.generation_mode == "orchestrator" else False,
                        )
                    elif USE_API and check_api_health():
                        # Use FastAPI
                        technical_params = {
                            "key": None if selected_key == "Auto" else selected_key,
                            "bpm": bpm,
                            "genre": None if selected_genre == "Auto" else selected_genre.lower(),
                        }
                        result = generate_via_api(emotion_text, technical_params)
                    else:
                        # Use direct API
                        technical_params = {
                            "key": None if selected_key == "Auto" else selected_key,
                            "bpm": bpm,
                            "genre": None if selected_genre == "Auto" else selected_genre.lower(),
                        }
                        result = generate_via_direct_api(emotion_text, technical_params)

                    # Store result
                    st.session_state.last_result = result

                    # Add to history
                    st.session_state.generation_history.append({
                        "emotion": emotion_text,
                        "key": selected_key,
                        "bpm": bpm,
                        "genre": selected_genre,
                        "timestamp": datetime.now().isoformat(),
                        "result": result,
                    })

                    st.success("✅ Generation complete!")
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Generation failed: {str(e)}")
                    st.exception(e)


with tab2:
    # Results Display
    if st.session_state.last_result:
        result = st.session_state.last_result

        # Status and Mode
        col_status1, col_status2, col_status3 = st.columns(3)
        with col_status1:
            st.metric("Status", "✅ Complete" if result.get("status") == "success" else "❌ Failed")
        with col_status2:
            st.metric("Mode", result.get("mode", "unknown").title())
        with col_status3:
            if result.get("mode") == "orchestrator" and result.get("intent"):
                st.metric("Mood", result["intent"].get("mood_primary", "N/A"))

        st.markdown("---")

        # MIDI Results
        if result.get("midi"):
            st.subheader("🎹 MIDI Generation")
            midi_data = result["midi"]
            if isinstance(midi_data, dict) and midi_data.get("status") == "completed":
                col_midi1, col_midi2 = st.columns(2)

                with col_midi1:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    if midi_data.get("key"):
                        st.write(f"**Key:** {midi_data['key']}")
                    if midi_data.get("tempo"):
                        st.write(f"**Tempo:** {midi_data['tempo']} BPM")
                    if midi_data.get("duration_bars"):
                        st.write(f"**Length:** {midi_data['duration_bars']} bars")
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_midi2:
                    midi_path = midi_data.get("file_path")
                    if midi_path and Path(midi_path).exists():
                        with open(midi_path, "rb") as f:
                            midi_bytes = f.read()
                        st.download_button(
                            "📥 Download MIDI",
                            data=midi_bytes,
                            file_name=f"KmiDi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mid",
                            mime="audio/midi",
                            use_container_width=True,
                        )
                    else:
                        st.info("MIDI file path not available")
            else:
                st.warning(f"MIDI generation: {midi_data.get('status', 'unknown')}")

        # Image Results (Orchestrator mode)
        if result.get("image") and result.get("mode") == "orchestrator":
            st.markdown("---")
            st.subheader("🖼️ Generated Image")
            image_data = result["image"]
            if image_data.get("status") == "completed":
                img_base64 = image_data.get("image_data_base64")
                if img_base64 and not img_base64.startswith("<"):
                    try:
                        img_bytes = base64.b64decode(img_base64)
                        st.image(img_bytes, use_container_width=True)
                        st.download_button(
                            "📥 Download Image",
                            data=img_bytes,
                            file_name=f"KmiDi_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                        )
                    except Exception as e:
                        st.error(f"Error displaying image: {e}")
                else:
                    st.info("Image data not available")
            else:
                st.info(f"Image generation: {image_data.get('status', 'not generated')}")

            if result.get("image_prompt"):
                with st.expander("📝 Image Prompt"):
                    st.write(result["image_prompt"])

        # Audio Results (Orchestrator mode)
        if result.get("audio") and result.get("mode") == "orchestrator":
            st.markdown("---")
            st.subheader("🔊 Generated Audio Texture")
            audio_data = result["audio"]
            if audio_data.get("status") == "completed":
                st.success("Audio texture generated successfully")
                if result.get("audio_prompt"):
                    with st.expander("📝 Audio Prompt"):
                        st.write(result["audio_prompt"])
            else:
                st.info(f"Audio generation: {audio_data.get('status', 'not generated')}")

        # Explanation (Orchestrator mode)
        if result.get("intent") and result["intent"].get("explanation"):
            st.markdown("---")
            st.subheader("💡 AI Explanation")
            st.info(result["intent"]["explanation"])

        # Full JSON (Expandable)
        with st.expander("🔍 View Full Result (JSON)"):
            st.json(result)

        # Regenerate
        st.markdown("---")
        if st.button("🔄 Regenerate with Same Settings", use_container_width=True):
            st.rerun()
    else:
        st.info("👆 Generate music to see results here")


with tab3:
    st.subheader("📖 How to Use KmiDi")

    st.markdown("""
    ### Quick Start

    1. **Enter Your Emotion** 💭
       - Describe the mood or feeling you want to express
       - Be specific: "grief hidden as love" is better than "sad"

    2. **Choose Generation Mode** 🎯
       - **Basic Mode**: Fast generation using music_brain API
       - **Orchestrator Mode**: Full AI pipeline with LLM reasoning, image, and audio generation

    3. **Adjust Parameters** ⚙️ (Optional)
       - Set key, tempo, and genre in the sidebar
       - Or leave on "Auto" for AI to decide

    4. **Generate & Download** 🎵
       - Click "Generate Music"
       - Download MIDI, images, and audio textures

    ### Generation Modes

    **Basic Mode:**
    - Fast generation using direct API
    - MIDI output only
    - Good for quick iterations

    **Orchestrator Mode:**
    - Full AI reasoning with Mistral 7B
    - MIDI + Image + Audio generation
    - Deep emotional understanding
    - Requires LLM model path configured

    ### Tips for Best Results

    - **Be Specific**: Detailed emotions produce better music
    - **Use Context**: Add core wound/desire for nuanced results
    - **Experiment**: Try different combinations
    - **Orchestrator**: Best for complex emotional expressions
    """)

    st.markdown("---")
    st.subheader("🔧 Technical Details")

    col_tech1, col_tech2 = st.columns(2)
    with col_tech1:
        st.write("**Generation Mode:**")
        st.code(st.session_state.generation_mode)
        st.write("**API Mode:**")
        st.code("FastAPI" if USE_API else "Direct")
    with col_tech2:
        st.write("**API URL:**")
        st.code(API_URL if USE_API else "N/A")
        st.write("**Orchestrator:**")
        st.code("Available" if initialize_orchestrator() else "Unavailable")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748b; font-size: 0.9rem;'>"
    "🎵 KmiDi - Intelligent Music Generation System | "
    "Version 2.0 | Built with ❤️ for creative expression"
    "</p>",
    unsafe_allow_html=True,
)
