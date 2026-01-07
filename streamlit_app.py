"""
Streamlit Demo Frontend for KmiDi Music Generation.

User-facing interface for emotion input → music output.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from music_brain.api import api as music_api, EMOTIONAL_PRESETS
    MUSIC_BRAIN_AVAILABLE = True
except ImportError:
    MUSIC_BRAIN_AVAILABLE = False
    st.error("Music brain API not available. Please install dependencies.")

st.set_page_config(
    page_title="KmiDi Music Generation",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 KmiDi Music Generation")
st.markdown("Generate music from emotional intent")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Emotion selection
    if MUSIC_BRAIN_AVAILABLE and EMOTIONAL_PRESETS:
        available_emotions = sorted(EMOTIONAL_PRESETS.keys())
        selected_emotion = st.selectbox(
            "Select Emotion",
            available_emotions,
            index=0 if available_emotions else None
        )
    else:
        selected_emotion = st.text_input("Enter Emotion", value="grief")
    
    # Technical parameters
    st.subheader("Technical Parameters")
    key = st.selectbox("Key", ["C", "D", "E", "F", "G", "A", "B"], index=4)
    bpm = st.slider("BPM", 60, 180, 120)
    genre = st.selectbox("Genre", ["pop", "rock", "jazz", "electronic", "acoustic"], index=0)
    
    # Generation button
    generate_button = st.button("Generate Music", type="primary", use_container_width=True)


# Main content
if not MUSIC_BRAIN_AVAILABLE:
    st.error("Music generation API is not available. Please check installation.")
else:
    # Emotion input
    st.subheader("Emotional Intent")
    emotional_intent = st.text_area(
        "Describe the emotional intent",
        value=selected_emotion if 'selected_emotion' in locals() else "grief",
        height=100
    )
    
    # Generate on button click
    if generate_button and emotional_intent:
        with st.spinner("Generating music..."):
            try:
                # Map to therapy session
                motivation = max(1, min(10, int(bpm / 20)))
                chaos = 0.5
                
                result = music_api.therapy_session(
                    text=emotional_intent,
                    motivation=motivation,
                    chaos_tolerance=chaos,
                    output_midi=None,
                )
                
                st.success("Music generated successfully!")
                
                # Display results
                st.subheader("Generated Music")
                
                if result and isinstance(result, dict):
                    # Display chord progression if available
                    if 'chords' in result:
                        st.write("**Chord Progression:**")
                        st.code(', '.join(result['chords']))
                    
                    # Display MIDI file if available
                    if 'midi_path' in result and Path(result['midi_path']).exists():
                        st.download_button(
                            label="Download MIDI",
                            data=open(result['midi_path'], 'rb').read(),
                            file_name="generated_music.mid",
                            mime="audio/midi"
                        )
                    
                    # Display full result
                    with st.expander("Full Result"):
                        st.json(result)
                else:
                    st.info("Generation complete. Check output for details.")
                    
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
                st.exception(e)
    
    # Instructions
    with st.expander("How to Use"):
        st.markdown("""
        1. **Select or enter an emotion** - Choose from presets or describe your own
        2. **Adjust technical parameters** - Set key, BPM, and genre
        3. **Describe emotional intent** - Provide context about the desired mood
        4. **Click Generate Music** - Create your personalized composition
        
        The system uses AI to generate music that matches your emotional intent.
        """)


# Footer
st.markdown("---")
st.markdown("KmiDi Music Generation System - Version 1.0.0")

