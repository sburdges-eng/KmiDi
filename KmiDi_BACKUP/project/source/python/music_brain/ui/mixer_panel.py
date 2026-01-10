"""
Streamlit Web Mixer Control Panel

Interactive mixer UI for controlling the Penta-Core mixer engine
through a web interface. Provides real-time control of channels,
sends, and master bus.

Run with:
    streamlit run music_brain/ui/mixer_panel.py
"""

import streamlit as st
import numpy as np
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from penta_core.mixer import MixerEngine, MixerState, apply_emotion_to_mixer
from music_brain.daw.mixer_params import (
    EmotionMapper,
    MixerParameters,
    MIXER_PRESETS
)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize Streamlit session state."""
    if 'mixer' not in st.session_state:
        st.session_state.mixer = MixerEngine(sample_rate=48000.0)
        st.session_state.mixer.set_num_channels(8)
        st.session_state.mixer.set_num_send_buses(4)

    if 'channel_names' not in st.session_state:
        st.session_state.channel_names = [
            "Drums", "Bass", "Guitar L", "Guitar R",
            "Vocals", "Keys", "Pad", "FX"
        ]

    if 'send_names' not in st.session_state:
        st.session_state.send_names = [
            "Reverb", "Delay", "Chorus", "FX"
        ]

    if 'emotion_mapper' not in st.session_state:
        st.session_state.emotion_mapper = EmotionMapper()


# =============================================================================
# UI Components
# =============================================================================

def render_channel_strip(channel: int, name: str):
    """Render a single channel strip."""
    mixer: MixerEngine = st.session_state.mixer

    with st.container():
        st.markdown(f"### 🎚️ {name}")

        # Gain fader
        current_gain = mixer._channel_gains[channel]
        gain = st.slider(
            "Gain (dB)",
            min_value=-60.0,
            max_value=12.0,
            value=current_gain,
            step=0.1,
            key=f"gain_{channel}"
        )
        mixer.set_channel_gain(channel, gain)

        # Pan knob
        current_pan = mixer._channel_pans[channel]
        pan = st.slider(
            "Pan",
            min_value=-1.0,
            max_value=1.0,
            value=current_pan,
            step=0.01,
            key=f"pan_{channel}",
            format="%.2f"
        )
        mixer.set_channel_pan(channel, pan)

        # Mute/Solo buttons
        col1, col2 = st.columns(2)

        with col1:
            muted = st.checkbox(
                "🔇 Mute",
                value=mixer._channel_mutes[channel],
                key=f"mute_{channel}"
            )
            mixer.set_channel_mute(channel, muted)

        with col2:
            soloed = st.checkbox(
                "🎯 Solo",
                value=mixer._channel_solos[channel],
                key=f"solo_{channel}"
            )
            mixer.set_channel_solo(channel, soloed)

        # Metering
        peak = mixer.get_channel_peak(channel)
        rms = mixer.get_channel_rms(channel)

        st.markdown("**Meters:**")
        st.progress(min(peak, 1.0), text=f"Peak: {peak:.3f}")
        st.progress(min(rms, 1.0), text=f"RMS: {rms:.3f}")

        # Send levels
        if mixer.num_send_buses > 0:
            st.markdown("**Sends:**")
            for send_idx in range(mixer.num_send_buses):
                send_name = st.session_state.send_names[send_idx]
                current_send = mixer._channel_sends[channel][send_idx]
                send_level = st.slider(
                    f"{send_name}",
                    min_value=0.0,
                    max_value=1.0,
                    value=current_send,
                    step=0.01,
                    key=f"send_{channel}_{send_idx}"
                )
                mixer.set_channel_send(channel, send_idx, send_level)

        st.divider()


def render_send_bus(send_idx: int, name: str):
    """Render a send/return bus."""
    mixer: MixerEngine = st.session_state.mixer

    st.markdown(f"#### 🔄 {name}")

    # Return level
    current_return = mixer._send_return_levels[send_idx]
    return_level = st.slider(
        "Return Level",
        min_value=0.0,
        max_value=2.0,
        value=current_return,
        step=0.01,
        key=f"return_{send_idx}"
    )
    mixer.set_send_return_level(send_idx, return_level)

    # Mute
    muted = st.checkbox(
        "🔇 Mute",
        value=mixer._send_mutes[send_idx],
        key=f"send_mute_{send_idx}"
    )
    mixer.set_send_mute(send_idx, muted)


def render_master_section():
    """Render master bus controls."""
    mixer: MixerEngine = st.session_state.mixer

    st.markdown("## 🎛️ Master Bus")

    # Master gain
    master_gain = st.slider(
        "Master Gain (dB)",
        min_value=-12.0,
        max_value=12.0,
        value=mixer._master_gain,
        step=0.1,
        key="master_gain"
    )
    mixer.set_master_gain(master_gain)

    # Limiter
    st.markdown("### 🛡️ Limiter")

    limiter_enabled = st.checkbox(
        "Enable Limiter",
        value=mixer._master_limiter_enabled,
        key="limiter_enabled"
    )

    limiter_threshold = st.slider(
        "Threshold (dB)",
        min_value=-12.0,
        max_value=0.0,
        value=mixer._master_limiter_threshold,
        step=0.1,
        key="limiter_threshold"
    )

    mixer.set_master_limiter(limiter_enabled, limiter_threshold)

    # Master meters
    st.markdown("### 📊 Master Meters")
    peak_l = mixer.get_master_peak_l()
    peak_r = mixer.get_master_peak_r()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Peak L", f"{peak_l:.3f}")
        st.progress(min(peak_l, 1.0))

    with col2:
        st.metric("Peak R", f"{peak_r:.3f}")
        st.progress(min(peak_r, 1.0))


def render_emotion_presets():
    """Render emotion preset selector."""
    st.markdown("## 🎭 Emotion Presets")

    mapper: EmotionMapper = st.session_state.emotion_mapper

    # Preset selector
    preset_names = mapper.list_presets()
    selected_preset = st.selectbox(
        "Select Emotion Preset",
        options=preset_names,
        key="emotion_preset"
    )

    # Target channel selector
    mixer: MixerEngine = st.session_state.mixer
    target_channel = st.selectbox(
        "Apply to Channel",
        options=list(range(mixer.num_channels)),
        format_func=lambda i: st.session_state.channel_names[i],
        key="emotion_target_channel"
    )

    # Apply button
    if st.button("🎨 Apply Emotion Preset", key="apply_emotion"):
        emotion_params = mapper.get_preset(selected_preset)
        if emotion_params:
            apply_emotion_to_mixer(mixer, emotion_params, target_channel)
            st.success(f"Applied '{selected_preset}' to {st.session_state.channel_names[target_channel]}")
            st.rerun()

    # Show preset details
    if selected_preset:
        preset = mapper.get_preset(selected_preset)
        if preset:
            with st.expander("📝 Preset Details"):
                st.markdown(f"**Description:** {preset.description}")
                st.markdown(f"**Tags:** {', '.join(preset.tags)}")
                st.markdown(f"**Emotional Justification:**")
                st.info(preset.emotional_justification)


def render_test_signal_generator():
    """Render test signal generator."""
    st.markdown("## 🎵 Test Signal Generator")

    # Signal type
    signal_type = st.selectbox(
        "Signal Type",
        options=["Sine Wave", "White Noise", "Pink Noise", "Impulse"],
        key="signal_type"
    )

    # Frequency (for sine wave)
    frequency = 440.0
    if signal_type == "Sine Wave":
        frequency = st.slider(
            "Frequency (Hz)",
            min_value=20.0,
            max_value=20000.0,
            value=440.0,
            step=1.0,
            key="signal_frequency"
        )

    # Duration
    duration = st.slider(
        "Duration (seconds)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        key="signal_duration"
    )

    # Generate button
    if st.button("🎼 Generate & Process", key="generate_signal"):
        mixer: MixerEngine = st.session_state.mixer

        num_frames = int(duration * mixer.sample_rate)
        t = np.linspace(0, duration, num_frames)

        # Generate test signals for each channel
        inputs = []
        for ch in range(mixer.num_channels):
            if signal_type == "Sine Wave":
                # Different frequency for each channel
                freq = frequency * (1.0 + ch * 0.1)
                signal = np.sin(2 * np.pi * freq * t)
            elif signal_type == "White Noise":
                signal = np.random.randn(num_frames) * 0.1
            elif signal_type == "Pink Noise":
                signal = np.random.randn(num_frames) * 0.1
                # Simple pink noise approximation
                signal = np.cumsum(signal) / 10.0
            else:  # Impulse
                signal = np.zeros(num_frames)
                signal[0] = 1.0

            inputs.append(signal.astype(np.float32))

        inputs = np.array(inputs)

        # Process through mixer
        output_l, output_r = mixer.process(inputs)

        # Display results
        st.success(f"Processed {num_frames} frames")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Master Peak L", f"{mixer.get_master_peak_l():.3f}")
        with col2:
            st.metric("Master Peak R", f"{mixer.get_master_peak_r():.3f}")
        with col3:
            st.metric("Output Frames", f"{len(output_l)}")

        # Show channel activity
        st.markdown("### Channel Activity")
        for ch in range(mixer.num_channels):
            peak = mixer.get_channel_peak(ch)
            rms = mixer.get_channel_rms(ch)
            st.text(f"{st.session_state.channel_names[ch]}: Peak={peak:.3f}, RMS={rms:.3f}")


def render_session_management():
    """Render session save/load controls."""
    st.markdown("## 💾 Session Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Save Session", key="save_session"):
            mixer: MixerEngine = st.session_state.mixer
            state = mixer.get_state()
            st.session_state.saved_mixer_state = state
            st.success("Session saved!")

    with col2:
        if st.button("📤 Load Session", key="load_session"):
            if 'saved_mixer_state' in st.session_state:
                mixer: MixerEngine = st.session_state.mixer
                mixer.load_state(st.session_state.saved_mixer_state)
                st.success("Session loaded!")
                st.rerun()
            else:
                st.warning("No saved session found")

    # Reset all
    if st.button("🔄 Reset All", key="reset_all"):
        mixer: MixerEngine = st.session_state.mixer
        mixer.set_num_channels(8)
        for ch in range(8):
            mixer.set_channel_gain(ch, 0.0)
            mixer.set_channel_pan(ch, 0.0)
            mixer.set_channel_mute(ch, False)
            mixer.set_channel_solo(ch, False)
        mixer.set_master_gain(0.0)
        mixer.reset_all_meters()
        st.success("Mixer reset!")
        st.rerun()


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="iDAW Mixer Console",
        page_icon="🎛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()

    # Header
    st.title("🎛️ iDAW Mixer Console")
    st.markdown("""
    Real-time mixer control panel for the Penta-Core audio engine.
    Adjust channels, sends, and master bus. Apply emotion-based presets
    for instant creative mixing.
    """)

    st.divider()

    # Sidebar
    with st.sidebar:
        st.markdown("## 🎨 Quick Actions")

        render_emotion_presets()

        st.divider()

        render_session_management()

        st.divider()

        render_test_signal_generator()

        st.divider()

        st.markdown("## ℹ️ About")
        st.info("""
        **iDAW Mixer Console**

        Part of the iDAW (Intelligent Digital Audio Workstation) suite.

        - Real-time safe mixer engine
        - Emotion-based preset system
        - Multi-channel mixing
        - Send/return buses
        - Master limiting
        """)

    # Main content area
    tabs = st.tabs(["📊 Channels", "🔄 Sends", "🎛️ Master"])

    with tabs[0]:
        st.markdown("# Channel Strips")

        mixer: MixerEngine = st.session_state.mixer
        num_channels = mixer.num_channels

        # Display channels in columns
        cols_per_row = 4
        for row_start in range(0, num_channels, cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, channel in enumerate(range(row_start, min(row_start + cols_per_row, num_channels))):
                with cols[col_idx]:
                    render_channel_strip(channel, st.session_state.channel_names[channel])

    with tabs[1]:
        st.markdown("# Send/Return Buses")

        mixer: MixerEngine = st.session_state.mixer
        num_sends = mixer.num_send_buses

        if num_sends > 0:
            cols = st.columns(min(num_sends, 4))
            for send_idx in range(num_sends):
                with cols[send_idx % 4]:
                    render_send_bus(send_idx, st.session_state.send_names[send_idx])
        else:
            st.info("No send buses configured")

    with tabs[2]:
        render_master_section()

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    iDAW Mixer Console | Powered by Penta-Core | "Interrogate Before Generate"
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
