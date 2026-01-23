"""
Example: Spectocloud 3D Visualization

Demonstrates the Spectocloud visualization system with sample MIDI data
and emotional trajectory.

This creates a 3D view showing:
- Stationary anchor points for musical parameters
- A conical lifeline traveling through time
- Storm cloud dynamics with emotional adhesion
- Lightning effects when charge accumulates
"""

import numpy as np
from music_brain.visualization import Spectocloud


def generate_sample_midi_events(duration=10.0, bpm=120):
    """
    Generate sample MIDI events for demonstration.
    
    Creates a simple progression with varying note density and pitch.
    """
    events = []
    beat_duration = 60.0 / bpm  # seconds per beat
    
    # Simple chord progression: I - IV - V - I
    chord_roots = [60, 65, 67, 60]  # C, F, G, C
    chord_patterns = [
        [0, 4, 7],      # Major triad
        [0, 4, 7],
        [0, 4, 7],
        [0, 4, 7],
    ]
    
    time = 0.0
    beat = 0
    
    while time < duration:
        # Determine current chord
        chord_idx = (beat // 4) % len(chord_roots)
        root = chord_roots[chord_idx]
        pattern = chord_patterns[chord_idx]
        
        # Add chord notes
        for offset in pattern:
            events.append({
                'time': time,
                'type': 'note_on',
                'note': root + offset,
                'velocity': 60 + int(20 * np.sin(2 * np.pi * time / duration)),
            })
        
        # Add some melody notes
        if beat % 2 == 0:
            melody_note = root + 12 + int(7 * np.sin(2 * np.pi * time / duration))
            events.append({
                'time': time + beat_duration * 0.25,
                'type': 'note_on',
                'note': melody_note,
                'velocity': 80,
            })
        
        time += beat_duration
        beat += 1
    
    return events


def generate_sample_emotion_trajectory(duration=10.0, window_size=0.2):
    """
    Generate sample emotional trajectory over time.
    
    Models an emotional journey from grief to hope.
    """
    n_windows = int(duration / window_size)
    trajectory = []
    
    for i in range(n_windows):
        t_norm = i / max(1, n_windows - 1)
        
        # Emotional arc: start negative, build to positive
        valence = -0.7 + 1.4 * t_norm
        valence += 0.2 * np.sin(2 * np.pi * 2.0 * t_norm)  # Add variation
        
        # Arousal builds in middle, settles at end
        arousal = 0.3 + 0.6 * np.sin(np.pi * t_norm)
        arousal += 0.15 * np.sin(2 * np.pi * 3.0 * t_norm)
        
        # Intensity follows arousal
        intensity = 0.4 + 0.5 * arousal
        
        # Add dramatic moment at 60% mark
        if 0.55 < t_norm < 0.65:
            valence -= 0.3
            arousal += 0.2
        
        trajectory.append({
            'valence': np.clip(valence, -1.0, 1.0),
            'arousal': np.clip(arousal, 0.0, 1.0),
            'intensity': np.clip(intensity, 0.0, 1.0),
        })
    
    return trajectory


def main():
    """Run Spectocloud example."""
    print("=" * 60)
    print("Spectocloud 3D Visualization Example")
    print("=" * 60)
    
    # Configuration
    duration = 10.0  # seconds
    bpm = 120
    
    print(f"\nGenerating sample data:")
    print(f"  Duration: {duration}s @ {bpm} BPM")
    
    # Generate sample MIDI
    print("  - Creating MIDI events...")
    midi_events = generate_sample_midi_events(duration=duration, bpm=bpm)
    print(f"    Generated {len(midi_events)} MIDI events")
    
    # Generate emotional trajectory
    print("  - Creating emotion trajectory...")
    emotion_trajectory = generate_sample_emotion_trajectory(duration=duration)
    print(f"    Generated {len(emotion_trajectory)} emotion windows")
    
    # Create Spectocloud visualization
    print("\nInitializing Spectocloud...")
    spectocloud = Spectocloud(
        anchor_density="normal",  # Can be "sparse", "normal", or "dense"
        n_particles=1500,         # Number of cloud particles
        window_size=0.2,          # Feature extraction window (seconds)
    )
    print(f"  - Anchor library: {len(spectocloud.anchor_library.anchors)} anchors")
    print(f"  - Particle count: {spectocloud.n_particles}")
    
    # Process MIDI with emotion trajectory
    print("\nProcessing MIDI events...")
    spectocloud.process_midi(
        midi_events=midi_events,
        duration=duration,
        emotion_trajectory=emotion_trajectory,
    )
    print(f"  - Generated {len(spectocloud.frames)} frames")
    print(f"  - Storm charge: {spectocloud.storm.charge:.2f}")
    
    # Render key frames
    print("\nRendering visualization frames...")
    
    # Beginning
    print("  - Rendering frame 0 (beginning)...")
    spectocloud.render_static_frame(
        frame_idx=0,
        output_path="/tmp/spectocloud_frame_0.png",
        show=False,
    )
    
    # Middle
    mid_idx = len(spectocloud.frames) // 2
    print(f"  - Rendering frame {mid_idx} (middle)...")
    spectocloud.render_static_frame(
        frame_idx=mid_idx,
        output_path="/tmp/spectocloud_frame_mid.png",
        show=False,
    )
    
    # End
    end_idx = len(spectocloud.frames) - 1
    print(f"  - Rendering frame {end_idx} (end)...")
    spectocloud.render_static_frame(
        frame_idx=end_idx,
        output_path="/tmp/spectocloud_frame_end.png",
        show=False,
    )
    
    # Export data
    print("\nExporting visualization data...")
    spectocloud.export_data("/tmp/spectocloud_data.json")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nOutput files:")
    print("  - /tmp/spectocloud_frame_0.png")
    print("  - /tmp/spectocloud_frame_mid.png")
    print("  - /tmp/spectocloud_frame_end.png")
    print("  - /tmp/spectocloud_data.json")
    print("\nKey observations:")
    print("  - Anchors remain stationary (musical parameter space)")
    print("  - Lifeline expands conically along time axis")
    print("  - Cloud spread increases with arousal and time")
    print("  - Storm charge accumulates from anchor proximity")
    print("  - Valence controls color (blue=negative, red=positive)")
    
    # Print some frame statistics
    print("\nFrame statistics:")
    valences = [f.valence for f in spectocloud.frames]
    arousals = [f.arousal for f in spectocloud.frames]
    spreads = [f.spread for f in spectocloud.frames]
    
    print(f"  Valence range: [{min(valences):.2f}, {max(valences):.2f}]")
    print(f"  Arousal range: [{min(arousals):.2f}, {max(arousals):.2f}]")
    print(f"  Spread range: [{min(spreads):.2f}, {max(spreads):.2f}]")
    print(f"  Final storm charge: {spectocloud.storm.charge:.2f}")


if __name__ == "__main__":
    main()
