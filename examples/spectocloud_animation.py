"""
Example: Spectocloud Animation

Demonstrates creating an animated GIF of the Spectocloud visualization
showing the conical lifeline traveling through musical space over time.
"""

import numpy as np
from music_brain.visualization import Spectocloud


def generate_chord_progression_midi(duration=8.0, bpm=100):
    """
    Generate MIDI events for a chord progression with emotional arc.
    
    Creates: i - VI - III - VII (minor progression with tension)
    """
    events = []
    beat_duration = 60.0 / bpm
    
    # Chord progression in Am
    chords = [
        {'root': 57, 'type': 'minor'},   # Am (i)
        {'root': 65, 'type': 'major'},   # F  (VI)
        {'root': 60, 'type': 'major'},   # C  (III)
        {'root': 67, 'type': 'major'},   # G  (VII)
    ]
    
    chord_patterns = {
        'minor': [0, 3, 7],              # Minor triad
        'major': [0, 4, 7],              # Major triad
    }
    
    time = 0.0
    beat = 0
    
    while time < duration:
        # Current chord (2 beats per chord)
        chord_idx = (beat // 2) % len(chords)
        chord = chords[chord_idx]
        root = chord['root']
        pattern = chord_patterns[chord['type']]
        
        # Add chord notes
        for offset in pattern:
            # Velocity varies with position in progression
            base_vel = 50 + int(30 * (time / duration))
            velocity = base_vel + int(10 * np.sin(2 * np.pi * time / (duration / 2)))
            
            events.append({
                'time': time,
                'type': 'note_on',
                'note': root + offset,
                'velocity': velocity,
            })
        
        # Add melody notes (varied rhythm)
        if beat % 2 == 0:
            # Melody rises with emotional arc
            melody_offset = int(12 + 7 * (time / duration))
            melody_note = root + melody_offset
            
            events.append({
                'time': time + beat_duration * 0.25,
                'type': 'note_on',
                'note': melody_note,
                'velocity': 70 + int(20 * (time / duration)),
            })
        
        # Add bass note
        events.append({
            'time': time,
            'type': 'note_on',
            'note': root - 12,
            'velocity': 60,
        })
        
        time += beat_duration
        beat += 1
    
    return events


def generate_emotional_journey(duration=8.0, window_size=0.2):
    """
    Generate emotional trajectory: grief → longing → hope → euphoria
    """
    n_windows = int(duration / window_size)
    trajectory = []
    
    for i in range(n_windows):
        t_norm = i / max(1, n_windows - 1)
        
        # Emotional journey stages
        if t_norm < 0.25:
            # Stage 1: Grief (low valence, low arousal)
            valence = -0.8 + 0.3 * (t_norm / 0.25)
            arousal = 0.2 + 0.3 * (t_norm / 0.25)
            intensity = 0.6
        elif t_norm < 0.5:
            # Stage 2: Longing (rising arousal, still negative)
            stage = (t_norm - 0.25) / 0.25
            valence = -0.5 + 0.3 * stage
            arousal = 0.5 + 0.3 * stage
            intensity = 0.7 + 0.2 * stage
        elif t_norm < 0.75:
            # Stage 3: Hope (crossing to positive)
            stage = (t_norm - 0.5) / 0.25
            valence = -0.2 + 0.8 * stage
            arousal = 0.8 - 0.1 * stage
            intensity = 0.9
        else:
            # Stage 4: Euphoria (high valence, high arousal)
            stage = (t_norm - 0.75) / 0.25
            valence = 0.6 + 0.3 * stage
            arousal = 0.7 + 0.2 * stage
            intensity = 0.9 + 0.1 * stage
        
        # Add micro-variations
        valence += 0.1 * np.sin(2 * np.pi * 3.0 * t_norm)
        arousal += 0.08 * np.sin(2 * np.pi * 5.0 * t_norm)
        
        trajectory.append({
            'valence': np.clip(valence, -1.0, 1.0),
            'arousal': np.clip(arousal, 0.0, 1.0),
            'intensity': np.clip(intensity, 0.0, 1.0),
        })
    
    return trajectory


def main():
    """Run Spectocloud animation example."""
    print("=" * 70)
    print("Spectocloud Animation Example")
    print("=" * 70)
    
    # Configuration
    duration = 8.0
    bpm = 100
    
    print(f"\nGenerating musical data:")
    print(f"  Duration: {duration}s @ {bpm} BPM")
    print(f"  Progression: Am - F - C - G (emotional journey)")
    
    # Generate MIDI
    print("  - Creating MIDI events...")
    midi_events = generate_chord_progression_midi(duration=duration, bpm=bpm)
    print(f"    Generated {len(midi_events)} MIDI events")
    
    # Generate emotion trajectory
    print("  - Creating emotional journey (grief → euphoria)...")
    emotion_trajectory = generate_emotional_journey(duration=duration)
    print(f"    Generated {len(emotion_trajectory)} emotion windows")
    
    # Create Spectocloud
    print("\nInitializing Spectocloud...")
    spectocloud = Spectocloud(
        anchor_density="normal",
        n_particles=1200,  # Fewer particles for faster animation
        window_size=0.2,
    )
    print(f"  - {len(spectocloud.anchor_library.anchors)} anchors")
    print(f"  - {spectocloud.n_particles} particles per frame")
    
    # Process MIDI
    print("\nProcessing MIDI with emotion trajectory...")
    spectocloud.process_midi(
        midi_events=midi_events,
        duration=duration,
        emotion_trajectory=emotion_trajectory,
    )
    print(f"  - Generated {len(spectocloud.frames)} frames")
    
    # Print emotional journey stats
    print("\nEmotional journey:")
    for stage, desc in [
        (0, "Start (Grief)"),
        (len(spectocloud.frames) // 4, "Quarter (Longing)"),
        (len(spectocloud.frames) // 2, "Middle (Hope)"),
        (3 * len(spectocloud.frames) // 4, "Three-quarter (Rising)"),
        (len(spectocloud.frames) - 1, "End (Euphoria)"),
    ]:
        if stage < len(spectocloud.frames):
            frame = spectocloud.frames[stage]
            print(f"  {desc:20s}: v={frame.valence:+.2f}, a={frame.arousal:.2f}, "
                  f"spread={frame.spread:.2f}")
    
    # Render animation
    print("\n" + "=" * 70)
    print("Rendering animation...")
    print("=" * 70)
    print("\nThis may take a minute...")
    
    output_path = "/tmp/spectocloud_journey.gif"
    
    spectocloud.render_animation(
        output_path=output_path,
        fps=15,              # 15 FPS for reasonable file size
        duration=None,       # Use all frames
        rotate=True,         # Rotate camera view
    )
    
    print("\n" + "=" * 70)
    print("Animation complete!")
    print("=" * 70)
    print(f"\nOutput: {output_path}")
    
    # File size check
    import os
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")
    
    print("\nVisualization features demonstrated:")
    print("  ✓ Conical lifeline traveling along time axis")
    print("  ✓ Color transition (blue → neutral → red) with valence")
    print("  ✓ Cloud expansion with arousal and time")
    print("  ✓ Stationary anchors (musical parameter space)")
    print("  ✓ Storm charge accumulation")
    print("  ✓ Rotating camera perspective")


if __name__ == "__main__":
    main()
