#!/usr/bin/env python
"""
Spectocloud CLI - Command line interface for 3D visualization

Usage:
    spectocloud_cli.py render <midi_file> [options]
    spectocloud_cli.py animate <midi_file> [options]
    spectocloud_cli.py --help
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict


def load_midi_from_file(midi_path: str) -> tuple:
    """
    Load MIDI file and convert to events.
    
    Returns:
        (events, duration) tuple
    """
    try:
        import mido
    except ImportError:
        print("Error: mido not installed. Install with: pip install mido")
        sys.exit(1)
    
    mid = mido.MidiFile(midi_path)
    events = []
    current_time = 0.0
    
    for msg in mid:
        current_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            events.append({
                'time': current_time,
                'type': 'note_on',
                'note': msg.note,
                'velocity': msg.velocity,
            })
    
    duration = current_time if events else 10.0
    return events, duration


def load_emotion_from_file(emotion_path: str) -> List[Dict]:
    """Load emotion trajectory from JSON file."""
    with open(emotion_path, 'r') as f:
        data = json.load(f)
    
    # Support both direct list and wrapped format
    if isinstance(data, list):
        return data
    elif 'snapshots' in data:
        # Convert from EmotionTrajectory format
        return [
            {
                'valence': s.get('valence', 0.0),
                'arousal': s.get('arousal', 0.5),
                'intensity': s.get('intensity', 0.5),
            }
            for s in data['snapshots']
        ]
    else:
        print(f"Error: Unrecognized emotion format in {emotion_path}")
        sys.exit(1)


def cmd_render(args):
    """Render static frame(s) from MIDI."""
    from music_brain.visualization import Spectocloud
    
    print(f"Loading MIDI from {args.midi}")
    events, duration = load_midi_from_file(args.midi)
    print(f"  Loaded {len(events)} events, duration {duration:.1f}s")
    
    # Load emotion if provided
    emotion_trajectory = None
    if args.emotion:
        print(f"Loading emotion trajectory from {args.emotion}")
        emotion_trajectory = load_emotion_from_file(args.emotion)
        print(f"  Loaded {len(emotion_trajectory)} emotion windows")
    
    # Create Spectocloud
    print(f"\nInitializing Spectocloud...")
    print(f"  Anchor density: {args.anchors}")
    print(f"  Particles: {args.particles}")
    
    spectocloud = Spectocloud(
        anchor_density=args.anchors,
        n_particles=args.particles,
        window_size=args.window,
    )
    
    # Process
    print("\nProcessing MIDI...")
    spectocloud.process_midi(
        midi_events=events,
        duration=duration,
        emotion_trajectory=emotion_trajectory,
    )
    print(f"  Generated {len(spectocloud.frames)} frames")
    
    # Render frame(s)
    if args.frame == 'all':
        # Render beginning, middle, end
        frames_to_render = [
            (0, 'begin'),
            (len(spectocloud.frames) // 2, 'middle'),
            (len(spectocloud.frames) - 1, 'end'),
        ]
    else:
        frame_idx = int(args.frame)
        if frame_idx < 0 or frame_idx >= len(spectocloud.frames):
            print(f"Error: Frame {frame_idx} out of range (0-{len(spectocloud.frames)-1})")
            sys.exit(1)
        frames_to_render = [(frame_idx, str(frame_idx))]
    
    print(f"\nRendering {len(frames_to_render)} frame(s)...")
    
    output_base = Path(args.output).stem
    output_dir = Path(args.output).parent
    output_ext = Path(args.output).suffix or '.png'
    
    for idx, label in frames_to_render:
        output_path = output_dir / f"{output_base}_{label}{output_ext}"
        print(f"  Frame {idx} -> {output_path}")
        
        spectocloud.render_static_frame(
            frame_idx=idx,
            output_path=str(output_path),
            show=False,
        )
    
    print("\nDone!")


def cmd_animate(args):
    """Render animation from MIDI."""
    from music_brain.visualization import Spectocloud
    
    print(f"Loading MIDI from {args.midi}")
    events, duration = load_midi_from_file(args.midi)
    print(f"  Loaded {len(events)} events, duration {duration:.1f}s")
    
    # Load emotion if provided
    emotion_trajectory = None
    if args.emotion:
        print(f"Loading emotion trajectory from {args.emotion}")
        emotion_trajectory = load_emotion_from_file(args.emotion)
        print(f"  Loaded {len(emotion_trajectory)} emotion windows")
    
    # Create Spectocloud
    print(f"\nInitializing Spectocloud...")
    print(f"  Anchor density: {args.anchors}")
    print(f"  Particles: {args.particles}")
    
    spectocloud = Spectocloud(
        anchor_density=args.anchors,
        n_particles=args.particles,
        window_size=args.window,
    )
    
    # Process
    print("\nProcessing MIDI...")
    spectocloud.process_midi(
        midi_events=events,
        duration=duration,
        emotion_trajectory=emotion_trajectory,
    )
    print(f"  Generated {len(spectocloud.frames)} frames")
    
    # Render animation
    print(f"\nRendering animation to {args.output}...")
    print(f"  FPS: {args.fps}")
    print(f"  Rotate: {args.rotate}")
    print("  (This may take a minute...)")
    
    spectocloud.render_animation(
        output_path=args.output,
        fps=args.fps,
        duration=args.duration,
        rotate=args.rotate,
    )
    
    # Show file size
    import os
    if os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"\nAnimation saved: {args.output} ({size_mb:.2f} MB)")
    
    print("Done!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Spectocloud 3D Visualization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render middle frame
  %(prog)s render song.mid -o output.png -f 25
  
  # Render beginning, middle, end
  %(prog)s render song.mid -o output.png -f all
  
  # Render with emotion trajectory
  %(prog)s render song.mid -e emotion.json -o output.png
  
  # Create animated GIF
  %(prog)s animate song.mid -o animation.gif --fps 15
  
  # High quality animation with many particles
  %(prog)s animate song.mid -o hq.gif --particles 3000 --fps 20
        """,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Render command
    render_parser = subparsers.add_parser('render', help='Render static frame(s)')
    render_parser.add_argument('midi', help='MIDI file path')
    render_parser.add_argument('-o', '--output', default='spectocloud.png',
                              help='Output image path (default: spectocloud.png)')
    render_parser.add_argument('-f', '--frame', default='all',
                              help='Frame index or "all" (default: all)')
    render_parser.add_argument('-e', '--emotion', default=None,
                              help='Emotion trajectory JSON file')
    render_parser.add_argument('-a', '--anchors', default='normal',
                              choices=['sparse', 'normal', 'dense'],
                              help='Anchor density (default: normal)')
    render_parser.add_argument('-p', '--particles', type=int, default=1500,
                              help='Number of particles (default: 1500)')
    render_parser.add_argument('-w', '--window', type=float, default=0.2,
                              help='Time window size in seconds (default: 0.2)')
    
    # Animate command
    animate_parser = subparsers.add_parser('animate', help='Render animation/GIF')
    animate_parser.add_argument('midi', help='MIDI file path')
    animate_parser.add_argument('-o', '--output', default='spectocloud.gif',
                               help='Output GIF path (default: spectocloud.gif)')
    animate_parser.add_argument('-e', '--emotion', default=None,
                               help='Emotion trajectory JSON file')
    animate_parser.add_argument('--fps', type=int, default=15,
                               help='Frames per second (default: 15)')
    animate_parser.add_argument('--duration', type=float, default=None,
                               help='Max duration in seconds (default: use all frames)')
    animate_parser.add_argument('--rotate', action='store_true', default=True,
                               help='Rotate camera (default: True)')
    animate_parser.add_argument('--no-rotate', dest='rotate', action='store_false',
                               help='Disable camera rotation')
    animate_parser.add_argument('-a', '--anchors', default='normal',
                               choices=['sparse', 'normal', 'dense'],
                               help='Anchor density (default: normal)')
    animate_parser.add_argument('-p', '--particles', type=int, default=1200,
                               help='Number of particles (default: 1200)')
    animate_parser.add_argument('-w', '--window', type=float, default=0.2,
                               help='Time window size in seconds (default: 0.2)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Dispatch to command handler
    if args.command == 'render':
        cmd_render(args)
    elif args.command == 'animate':
        cmd_animate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
