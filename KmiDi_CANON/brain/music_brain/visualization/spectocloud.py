"""
Spectocloud: 3D Song Lifeline Visualization

A visualization that shows what musical "space" a song occupies over time,
with emotions accumulating and "sticking" to musically meaningful regions
through a snowball/memory effect.

Spine-included per docs/CONTRACTS.md §5b. Consumed by body via POST /spectocloud/render.
Input: midi_events or midi_file_path; optional duration, emotion_trajectory, mode (static|animation), etc.
Output: status, mode, output_path, frames.

Core concepts:
- Hard-truth musical parameters as stationary anchors (coordinate system)
- Song lifeline as a moving trajectory (time-ordered, conical expansion)
- Emotional adhesion with sticky accumulation (electrostatic storm model)

Design: X=time, Y=frequency/pitch, Z=musical parameter composite
Additional channels: color (emotion), size (energy), opacity (recency/charge)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import tempfile

# Optional rendering dependencies
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    Axes3D = None
    MATPLOTLIB_AVAILABLE = False

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    mido = None
    MIDO_AVAILABLE = False


class AnchorFamily(Enum):
    """Musical parameter families for anchors."""
    HARMONY = "harmony"
    RHYTHM = "rhythm"
    DYNAMICS = "dynamics"
    TIMBRE = "timbre"
    TEXTURE = "texture"


@dataclass
class Anchor:
    """
    A stationary point in musical space representing a specific parameter subset.
    
    Anchors serve as:
    - Coordinate system for musical meaning
    - Attractors for emotional adhesion
    - Stable backdrop for interpretable song paths
    """
    id: str
    name: str
    family: AnchorFamily
    
    position_y: float  # Frequency/pitch dimension
    position_z: float  # Musical parameter composite
    
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    base_charge: float = 1.0
    polarity: float = 1.0
    radius: float = 0.35
    
    color: Tuple[float, float, float] = (0.7, 0.7, 0.7)
    base_opacity: float = 0.06
    base_size: float = 5.0
    
    activation: float = 0.0
    accumulated_glow: float = 0.0


@dataclass
class Frame:
    """A single time window in the song lifeline."""
    time: float
    x: float  # Time
    y: float  # Frequency/pitch centroid
    z: float  # Musical parameter composite
    
    features: Dict[str, float] = field(default_factory=dict)
    top_anchors: List[Tuple[str, float]] = field(default_factory=list)
    
    valence: float = 0.0
    arousal: float = 0.5
    intensity: float = 0.5
    emotion_label: Optional[str] = None
    
    spread: float = 0.16
    opacity: float = 0.02


@dataclass
class EmotionParticle:
    """A particle representing emotional mass that can attach to anchors."""
    position: np.ndarray
    velocity: np.ndarray
    mass: float = 1.0
    attached_anchors: Dict[str, float] = field(default_factory=dict)
    valence: float = 0.0
    arousal: float = 0.5
    
    def __post_init__(self):
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=float)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=float)


@dataclass
class StormState:
    """Electrostatic storm accumulator for lightning and charge dynamics."""
    charge: float = 0.0
    energy: float = 0.0
    active_arcs: List[Tuple[int, str]] = field(default_factory=list)
    leak: float = 0.06
    gain: float = 1.2
    threshold: float = 14.0
    discharge_factor: float = 0.45


class MusicalParameterExtractor:
    """Extracts musical features from MIDI data."""
    
    def extract_from_midi_window(
        self,
        midi_events: List[Dict],
        window_start: float,
        window_end: float
    ) -> Dict[str, float]:
        """Extract features from MIDI events in a time window."""
        notes = [e for e in midi_events 
                if e.get('type') == 'note_on' 
                and window_start <= e.get('time', 0) < window_end]
        
        if not notes:
            return {
                'note_density': 0.0,
                'pitch_centroid': 0.5,
                'velocity_centroid': 0.5,
                'pitch_range': 0.0,
                'tension': 0.0,
                'rhythmic_regularity': 0.5,
            }
        
        pitches = [n['note'] for n in notes]
        velocities = [n.get('velocity', 64) for n in notes]
        
        return {
            'note_density': min(1.0, len(notes) / 20.0),
            'pitch_centroid': (np.mean(pitches) - 36) / 72,
            'pitch_range': (np.max(pitches) - np.min(pitches)) / 48.0,
            'velocity_centroid': np.mean(velocities) / 127.0,
            'tension': min(1.0, np.std(pitches) / 12.0),
            'rhythmic_regularity': 0.5,
        }


class AnchorLibrary:
    """Default musical parameter anchors."""
    
    def __init__(self):
        self.anchors = self._create_default_anchors()
    
    def _create_default_anchors(self) -> List[Anchor]:
        """Create standard musical space anchors."""
        return [
            Anchor("harmony_simple", "Simple Harmony", AnchorFamily.HARMONY, 0.3, 0.2),
            Anchor("harmony_complex", "Complex Harmony", AnchorFamily.HARMONY, 0.7, 0.8),
            Anchor("rhythm_sparse", "Sparse Rhythm", AnchorFamily.RHYTHM, 0.2, 0.3),
            Anchor("rhythm_dense", "Dense Rhythm", AnchorFamily.RHYTHM, 0.8, 0.7),
            Anchor("dynamics_soft", "Soft Dynamics", AnchorFamily.DYNAMICS, 0.3, 0.4),
            Anchor("dynamics_loud", "Loud Dynamics", AnchorFamily.DYNAMICS, 0.7, 0.9),
        ]


class Renderer3D:
    """3D matplotlib renderer for Spectocloud."""
    
    def __init__(self, figsize=(12, 9), dpi=120):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib not available for rendering")
        self.figsize = figsize
        self.dpi = dpi
        self.plt = plt
    
    def render_frame(
        self,
        anchors: List[Anchor],
        frames: List[Frame],
        particles: List[EmotionParticle],
        storm: StormState,
        current_frame_idx: int,
        title: str = "Spectocloud",
        elev: float = 22.0,
        azim: float = 45.0,
    ):
        """Render a single frame to matplotlib figure."""
        fig = self.plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw anchors
        for anchor in anchors:
            ax.scatter(0.5, anchor.position_y, anchor.position_z,
                      c=[anchor.color], s=anchor.base_size * 10,
                      alpha=anchor.base_opacity + anchor.activation * 0.3,
                      marker='o')
        
        # Draw lifeline (frames up to current)
        if current_frame_idx > 0:
            past_frames = frames[:current_frame_idx + 1]
            xs = [f.x for f in past_frames]
            ys = [f.y for f in past_frames]
            zs = [f.z for f in past_frames]
            ax.plot(xs, ys, zs, 'b-', alpha=0.3, linewidth=1)
        
        # Draw particles
        if particles:
            positions = np.array([p.position for p in particles])
            colors = [(0.9, 0.5 + p.valence * 0.5, 0.3) for p in particles]
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c=colors, s=2, alpha=0.4, marker='.')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Pitch/Frequency')
        ax.set_zlabel('Musical Parameters')
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        
        return fig


class Spectocloud:
    """Main Spectocloud visualization engine."""
    
    def __init__(
        self,
        n_particles: int = 400,
        window_size: float = 0.5,
        anchor_density: str = "default",
    ):
        self.n_particles = n_particles
        self.window_size = window_size
        self.anchor_library = AnchorLibrary()
        self.extractor = MusicalParameterExtractor()
        self.storm = StormState()
        self.frames: List[Frame] = []
        
        if MATPLOTLIB_AVAILABLE:
            self.renderer = Renderer3D()
        else:
            self.renderer = None
    
    def process_midi(
        self,
        midi_events: List[Dict[str, Any]],
        duration: Optional[float] = None,
        emotion_trajectory: Optional[List[Dict[str, Any]]] = None,
    ):
        """Process MIDI events into frames."""
        if not midi_events:
            return
        
        max_time = max(e.get('time', 0) for e in midi_events)
        duration = duration or max_time
        
        # Create frames
        num_windows = max(1, int(duration / self.window_size))
        self.frames = []
        
        for i in range(num_windows):
            window_start = i * self.window_size
            window_end = (i + 1) * self.window_size
            
            features = self.extractor.extract_from_midi_window(
                midi_events, window_start, window_end
            )
            
            frame = Frame(
                time=window_start,
                x=window_start / duration if duration > 0 else 0.5,
                y=features.get('pitch_centroid', 0.5),
                z=features.get('note_density', 0.5),
                features=features,
                valence=0.0,
                arousal=0.5,
                intensity=features.get('velocity_centroid', 0.5),
            )
            
            # Match to anchors
            for anchor in self.anchor_library.anchors:
                dist = np.sqrt((frame.y - anchor.position_y)**2 + 
                              (frame.z - anchor.position_z)**2)
                if dist < anchor.radius:
                    weight = 1.0 - (dist / anchor.radius)
                    frame.top_anchors.append((anchor.id, weight))
                    anchor.activation = max(anchor.activation, weight)
            
            frame.top_anchors.sort(key=lambda x: x[1], reverse=True)
            self.frames.append(frame)
    
    def process_midi_file(self, midi_file_path: str, **kwargs):
        """Load MIDI file and process."""
        if not MIDO_AVAILABLE:
            raise ImportError("mido not available for MIDI file loading")
        
        mid = mido.MidiFile(midi_file_path)
        events = []
        time = 0.0
        
        for msg in mid:
            time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                events.append({
                    'type': 'note_on',
                    'time': time,
                    'note': msg.note,
                    'velocity': msg.velocity,
                })
        
        self.process_midi(events, **kwargs)
    
    def render_static_frame(
        self,
        frame_idx: int,
        output_path: Optional[str] = None,
    ):
        """Render a single static frame."""
        if not self.frames:
            raise ValueError("No frames to render. Call process_midi() first.")
        if not self.renderer:
            raise ImportError("matplotlib not available for rendering")
        
        frame_idx = min(frame_idx, len(self.frames) - 1)
        particles = []
        
        # Generate particles around current frame
        if frame_idx < len(self.frames):
            frame = self.frames[frame_idx]
            for _ in range(self.n_particles):
                offset = np.random.normal(size=3) * frame.spread
                position = np.array([frame.x, frame.y, frame.z]) + offset
                particles.append(EmotionParticle(
                    position=position,
                    velocity=np.zeros(3),
                    valence=frame.valence,
                    arousal=frame.arousal,
                ))
        
        fig = self.renderer.render_frame(
            anchors=self.anchor_library.anchors,
            frames=self.frames,
            particles=particles,
            storm=self.storm,
            current_frame_idx=frame_idx,
            title="Spectocloud: Musical Space",
            elev=22.0,
            azim=45.0,
        )
        
        if fig and output_path:
            fig.savefig(output_path, dpi=self.renderer.dpi, bbox_inches='tight')
            self.renderer.plt.close(fig)
        
        return output_path


def render_spectocloud(
    midi_events: Optional[List[Dict[str, Any]]] = None,
    midi_file_path: Optional[str] = None,
    duration: Optional[float] = None,
    emotion_trajectory: Optional[List[Dict[str, Any]]] = None,
    mode: str = "static",
    frame_idx: Optional[int] = None,
    output_path: Optional[str] = None,
    fps: int = 24,
    rotate: bool = False,
    anchor_density: Optional[str] = None,
    n_particles: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Render spectocloud from MIDI events or file.

    Matches body contract: SpectocloudRenderRequest → SpectocloudRenderResponse
    (status, mode, output_path, frames).

    Args:
        midi_events: List of MIDI event dicts (e.g. from body).
        midi_file_path: Path to MIDI file (alternative to midi_events).
        duration: Optional duration in seconds.
        emotion_trajectory: Optional list of emotion waypoints.
        mode: "static" or "animation".
        frame_idx: For static mode, which frame to render (default: last).
        output_path: Where to write output (image or frame dir).
        fps: Frames per second for animation.
        rotate: Whether to rotate cloud.
        anchor_density: Density preset name.
        n_particles: Number of particles (if supported).
        **kwargs: Ignored extra keys from API.

    Returns:
        Dict with keys: status, mode, output_path, frames.
    """
    # Validation
    if (not midi_events or len(midi_events) == 0) and not midi_file_path:
        return {
            "status": "error",
            "mode": mode,
            "output_path": "",
            "frames": 0,
            "details": "provide midi_events or midi_file_path",
        }
    
    try:
        # Initialize engine
        engine = Spectocloud(
            n_particles=n_particles or 400,
            anchor_density=anchor_density or "default",
        )
        
        # Process MIDI
        if midi_file_path:
            engine.process_midi_file(midi_file_path, duration=duration, emotion_trajectory=emotion_trajectory)
        else:
            engine.process_midi(midi_events, duration=duration, emotion_trajectory=emotion_trajectory)
        
        # Determine output path
        if output_path is None:
            out_dir = Path(tempfile.gettempdir()) / "spectocloud"
            out_dir.mkdir(parents=True, exist_ok=True)
            if mode == "animation":
                output_path = str(out_dir / "spectocloud_frames")
            else:
                output_path = str(out_dir / "spectocloud_static.png")
        
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        
        # Render
        if mode == "static":
            if not MATPLOTLIB_AVAILABLE:
                # Fallback: write manifest
                manifest = {
                    "mode": mode,
                    "n_frames": len(engine.frames),
                    "n_particles": n_particles or 400,
                    "note": "matplotlib not available; full render skipped",
                }
                manifest_path = out_p.with_suffix(out_p.suffix + ".manifest.json")
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)
                return {
                    "status": "completed",
                    "mode": mode,
                    "output_path": str(manifest_path),
                    "frames": 1,
                    "details": "matplotlib unavailable; manifest only",
                }
            
            idx = frame_idx if frame_idx is not None else len(engine.frames) - 1
            engine.render_static_frame(idx, str(out_p))
            frames = 1
        else:
            # Animation: render multiple frames
            if not MATPLOTLIB_AVAILABLE:
                return {
                    "status": "error",
                    "mode": mode,
                    "output_path": "",
                    "frames": 0,
                    "details": "matplotlib required for animation",
                }
            
            out_p.mkdir(parents=True, exist_ok=True)
            frames = len(engine.frames)
            
            for i in range(frames):
                frame_path = out_p / f"frame_{i:04d}.png"
                engine.render_static_frame(i, str(frame_path))
        
        return {
            "status": "completed",
            "mode": mode,
            "output_path": str(output_path),
            "frames": frames,
        }
    
    except Exception as e:
        return {
            "status": "error",
            "mode": mode,
            "output_path": "",
            "frames": 0,
            "details": str(e),
        }


def render_spectocloud_from_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point for API: accept full request dict, return response dict."""
    return render_spectocloud(
        midi_events=payload.get("midi_events"),
        midi_file_path=payload.get("midi_file_path"),
        duration=payload.get("duration"),
        emotion_trajectory=payload.get("emotion_trajectory"),
        mode=payload.get("mode") or "static",
        frame_idx=payload.get("frame_idx"),
        output_path=payload.get("output_path"),
        fps=payload.get("fps", 24),
        rotate=payload.get("rotate", False),
        anchor_density=payload.get("anchor_density"),
        n_particles=payload.get("n_particles"),
        **{k: v for k, v in payload.items() if k not in (
            "midi_events", "midi_file_path", "duration", "emotion_trajectory",
            "mode", "frame_idx", "output_path", "fps", "rotate",
            "anchor_density", "n_particles",
        )},
    )
