"""
Spectocloud: 3D Song Lifeline Visualization

A visualization that shows what musical "space" a song occupies over time,
with emotions accumulating and "sticking" to musically meaningful regions
through a snowball/memory effect.

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


class AnchorFamily(Enum):
    """Musical parameter families for anchors."""
    HARMONY = "harmony"           # Key, mode, chord complexity, tension
    RHYTHM = "rhythm"             # Note density, syncopation, swing
    DYNAMICS = "dynamics"         # RMS energy, crest factor, velocity
    TIMBRE = "timbre"            # Spectral centroid, flux, roll-off
    TEXTURE = "texture"          # Polyphony, layer count, repetition


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
    
    # Normalized position in 3D space (Y, Z) - X is time, not used for anchors
    position_y: float  # Frequency/pitch dimension
    position_z: float  # Musical parameter composite
    
    # Musical parameter constraints
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Electrostatic properties
    base_charge: float = 1.0      # Static influence strength
    polarity: float = 1.0         # -1.0 (repel) to +1.0 (attract)
    radius: float = 0.35          # Influence falloff distance
    
    # Visual properties
    color: Tuple[float, float, float] = (0.7, 0.7, 0.7)
    base_opacity: float = 0.06
    base_size: float = 5.0
    
    # Runtime state
    activation: float = 0.0       # Current activation level (0-1)
    accumulated_glow: float = 0.0  # Memory of past activations


@dataclass
class Frame:
    """
    A single time window in the song lifeline.
    
    Represents the musical state at a specific moment in time.
    """
    time: float                   # Time in seconds or beats
    
    # 3D position
    x: float  # Time (matches self.time)
    y: float  # Frequency/pitch centroid
    z: float  # Musical parameter composite
    
    # Musical features
    features: Dict[str, float] = field(default_factory=dict)
    
    # Top matching anchors and their weights
    top_anchors: List[Tuple[str, float]] = field(default_factory=list)
    
    # Emotional state
    valence: float = 0.0          # -1.0 to 1.0
    arousal: float = 0.5          # 0.0 to 1.0
    intensity: float = 0.5        # 0.0 to 1.0
    emotion_label: Optional[str] = None
    
    # Conical expansion
    spread: float = 0.16          # Cloud spread radius
    opacity: float = 0.02         # Base particle opacity


@dataclass
class EmotionParticle:
    """
    A particle representing emotional mass that can attach to anchors.
    
    Implements the "snowball effect" where emotions accumulate and gain inertia.
    """
    position: np.ndarray          # Current (x, y, z) position
    velocity: np.ndarray          # Motion vector
    mass: float = 1.0             # Increases with attachment strength (snowball)
    
    # Attachment state
    attached_anchors: Dict[str, float] = field(default_factory=dict)  # anchor_id -> strength
    
    # Emotion properties
    valence: float = 0.0
    arousal: float = 0.5
    
    def __post_init__(self):
        """Ensure numpy arrays are properly initialized."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=float)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=float)


@dataclass
class StormState:
    """
    Electrostatic storm accumulator for lightning and charge dynamics.
    
    Models the capacitor-like behavior where charge accumulates until
    discharge threshold is reached, triggering lightning events.
    """
    charge: float = 0.0           # Accumulated charge
    energy: float = 0.0           # Current storm energy
    
    # Arc/lightning state
    active_arcs: List[Tuple[int, str]] = field(default_factory=list)  # (ttl, anchor_id)
    
    # Parameters
    leak: float = 0.06            # Charge decay rate
    gain: float = 1.2             # Energy to charge conversion
    threshold: float = 14.0       # Discharge threshold
    discharge_factor: float = 0.45  # Charge retention after discharge


class MusicalParameterExtractor:
    """
    Extracts musical features from MIDI and/or audio data.
    
    Computes harmony, rhythm, dynamics, timbre, and texture metrics
    for use in anchor matching and lifeline positioning.
    """
    
    def __init__(self):
        pass
    
    def extract_from_midi_window(
        self,
        midi_events: List[Dict],
        window_start: float,
        window_end: float
    ) -> Dict[str, float]:
        """
        Extract features from MIDI events in a time window.
        
        Args:
            midi_events: List of MIDI events with 'time', 'type', 'note', 'velocity'
            window_start: Window start time in seconds
            window_end: Window end time in seconds
            
        Returns:
            Dictionary of normalized features (0-1 range)
        """
        features = {}
        
        # Filter events in window
        notes = [e for e in midi_events 
                if e.get('type') == 'note_on' 
                and window_start <= e.get('time', 0) < window_end]
        
        if not notes:
            # Return neutral features for empty window
            return {
                'note_density': 0.0,
                'pitch_centroid': 0.5,
                'velocity_centroid': 0.5,
                'pitch_range': 0.0,
                'tension': 0.0,
                'rhythmic_regularity': 0.5,
            }
        
        # Rhythm features
        features['note_density'] = min(1.0, len(notes) / 20.0)  # Normalize to ~20 notes
        
        # Pitch features
        pitches = [n['note'] for n in notes]
        features['pitch_centroid'] = (np.mean(pitches) - 36) / 72  # MIDI 36-108 -> 0-1
        features['pitch_range'] = (np.max(pitches) - np.min(pitches)) / 48.0
        
        # Dynamics
        velocities = [n.get('velocity', 64) for n in notes]
        features['velocity_centroid'] = np.mean(velocities) / 127.0
        
        # Tension (simplified - based on pitch variance)
        features['tension'] = min(1.0, np.std(pitches) / 12.0)
        
        # Rhythmic regularity (simplified - inverse of timing variance)
        if len(notes) > 1:
            timings = [n['time'] for n in notes]
            intervals = np.diff(timings)
            if len(intervals) > 0:
                features['rhythmic_regularity'] = 1.0 - min(1.0, np.std(intervals))
            else:
                features['rhythmic_regularity'] = 0.5
        else:
            features['rhythmic_regularity'] = 0.5
        
        return features
    
    def extract_from_audio_window(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        window_start: float,
        window_end: float
    ) -> Dict[str, float]:
        """
        Extract spectral features from audio in a time window.
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            window_start: Window start time in seconds
            window_end: Window end time in seconds
            
        Returns:
            Dictionary of normalized spectral features (0-1 range)
        """
        start_sample = int(window_start * sample_rate)
        end_sample = int(window_end * sample_rate)
        window = audio_data[start_sample:end_sample]
        
        if len(window) == 0:
            return {
                'spectral_centroid': 0.5,
                'spectral_flux': 0.0,
                'rms_energy': 0.0,
            }
        
        # Compute spectrum
        spectrum = np.abs(np.fft.rfft(window))
        freqs = np.fft.rfftfreq(len(window), 1.0 / sample_rate)
        
        # Spectral centroid (normalized to 0-1, assuming 0-8kHz range)
        if np.sum(spectrum) > 0:
            centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
            features = {'spectral_centroid': min(1.0, centroid / 8000.0)}
        else:
            features = {'spectral_centroid': 0.5}
        
        # RMS energy
        rms = np.sqrt(np.mean(window ** 2))
        features['rms_energy'] = min(1.0, rms * 10.0)  # Assuming normalized audio
        
        # Spectral flux (change over time - needs previous frame, simplified here)
        features['spectral_flux'] = min(1.0, np.std(spectrum) / np.mean(spectrum + 1e-10))
        
        return features


class AnchorLibrary:
    """
    Manages the library of stationary musical parameter anchors.
    
    Provides predefined anchor templates and similarity computation.
    """
    
    def __init__(self, density: str = "normal"):
        """
        Initialize anchor library.
        
        Args:
            density: "sparse" (50-200), "normal" (200-600), "dense" (600-2000)
        """
        self.density = density
        self.anchors: List[Anchor] = []
        self._generate_anchors()
    
    def _generate_anchors(self):
        """Generate anchor grid based on density setting."""
        if self.density == "sparse":
            n_harmony = 15
            n_rhythm = 10
            n_dynamics = 10
            n_timbre = 10
            n_texture = 5
        elif self.density == "dense":
            n_harmony = 60
            n_rhythm = 40
            n_dynamics = 40
            n_timbre = 40
            n_texture = 20
        else:  # normal
            n_harmony = 30
            n_rhythm = 20
            n_dynamics = 20
            n_timbre = 20
            n_texture = 10
        
        anchor_id = 0
        
        # Harmony anchors (tension × chord complexity grid)
        for i, tension in enumerate(np.linspace(0.0, 1.0, n_harmony // 3)):
            for j, complexity in enumerate(np.linspace(0.0, 1.0, 3)):
                y = tension * 0.6 + 0.2  # Map to 0.2-0.8 range
                z = complexity * 0.4 + 0.3  # Map to 0.3-0.7 range
                
                self.anchors.append(Anchor(
                    id=f"HARM_{anchor_id}",
                    name=f"Harmony T{tension:.2f} C{complexity:.2f}",
                    family=AnchorFamily.HARMONY,
                    position_y=y,
                    position_z=z,
                    constraints={'tension': tension, 'complexity': complexity},
                    base_charge=1.2,
                    polarity=1.0 if tension < 0.5 else -0.5,
                ))
                anchor_id += 1
        
        # Rhythm anchors (density × regularity grid)
        for i, density in enumerate(np.linspace(0.0, 1.0, n_rhythm // 2)):
            for j, regularity in enumerate(np.linspace(0.0, 1.0, 2)):
                y = 0.3 + density * 0.4
                z = 0.6 + regularity * 0.3
                
                self.anchors.append(Anchor(
                    id=f"RHYT_{anchor_id}",
                    name=f"Rhythm D{density:.2f} R{regularity:.2f}",
                    family=AnchorFamily.RHYTHM,
                    position_y=y,
                    position_z=z,
                    constraints={'density': density, 'regularity': regularity},
                    base_charge=1.0,
                ))
                anchor_id += 1
        
        # Dynamics anchors (energy levels)
        for i, energy in enumerate(np.linspace(0.0, 1.0, n_dynamics)):
            y = 0.5
            z = energy
            
            self.anchors.append(Anchor(
                id=f"DYN_{anchor_id}",
                name=f"Dynamics E{energy:.2f}",
                family=AnchorFamily.DYNAMICS,
                position_y=y,
                position_z=z,
                constraints={'energy': energy},
                base_charge=0.8,
            ))
            anchor_id += 1
        
        # Timbre anchors (spectral brightness)
        for i, brightness in enumerate(np.linspace(0.0, 1.0, n_timbre)):
            y = brightness
            z = 0.5
            
            self.anchors.append(Anchor(
                id=f"TIMB_{anchor_id}",
                name=f"Timbre B{brightness:.2f}",
                family=AnchorFamily.TIMBRE,
                position_y=y,
                position_z=z,
                constraints={'brightness': brightness},
                base_charge=0.9,
            ))
            anchor_id += 1
        
        # Texture anchors (complexity)
        for i, complexity in enumerate(np.linspace(0.0, 1.0, n_texture)):
            y = 0.7 + complexity * 0.2
            z = 0.7 + complexity * 0.2
            
            self.anchors.append(Anchor(
                id=f"TEXT_{anchor_id}",
                name=f"Texture C{complexity:.2f}",
                family=AnchorFamily.TEXTURE,
                position_y=y,
                position_z=z,
                constraints={'complexity': complexity},
                base_charge=0.7,
            ))
            anchor_id += 1
    
    def compute_anchor_similarities(
        self,
        features: Dict[str, float],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Compute similarity between features and all anchors.
        
        Args:
            features: Normalized feature dictionary
            top_k: Number of top anchors to return
            
        Returns:
            List of (anchor_id, similarity) tuples, sorted by similarity
        """
        similarities = []
        
        for anchor in self.anchors:
            # Compute Gaussian similarity based on constraint matching
            sim = 0.0
            count = 0
            
            for param, value in anchor.constraints.items():
                if param in features:
                    # Gaussian kernel
                    diff = features[param] - value
                    sim += np.exp(-(diff ** 2) / (2 * 0.1 ** 2))
                    count += 1
            
            if count > 0:
                sim /= count
                similarities.append((anchor.id, sim))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class SpectocloudRenderer:
    """
    CPU-based renderer for Spectocloud visualization using matplotlib.
    
    Renders the 3D musical space with anchors, lifeline, cloud particles,
    and lightning effects.
    """
    
    def __init__(self, figsize=(10, 8), dpi=120):
        """Initialize renderer with matplotlib."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            self.plt = plt
            self.has_mpl = True
        except ImportError:
            self.plt = None
            self.has_mpl = False
            print("matplotlib not available. Install with: pip install matplotlib")
        
        self.figsize = figsize
        self.dpi = dpi
        self.cmap = None
        if self.has_mpl:
            self.cmap = plt.get_cmap("coolwarm")
    
    def rgba_for_valence(self, valence: float, alpha: float = 1.0) -> Tuple:
        """Convert valence to RGBA color using coolwarm colormap."""
        if not self.has_mpl:
            return (0.7, 0.7, 0.7, alpha)
        
        u = (valence + 1.0) * 0.5  # Map -1..1 to 0..1
        r, g, b, _ = self.cmap(u)
        return (r, g, b, alpha)
    
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
        """
        Render a single frame of the visualization.
        
        Args:
            anchors: List of anchor points
            frames: List of all frames (for trail)
            particles: Current emotion particles
            storm: Storm state
            current_frame_idx: Index of current frame
            title: Plot title
            elev: Camera elevation angle
            azim: Camera azimuth angle
            
        Returns:
            matplotlib figure
        """
        if not self.has_mpl:
            print("matplotlib not available")
            return None
        
        fig = self.plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set up axes
        if frames:
            max_time = max(f.time for f in frames)
            ax.set_xlim(0, max_time * 1.1)
        else:
            ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.0)
        ax.set_zlim(0, 1.0)
        
        ax.set_xlabel("Time (beats)")
        ax.set_ylabel("Frequency/Pitch")
        ax.set_zlabel("Musical Parameter")
        
        # White background
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        
        # Render anchors (low opacity, activated ones glow)
        if anchors:
            anchor_positions = np.array([[0, a.position_y, a.position_z] for a in anchors])
            anchor_colors = np.zeros((len(anchors), 4))
            anchor_sizes = np.zeros(len(anchors))
            
            for i, anchor in enumerate(anchors):
                alpha = anchor.base_opacity + 0.22 * (anchor.activation ** 2)
                anchor_colors[i] = (*anchor.color, min(0.3, alpha))
                anchor_sizes[i] = anchor.base_size + 16 * (anchor.activation ** 3)
            
            ax.scatter(
                anchor_positions[:, 0],
                anchor_positions[:, 1],
                anchor_positions[:, 2],
                s=anchor_sizes,
                c=anchor_colors,
                alpha=0.6,
            )
        
        # Render lifeline trail
        if frames and current_frame_idx > 0:
            trail_frames = frames[:current_frame_idx + 1]
            trail_x = [f.x for f in trail_frames]
            trail_y = [f.y for f in trail_frames]
            trail_z = [f.z for f in trail_frames]
            
            ax.plot(trail_x, trail_y, trail_z, 
                   linewidth=1.5, alpha=0.4, color='gray')
        
        # Render current cloud particles
        if particles and frames:
            current_frame = frames[current_frame_idx]
            particle_positions = np.array([p.position for p in particles])
            
            # Color particles by valence
            particle_colors = np.array([
                self.rgba_for_valence(
                    current_frame.valence,
                    alpha=current_frame.opacity * (1.0 + current_frame.arousal)
                )
                for _ in particles
            ])
            
            ax.scatter(
                particle_positions[:, 0],
                particle_positions[:, 1],
                particle_positions[:, 2],
                s=8,
                c=particle_colors,
                alpha=0.5,
            )
        
        # Render current frame center point
        if frames:
            current_frame = frames[current_frame_idx]
            center_color = self.rgba_for_valence(current_frame.valence, alpha=0.7)
            ax.scatter(
                [current_frame.x],
                [current_frame.y],
                [current_frame.z],
                s=40,
                c=[center_color],
                edgecolors='black',
                linewidths=0.5,
            )
        
        # Title with current state
        if frames:
            current_frame = frames[current_frame_idx]
            ax.set_title(
                f"{title} | t={current_frame.time:.1f} | "
                f"v={current_frame.valence:+.2f} a={current_frame.arousal:.2f} | "
                f"charge={storm.charge:.1f}",
                fontsize=10
            )
        else:
            ax.set_title(title, fontsize=10)
        
        # Set view angle
        ax.view_init(elev=elev, azim=azim)
        
        self.plt.tight_layout()
        
        return fig


class Spectocloud:
    """
    Main Spectocloud visualization engine.
    
    Combines MIDI/audio analysis, anchor matching, storm dynamics,
    and conical lifeline rendering into a complete 3D visualization.
    """
    
    def __init__(
        self,
        anchor_density: str = "normal",
        n_particles: int = 2200,
        window_size: float = 0.2,  # seconds
    ):
        """
        Initialize Spectocloud engine.
        
        Args:
            anchor_density: Anchor library density ("sparse", "normal", "dense")
            n_particles: Number of emotion particles per frame
            window_size: Time window size for feature extraction (seconds)
        """
        self.anchor_library = AnchorLibrary(density=anchor_density)
        self.feature_extractor = MusicalParameterExtractor()
        self.renderer = SpectocloudRenderer()
        
        self.n_particles = n_particles
        self.window_size = window_size
        
        # Simulation state
        self.frames: List[Frame] = []
        self.storm = StormState()
        
        # Storm parameters
        self.sigma = 0.35           # Anchor influence radius
        self.leak = 0.06            # Charge leak rate
        self.gain = 1.2             # Storm energy to charge gain
        self.threshold = 14.0       # Lightning threshold
        self.discharge_factor = 0.45
        
        # Conical expansion parameters
        self.base_spread = 0.16
        self.spread_growth = 0.22
        
    def process_midi(
        self,
        midi_events: List[Dict],
        duration: float,
        emotion_trajectory: Optional[List[Dict]] = None
    ):
        """
        Process MIDI events to generate frames along the lifeline.
        
        Args:
            midi_events: List of MIDI events with 'time', 'type', 'note', 'velocity'
            duration: Total duration in seconds
            emotion_trajectory: Optional list of emotion states over time
        """
        self.frames = []
        
        # Generate windows
        n_windows = int(duration / self.window_size)
        
        for i in range(n_windows):
            window_start = i * self.window_size
            window_end = (i + 1) * self.window_size
            
            # Extract features
            features = self.feature_extractor.extract_from_midi_window(
                midi_events, window_start, window_end
            )
            
            # Get emotion for this window
            valence, arousal, intensity = 0.0, 0.5, 0.5
            if emotion_trajectory and i < len(emotion_trajectory):
                emo = emotion_trajectory[i]
                valence = emo.get('valence', 0.0)
                arousal = emo.get('arousal', 0.5)
                intensity = emo.get('intensity', 0.5)
            
            # Compute anchor similarities
            top_anchors = self.anchor_library.compute_anchor_similarities(features, top_k=5)
            
            # Compute frame position
            x = window_start
            y = features.get('pitch_centroid', 0.5)
            z = self._compute_composite_z(features)
            
            # Conical spread based on time and emotion
            t_normalized = i / max(1, n_windows - 1)
            spread = self.base_spread + self.spread_growth * t_normalized
            spread *= (1.0 + 0.5 * arousal)  # Arousal increases spread
            
            # Base opacity grows with accumulated charge
            opacity = 0.02 + 0.03 * min(1.0, self.storm.charge / self.threshold)
            
            frame = Frame(
                time=window_start,
                x=x, y=y, z=z,
                features=features,
                top_anchors=top_anchors,
                valence=valence,
                arousal=arousal,
                intensity=intensity,
                spread=spread,
                opacity=opacity,
            )
            
            self.frames.append(frame)
            
            # Update storm state
            self._update_storm(frame)
    
    def _compute_composite_z(self, features: Dict[str, float]) -> float:
        """
        Compute composite Z coordinate from multiple features.
        
        Higher Z = more intense/complex/bright/dense
        """
        z = 0.0
        z += 0.35 * features.get('tension', 0.0)
        z += 0.25 * features.get('note_density', 0.0)
        z += 0.20 * features.get('velocity_centroid', 0.5)
        z += 0.20 * features.get('pitch_range', 0.0)
        return min(1.0, max(0.0, z))
    
    def _update_storm(self, frame: Frame):
        """Update storm accumulator based on current frame."""
        # Compute storm energy from anchor proximity
        energy = 0.0
        
        for anchor_id, similarity in frame.top_anchors:
            anchor = next((a for a in self.anchor_library.anchors if a.id == anchor_id), None)
            if anchor:
                # Update anchor activation
                anchor.activation = max(anchor.activation, similarity)
                
                # Add to storm energy
                energy += similarity * anchor.base_charge * (0.35 + 0.95 * frame.arousal)
        
        self.storm.energy = energy
        
        # Update charge accumulator
        self.storm.charge = self.storm.charge * (1.0 - self.leak) + energy * self.gain * 0.04
        
        # Check for lightning discharge
        if self.storm.charge > self.threshold:
            # Create arcs to top anchors
            for anchor_id, _ in frame.top_anchors[:3]:
                self.storm.active_arcs.append((9, anchor_id))  # TTL=9 frames
            
            # Discharge
            self.storm.charge *= self.discharge_factor
        
        # Decay arcs
        self.storm.active_arcs = [(ttl - 1, aid) for ttl, aid in self.storm.active_arcs if ttl > 1]
        
        # Decay anchor activations
        for anchor in self.anchor_library.anchors:
            anchor.activation *= 0.85
    
    def generate_particles_for_frame(self, frame_idx: int) -> List[EmotionParticle]:
        """Generate emotion particles around a specific frame."""
        if frame_idx >= len(self.frames):
            return []
        
        frame = self.frames[frame_idx]
        particles = []
        
        # Generate particles in a cloud around frame center
        for _ in range(self.n_particles):
            offset = np.random.normal(size=3) * frame.spread
            position = np.array([frame.x, frame.y, frame.z]) + offset
            
            particle = EmotionParticle(
                position=position,
                velocity=np.zeros(3),
                valence=frame.valence,
                arousal=frame.arousal,
            )
            particles.append(particle)
        
        return particles
    
    def render_static_frame(
        self,
        frame_idx: int,
        output_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Render a single static frame.
        
        Args:
            frame_idx: Index of frame to render
            output_path: Optional path to save image
            show: Whether to display interactively
        """
        if not self.frames:
            print("No frames to render. Call process_midi() first.")
            return
        
        particles = self.generate_particles_for_frame(frame_idx)
        
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
            print(f"Saved frame to {output_path}")
        
        if fig and show:
            self.renderer.plt.show()
        elif fig:
            self.renderer.plt.close(fig)
    
    def render_animation(
        self,
        output_path: str,
        fps: int = 20,
        duration: Optional[float] = None,
        rotate: bool = True,
    ):
        """
        Render animation/GIF of the visualization.
        
        Args:
            output_path: Path to save GIF (should end with .gif)
            fps: Frames per second
            duration: Optional duration in seconds (if None, uses all frames)
            rotate: Whether to rotate camera during animation
        """
        if not self.renderer.has_mpl:
            print("matplotlib not available for animation")
            return
        
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            print("matplotlib.animation not available")
            return
        
        if not self.frames:
            print("No frames to animate. Call process_midi() first.")
            return
        
        # Determine frame range
        if duration is not None:
            max_frame_idx = min(len(self.frames), int(duration / self.window_size))
        else:
            max_frame_idx = len(self.frames)
        
        print(f"Rendering animation with {max_frame_idx} frames...")
        
        # Create figure
        fig = self.renderer.plt.figure(figsize=self.renderer.figsize, dpi=self.renderer.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set up axes
        max_time = max(f.time for f in self.frames[:max_frame_idx])
        ax.set_xlim(0, max_time * 1.1)
        ax.set_ylim(0, 1.0)
        ax.set_zlim(0, 1.0)
        ax.set_xlabel("Time (beats)")
        ax.set_ylabel("Frequency/Pitch")
        ax.set_zlabel("Musical Parameter")
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        
        # Initialize plot elements
        anchor_positions = np.array([[0, a.position_y, a.position_z] 
                                     for a in self.anchor_library.anchors])
        anchor_sc = ax.scatter([], [], [], s=5, alpha=0.6)
        particle_sc = ax.scatter([], [], [], s=8, alpha=0.5)
        trail_line, = ax.plot([], [], [], linewidth=1.5, alpha=0.4, color='gray')
        center_sc = ax.scatter([], [], [], s=40, edgecolors='black', linewidths=0.5)
        
        def update(frame_idx):
            """Update function for animation."""
            if frame_idx >= len(self.frames):
                return anchor_sc, particle_sc, trail_line, center_sc
            
            frame = self.frames[frame_idx]
            
            # Update anchors
            anchor_colors = np.zeros((len(self.anchor_library.anchors), 4))
            anchor_sizes = np.zeros(len(self.anchor_library.anchors))
            for i, anchor in enumerate(self.anchor_library.anchors):
                alpha = anchor.base_opacity + 0.22 * (anchor.activation ** 2)
                anchor_colors[i] = (*anchor.color, min(0.3, alpha))
                anchor_sizes[i] = anchor.base_size + 16 * (anchor.activation ** 3)
            
            anchor_sc._offsets3d = (anchor_positions[:, 0], 
                                     anchor_positions[:, 1],
                                     anchor_positions[:, 2])
            anchor_sc._facecolor3d = anchor_colors
            anchor_sc._edgecolor3d = anchor_colors
            anchor_sc.set_sizes(anchor_sizes)
            
            # Generate particles for current frame
            particles = self.generate_particles_for_frame(frame_idx)
            if particles:
                particle_positions = np.array([p.position for p in particles])
                particle_colors = np.array([
                    self.renderer.rgba_for_valence(
                        frame.valence,
                        alpha=frame.opacity * (1.0 + frame.arousal)
                    )
                    for _ in particles
                ])
                particle_sc._offsets3d = (particle_positions[:, 0],
                                          particle_positions[:, 1],
                                          particle_positions[:, 2])
                particle_sc._facecolor3d = particle_colors
                particle_sc._edgecolor3d = particle_colors
            
            # Update trail
            if frame_idx > 0:
                trail_frames = self.frames[:frame_idx + 1]
                trail_x = [f.x for f in trail_frames]
                trail_y = [f.y for f in trail_frames]
                trail_z = [f.z for f in trail_frames]
                trail_line.set_data(trail_x, trail_y)
                trail_line.set_3d_properties(trail_z)
            
            # Update center
            center_color = self.renderer.rgba_for_valence(frame.valence, alpha=0.7)
            center_sc._offsets3d = ([frame.x], [frame.y], [frame.z])
            center_sc._facecolor3d = [center_color]
            center_sc._edgecolor3d = [center_color]
            
            # Update title
            ax.set_title(
                f"Spectocloud | t={frame.time:.1f} | "
                f"v={frame.valence:+.2f} a={frame.arousal:.2f} | "
                f"charge={self.storm.charge:.1f}",
                fontsize=10
            )
            
            # Rotate camera if enabled
            if rotate:
                t_norm = frame_idx / max(1, max_frame_idx - 1)
                elev = 22.0 + 10 * np.sin(2 * np.pi * t_norm)
                azim = 40.0 + 120 * t_norm
                ax.view_init(elev=elev, azim=azim)
            
            return anchor_sc, particle_sc, trail_line, center_sc
        
        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=max_frame_idx,
            interval=int(1000 / fps),
            blit=False
        )
        
        # Save as GIF
        print(f"Saving animation to {output_path}...")
        anim.save(output_path, writer=PillowWriter(fps=fps))
        print(f"Animation saved successfully!")
        
        self.renderer.plt.close(fig)
    
    def export_data(self, output_path: str):
        """Export visualization data to JSON."""
        data = {
            'anchors': [
                {
                    'id': a.id,
                    'name': a.name,
                    'family': a.family.value,
                    'position': [a.position_y, a.position_z],
                    'base_charge': a.base_charge,
                }
                for a in self.anchor_library.anchors
            ],
            'frames': [
                {
                    'time': f.time,
                    'position': [f.x, f.y, f.z],
                    'valence': f.valence,
                    'arousal': f.arousal,
                    'intensity': f.intensity,
                    'spread': f.spread,
                    'features': f.features,
                }
                for f in self.frames
            ],
        }
        
        with open(output_path, 'w') as fp:
            json.dump(data, fp, indent=2)
        
        print(f"Exported data to {output_path}")
