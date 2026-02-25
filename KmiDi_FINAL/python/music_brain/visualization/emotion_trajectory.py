"""
Emotion Trajectory Visualization

Visualizes how emotions change over time in a composition or session.
Creates timeline graphs showing emotional evolution.

Part of the "New Features" implementation for Kelly MIDI Companion.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
    NUMPY_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    NUMPY_AVAILABLE = False
    np = None  # Placeholder


@dataclass
class EmotionSnapshot:
    """Emotion state at a specific time point."""
    timestamp: float  # Time in seconds
    valence: float    # -1.0 to 1.0 (negative to positive)
    arousal: float   # 0.0 to 1.0 (calm to excited)
    intensity: float # 0.0 to 1.0 (weak to intense)
    emotion_label: Optional[str] = None  # Primary emotion name
    metadata: Optional[Dict] = None


@dataclass
class EmotionTrajectory:
    """Complete emotion trajectory over time."""
    snapshots: List[EmotionSnapshot]
    title: str = "Emotion Trajectory"
    duration_seconds: float = 0.0

    def get_valence_over_time(self) -> Tuple[List[float], List[float]]:
        """Get (timestamps, valence_values) for plotting."""
        timestamps = [s.timestamp for s in self.snapshots]
        values = [s.valence for s in self.snapshots]
        return timestamps, values

    def get_arousal_over_time(self) -> Tuple[List[float], List[float]]:
        """Get (timestamps, arousal_values) for plotting."""
        timestamps = [s.timestamp for s in self.snapshots]
        values = [s.arousal for s in self.snapshots]
        return timestamps, values

    def get_intensity_over_time(self) -> Tuple[List[float], List[float]]:
        """Get (timestamps, intensity_values) for plotting."""
        timestamps = [s.timestamp for s in self.snapshots]
        values = [s.intensity for s in self.snapshots]
        return timestamps, values

    def get_emotion_labels(self) -> List[Optional[str]]:
        """Get emotion labels at each snapshot."""
        return [s.emotion_label for s in self.snapshots]


class EmotionTrajectoryVisualizer:
    """
    Visualizes emotion trajectories over time.

    Usage:
        visualizer = EmotionTrajectoryVisualizer()
        trajectory = visualizer.load_from_intent(intent_data)
        visualizer.plot_trajectory(trajectory, output_path="emotion_timeline.png")
    """

    def __init__(self):
        self.color_map = {
            "grief": "#1f77b4",      # Blue
            "longing": "#9467bd",    # Purple
            "hope": "#2ca02c",       # Green
            "rage": "#d62728",       # Red
            "tenderness": "#ff7f0e", # Orange
            "anxiety": "#bcbd22",    # Yellow-green
            "euphoria": "#17becf",   # Cyan
            "melancholy": "#7f7f7f", # Gray
            "nostalgia": "#e377c2",  # Pink
            "catharsis": "#8c564b",  # Brown
            "dissociation": "#c49c94", # Tan
            "determination": "#ff9896", # Light red
            "surrender": "#c5b0d5",  # Lavender
        }

    def create_trajectory_from_snapshots(
        self,
        snapshots: List[EmotionSnapshot],
        title: str = "Emotion Trajectory"
    ) -> EmotionTrajectory:
        """Create trajectory from list of snapshots."""
        duration = max(s.timestamp for s in snapshots) if snapshots else 0.0
        return EmotionTrajectory(
            snapshots=snapshots,
            title=title,
            duration_seconds=duration
        )

    def create_trajectory_from_intent(
        self,
        intent_data: Dict,
        time_resolution: float = 1.0  # Sample every N seconds
    ) -> EmotionTrajectory:
        """
        Create trajectory from song intent data.

        Assumes intent has emotion/VAD data that can be sampled over time.
        """
        snapshots = []

        # Extract base emotion from intent
        emotion_label = None
        base_valence = 0.0
        base_arousal = 0.5
        base_intensity = 0.5

        # Try to get emotion from intent
        if "mood_primary" in intent_data:
            emotion_label = intent_data["mood_primary"]
        elif "emotion" in intent_data:
            emotion_label = intent_data["emotion"]

        # Map emotion to VAD if available
        if emotion_label:
            vad_map = self._get_emotion_vad_map()
            if emotion_label.lower() in vad_map:
                base_valence, base_arousal, base_intensity = vad_map[emotion_label.lower()]

        # Estimate duration (default 3 minutes if not specified)
        duration = intent_data.get("duration_seconds", 180.0)
        if "estimated_duration" in intent_data:
            duration = intent_data["estimated_duration"]

        # Create snapshots at regular intervals
        # Simple linear trajectory for now (can be enhanced with actual composition data)
        num_snapshots = int(duration / time_resolution)

        for i in range(num_snapshots):
            t = i * time_resolution

            # Simple modulation: add slight variation to create trajectory
            # In production, this would use actual composition analysis
            import math
            if num_snapshots > 1:
                variation = 0.1 * math.sin(2 * math.pi * t / (duration / 3))
            else:
                variation = 0.0

            snapshot = EmotionSnapshot(
                timestamp=t,
                valence=base_valence + variation * 0.3,
                arousal=base_arousal + variation * 0.2,
                intensity=base_intensity + abs(variation) * 0.3,
                emotion_label=emotion_label
            )
            snapshots.append(snapshot)

        return EmotionTrajectory(
            snapshots=snapshots,
            title=intent_data.get("title", "Untitled"),
            duration_seconds=duration
        )

    def _get_emotion_vad_map(self) -> Dict[str, Tuple[float, float, float]]:
        """Map emotion names to VAD (Valence, Arousal, Dominance/Intensity) values."""
        return {
            "grief": (-0.8, 0.3, 0.7),
            "longing": (-0.5, 0.6, 0.6),
            "hope": (0.7, 0.7, 0.6),
            "rage": (-0.9, 0.95, 0.9),
            "tenderness": (0.6, 0.4, 0.5),
            "anxiety": (-0.6, 0.85, 0.7),
            "euphoria": (0.9, 0.95, 0.8),
            "melancholy": (-0.7, 0.4, 0.6),
            "nostalgia": (0.2, 0.5, 0.5),
            "catharsis": (0.3, 0.8, 0.8),
            "dissociation": (-0.4, 0.2, 0.3),
            "determination": (0.5, 0.8, 0.9),
            "surrender": (0.0, 0.3, 0.4),
        }

    def plot_trajectory(
        self,
        trajectory: EmotionTrajectory,
        output_path: Optional[str] = None,
        show_plot: bool = True,
        style: str = "comprehensive"  # "simple", "comprehensive", "3d"
    ):
        """
        Plot emotion trajectory.

        Args:
            trajectory: EmotionTrajectory to plot
            output_path: Optional path to save figure
            show_plot: Whether to display plot (requires interactive environment)
            style: Plot style ("simple", "comprehensive", "3d")
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available. Install with: pip install matplotlib numpy")
            return

        if style == "simple":
            self._plot_simple(trajectory, output_path, show_plot)
        elif style == "comprehensive":
            self._plot_comprehensive(trajectory, output_path, show_plot)
        elif style == "3d":
            self._plot_3d(trajectory, output_path, show_plot)
        else:
            self._plot_comprehensive(trajectory, output_path, show_plot)

    def _plot_simple(
        self,
        trajectory: EmotionTrajectory,
        output_path: Optional[str],
        show_plot: bool
    ):
        """Simple line plot of valence, arousal, intensity over time."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(trajectory.title, fontsize=14, fontweight='bold')

        timestamps, valence = trajectory.get_valence_over_time()
        timestamps, arousal = trajectory.get_arousal_over_time()
        timestamps, intensity = trajectory.get_intensity_over_time()

        # Valence plot
        axes[0].plot(timestamps, valence, 'b-', linewidth=2, label='Valence')
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0].set_ylabel('Valence', fontsize=10)
        axes[0].set_ylim(-1.1, 1.1)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Arousal plot
        axes[1].plot(timestamps, arousal, 'r-', linewidth=2, label='Arousal')
        axes[1].set_ylabel('Arousal', fontsize=10)
        axes[1].set_ylim(0, 1.1)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # Intensity plot
        axes[2].plot(timestamps, intensity, 'g-', linewidth=2, label='Intensity')
        axes[2].set_ylabel('Intensity', fontsize=10)
        axes[2].set_xlabel('Time (seconds)', fontsize=10)
        axes[2].set_ylim(0, 1.1)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved emotion trajectory to {output_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def _plot_comprehensive(
        self,
        trajectory: EmotionTrajectory,
        output_path: Optional[str],
        show_plot: bool
    ):
        """Comprehensive plot with VAD dimensions and emotion regions."""
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy is required for comprehensive plotting")

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        timestamps, valence = trajectory.get_valence_over_time()
        timestamps, arousal = trajectory.get_arousal_over_time()
        timestamps, intensity = trajectory.get_intensity_over_time()
        labels = trajectory.get_emotion_labels()

        # Main title
        fig.suptitle(trajectory.title, fontsize=16, fontweight='bold', y=0.98)

        # Valence plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(timestamps, valence, 'b-', linewidth=2.5, label='Valence')
        ax1.fill_between(timestamps, 0, valence, where=np.array(valence) >= 0,
                         alpha=0.3, color='green', label='Positive')
        ax1.fill_between(timestamps, 0, valence, where=np.array(valence) < 0,
                         alpha=0.3, color='red', label='Negative')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Valence\n(Negative ↔ Positive)', fontsize=10, fontweight='bold')
        ax1.set_ylim(-1.1, 1.1)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # Arousal plot
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(timestamps, arousal, 'r-', linewidth=2.5, label='Arousal')
        ax2.fill_between(timestamps, arousal, alpha=0.3, color='orange')
        ax2.set_ylabel('Arousal\n(Calm ↔ Excited)', fontsize=10, fontweight='bold')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Intensity plot
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(timestamps, intensity, 'g-', linewidth=2.5, label='Intensity')
        ax3.fill_between(timestamps, intensity, alpha=0.3, color='green')
        ax3.set_ylabel('Intensity\n(Weak ↔ Intense)', fontsize=10, fontweight='bold')
        ax3.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 2D Valence-Arousal scatter over time (colored by intensity)
        ax4 = fig.add_subplot(gs[2, 0])
        scatter = ax4.scatter(valence, arousal, c=intensity, cmap='viridis',
                             s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('Valence', fontsize=10, fontweight='bold')
        ax4.set_ylabel('Arousal', fontsize=10, fontweight='bold')
        ax4.set_xlim(-1.1, 1.1)
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
        ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Intensity')
        ax4.set_title('Emotional Space Trajectory', fontsize=11, fontweight='bold')

        # Emotion labels timeline
        ax5 = fig.add_subplot(gs[2, 1])
        if labels and any(labels):
            unique_labels = list(set(filter(None, labels)))
            if unique_labels:
                for i, label in enumerate(unique_labels):
                    label_indices = [j for j, l in enumerate(labels) if l == label]
                    if label_indices:
                        label_times = [timestamps[j] for j in label_indices]
                        color = self.color_map.get(label.lower(), '#7f7f7f')
                        ax5.scatter(label_times, [i] * len(label_times),
                                  c=color, s=100, label=label, alpha=0.7)
                ax5.set_yticks(range(len(unique_labels)))
                ax5.set_yticklabels(unique_labels)
                ax5.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
                ax5.set_title('Emotion Labels Over Time', fontsize=11, fontweight='bold')
                ax5.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved emotion trajectory to {output_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def _plot_3d(
        self,
        trajectory: EmotionTrajectory,
        output_path: Optional[str],
        show_plot: bool
    ):
        """3D plot of VAD space."""
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        timestamps, valence = trajectory.get_valence_over_time()
        timestamps, arousal = trajectory.get_arousal_over_time()
        timestamps, intensity = trajectory.get_intensity_over_time()

        # Color by time
        scatter = ax.scatter(valence, arousal, intensity, c=timestamps,
                           cmap='viridis', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Plot trajectory line
        ax.plot(valence, arousal, intensity, 'k-', linewidth=1, alpha=0.3)

        ax.set_xlabel('Valence', fontsize=10, fontweight='bold')
        ax.set_ylabel('Arousal', fontsize=10, fontweight='bold')
        ax.set_zlabel('Intensity', fontsize=10, fontweight='bold')
        ax.set_title(trajectory.title, fontsize=12, fontweight='bold', pad=20)

        plt.colorbar(scatter, ax=ax, label='Time (seconds)', shrink=0.8)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved 3D emotion trajectory to {output_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def export_trajectory_json(
        self,
        trajectory: EmotionTrajectory,
        output_path: str
    ):
        """Export trajectory data to JSON for external use."""
        data = {
            "title": trajectory.title,
            "duration_seconds": trajectory.duration_seconds,
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "valence": s.valence,
                    "arousal": s.arousal,
                    "intensity": s.intensity,
                    "emotion_label": s.emotion_label,
                    "metadata": s.metadata
                }
                for s in trajectory.snapshots
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Exported trajectory data to {output_path}")


def main():
    """Example usage."""
    visualizer = EmotionTrajectoryVisualizer()

    # Create sample trajectory
    snapshots = [
        EmotionSnapshot(0.0, -0.7, 0.4, 0.6, "grief"),
        EmotionSnapshot(30.0, -0.5, 0.5, 0.7, "longing"),
        EmotionSnapshot(60.0, 0.0, 0.6, 0.8, "catharsis"),
        EmotionSnapshot(90.0, 0.5, 0.7, 0.9, "hope"),
        EmotionSnapshot(120.0, 0.7, 0.8, 0.85, "euphoria"),
    ]

    trajectory = visualizer.create_trajectory_from_snapshots(
        snapshots,
        title="Sample Emotional Journey"
    )

    # Plot and export
    visualizer.plot_trajectory(trajectory, output_path="emotion_trajectory.png", show_plot=False)
    visualizer.export_trajectory_json(trajectory, "emotion_trajectory.json")


if __name__ == "__main__":
    main()
