"""
Complete iDAW Workflow Example: Tier 1 + Tier 2 Integration

Demonstrates full pipeline:
  1. Emotion Intent → Emotion Embedding (using existing iDAW emotion recognition)
  2. MIDI Generation (Tier 1 pretrained melody + harmony + groove)
  3. Audio Synthesis (Tier 1 pretrained synthesis)
  4. Voice Generation (Tier 1 TTS with emotion)
  5. Mac optimization (MPS acceleration, memory management)

Optional Tier 2 enhancement:
  - Fine-tune generators on custom therapy dataset with LoRA

Usage:
    python complete_workflow_example.py --emotion GRIEF --duration 8
"""

import argparse
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class iDAWWorkflow:
    """
    Complete iDAW workflow: Intent → Music + Voice.

    Integrates:
      - Tier 1 generators (MIDI, Audio, Voice)
      - Tier 2 optional fine-tuning
      - Mac optimization (MPS acceleration)
      - iDAW emotion schema
    """

    def __init__(
        self,
        device: str = "auto",
        tier: int = 1,
        use_optimization: bool = True
    ):
        """
        Initialize workflow.

        Args:
            device: "auto" (detect), "mps", "cuda", "cpu"
            tier: 1 (pretrained) or 2 (with LoRA fine-tuning)
            use_optimization: Apply Mac optimizations
        """
        self.device = device
        self.tier = tier
        self.use_optimization = use_optimization

        self._load_generators()

        if use_optimization:
            self._setup_mac_optimization()

        logger.info(f"✓ iDAW Workflow initialized (Tier {tier})")

    def _load_generators(self):
        """Load Tier 1 generators"""
        try:
            from music_brain.tier1 import (
                Tier1MIDIGenerator,
                Tier1AudioGenerator,
                Tier1VoiceGenerator
            )

            self.midi_gen = Tier1MIDIGenerator(device=self.device, verbose=False)
            self.audio_gen = Tier1AudioGenerator(device=self.device, verbose=False)
            self.voice_gen = Tier1VoiceGenerator(device=self.device, verbose=False)

        except ImportError as e:
            logger.error(f"Failed to load generators: {e}")
            raise

    def _setup_mac_optimization(self):
        """Setup Mac optimization if available"""
        try:
            from music_brain.mac_optimization import MacOptimizationLayer

            self.mac_opt = MacOptimizationLayer(verbose=False)
            logger.info(f"✓ Mac optimization enabled ({self.mac_opt.device})")

        except ImportError:
            logger.warning("Mac optimization not available")
            self.mac_opt = None

    def emotion_intent_to_embedding(
        self,
        wound: str,
        emotion_label: str,
        intensity: float = 0.7
    ) -> np.ndarray:
        """
        Convert iDAW intent to emotion embedding.

        Args:
            wound: Core wound description (e.g., "I feel lost in grief")
            emotion_label: Emotion category (GRIEF, JOY, CALM, ANGER, etc.)
            intensity: Intensity scale [0, 1]

        Returns:
            embedding: (64,) emotion vector
        """
        # Map emotion label to embedding
        emotion_map = {
            "GRIEF": {
                "valence": -0.9,      # Very negative
                "arousal": -0.6,      # Low energy
                "intensity": intensity,
                "description": "deep loss, sadness"
            },
            "JOY": {
                "valence": 0.95,      # Highly positive
                "arousal": 0.85,      # High energy
                "intensity": intensity,
                "description": "celebration, happiness"
            },
            "CALM": {
                "valence": 0.4,       # Slightly positive
                "arousal": -0.3,      # Low energy
                "intensity": intensity * 0.5,  # Muted
                "description": "peace, stillness"
            },
            "ANGER": {
                "valence": -0.8,      # Negative
                "arousal": 0.95,      # Very high energy
                "intensity": intensity,
                "description": "righteous fury, power"
            },
            "ANXIETY": {
                "valence": -0.5,      # Negative
                "arousal": 0.8,       # High agitation
                "intensity": intensity,
                "description": "unease, restlessness"
            },
            "HOPE": {
                "valence": 0.7,       # Positive
                "arousal": 0.6,       # Moderate energy
                "intensity": intensity,
                "description": "yearning, reaching"
            }
        }

        emotion_config = emotion_map.get(emotion_label, emotion_map["CALM"])

        # Create 64-dim embedding
        embedding = np.zeros(64, dtype=np.float32)

        # First 32 dims: valence (positive/negative)
        valence_intensity = emotion_config["valence"] * emotion_config["intensity"]
        embedding[:32] = np.linspace(valence_intensity, 0, 32)

        # Next 32 dims: arousal (energy level)
        arousal_intensity = emotion_config["arousal"] * emotion_config["intensity"]
        embedding[32:] = np.linspace(arousal_intensity, 0, 32)

        logger.info(f"Created emotion embedding: {emotion_label} "
                   f"(valence={emotion_config['valence']:.2f}, "
                   f"arousal={emotion_config['arousal']:.2f})")

        return embedding

    def generate_complete_music(
        self,
        wound: str,
        emotion_label: str,
        genre: str = "ballad",
        duration_bars: int = 8,
        intensity: float = 0.7
    ) -> Dict:
        """
        Generate complete music: MIDI + Audio + Voice.

        Args:
            wound: Core wound description
            emotion_label: Emotion category
            genre: Music genre (influences groove parameters)
            duration_bars: Duration in bars
            intensity: Emotional intensity [0, 1]

        Returns:
            result: Dict with all generated components
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"iDAW Generation: {emotion_label}")
        logger.info(f"Duration: {duration_bars} bars | Genre: {genre}")
        logger.info(f"{'='*60}")

        # Step 1: Convert intent to emotion embedding
        logger.info("\n[1/5] Processing emotional intent...")
        emotion_embedding = self.emotion_intent_to_embedding(
            wound, emotion_label, intensity
        )

        # Step 2: Generate MIDI
        logger.info("[2/5] Generating MIDI (melody, harmony, groove)...")
        midi_result = self.midi_gen.full_pipeline(
            emotion_embedding,
            length=duration_bars * 8
        )

        # Step 3: Generate audio
        logger.info("[3/5] Synthesizing audio texture...")
        audio = self.audio_gen.synthesize_texture(
            midi_result["melody"],
            midi_result["groove"],
            emotion_embedding,
            duration_seconds=duration_bars * 2  # ~2 sec per bar @ 120 BPM
        )

        # Step 4: Generate voice
        logger.info("[4/5] Generating therapeutic voice guidance...")
        voice_text = self._generate_therapeutic_text(emotion_label, wound)
        voice = self.voice_gen.speak_emotion(
            voice_text,
            emotion=emotion_label.lower()
        )

        # Step 5: Compose to MIDI file
        logger.info("[5/5] Composing MIDI file...")
        midi_bytes = self._notes_to_midi_file(
            midi_result["melody"],
            midi_result["groove"],
            tempo_bpm=120
        )

        result = {
            "midi": midi_bytes,
            "audio": audio,
            "voice": voice,
            "emotion_embedding": emotion_embedding,
            "metadata": {
                "wound": wound,
                "emotion": emotion_label,
                "genre": genre,
                "duration_bars": duration_bars,
                "intensity": intensity,
                "tier": self.tier,
                "device": self.device,
                "voice_text": voice_text
            },
            "components": midi_result
        }

        logger.info(f"\n✓ Generation complete!")
        logger.info(f"  Audio: {len(audio)} samples")
        logger.info(f"  MIDI: {len(midi_result['melody'])} notes")

        return result

    def _generate_therapeutic_text(self, emotion: str, wound: str) -> str:
        """Generate therapeutic guidance based on emotion + wound"""
        responses = {
            "GRIEF": (
                "Your grief is valid and real. What you've lost matters. "
                "This music honors the depth of your feeling. "
                "Let yourself feel what needs to be felt."
            ),
            "JOY": (
                "Your joy is a gift—to yourself and to others. "
                "This music celebrates everything bright within you. "
                "You deserve this happiness."
            ),
            "CALM": (
                "You are safe in this moment. Your breath is your anchor. "
                "Let this music wrap around you like a blanket of peace. "
                "You are exactly where you need to be."
            ),
            "ANGER": (
                "Your anger has power. It speaks truth when words fail. "
                "This music gives your anger a voice—strong, honest, and true. "
                "Channel it forward."
            ),
            "ANXIETY": (
                "What you feel is real, and it will pass. "
                "This moment is not forever. "
                "Let this music anchor you to now, to your breath, to safety."
            ),
            "HOPE": (
                "Even in darkness, you reach toward light. "
                "Your hope is a strength, a quiet revolution. "
                "This music believes in your tomorrow."
            )
        }

        return responses.get(emotion, responses["CALM"])

    def _notes_to_midi_file(
        self,
        notes: np.ndarray,
        groove_params: Dict,
        tempo_bpm: int = 120
    ) -> bytes:
        """Convert notes to MIDI bytes"""
        try:
            from music21 import stream, note, tempo, instrument, environment

            # Suppress music21 display
            environment.set('musicxmlPath', None)

            # Create score
            s = stream.Score()
            part = stream.Part()
            part.instrument = instrument.Piano()

            # Add tempo
            part.append(tempo.MetronomeMark(number=tempo_bpm))

            # Add notes
            for midi_note in notes:
                n = note.Note(int(midi_note))
                n.quarterLength = 0.5
                part.append(n)

            s.append(part)

            # Convert to MIDI bytes
            from io import BytesIO
            output = BytesIO()
            s.write('midi', fp=output)
            return output.getvalue()

        except ImportError:
            logger.warning("music21 not installed; returning empty MIDI")
            return b''

    def save_outputs(self, result: Dict, output_dir: str = "./generated_music"):
        """Save all outputs to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = result["metadata"].copy()
        metadata["emotion_embedding"] = result["emotion_embedding"].tolist()
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save audio
        try:
            from scipy.io import wavfile
            audio_data = (result["audio"] * 32767).astype(np.int16)
            wavfile.write(
                output_path / "generated_music.wav",
                22050,
                audio_data
            )

            voice_data = (result["voice"] * 32767).astype(np.int16)
            wavfile.write(
                output_path / "voice_guidance.wav",
                22050,
                voice_data
            )

        except ImportError:
            logger.warning("scipy not installed; skipping WAV export")

        # Save MIDI
        if result["midi"]:
            with open(output_path / "generated_music.mid", "wb") as f:
                f.write(result["midi"])

        logger.info(f"✓ Saved outputs to {output_path}/")

        return output_path


def main():
    parser = argparse.ArgumentParser(description="iDAW Complete Workflow")
    parser.add_argument("--wound", type=str, default="I feel lost and alone")
    parser.add_argument("--emotion", type=str, default="GRIEF",
                       choices=["GRIEF", "JOY", "CALM", "ANGER", "ANXIETY", "HOPE"])
    parser.add_argument("--genre", type=str, default="ballad")
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--intensity", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2])
    parser.add_argument("--output-dir", type=str, default="./generated_music")
    parser.add_argument("--no-save", action="store_true")

    args = parser.parse_args()

    # Initialize workflow
    workflow = iDAWWorkflow(device=args.device, tier=args.tier)

    # Generate music
    result = workflow.generate_complete_music(
        wound=args.wound,
        emotion_label=args.emotion,
        genre=args.genre,
        duration_bars=args.duration,
        intensity=args.intensity
    )

    # Save outputs
    if not args.no_save:
        workflow.save_outputs(result, args.output_dir)

    logger.info(f"\n{'='*60}")
    logger.info(f"Generation Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Emotion: {result['metadata']['emotion']}")
    logger.info(f"Duration: {result['metadata']['duration_bars']} bars")
    logger.info(f"Device: {result['metadata']['device']}")
    logger.info(f"Output: {args.output_dir}/")
    logger.info(f"{'='*60}\n")

    return result


if __name__ == "__main__":
    main()
