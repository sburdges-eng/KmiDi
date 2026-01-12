"""
Voice Learning - Learn from Voice Samples

Stores voice samples, extracts features, and builds voice profiles
for voice cloning and mimicking.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime
import soundfile as sf

from music_brain.voice.voice_input import VoiceMimic
from music_brain.voice.pitch_controller import PitchController


@dataclass
class VoiceSample:
    """Represents a single voice sample."""
    audio: np.ndarray
    sample_rate: int
    text: Optional[str] = None  # Optional transcript
    metadata: Optional[Dict] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LearnedVoiceProfile:
    """Voice profile learned from samples."""
    name: str
    characteristics: Dict
    sample_count: int
    total_duration: float  # seconds
    created: str
    updated: str

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "characteristics": self.characteristics,
            "sample_count": self.sample_count,
            "total_duration": self.total_duration,
            "created": self.created,
            "updated": self.updated
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LearnedVoiceProfile":
        return cls(**data)


class VoiceSampleStore:
    """
    Stores and manages voice samples for learning.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize voice sample store.

        Args:
            storage_dir: Directory to store samples (default: ~/.parrot/voice_samples)
        """
        if storage_dir is None:
            storage_dir = Path.home() / ".parrot" / "voice_samples"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.samples_dir = self.storage_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)

        self.profiles_dir = self.storage_dir / "profiles"
        self.profiles_dir.mkdir(exist_ok=True)

    def save_sample(
        self,
        audio: np.ndarray,
        sample_rate: int,
        sample_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Save a voice sample.

        Args:
            audio: Audio signal
            sample_rate: Sample rate
            sample_id: Unique identifier for sample
            text: Optional transcript
            metadata: Optional metadata

        Returns:
            Path to saved sample
        """
        # Save audio
        audio_path = self.samples_dir / f"{sample_id}.wav"
        sf.write(str(audio_path), audio, sample_rate)

        # Save metadata
        metadata_path = self.samples_dir / f"{sample_id}.json"
        metadata_dict = {
            "sample_id": sample_id,
            "sample_rate": sample_rate,
            "duration": len(audio) / sample_rate,
            "text": text,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        return audio_path

    def load_sample(self, sample_id: str) -> Optional[VoiceSample]:
        """
        Load a voice sample.

        Args:
            sample_id: Sample identifier

        Returns:
            VoiceSample or None if not found
        """
        audio_path = self.samples_dir / f"{sample_id}.wav"
        metadata_path = self.samples_dir / f"{sample_id}.json"

        if not audio_path.exists():
            return None

        try:
            audio, sample_rate = sf.read(str(audio_path))

            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata_dict = json.load(f)
                    metadata = metadata_dict.get("metadata", {})
                    text = metadata_dict.get("text")
            else:
                text = None

            return VoiceSample(
                audio=audio,
                sample_rate=sample_rate,
                text=text,
                metadata=metadata
            )
        except Exception as e:
            print(f"Error loading sample {sample_id}: {e}")
            return None

    def list_samples(self) -> List[str]:
        """List all sample IDs."""
        sample_ids = []
        for wav_file in self.samples_dir.glob("*.wav"):
            sample_id = wav_file.stem
            sample_ids.append(sample_id)
        return sorted(sample_ids)

    def delete_sample(self, sample_id: str) -> bool:
        """Delete a sample."""
        audio_path = self.samples_dir / f"{sample_id}.wav"
        metadata_path = self.samples_dir / f"{sample_id}.json"

        deleted = False
        if audio_path.exists():
            audio_path.unlink()
            deleted = True
        if metadata_path.exists():
            metadata_path.unlink()

        return deleted


class VoiceLearner:
    """
    Learns voice characteristics from samples.
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize voice learner.

        Args:
            sample_rate: Sample rate
        """
        self.sample_rate = sample_rate
        self.voice_mimic = VoiceMimic(sample_rate)
        self.pitch_controller = PitchController(sample_rate)

    def extract_features(self, audio: np.ndarray) -> Dict:
        """
        Extract comprehensive features from audio.

        Args:
            audio: Audio signal

        Returns:
            Dictionary of extracted features
        """
        # Basic voice characteristics
        characteristics = self.voice_mimic.extract_voice_characteristics(audio)

        # Additional features
        features = {
            **characteristics,
            "duration": len(audio) / self.sample_rate,
            "rms_energy": float(np.sqrt(np.mean(audio ** 2))),
            "zero_crossing_rate": float(np.mean(np.abs(np.diff(np.sign(audio))))) / 2.0,
        }

        # Spectral features
        try:
            import librosa
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
            features["spectral_centroid_std"] = float(np.std(spectral_centroids))

            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))

            # MFCCs (mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            features["mfcc_mean"] = [float(x) for x in np.mean(mfccs, axis=1)]
            features["mfcc_std"] = [float(x) for x in np.std(mfccs, axis=1)]

            # Chroma (pitch class)
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            features["chroma_mean"] = [float(x) for x in np.mean(chroma, axis=1)]
        except ImportError:
            # Fallback: simple spectral analysis
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio), 1 / self.sample_rate)

            if np.sum(magnitude) > 0:
                features["spectral_centroid_mean"] = float(np.sum(freqs * magnitude) / np.sum(magnitude))
            else:
                features["spectral_centroid_mean"] = 2000.0

            features["spectral_centroid_std"] = 0.0
            features["spectral_rolloff_mean"] = 5000.0
            features["mfcc_mean"] = [0.0] * 13
            features["mfcc_std"] = [0.0] * 13
            features["chroma_mean"] = [0.0] * 13

        return features

    def learn_from_samples(
        self,
        samples: List[VoiceSample],
        profile_name: str
    ) -> LearnedVoiceProfile:
        """
        Learn voice profile from multiple samples.

        Args:
            samples: List of voice samples
            profile_name: Name for the learned profile

        Returns:
            LearnedVoiceProfile
        """
        if not samples:
            raise ValueError("No samples provided")

        # Extract features from all samples
        all_features = []
        total_duration = 0.0

        for sample in samples:
            features = self.extract_features(sample.audio)
            all_features.append(features)
            total_duration += len(sample.audio) / sample.sample_rate

        # Aggregate features (weighted by duration)
        aggregated = {}

        for key in all_features[0].keys():
            if isinstance(all_features[0][key], (int, float)):
                # Average numeric features
                values = [f[key] for f in all_features]
                aggregated[key] = float(np.mean(values))
            elif isinstance(all_features[0][key], list):
                # Average list features
                values = [np.array(f[key]) for f in all_features]
                aggregated[key] = [float(x) for x in np.mean(values, axis=0)]
            else:
                # Use first value for other types
                aggregated[key] = all_features[0][key]

        # Create learned profile
        profile = LearnedVoiceProfile(
            name=profile_name,
            characteristics=aggregated,
            sample_count=len(samples),
            total_duration=total_duration,
            created=datetime.now().isoformat(),
            updated=datetime.now().isoformat()
        )

        return profile

    def update_profile(
        self,
        existing_profile: LearnedVoiceProfile,
        new_samples: List[VoiceSample]
    ) -> LearnedVoiceProfile:
        """
        Update existing profile with new samples.

        Args:
            existing_profile: Existing learned profile
            new_samples: New samples to add

        Returns:
            Updated profile
        """
        # Combine old and new samples
        # For simplicity, we'll re-learn from all samples
        # In production, would use incremental learning

        # Extract features from new samples
        new_features = []
        new_duration = 0.0

        for sample in new_samples:
            features = self.extract_features(sample.audio)
            new_features.append(features)
            new_duration += len(sample.audio) / sample.sample_rate

        # Combine with existing characteristics
        # Weight by sample count
        old_weight = existing_profile.sample_count
        new_weight = len(new_samples)
        total_weight = old_weight + new_weight

        updated_characteristics = {}

        for key in existing_profile.characteristics.keys():
            if isinstance(existing_profile.characteristics[key], (int, float)):
                # Weighted average
                old_val = existing_profile.characteristics[key]
                new_vals = [f[key] for f in new_features if key in f]
                if new_vals:
                    new_val = np.mean(new_vals)
                    updated_characteristics[key] = float(
                        (old_val * old_weight + new_val * new_weight) / total_weight
                    )
                else:
                    updated_characteristics[key] = old_val
            elif isinstance(existing_profile.characteristics[key], list):
                # Weighted average for lists
                old_val = np.array(existing_profile.characteristics[key])
                new_vals = [np.array(f[key]) for f in new_features if key in f]
                if new_vals:
                    new_val = np.mean(new_vals, axis=0)
                    updated = (old_val * old_weight + new_val * new_weight) / total_weight
                    updated_characteristics[key] = [float(x) for x in updated]
                else:
                    updated_characteristics[key] = existing_profile.characteristics[key]
            else:
                updated_characteristics[key] = existing_profile.characteristics[key]

        return LearnedVoiceProfile(
            name=existing_profile.name,
            characteristics=updated_characteristics,
            sample_count=existing_profile.sample_count + len(new_samples),
            total_duration=existing_profile.total_duration + new_duration,
            created=existing_profile.created,
            updated=datetime.now().isoformat()
        )


class VoiceLearningManager:
    """
    Manages voice learning workflow.
    """

    def __init__(self, storage_dir: Optional[Path] = None, sample_rate: int = 44100):
        """
        Initialize learning manager.

        Args:
            storage_dir: Storage directory for samples
            sample_rate: Sample rate
        """
        self.store = VoiceSampleStore(storage_dir)
        self.learner = VoiceLearner(sample_rate)
        self.sample_rate = sample_rate

    def add_sample(
        self,
        audio: np.ndarray,
        sample_id: Optional[str] = None,
        text: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add a voice sample for learning.

        Args:
            audio: Audio signal
            sample_id: Optional sample ID (auto-generated if None)
            text: Optional transcript
            metadata: Optional metadata

        Returns:
            Sample ID
        """
        if sample_id is None:
            sample_id = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        self.store.save_sample(audio, self.sample_rate, sample_id, text, metadata)
        return sample_id

    def learn_profile(
        self,
        profile_name: str,
        sample_ids: Optional[List[str]] = None
    ) -> LearnedVoiceProfile:
        """
        Learn a voice profile from samples.

        Args:
            profile_name: Name for the profile
            sample_ids: Optional list of sample IDs (uses all if None)

        Returns:
            LearnedVoiceProfile
        """
        if sample_ids is None:
            sample_ids = self.store.list_samples()

        # Load samples
        samples = []
        for sample_id in sample_ids:
            sample = self.store.load_sample(sample_id)
            if sample:
                samples.append(sample)

        if not samples:
            raise ValueError("No samples found")

        # Learn profile
        profile = self.learner.learn_from_samples(samples, profile_name)

        # Save profile
        self.save_profile(profile)

        return profile

    def save_profile(self, profile: LearnedVoiceProfile) -> Path:
        """Save a learned profile."""
        profile_path = self.store.profiles_dir / f"{profile.name}.json"
        with open(profile_path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        return profile_path

    def load_profile(self, profile_name: str) -> Optional[LearnedVoiceProfile]:
        """Load a learned profile."""
        profile_path = self.store.profiles_dir / f"{profile_name}.json"
        if not profile_path.exists():
            return None

        with open(profile_path, "r") as f:
            data = json.load(f)
            return LearnedVoiceProfile.from_dict(data)

    def list_profiles(self) -> List[str]:
        """List all learned profiles."""
        profiles = []
        for json_file in self.store.profiles_dir.glob("*.json"):
            profiles.append(json_file.stem)
        return sorted(profiles)

    def update_profile_from_samples(
        self,
        profile_name: str,
        new_sample_ids: List[str]
    ) -> LearnedVoiceProfile:
        """
        Update existing profile with new samples.

        Args:
            profile_name: Profile name
            new_sample_ids: New sample IDs

        Returns:
            Updated profile
        """
        # Load existing profile
        existing = self.load_profile(profile_name)
        if not existing:
            raise ValueError(f"Profile '{profile_name}' not found")

        # Load new samples
        new_samples = []
        for sample_id in new_sample_ids:
            sample = self.store.load_sample(sample_id)
            if sample:
                new_samples.append(sample)

        if not new_samples:
            raise ValueError("No new samples found")

        # Update profile
        updated = self.learner.update_profile(existing, new_samples)

        # Save updated profile
        self.save_profile(updated)

        return updated

    def get_profile_characteristics(self, profile_name: str) -> Optional[Dict]:
        """
        Get voice characteristics from a learned profile.

        Args:
            profile_name: Profile name

        Returns:
            Voice characteristics dict (compatible with Parrot.sing_with_voice)
        """
        profile = self.load_profile(profile_name)
        if not profile:
            return None

        # Extract key characteristics for synthesis
        chars = profile.characteristics
        return {
            "mean_pitch": chars.get("mean_pitch", 220.0),
            "pitch_range": chars.get("pitch_range", 200.0),
            "brightness": chars.get("brightness", chars.get("spectral_centroid_mean", 2000.0)),
            "breathiness": chars.get("breathiness", 0.2),
            "formant_emphasis": chars.get("formant_emphasis", 0.5),
            "pitch_variance": chars.get("pitch_variance", 0.0),
        }
