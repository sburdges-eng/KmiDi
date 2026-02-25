"""
Emotion-Enhanced Stem Exporter

Extends stem export functionality to include emotional metadata in filenames
and embedded audio metadata (ID3, BWF, etc.).

Part of the "New Features" implementation for Kelly MIDI Companion.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import re
from dataclasses import dataclass, asdict


@dataclass
class EmotionMetadata:
    """Emotion metadata for a stem."""
    emotion_label: Optional[str] = None  # Primary emotion (e.g., "grief", "hope")
    valence: Optional[float] = None      # -1.0 to 1.0
    arousal: Optional[float] = None      # 0.0 to 1.0
    intensity: Optional[float] = None    # 0.0 to 1.0
    mood_tags: List[str] = None          # Additional mood tags
    rule_breaks: List[str] = None        # Applied rule-breaking techniques

    def __post_init__(self):
        if self.mood_tags is None:
            self.mood_tags = []
        if self.rule_breaks is None:
            self.rule_breaks = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'EmotionMetadata':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class StemExportInfo:
    """Information about a stem export with emotion metadata."""
    original_filepath: str
    emotion_metadata: EmotionMetadata
    track_name: str
    track_index: int

    def generate_emotion_filename(
        self,
        output_directory: str,
        include_emotion: bool = True,
        include_vad: bool = False,
        format_extension: str = ".wav"
    ) -> str:
        """
        Generate filename with emotion labels.

        Examples:
        - "melancholy_bass.wav"
        - "grief_valence-0.7_bass.wav"
        - "hope_arousal0.8_intensity0.9_vocals.wav"
        """
        output_dir = Path(output_directory)
        base_name = self.track_name.lower()

        # Sanitize base name
        base_name = re.sub(r'[^\w\s-]', '', base_name)
        base_name = re.sub(r'\s+', '_', base_name).strip('_')

        if not base_name:
            base_name = f"track_{self.track_index}"

        parts = []

        # Add emotion label if available
        if include_emotion and self.emotion_metadata.emotion_label:
            emotion = self.emotion_metadata.emotion_label.lower()
            emotion = re.sub(r'[^\w]', '', emotion)  # Sanitize
            parts.append(emotion)

        # Add VAD values if requested
        if include_vad:
            if self.emotion_metadata.valence is not None:
                parts.append(f"v{self.emotion_metadata.valence:.2f}".replace('.', '').replace('-', 'n'))
            if self.emotion_metadata.arousal is not None:
                parts.append(f"a{self.emotion_metadata.arousal:.2f}".replace('.', ''))
            if self.emotion_metadata.intensity is not None:
                parts.append(f"i{self.emotion_metadata.intensity:.2f}".replace('.', ''))

        # Build filename
        if parts:
            filename = "_".join(parts) + "_" + base_name
        else:
            filename = base_name

        filename += format_extension

        return str(output_dir / filename)

    def generate_display_name(self) -> str:
        """Generate human-readable display name with emotion."""
        parts = []

        if self.emotion_metadata.emotion_label:
            emotion = self.emotion_metadata.emotion_label.title()
            parts.append(emotion)

        parts.append(self.track_name)

        return " - ".join(parts) if len(parts) > 1 else self.track_name


class EmotionStemExporter:
    """
    Enhanced stem exporter with emotion metadata support.

    Usage:
        exporter = EmotionStemExporter()

        # Create emotion metadata
        metadata = EmotionMetadata(
            emotion_label="grief",
            valence=-0.7,
            arousal=0.4,
            intensity=0.6,
            mood_tags=["melancholy", "sad"],
            rule_breaks=["HARMONY_AvoidTonicResolution"]
        )

        # Create export info
        export_info = StemExportInfo(
            original_filepath="bass.wav",
            emotion_metadata=metadata,
            track_name="Bass",
            track_index=0
        )

        # Generate emotion-labeled filename
        new_path = export_info.generate_emotion_filename("./exports", include_emotion=True)

        # Export with metadata
        exporter.export_with_emotion_metadata(export_info, new_path)
    """

    def __init__(self):
        self.emotion_map = {
            # Common emotion synonyms for filename sanitization
            "sadness": "sad",
            "happiness": "happy",
            "anger": "angry",
            "fear": "afraid",
            "joy": "joyful",
            "sorrow": "sorrowful",
        }

    def sanitize_emotion_label(self, emotion: str) -> str:
        """Sanitize emotion label for use in filenames."""
        emotion = emotion.lower().strip()

        # Normalize common variations
        emotion = self.emotion_map.get(emotion, emotion)

        # Remove special characters
        emotion = re.sub(r'[^\w]', '', emotion)

        return emotion

    def generate_emotion_tags(self, metadata: EmotionMetadata) -> List[str]:
        """Generate list of tags from emotion metadata."""
        tags = []

        if metadata.emotion_label:
            tags.append(metadata.emotion_label.lower())

        tags.extend(metadata.mood_tags)

        # Add VAD-based tags
        if metadata.valence is not None:
            if metadata.valence > 0.5:
                tags.append("positive")
            elif metadata.valence < -0.5:
                tags.append("negative")
            else:
                tags.append("neutral")

        if metadata.arousal is not None:
            if metadata.arousal > 0.7:
                tags.append("high-energy")
            elif metadata.arousal < 0.3:
                tags.append("calm")

        if metadata.intensity is not None:
            if metadata.intensity > 0.7:
                tags.append("intense")
            elif metadata.intensity < 0.3:
                tags.append("subtle")

        return list(set(tags))  # Remove duplicates

    def create_metadata_json(
        self,
        export_info: StemExportInfo,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Create JSON metadata file for stem export.

        Returns metadata dict, optionally saves to file.
        """
        metadata = {
            "stem_export": {
                "track_name": export_info.track_name,
                "track_index": export_info.track_index,
                "original_filepath": export_info.original_filepath,
                "display_name": export_info.generate_display_name(),
            },
            "emotion": export_info.emotion_metadata.to_dict(),
            "tags": self.generate_emotion_tags(export_info.emotion_metadata),
            "export_info": {
                "format": "stem_with_emotion_metadata",
                "version": "1.0.0"
            }
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata to {output_path}")

        return metadata

    def export_with_emotion_metadata(
        self,
        export_info: StemExportInfo,
        output_filepath: str,
        create_metadata_file: bool = True,
        copy_audio: bool = False
    ) -> str:
        """
        Export stem with emotion metadata.

        Args:
            export_info: Stem export information with emotion metadata
            output_filepath: Path for output stem file
            create_metadata_file: Whether to create .json metadata file
            copy_audio: Whether to copy audio file (if False, just creates metadata)

        Returns:
            Path to exported file
        """
        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy audio file if requested
        if copy_audio and Path(export_info.original_filepath).exists():
            import shutil
            shutil.copy2(export_info.original_filepath, output_path)
            print(f"Copied audio to {output_path}")
        elif copy_audio:
            print(f"Warning: Source audio file not found: {export_info.original_filepath}")

        # Create metadata JSON file
        if create_metadata_file:
            metadata_path = output_path.with_suffix('.json')
            self.create_metadata_json(export_info, str(metadata_path))

        # TODO: Embed metadata in audio file (requires mutagen or similar library)
        # This would add ID3 tags, BWF chunks, etc.

        return str(output_path)

    def batch_export_with_emotions(
        self,
        export_infos: List[StemExportInfo],
        output_directory: str,
        filename_style: str = "emotion"  # "emotion", "vad", "simple"
    ) -> List[str]:
        """
        Batch export multiple stems with emotion metadata.

        Args:
            export_infos: List of stem export information
            output_directory: Directory for output files
            filename_style: Style of filename ("emotion", "vad", "simple")

        Returns:
            List of output file paths
        """
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = []

        for export_info in export_infos:
            if filename_style == "emotion":
                filename = export_info.generate_emotion_filename(
                    str(output_dir),
                    include_emotion=True,
                    include_vad=False
                )
            elif filename_style == "vad":
                filename = export_info.generate_emotion_filename(
                    str(output_dir),
                    include_emotion=True,
                    include_vad=True
                )
            else:  # simple
                filename = export_info.generate_emotion_filename(
                    str(output_dir),
                    include_emotion=False,
                    include_vad=False
                )

            output_path = self.export_with_emotion_metadata(
                export_info,
                filename,
                create_metadata_file=True,
                copy_audio=True
            )

            output_paths.append(output_path)

        return output_paths

    def create_export_manifest(
        self,
        export_infos: List[StemExportInfo],
        output_path: str
    ) -> Dict:
        """
        Create manifest file listing all exported stems with their emotion metadata.

        Useful for batch processing and documentation.
        """
        manifest = {
            "export_manifest": {
                "version": "1.0.0",
                "total_stems": len(export_infos),
                "export_date": str(Path(output_path).parent),
            },
            "stems": []
        }

        for export_info in export_infos:
            stem_data = {
                "track_name": export_info.track_name,
                "track_index": export_info.track_index,
                "display_name": export_info.generate_display_name(),
                "emotion": export_info.emotion_metadata.emotion_label,
                "vad": {
                    "valence": export_info.emotion_metadata.valence,
                    "arousal": export_info.emotion_metadata.arousal,
                    "intensity": export_info.emotion_metadata.intensity,
                },
                "tags": self.generate_emotion_tags(export_info.emotion_metadata),
                "rule_breaks": export_info.emotion_metadata.rule_breaks,
            }
            manifest["stems"].append(stem_data)

        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"Created export manifest: {output_path}")
        return manifest


def create_emotion_metadata_from_intent(
    intent_data: Dict,
    track_name: str,
    track_index: int = 0
) -> Tuple[EmotionMetadata, StemExportInfo]:
    """
    Helper function to create emotion metadata from song intent data.

    Args:
        intent_data: Song intent dictionary (from intent_schema.py)
        track_name: Name of the track
        track_index: Index of the track

    Returns:
        Tuple of (EmotionMetadata, StemExportInfo)
    """
    # Extract emotion from intent
    emotion_label = None
    if "mood_primary" in intent_data:
        emotion_label = intent_data["mood_primary"]
    elif "emotion" in intent_data:
        emotion_label = intent_data["emotion"]

    # Extract rule breaks
    rule_breaks = []
    if "rule_breaks" in intent_data:
        rule_breaks = intent_data["rule_breaks"]
    elif "technical_rule_to_break" in intent_data:
        if intent_data["technical_rule_to_break"]:
            rule_breaks = [intent_data["technical_rule_to_break"]]

    # Extract mood tags
    mood_tags = []
    if "mood_tags" in intent_data:
        mood_tags = intent_data["mood_tags"]
    elif "mood_secondary" in intent_data:
        mood_tags = [intent_data["mood_secondary"]]

    # Create metadata
    metadata = EmotionMetadata(
        emotion_label=emotion_label,
        mood_tags=mood_tags,
        rule_breaks=rule_breaks
    )

    # Create export info (will need original_filepath to be set later)
    export_info = StemExportInfo(
        original_filepath="",  # Set later
        emotion_metadata=metadata,
        track_name=track_name,
        track_index=track_index
    )

    return metadata, export_info


def main():
    """Example usage."""
    exporter = EmotionStemExporter()

    # Create sample export info
    metadata = EmotionMetadata(
        emotion_label="grief",
        valence=-0.7,
        arousal=0.4,
        intensity=0.6,
        mood_tags=["melancholy", "sad"],
        rule_breaks=["HARMONY_AvoidTonicResolution"]
    )

    export_info = StemExportInfo(
        original_filepath="bass_track.wav",
        emotion_metadata=metadata,
        track_name="Bass",
        track_index=0
    )

    # Generate filename
    filename = export_info.generate_emotion_filename("./exports", include_emotion=True)
    print(f"Generated filename: {filename}")
    print(f"Display name: {export_info.generate_display_name()}")

    # Create metadata JSON
    exporter.create_metadata_json(export_info, "./exports/bass_metadata.json")


if __name__ == "__main__":
    main()
