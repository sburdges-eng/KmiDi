"""
Social Platform Export Optimization

Exports music optimized for social media platforms (TikTok, Instagram Reels, YouTube Shorts, etc.).

Part of the "New Features" implementation for Kelly MIDI Companion.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json


class SocialPlatform(Enum):
    """Supported social media platforms."""
    TIKTOK = "tiktok"
    INSTAGRAM_REELS = "instagram_reels"
    INSTAGRAM_STORIES = "instagram_stories"
    YOUTUBE_SHORTS = "youtube_shorts"
    TWITTER = "twitter"  # X
    FACEBOOK = "facebook"
    SNAPCHAT = "snapchat"


@dataclass
class PlatformSpec:
    """Specifications for a social media platform."""
    name: str
    max_duration_seconds: int
    preferred_duration_seconds: Optional[int] = None  # Optimal length
    video_dimensions: Tuple[int, int] = (1080, 1920)  # width, height (portrait)
    audio_format: str = "mp3"
    audio_bitrate_kbps: int = 128
    max_file_size_mb: Optional[int] = None
    supports_looping: bool = True
    supports_transitions: bool = True
    requires_visual: bool = True  # Whether video/image is required
    aspect_ratio: str = "9:16"  # Portrait (TikTok, Reels) or "16:9" (YouTube Shorts)
    description_template: Optional[str] = None  # Template for description/hashtags


# Platform specifications
PLATFORM_SPECS = {
    SocialPlatform.TIKTOK: PlatformSpec(
        name="TikTok",
        max_duration_seconds=600,  # 10 minutes
        preferred_duration_seconds=15,  # 15-60 seconds is optimal
        video_dimensions=(1080, 1920),  # 9:16 portrait
        audio_format="mp3",
        audio_bitrate_kbps=128,
        max_file_size_mb=287,  # 287 MB
        supports_looping=True,
        supports_transitions=True,
        requires_visual=True,
        aspect_ratio="9:16",
        description_template="#music #originalmusic #fyp #viral"
    ),

    SocialPlatform.INSTAGRAM_REELS: PlatformSpec(
        name="Instagram Reels",
        max_duration_seconds=90,
        preferred_duration_seconds=30,
        video_dimensions=(1080, 1920),  # 9:16 portrait
        audio_format="mp3",
        audio_bitrate_kbps=128,
        max_file_size_mb=100,
        supports_looping=True,
        supports_transitions=True,
        requires_visual=True,
        aspect_ratio="9:16",
        description_template="#music #reels #originalmusic #musicproducer"
    ),

    SocialPlatform.INSTAGRAM_STORIES: PlatformSpec(
        name="Instagram Stories",
        max_duration_seconds=15,
        preferred_duration_seconds=15,
        video_dimensions=(1080, 1920),
        audio_format="mp3",
        audio_bitrate_kbps=128,
        max_file_size_mb=100,
        supports_looping=True,
        supports_transitions=False,
        requires_visual=True,
        aspect_ratio="9:16",
        description_template="#music #story"
    ),

    SocialPlatform.YOUTUBE_SHORTS: PlatformSpec(
        name="YouTube Shorts",
        max_duration_seconds=60,
        preferred_duration_seconds=15,
        video_dimensions=(1080, 1920),  # 9:16 portrait
        audio_format="mp3",
        audio_bitrate_kbps=128,
        max_file_size_mb=None,  # No strict limit
        supports_looping=False,
        supports_transitions=True,
        requires_visual=True,
        aspect_ratio="9:16",
        description_template="#shorts #music #originalmusic"
    ),

    SocialPlatform.TWITTER: PlatformSpec(
        name="Twitter/X",
        max_duration_seconds=140,  # 2:20
        preferred_duration_seconds=30,
        video_dimensions=(1280, 720),  # 16:9 landscape
        audio_format="mp3",
        audio_bitrate_kbps=128,
        max_file_size_mb=512,  # 512 MB
        supports_looping=False,
        supports_transitions=True,
        requires_visual=True,
        aspect_ratio="16:9",
        description_template="#music #newmusic"
    ),

    SocialPlatform.FACEBOOK: PlatformSpec(
        name="Facebook",
        max_duration_seconds=240,  # 4 minutes
        preferred_duration_seconds=60,
        video_dimensions=(1280, 720),
        audio_format="mp3",
        audio_bitrate_kbps=128,
        max_file_size_mb=1024,  # 1 GB
        supports_looping=False,
        supports_transitions=True,
        requires_visual=True,
        aspect_ratio="16:9",
        description_template=""
    ),
}


class SocialPlatformExporter:
    """
    Export music optimized for social media platforms.

    Usage:
        exporter = SocialPlatformExporter()

        # Export for TikTok
        result = exporter.export_for_platform(
            audio_file="song.wav",
            platform=SocialPlatform.TIKTOK,
            output_path="tiktok_export.mp4",
            start_time=0,
            duration=15
        )
    """

    def __init__(self):
        self.platform_specs = PLATFORM_SPECS

    def get_platform_spec(self, platform: SocialPlatform) -> PlatformSpec:
        """Get specifications for a platform."""
        return self.platform_specs[platform]

    def validate_audio_for_platform(
        self,
        audio_file: str,
        platform: SocialPlatform
    ) -> Tuple[bool, List[str]]:
        """
        Validate if audio file meets platform requirements.

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        spec = self.get_platform_spec(platform)
        warnings = []

        # Check if file exists
        if not Path(audio_file).exists():
            return False, [f"Audio file not found: {audio_file}"]

        # TODO: Check audio duration (requires audio library like librosa or mutagen)
        # For now, just return warnings about duration limits
        warnings.append(f"Max duration: {spec.max_duration_seconds}s")
        if spec.preferred_duration_seconds:
            warnings.append(f"Preferred duration: {spec.preferred_duration_seconds}s")

        # Check file size
        file_size_mb = Path(audio_file).stat().st_size / (1024 * 1024)
        if spec.max_file_size_mb and file_size_mb > spec.max_file_size_mb:
            return False, [f"File size {file_size_mb:.1f}MB exceeds max {spec.max_file_size_mb}MB"]

        return True, warnings

    def create_export_config(
        self,
        platform: SocialPlatform,
        audio_file: str,
        start_time: float = 0.0,
        duration: Optional[float] = None,
        fade_in: float = 0.1,
        fade_out: float = 0.1,
        normalize: bool = True,
        target_lufs: float = -14.0  # TikTok/Instagram standard
    ) -> Dict:
        """
        Create export configuration for a platform.

        Args:
            platform: Target platform
            audio_file: Source audio file
            start_time: Start time in seconds (for trimming)
            duration: Duration in seconds (None = use preferred)
            fade_in: Fade in duration in seconds
            fade_out: Fade out duration in seconds
            normalize: Whether to normalize audio
            target_lufs: Target loudness (LUFS)

        Returns:
            Export configuration dictionary
        """
        spec = self.get_platform_spec(platform)

        # Use preferred duration if not specified
        if duration is None:
            duration = spec.preferred_duration_seconds or spec.max_duration_seconds

        # Ensure duration doesn't exceed max
        duration = min(duration, spec.max_duration_seconds)

        config = {
            "platform": platform.value,
            "platform_name": spec.name,
            "source_audio": audio_file,
            "start_time": start_time,
            "duration": duration,
            "fade_in": fade_in,
            "fade_out": fade_out,
            "normalize": normalize,
            "target_lufs": target_lufs,
            "output_format": spec.audio_format,
            "bitrate_kbps": spec.audio_bitrate_kbps,
            "video_dimensions": spec.video_dimensions,
            "aspect_ratio": spec.aspect_ratio,
            "requires_visual": spec.requires_visual,
        }

        return config

    def export_for_platform(
        self,
        audio_file: str,
        platform: SocialPlatform,
        output_path: str,
        start_time: float = 0.0,
        duration: Optional[float] = None,
        create_video: bool = False,
        visual_file: Optional[str] = None,  # Image or video for visual
        **kwargs
    ) -> Dict:
        """
        Export audio optimized for a social media platform.

        Args:
            audio_file: Source audio file path
            platform: Target platform
            output_path: Output file path
            start_time: Start time in seconds
            duration: Duration in seconds
            create_video: Whether to create video file (requires visual)
            visual_file: Path to image/video for visual component
            **kwargs: Additional export options

        Returns:
            Dictionary with export information
        """
        spec = self.get_platform_spec(platform)
        config = self.create_export_config(platform, audio_file, start_time, duration, **kwargs)

        # Validate
        is_valid, warnings = self.validate_audio_for_platform(audio_file, platform)
        if not is_valid:
            raise ValueError(f"Audio validation failed: {warnings}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # TODO: Actual audio processing would happen here
        # This would use libraries like:
        # - librosa/soundfile for audio processing
        # - pydub for format conversion
        # - moviepy or ffmpeg for video creation
        # - pyloudnorm for loudness normalization

        # For now, create a configuration file
        config_path = output_path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        result = {
            "success": True,
            "platform": platform.value,
            "output_path": str(output_path),
            "config_path": str(config_path),
            "warnings": warnings,
            "spec": {
                "max_duration": spec.max_duration_seconds,
                "preferred_duration": spec.preferred_duration_seconds,
                "dimensions": spec.video_dimensions,
                "aspect_ratio": spec.aspect_ratio,
            },
            "export_config": config,
        }

        print(f"Export configuration created for {spec.name}")
        print(f"  Output: {output_path}")
        print(f"  Duration: {config['duration']}s")
        print(f"  Format: {spec.audio_format}")
        print(f"  Bitrate: {spec.audio_bitrate_kbps} kbps")

        if create_video:
            if not visual_file and spec.requires_visual:
                print(f"  Warning: {spec.name} requires a visual component")
                print(f"  Provide a visual_file or use create_video=False for audio-only")

        return result

    def batch_export_for_platforms(
        self,
        audio_file: str,
        platforms: List[SocialPlatform],
        output_directory: str,
        base_name: str = "export",
        start_time: float = 0.0,
        duration: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Dict]:
        """
        Export audio for multiple platforms at once.

        Returns:
            Dictionary mapping platform names to export results
        """
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for platform in platforms:
            spec = self.get_platform_spec(platform)
            output_path = output_dir / f"{base_name}_{platform.value}.{spec.audio_format}"

            try:
                result = self.export_for_platform(
                    audio_file=audio_file,
                    platform=platform,
                    output_path=str(output_path),
                    start_time=start_time,
                    duration=duration,
                    **kwargs
                )
                results[platform.value] = result
            except Exception as e:
                results[platform.value] = {
                    "success": False,
                    "error": str(e)
                }

        return results

    def generate_description(
        self,
        platform: SocialPlatform,
        title: Optional[str] = None,
        artist: Optional[str] = None,
        custom_tags: Optional[List[str]] = None
    ) -> str:
        """
        Generate platform-optimized description with hashtags.

        Args:
            platform: Target platform
            title: Song title
            artist: Artist name
            custom_tags: Additional custom hashtags

        Returns:
            Description text with hashtags
        """
        spec = self.get_platform_spec(platform)

        parts = []

        if title:
            parts.append(title)
        if artist:
            parts.append(f"by {artist}")

        # Add platform-specific template
        if spec.description_template:
            parts.append(spec.description_template)

        # Add custom tags
        if custom_tags:
            tags_str = " ".join(f"#{tag.replace(' ', '')}" for tag in custom_tags)
            parts.append(tags_str)

        return " ".join(parts)

    def suggest_clip_segment(
        self,
        audio_file: str,
        platform: SocialPlatform,
        total_duration: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Suggest the best segment to clip from a longer audio file.

        Returns:
            Tuple of (start_time, duration)

        TODO: Implement intelligent segment selection based on:
        - Energy levels
        - Hook detection
        - Tempo/momentum
        - Silence detection
        """
        spec = self.get_platform_spec(platform)
        preferred_duration = spec.preferred_duration_seconds or 15

        # For now, return the first segment
        # In production, this would analyze the audio to find the best hook
        return (0.0, float(preferred_duration))


def main():
    """Example usage."""
    exporter = SocialPlatformExporter()

    # Example: Export for TikTok
    result = exporter.export_for_platform(
        audio_file="song.wav",
        platform=SocialPlatform.TIKTOK,
        output_path="tiktok_export.mp3",
        duration=15
    )

    print("\nExport result:")
    print(json.dumps(result, indent=2))

    # Generate description
    description = exporter.generate_description(
        platform=SocialPlatform.TIKTOK,
        title="My Song",
        artist="Artist Name",
        custom_tags=["original", "indie", "music"]
    )
    print(f"\nDescription: {description}")


if __name__ == "__main__":
    main()
