"""
Video Generator - Main orchestrator for music video generation.

Integrates emotion-driven music with visual generation through
Unreal Engine and Jespa to create synchronized music videos.
"""

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

# Known subdirectory prefix for generator-created temp dirs. Used to ensure
# cleanup only removes dirs created by this generator.
TEMP_SUBDIR_PREFIX = "kmidi_video_gen_"


class VideoFormat(Enum):
    """Supported video output formats."""
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    WEBM = "webm"


class VideoQuality(Enum):
    """Video quality presets."""
    LOW = "low"           # 720p, lower bitrate
    MEDIUM = "medium"     # 1080p, standard bitrate
    HIGH = "high"         # 1080p, high bitrate
    ULTRA = "ultra"       # 4K, very high bitrate


@dataclass
class VideoConfig:
    """Configuration for video generation."""

    # Output settings
    output_path: Optional[Path] = None
    format: VideoFormat = VideoFormat.MP4
    quality: VideoQuality = VideoQuality.MEDIUM

    # Resolution and frame rate
    width: int = 1920
    height: int = 1080
    fps: int = 30

    # Rendering backend
    use_unreal: bool = True
    use_jespa: bool = True

    # Performance
    use_gpu: bool = True
    max_render_time: Optional[float] = None  # seconds, None = unlimited

    # Scene settings
    auto_sync_to_music: bool = True
    scene_transition_style: str = "crossfade"

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoGenerationResult:
    """Result from video generation process."""

    success: bool
    output_path: Optional[Path] = None
    duration: float = 0.0  # seconds
    frame_count: int = 0
    resolution: tuple = (0, 0)
    file_size: int = 0  # bytes
    render_time: float = 0.0  # seconds
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VideoGenerator:
    """
    Main orchestrator for emotion-driven music video generation.

    Coordinates between:
    - Emotion analysis from music_brain
    - Visual parameter mapping
    - Scene composition
    - Unreal Engine rendering
    - Jespa post-processing

    Example:
        >>> from music_brain.video import VideoGenerator, VideoConfig
        >>> gen = VideoGenerator()
        >>> result = gen.generate_from_emotion(
        ...     emotion="grief",
        ...     music_path="song.wav",
        ...     config=VideoConfig(quality=VideoQuality.HIGH)
        ... )
        >>> print(f"Video saved to: {result.output_path}")
    """

    def __init__(self, config: Optional[VideoConfig] = None):
        """
        Initialize the video generator.

        Args:
            config: Optional video configuration. Defaults to VideoConfig().
        """
        self.config = config or VideoConfig()
        self._initialized = False
        self._temp_dir: Optional[Path] = None

    def initialize(self) -> bool:
        """
        Initialize all video generation backends.

        Returns:
            True if initialization successful, False otherwise.

        Note:
            This is a stub. Future implementation will:
            - Initialize Unreal Engine connection
            - Set up Jespa processing pipeline
            - Load visual parameter mappings
            - Verify GPU availability
        """
        # TODO: Initialize Unreal Engine bridge
        # TODO: Initialize Jespa connector
        # TODO: Load emotion-visual mappings
        # TODO: Verify rendering capabilities

        self._initialized = True
        return True

    def generate_from_emotion(
        self,
        emotion: str,
        music_path: Optional[Path] = None,
        intensity: float = 0.5,
        config: Optional[VideoConfig] = None
    ) -> VideoGenerationResult:
        """
        Generate a music video from an emotion and optional music file.

        Args:
            emotion: Primary emotion (e.g., "grief", "joy", "fear")
            music_path: Path to music audio file. If None, generates video from emotion only.
            intensity: Emotion intensity from 0.0 to 1.0
            config: Optional config override

        Returns:
            VideoGenerationResult with generation details

        Note:
            This is a stub. Future implementation will:
            - Analyze music for structure and timing
            - Map emotion to visual parameters
            - Generate scene definitions
            - Render with Unreal Engine
            - Post-process with Jespa
        """
        if not self._initialized:
            self.initialize()

        # TODO: Implement actual generation
        # 1. Analyze music (if provided) for beats, sections, dynamics
        # 2. Map emotion + intensity to visual parameters
        # 3. Generate scene timeline from music structure
        # 4. Render scenes with Unreal Engine
        # 5. Apply Jespa effects and transitions
        # 6. Export final video

        return VideoGenerationResult(
            success=False,
            error_message="Video generation not yet implemented - this is a stub"
        )

    def generate_from_intent(
        self,
        intent: Dict[str, Any],
        music_path: Optional[Path] = None,
        config: Optional[VideoConfig] = None
    ) -> VideoGenerationResult:
        """
        Generate a music video from a complete song intent.

        Args:
            intent: SongIntent dictionary with emotional and musical parameters
            music_path: Path to generated or existing music file
            config: Optional config override

        Returns:
            VideoGenerationResult with generation details

        Note:
            This is a stub. Future implementation will use the full
            intent schema to create sophisticated, emotionally-aligned visuals.
        """
        if not self._initialized:
            self.initialize()

        # TODO: Extract emotion from intent
        # TODO: Use musical structure from intent for scene timing
        # TODO: Apply rule-breaking justifications to visual choices
        # TODO: Generate video with emotional coherence

        return VideoGenerationResult(
            success=False,
            error_message="Intent-based video generation not yet implemented - this is a stub"
        )

    def preview_scene(
        self,
        emotion: str,
        timestamp: float = 0.0,
        intensity: float = 0.5
    ) -> Optional[bytes]:
        """
        Generate a single preview frame for a given emotion and time.

        Args:
            emotion: Emotion to visualize
            timestamp: Time position in seconds
            intensity: Emotion intensity 0.0-1.0

        Returns:
            PNG image data as bytes, or None if preview fails

        Note:
            This is a stub for real-time preview functionality.
        """
        # TODO: Implement preview rendering
        # - Use Unreal Engine's quick render mode
        # - Return single frame as PNG

        return None

    def _ensure_temp_dir(self) -> Path:
        """
        Create and return a temp directory for this generator. Uses a known
        prefix so cleanup can safely identify and remove only our dirs.

        Returns:
            Path to the created temp directory.
        """
        if self._temp_dir is not None and self._temp_dir.exists():
            return self._temp_dir
        base = Path(tempfile.gettempdir())
        self._temp_dir = Path(tempfile.mkdtemp(prefix=TEMP_SUBDIR_PREFIX, dir=base))
        logger.debug("Created temp dir for video generation: %s", self._temp_dir)
        return self._temp_dir

    def _is_safe_to_delete(self, path: Path) -> bool:
        """
        Check if the given path is safe to delete (inside system temp and
        created by this generator).

        Returns:
            True if path is safe to delete.
        """
        try:
            resolved = path.resolve()
            system_temp = Path(tempfile.gettempdir()).resolve()
            # Must be inside system temp directory
            if not str(resolved).startswith(str(system_temp)):
                logger.warning(
                    "Refusing to delete path outside system temp: %s (system temp: %s)",
                    resolved,
                    system_temp,
                )
                return False
            # Must contain our known subdirectory prefix
            if TEMP_SUBDIR_PREFIX not in resolved.name:
                logger.warning(
                    "Refusing to delete path not created by this generator: %s "
                    "(expected prefix: %s)",
                    resolved,
                    TEMP_SUBDIR_PREFIX,
                )
                return False
            return True
        except (OSError, RuntimeError) as e:
            logger.warning("Could not resolve path for safety check: %s - %s", path, e)
            return False

    def cleanup(self) -> None:
        """
        Clean up video generation resources.

        Safely removes only temp directories created by this generator:
        - Only deletes paths inside the system temp directory
        - Only deletes dirs with our known subdirectory prefix
        - Uses structured logging and proper error handling
        """
        # TODO: Cleanup Unreal Engine connection
        # TODO: Cleanup Jespa resources

        if self._temp_dir is not None:
            path = self._temp_dir
            self._temp_dir = None
            if path.exists() and self._is_safe_to_delete(path):
                try:
                    shutil.rmtree(path)
                    logger.info("Cleaned up video generator temp dir: %s", path)
                except OSError as e:
                    logger.exception(
                        "Failed to remove temp dir %s: %s",
                        path,
                        e,
                        exc_info=True,
                    )
            elif path.exists():
                logger.debug(
                    "Skipped deletion of temp dir (safety check failed): %s", path
                )

        self._initialized = False
