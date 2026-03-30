"""Preset manager for saving, loading, and managing presets.

Presets are stored in a presets/ subdirectory relative to the project file.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional
from .preset import Preset

logger = logging.getLogger(__name__)


class PresetManager:
    """Manages preset storage and retrieval.

    Presets are stored as JSON files in a presets/ subdirectory.
    Each preset is saved as {preset_id}.json.
    """

    def __init__(self, project_path: Optional[Path] = None):
        """Initialize preset manager.

        Args:
            project_path: Path to project file or directory. If None, uses current directory.
        """
        self.project_path = project_path or Path.cwd()
        if self.project_path.is_file():
            # If project_path is a file, use its parent directory
            self.project_path = self.project_path.parent

        self.presets_dir = self.project_path / "presets"
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def save_preset(self, preset: Preset, project_path: Optional[Path] = None) -> Path:
        """Save preset to disk.

        Args:
            preset: Preset to save
            project_path: Optional project path override

        Returns:
            Path to saved preset file

        Raises:
            IOError: If file cannot be written
            ValueError: If preset validation fails
        """
        if project_path:
            self.project_path = project_path if project_path.is_dir() else project_path.parent
            self.presets_dir = self.project_path / "presets"
            self.presets_dir.mkdir(parents=True, exist_ok=True)

        # Validate preset
        issues = preset.validate()
        if issues:
            raise ValueError(f"Preset validation failed: {', '.join(issues)}")

        # Save to JSON file
        preset_path = self.presets_dir / f"{preset.id}.json"

        try:
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(preset.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved preset: {preset.name} ({preset.id})")
            return preset_path
        except IOError as e:
            logger.error(f"Failed to save preset: {e}")
            raise

    def load_preset(self, preset_id: str, project_path: Optional[Path] = None) -> Optional[Preset]:
        """Load preset from disk.

        Args:
            preset_id: UUID of preset to load
            project_path: Optional project path override

        Returns:
            Loaded preset, or None if not found
        """
        if project_path:
            self.project_path = project_path if project_path.is_dir() else project_path.parent
            self.presets_dir = self.project_path / "presets"

        preset_path = self.presets_dir / f"{preset_id}.json"

        if not preset_path.exists():
            logger.warning(f"Preset not found: {preset_id}")
            return None

        try:
            with open(preset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            preset = Preset.from_dict(data)

            # Handle version migration if needed
            preset = self._migrate_preset(preset)

            logger.info(f"Loaded preset: {preset.name} ({preset.id})")
            return preset
        except (IOError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load preset {preset_id}: {e}")
            return None

    def list_presets(self, project_path: Optional[Path] = None) -> List[Preset]:
        """List all presets for the project.

        Args:
            project_path: Optional project path override

        Returns:
            List of presets (may be empty)
        """
        if project_path:
            self.project_path = project_path if project_path.is_dir() else project_path.parent
            self.presets_dir = self.project_path / "presets"

        if not self.presets_dir.exists():
            return []

        presets = []
        for preset_file in self.presets_dir.glob("*.json"):
            try:
                preset_id = preset_file.stem
                preset = self.load_preset(preset_id)
                if preset:
                    presets.append(preset)
            except Exception as e:
                logger.warning(f"Skipping invalid preset file {preset_file.name}: {e}")

        # Sort by timestamp (newest first)
        presets.sort(key=lambda p: p.timestamp, reverse=True)
        return presets

    def delete_preset(self, preset_id: str, project_path: Optional[Path] = None) -> bool:
        """Delete preset from disk.

        Args:
            preset_id: UUID of preset to delete
            project_path: Optional project path override

        Returns:
            True if deleted, False if not found
        """
        if project_path:
            self.project_path = project_path if project_path.is_dir() else project_path.parent
            self.presets_dir = self.project_path / "presets"

        preset_path = self.presets_dir / f"{preset_id}.json"

        if not preset_path.exists():
            logger.warning(f"Preset not found for deletion: {preset_id}")
            return False

        try:
            preset_path.unlink()
            logger.info(f"Deleted preset: {preset_id}")
            return True
        except IOError as e:
            logger.error(f"Failed to delete preset {preset_id}: {e}")
            return False

    def _migrate_preset(self, preset: Preset) -> Preset:
        """Migrate preset to current version if needed.

        Args:
            preset: Preset to migrate

        Returns:
            Migrated preset (may be same object if no migration needed)
        """
        # Future: Add version migration logic here
        # For now, version 1.0.0 requires no migration
        if preset.version == "1.0.0":
            return preset

        # Unknown versions: log warning but try to load anyway
        logger.warning(f"Unknown preset version: {preset.version}, attempting to load anyway")
        return preset
