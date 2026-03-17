"""Unit tests for VideoGenerator.cleanup() temp dir removal and safety."""

import pytest
from pathlib import Path

from music_brain.video.video_generator import (
    VideoGenerator,
    TEMP_SUBDIR_PREFIX,
)


class TestVideoGeneratorCleanup:
    """Tests for VideoGenerator.cleanup() behavior."""

    def test_cleanup_removes_temp_dir_and_contents(self):
        """Cleanup removes generator-created temp dir and all contents (including hidden)."""
        gen = VideoGenerator()
        gen.initialize()
        temp_dir = gen._ensure_temp_dir()
        assert temp_dir.exists()
        assert TEMP_SUBDIR_PREFIX in temp_dir.name

        # Add files and subdirs including hidden
        (temp_dir / "file1.txt").write_text("dummy")
        (temp_dir / ".hidden_file").write_text("hidden")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "file2.txt").write_text("dummy2")
        (temp_dir / ".hidden_subdir").mkdir()
        (temp_dir / ".hidden_subdir" / "file3.txt").write_text("hidden2")

        gen.cleanup()

        assert gen._initialized is False
        assert gen._temp_dir is None
        assert not temp_dir.exists()

    def test_cleanup_when_no_temp_dir(self):
        """Cleanup when _ensure_temp_dir was never called does not raise."""
        gen = VideoGenerator()
        gen.initialize()
        assert gen._temp_dir is None
        gen.cleanup()
        assert gen._initialized is False
        assert gen._temp_dir is None

    def test_cleanup_when_temp_dir_already_removed(self):
        """Cleanup when temp dir was already deleted (e.g. externally) does not raise."""
        gen = VideoGenerator()
        gen.initialize()
        temp_dir = gen._ensure_temp_dir()
        assert temp_dir.exists()
        import shutil
        shutil.rmtree(temp_dir)
        gen.cleanup()
        assert gen._initialized is False
        assert gen._temp_dir is None

    def test_cleanup_skips_unsafe_path(self, tmp_path):
        """Cleanup skips deletion when path fails safety check (e.g. wrong prefix)."""
        # Use a path that exists but does not have our prefix so _is_safe_to_delete returns False
        unsafe_dir = tmp_path / "other_dir"
        unsafe_dir.mkdir()
        (unsafe_dir / "file.txt").write_text("content")

        gen = VideoGenerator()
        gen.initialize()
        gen._temp_dir = unsafe_dir
        gen.cleanup()

        assert gen._initialized is False
        assert gen._temp_dir is None
        # Dir should still exist (we skipped deletion)
        assert unsafe_dir.exists()
        assert (unsafe_dir / "file.txt").read_text() == "content"
