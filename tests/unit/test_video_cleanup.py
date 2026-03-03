import sys
from unittest.mock import MagicMock

# Mock dependencies
sys.modules['numpy'] = MagicMock()
sys.modules['pydantic'] = MagicMock()
sys.modules['music21'] = MagicMock()
sys.modules['mido'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['librosa'] = MagicMock()
sys.modules['yaml'] = MagicMock()
sys.modules['scipy'] = MagicMock()

import os
import shutil
from pathlib import Path
from music_brain.video.video_generator import VideoGenerator, VideoConfig

def test_video_generator_cleanup(tmp_path):
    # Setup
    temp_dir = tmp_path / "video_temp"
    temp_dir.mkdir()

    # Create some dummy files and directories (including hidden ones)
    (temp_dir / "file1.txt").write_text("dummy content")
    (temp_dir / ".hidden_file").write_text("hidden content")
    (temp_dir / "subdir").mkdir()
    (temp_dir / "subdir" / "file2.txt").write_text("dummy content 2")
    (temp_dir / ".hidden_subdir").mkdir()
    (temp_dir / ".hidden_subdir" / "file3.txt").write_text("hidden content 2")

    config = VideoConfig(temp_dir=temp_dir)
    generator = VideoGenerator(config=config)
    generator.initialize()

    assert generator._initialized is True
    assert (temp_dir / "file1.txt").exists()
    assert (temp_dir / ".hidden_file").exists()
    assert (temp_dir / "subdir").exists()
    assert (temp_dir / ".hidden_subdir").exists()

    # Cleanup
    generator.cleanup()

    # Verify
    assert generator._initialized is False
    # Check that contents are gone (including hidden ones)
    assert not (temp_dir / "file1.txt").exists()
    assert not (temp_dir / ".hidden_file").exists()
    assert not (temp_dir / "subdir").exists()
    assert not (temp_dir / ".hidden_subdir").exists()
    assert temp_dir.exists()
    assert len(list(temp_dir.iterdir())) == 0

def test_cleanup_no_temp_dir():
    config = VideoConfig(temp_dir=None)
    generator = VideoGenerator(config=config)
    generator.initialize()

    # Should not raise any error
    generator.cleanup()
    assert generator._initialized is False

def test_cleanup_nonexistent_temp_dir(tmp_path):
    temp_dir = tmp_path / "nonexistent"
    config = VideoConfig(temp_dir=temp_dir)
    generator = VideoGenerator(config=config)
    generator.initialize()

    # Should not raise any error
    generator.cleanup()
    assert generator._initialized is False

def test_cleanup_permission_error(tmp_path, monkeypatch):
    temp_dir = tmp_path / "permission_test"
    temp_dir.mkdir()
    (temp_dir / "locked_file.txt").write_text("content")

    def mock_unlink(self):
        raise PermissionError("Locked")

    # Mock unlink to raise PermissionError
    monkeypatch.setattr(Path, "unlink", mock_unlink)

    config = VideoConfig(temp_dir=temp_dir)
    generator = VideoGenerator(config=config)
    generator.initialize()

    # Should not raise any error despite mock failure
    generator.cleanup()
    assert generator._initialized is False
    # File should still exist because we mocked failure
    assert (temp_dir / "locked_file.txt").exists()

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
