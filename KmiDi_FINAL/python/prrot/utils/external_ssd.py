"""
External SSD Path Management - Manage paths and batching for USB 2.0 bandwidth

Optimizes file I/O for external SSD connected over USB 2.0 by batching operations
and reusing cached data.
"""

import os
from pathlib import Path
from typing import Optional, List
from functools import lru_cache


class ExternalSSDManager:
    """Manage external SSD paths and optimize I/O for USB 2.0 bandwidth"""

    # USB 2.0 bandwidth: ~35-40 MB/s theoretical, ~30 MB/s practical
    # Batch operations to minimize overhead
    BATCH_SIZE_MB = 10  # Read/write in 10MB batches

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize external SSD manager"""
        if base_path is None:
            # Try to get from environment variable
            base_path = os.getenv("PRROT_EXTERNAL_SSD_PATH")
            if base_path:
                base_path = Path(base_path)
            else:
                # Default path for macOS
                base_path = Path("/Volumes/ExternalSSD/prrot")

        self.base_path = Path(base_path)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist"""
        directories = ["reference_audio", "voice_profiles", "models", "jobs", "caches", "outputs"]

        for directory in directories:
            path = self.base_path / directory
            path.mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=128)
    def get_reference_audio_path(self, profile_id: str) -> Path:
        """Get path for reference audio directory"""
        return self.base_path / "reference_audio" / profile_id / "samples"

    @lru_cache(maxsize=128)
    def get_voice_profile_path(self, profile_id: str) -> Path:
        """Get path for voice profile directory"""
        return self.base_path / "voice_profiles" / profile_id

    @lru_cache(maxsize=32)
    def get_model_path(self, model_name: str) -> Path:
        """Get path for model file"""
        return self.base_path / "models" / model_name

    @lru_cache(maxsize=128)
    def get_job_path(self, job_id: str) -> Path:
        """Get path for job directory"""
        return self.base_path / "jobs" / job_id

    @lru_cache(maxsize=128)
    def get_cache_path(self, profile_id: str) -> Path:
        """Get path for cache directory"""
        return self.base_path / "caches" / profile_id

    @lru_cache(maxsize=128)
    def get_output_path(self, output_id: str) -> Path:
        """Get path for output directory"""
        return self.base_path / "outputs" / output_id

    def list_reference_audio_files(self, profile_id: str) -> List[Path]:
        """List all reference audio files for a profile"""
        audio_path = self.get_reference_audio_path(profile_id)
        if not audio_path.exists():
            return []

        audio_files = []
        for ext in [".wav", ".flac", ".mp3"]:
            audio_files.extend(audio_path.glob(f"*{ext}"))

        return sorted(audio_files)

    def batch_read_files(
        self, file_paths: List[Path], chunk_size_mb: Optional[int] = None
    ) -> List[bytes]:
        """
        Batch read files to optimize USB 2.0 bandwidth

        Args:
            file_paths: List of file paths to read
            chunk_size_mb: Chunk size in MB (default: BATCH_SIZE_MB)

        Returns:
            List of file contents as bytes
        """
        if chunk_size_mb is None:
            chunk_size_mb = self.BATCH_SIZE_MB

        chunk_size_bytes = chunk_size_mb * 1024 * 1024
        results = []

        for file_path in file_paths:
            if not file_path.exists():
                results.append(None)
                continue

            # Read in chunks for large files
            file_size = file_path.stat().st_size
            if file_size > chunk_size_bytes:
                chunks = []
                with open(file_path, "rb") as f:
                    while True:
                        chunk = f.read(chunk_size_bytes)
                        if not chunk:
                            break
                        chunks.append(chunk)
                results.append(b"".join(chunks))
            else:
                # Read entire file for small files
                with open(file_path, "rb") as f:
                    results.append(f.read())

        return results

    def batch_write_files(self, file_paths: List[Path], file_contents: List[bytes]) -> None:
        """
        Batch write files to optimize USB 2.0 bandwidth

        Args:
            file_paths: List of file paths to write
            file_contents: List of file contents as bytes
        """
        if len(file_paths) != len(file_contents):
            raise ValueError("Number of file paths must match number of file contents")

        for file_path, contents in zip(file_paths, file_contents):
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(contents)

    def get_cache_file_path(self, profile_id: str, cache_type: str) -> Path:
        """Get path for cache file"""
        cache_path = self.get_cache_path(profile_id)
        if cache_type == "embeddings":
            return cache_path / "embeddings.bin"
        elif cache_type == "features":
            return cache_path / "features.json"
        else:
            return cache_path / f"{cache_type}.bin"

    def is_cache_valid(
        self, profile_id: str, cache_type: str, max_age_seconds: Optional[int] = None
    ) -> bool:
        """Check if cache file exists and is valid"""
        cache_file = self.get_cache_file_path(profile_id, cache_type)
        if not cache_file.exists():
            return False

        if max_age_seconds is not None:
            import time

            file_age = time.time() - cache_file.stat().st_mtime
            if file_age > max_age_seconds:
                return False

        return True
