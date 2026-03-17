#!/usr/bin/env python3
"""
Real Dataset Preparation Script for Kelly ML Training.

Downloads, preprocesses, and prepares real audio datasets for training:
- Emotion recognition datasets
- MIDI melody datasets
- Chord progression datasets
- Groove/timing datasets

All data is stored in a repository-relative default location unless overridden via
AUDIO_DATA_ROOT or KMIDI_DATA_ROOT.

Usage:
    python scripts/prepare_datasets.py --dataset emotion --download
    python scripts/prepare_datasets.py --dataset all --preprocess
    python scripts/prepare_datasets.py --list
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Audio data root (override via env AUDIO_DATA_ROOT, KMIDI_DATA_ROOT, or --root)
def _detect_default_root() -> Path:
    """Pick the best available mount point for large datasets."""
    candidates = [
        Path(os.environ.get("AUDIO_DATA_ROOT")) if os.environ.get("AUDIO_DATA_ROOT") else None,
        # KmiDi .env: use Datasets on external drive so prepare_datasets writes there
        (Path(os.environ.get("KMIDI_DATA_ROOT")) / "Datasets") if os.environ.get("KMIDI_DATA_ROOT") else None,
        ROOT / "kmidi_audio_data",
        ROOT / "datasets" / "kmidi_audio_data",
        Path.home() / "kmidi_audio_data",
    ]
    for path in candidates:
        if not path:
            continue
        candidate = path.expanduser()
        if candidate.exists():
            return candidate
        if candidate.parent.exists():
            return path
    # Fallback to workspace
    return Path.cwd() / "kmidi_audio_data"


DEFAULT_AUDIO_ROOT = _detect_default_root()
AUDIO_DATA_ROOT = DEFAULT_AUDIO_ROOT.expanduser()


# =============================================================================
# Dataset Configurations
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    task: str
    description: str
    sources: List[Dict[str, Any]]
    output_dir: str
    sample_rate: int = 16000
    max_duration: float = 10.0
    min_duration: float = 0.5


DATASETS = {
    "emotion_ravdess": DatasetConfig(
        name="RAVDESS",
        task="emotion",
        description="Ryerson Audio-Visual Database of Emotional Speech and Song",
        sources=[
            {
                "type": "kaggle",
                "dataset": "uwrfkaggler/ravdess-emotional-speech-audio",
                "files": ["*.wav"],
            }
        ],
        output_dir="emotions/ravdess",
        sample_rate=16000,
        max_duration=5.0,
    ),
    "emotion_cremad": DatasetConfig(
        name="CREMA-D",
        task="emotion",
        description="Crowd-sourced Emotional Multimodal Actors Dataset",
        sources=[
            {
                "type": "url",
                "url": "https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip",
            }
        ],
        output_dir="emotions/cremad",
        sample_rate=16000,
    ),
    "emotion_tess": DatasetConfig(
        name="TESS",
        task="emotion",
        description="Toronto Emotional Speech Set",
        sources=[
            {
                "type": "kaggle",
                "dataset": "ejlok1/toronto-emotional-speech-set-tess",
            }
        ],
        output_dir="emotions/tess",
        sample_rate=16000,
    ),
    "groove_midi": DatasetConfig(
        name="Groove MIDI Dataset",
        task="groove",
        description="Expressive drum performances from Magenta",
        sources=[
            {
                "type": "url",
                "url": "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip",
            }
        ],
        output_dir="grooves/groove_midi",
    ),
    "maestro": DatasetConfig(
        name="MAESTRO",
        task="melody",
        description="MIDI and Audio Edited for Synchronous TRacks and Organization",
        sources=[
            {
                "type": "url",
                "url": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
            }
        ],
        output_dir="melodies/maestro",
    ),
    "lakh_midi": DatasetConfig(
        name="Lakh MIDI Dataset (Clean)",
        task="harmony",
        description="Clean subset of Lakh MIDI Dataset",
        sources=[
            {
                "type": "url",
                "url": "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz",
            }
        ],
        output_dir="chord_progressions/lakh",
    ),
    "musicnet": DatasetConfig(
        name="MusicNet",
        task="melody",
        description="Classical music with note annotations (168GB)",
        sources=[
            {
                "type": "url",
                "url": "https://zenodo.org/record/5120004/files/musicnet.tar.gz",
            }
        ],
        output_dir="melodies/musicnet",
    ),
    "gtzan": DatasetConfig(
        name="GTZAN",
        task="emotion",
        description="Music genre classification dataset (10 genres)",
        sources=[
            {
                "type": "url",
                "url": "https://mirg.city.ac.uk/datasets/gtzan/genres.tar.gz",
            }
        ],
        output_dir="emotions/gtzan",
        sample_rate=22050,
        max_duration=30.0,
    ),
    "fma_small": DatasetConfig(
        name="Free Music Archive (Small)",
        task="all",
        description="8,000 tracks of 30s, 8 genres (7.2GB)",
        sources=[
            {
                "type": "url",
                "url": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
            }
        ],
        output_dir="raw/fma_small",
    ),
    "fma_medium": DatasetConfig(
        name="Free Music Archive (Medium)",
        task="all",
        description="25,000 tracks of 30s, 16 genres (22GB)",
        sources=[
            {
                "type": "url",
                "url": "https://os.unil.cloud.switch.ch/fma/fma_medium.zip",
            }
        ],
        output_dir="raw/fma_medium",
    ),
    "fma_full": DatasetConfig(
        name="Free Music Archive (Full)",
        task="all",
        description="Massive collection of 106,574 tracks (~900GB)",
        sources=[
            {
                "type": "url",
                "url": "https://os.unil.cloud.switch.ch/fma/fma_full.zip",
            }
        ],
        output_dir="raw/fma_full",
    ),
    "mtg_jamendo": DatasetConfig(
        name="MTG-Jamendo",
        task="all",
        description="Multi-label dataset for music classification (~1TB)",
        sources=[
            {
                "type": "url",
                "url": "https://mtg.github.io/mtg-jamendo-dataset/data/autotagging_moodtheme.tsv",
            },
            {
                "type": "url",
                "url": "https://mtg.github.io/mtg-jamendo-dataset/data/autotagging_genre.tsv",
            }
        ],
        output_dir="raw/mtg_jamendo",
    ),
    "nsynth_full": DatasetConfig(
        name="NSynth (Full)",
        task="instrument",
        description="Large-scale dataset of musical notes (~30GB)",
        sources=[
            {
                "type": "url",
                "url": "https://storage.googleapis.com/magentadata/datasets/nsynth/nsynth-train.jsonwav.tar.gz",
            }
        ],
        output_dir="raw/nsynth",
    ),
    "musdb18": DatasetConfig(
        name="MUSDB18",
        task="source_separation",
        description="Dataset for music source separation (~10GB compressed)",
        sources=[
            {
                "type": "url",
                "url": "https://zenodo.org/record/1117372/files/musdb18.zip",
            }
        ],
        output_dir="raw/musdb18",
    ),
    "local_music": DatasetConfig(
        name="Local Music Library",
        task="all",
        description="Files copied from the local ~/Music directory",
        sources=[
            {
                "type": "local",
                "path": "~/Music",
            }
        ],
        output_dir="raw/local_music",
    ),
}


# =============================================================================
# Download Functions
# =============================================================================

def download_from_url(url: str, output_dir: Path) -> Optional[Path]:
    """Download file from URL."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        logger.error("requests and tqdm required: pip install requests tqdm")
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get filename from URL
    filename = url.split("/")[-1]
    output_path = output_dir / filename
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    
    if output_path.exists():
        logger.info(f"Already downloaded: {output_path}")
        return output_path
    
    logger.info(f"Downloading: {url}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    total_size = 0
    try:
        response = requests.get(url, stream=True, headers=headers, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        
        with open(partial_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            f.flush()
            os.fsync(f.fileno())
        
        if partial_path.exists():
            shutil.move(str(partial_path), str(output_path))
        elif output_path.exists() and output_path.stat().st_size >= total_size:
            # Partial was already moved or renamed (e.g. external drive sync)
            pass
        else:
            raise FileNotFoundError(f"Partial file missing after write: {partial_path}")
        logger.info(f"Downloaded: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        if partial_path.exists():
            partial_path.unlink()
        if total_size and output_path.exists() and output_path.stat().st_size >= total_size:
            logger.info(f"Using existing file: {output_path}")
            return output_path
        return None


def download_from_kaggle(dataset: str, output_dir: Path) -> bool:
    """Download dataset from Kaggle."""
    try:
        import kaggle
    except ImportError:
        logger.error("kaggle package required: pip install kaggle")
        logger.info("Also set up ~/.kaggle/kaggle.json with your API key")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading from Kaggle: {dataset}")
    
    try:
        kaggle.api.dataset_download_files(
            dataset,
            path=str(output_dir),
            unzip=True,
        )
        logger.info(f"Downloaded to: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        return False


def download_from_huggingface(dataset_name: str, output_dir: Path, split: str = "train") -> bool:
    """Download dataset from Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets package required: pip install datasets")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading from Hugging Face: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
        
        # Save to disk
        dataset.save_to_disk(str(output_dir))
        logger.info(f"Saved to: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Hugging Face download failed: {e}")
        return False


def _is_safe_relative_path(path: str) -> bool:
    norm_path = Path(path).as_posix()
    return (
        not norm_path.startswith("/")
        and not norm_path.startswith("\\")
        and ".." not in norm_path.split("/")
    )


def _is_safe_extracted_path(output_dir: Path, path: str) -> bool:
    base_path = output_dir.resolve()
    target = (output_dir / path).resolve()
    return str(target).startswith(f"{base_path}{os.sep}") or target == base_path


def extract_archive(archive_path: Path, output_dir: Path) -> bool:
    """Extract zip/tar archive with traversal-safe validation."""
    import tarfile
    import zipfile

    logger.info(f"Extracting: {archive_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                members = sorted(zf.infolist(), key=lambda item: item.filename)
                for member in members:
                    if not _is_safe_relative_path(member.filename):
                        logger.error(f"Unsafe zip entry blocked: {member.filename}")
                        return False
                    if not _is_safe_extracted_path(output_dir, member.filename):
                        logger.error(f"Unsafe zip path blocked: {member.filename}")
                        return False
                    zf.extract(member, output_dir)
        elif archive_path.suffix in [".tar", ".gz", ".tgz"] or ".tar" in archive_path.name:
            mode = "r:gz" if ".gz" in archive_path.name else "r"
            with tarfile.open(archive_path, mode) as tf:
                members = sorted(tf.getmembers(), key=lambda item: item.name)
                for member in members:
                    if not _is_safe_relative_path(member.name):
                        logger.error(f"Unsafe tar entry blocked: {member.name}")
                        return False
                    if not _is_safe_extracted_path(output_dir, member.name):
                        logger.error(f"Unsafe tar path blocked: {member.name}")
                        return False
                    tf.extract(member, output_dir)
        else:
            logger.error(f"Unknown archive format: {archive_path}")
            return False
        
        logger.info(f"Extracted to: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def download_dataset(config: DatasetConfig) -> bool:
    """Download a dataset based on its configuration."""
    downloads_dir = AUDIO_DATA_ROOT / "downloads"
    raw_dir = AUDIO_DATA_ROOT / "raw" / config.output_dir
    
    success = True
    
    for source in config.sources:
        source_type = source.get("type")
        
        if source_type == "url":
            archive_path = download_from_url(source["url"], downloads_dir)
            if archive_path:
                extract_archive(archive_path, raw_dir)
            else:
                success = False
                
        elif source_type == "kaggle":
            if not download_from_kaggle(source["dataset"], raw_dir):
                success = False
                
        elif source_type == "huggingface":
            if not download_from_huggingface(source["dataset"], raw_dir):
                success = False
                
        elif source_type == "local":
            source_path = Path(source["path"]).expanduser()
            if source_path.exists():
                logger.info(f"Copying local files from: {source_path}")
                # Recursively copy audio files
                extensions = [".wav", ".mp3", ".flac", ".ogg", ".mid", ".midi"]
                for ext in extensions:
                    for src_file in source_path.rglob(f"*{ext}"):
                        rel_path = src_file.relative_to(source_path)
                        dst_file = raw_dir / rel_path
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        if not dst_file.exists():
                            shutil.copy2(src_file, dst_file)
            else:
                logger.error(f"Local path not found: {source_path}")
                success = False
    
    return success


# =============================================================================
# Preprocessing Functions
# =============================================================================

def preprocess_audio_file(
    input_path: Path,
    output_path: Path,
    target_sr: int = 16000,
    max_duration: float = 10.0,
    normalize: bool = True,
) -> bool:
    """Preprocess a single audio file."""
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        logger.error("librosa and soundfile required")
        return False
    
    try:
        # Load audio
        y, sr = librosa.load(str(input_path), sr=target_sr, duration=max_duration)
        
        # Normalize
        if normalize:
            y = y / (np.max(np.abs(y)) + 1e-8)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), y, target_sr)
        
        return True
    except Exception as e:
        logger.debug(f"Failed to process {input_path}: {e}")
        return False


def extract_mel_spectrogram(
    audio_path: Path,
    output_path: Path,
    sr: int = 16000,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> bool:
    """Extract and save mel spectrogram."""
    try:
        import librosa
    except ImportError:
        return False
    
    try:
        y, _ = librosa.load(str(audio_path), sr=sr)
        
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output_path), mel_spec_db)
        
        return True
    except Exception as e:
        logger.debug(f"Failed to extract mel: {e}")
        return False


def parse_ravdess_filename(filename: str) -> Dict[str, Any]:
    """Parse RAVDESS filename to extract metadata."""
    # Format: 03-01-06-01-02-01-12.wav
    # Modality-Vocal-Emotion-Intensity-Statement-Repetition-Actor
    
    parts = filename.replace(".wav", "").split("-")
    if len(parts) != 7:
        return {}
    
    emotion_map = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fear",
        "07": "disgust",
        "08": "surprise",
    }
    
    return {
        "modality": parts[0],
        "vocal_channel": parts[1],
        "emotion": emotion_map.get(parts[2], "unknown"),
        "intensity": "normal" if parts[3] == "01" else "strong",
        "statement": parts[4],
        "repetition": parts[5],
        "actor": parts[6],
    }


def parse_cremad_filename(filename: str) -> str:
    """Parse CREMA-D filename to extract emotion."""
    # Format: 1001_DFA_ANG_XX.wav
    parts = filename.split("_")
    if len(parts) < 3:
        return "unknown"
    
    code = parts[2]
    emotion_map = {
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fear",
        "HAP": "happy",
        "NEU": "neutral",
        "SAD": "sad",
    }
    return emotion_map.get(code, "unknown")


def file_content_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 content hash for deduplication."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def dedupe_metadata(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate samples by source path or output file path."""
    deduped: List[Dict[str, Any]] = []
    seen_keys = set()
    for sample in samples:
        key = sample.get("source_content_hash") or sample.get("source_path") or sample.get("file")
        if not key:
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(sample)
    return deduped


def save_metadata_files(output_dir: Path, samples: List[Dict[str, Any]]) -> None:
    """Persist metadata in JSON + CSV with dynamic columns."""
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, indent=2)

    csv_path = output_dir / "metadata.csv"
    fieldnames = sorted({k for s in samples for k in s.keys()}) if samples else ["file"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)


def create_split_manifests(
    output_dir: Path,
    samples: List[Dict[str, Any]],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, int]:
    """Create train/val/test JSONL manifests with optional stratification."""
    if not samples:
        return {"train": 0, "val": 0, "test": 0}

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    rng = random.Random(seed)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        label = sample.get("emotion") or sample.get("instrument_family") or sample.get("label") or "__all__"
        grouped.setdefault(str(label), []).append(sample)

    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []
    test_samples: List[Dict[str, Any]] = []

    for group in grouped.values():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        if n > 0 and n_train == 0:
            n_train = 1
            if n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1

        train_samples.extend(group[:n_train])
        val_samples.extend(group[n_train : n_train + n_val])
        test_samples.extend(group[n_train + n_val : n_train + n_val + n_test])

    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }
    for split_name, split_samples in split_map.items():
        path = splits_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for sample in split_samples:
                f.write(json.dumps(sample, ensure_ascii=True) + "\n")

    summary = {name: len(data) for name, data in split_map.items()}
    with open(splits_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def preprocess_emotion_dataset(
    input_dir: Path,
    output_dir: Path,
    config: DatasetConfig,
) -> Tuple[int, int]:
    """Preprocess emotion dataset and create metadata."""
    from tqdm import tqdm
    
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    success_count = 0
    fail_count = 0
    duplicate_count = 0
    seen_content_hashes: set[str] = set()

    # Find all audio files
    audio_files = list(input_dir.rglob("*.wav")) + list(input_dir.rglob("*.mp3"))

    logger.info(f"Found {len(audio_files)} audio files")

    for audio_path in tqdm(audio_files, desc="Processing"):
        # Parse filename for metadata
        if "ravdess" in str(input_dir).lower():
            meta = parse_ravdess_filename(audio_path.name)
            emotion = meta.get("emotion", "unknown")
        elif "cremad" in str(input_dir).lower():
            emotion = parse_cremad_filename(audio_path.name)
        else:
            # Try to infer emotion from directory structure
            emotion = audio_path.parent.name.lower()

        if emotion == "unknown":
            fail_count += 1
            continue

        try:
            source_hash = file_content_hash(audio_path)
        except Exception as e:
            logger.debug(f"Failed to hash {audio_path}: {e}")
            fail_count += 1
            continue
        if source_hash in seen_content_hashes:
            duplicate_count += 1
            continue

        # Create output path
        output_filename = f"{audio_path.stem}.wav"
        emotion_dir = processed_dir / emotion
        output_path = emotion_dir / output_filename

        # Process audio
        if preprocess_audio_file(
            audio_path,
            output_path,
            target_sr=config.sample_rate,
            max_duration=config.max_duration,
        ):
            metadata.append({
                "file": str(output_path.relative_to(output_dir)),
                "emotion": emotion,
                "original_file": audio_path.name,
                "source_path": str(audio_path),
                "source_content_hash": source_hash,
            })
            seen_content_hashes.add(source_hash)
            success_count += 1
        else:
            fail_count += 1

    metadata = dedupe_metadata(metadata)
    save_metadata_files(output_dir, metadata)

    logger.info(f"Saved metadata: {len(metadata)} samples")
    if duplicate_count > 0:
        logger.info(f"Skipped duplicate files by content hash: {duplicate_count}")

    return success_count, fail_count


def preprocess_midi_dataset(
    input_dir: Path,
    output_dir: Path,
    config: DatasetConfig,
) -> Tuple[int, int]:
    """Preprocess MIDI dataset for melody/harmony training."""
    try:
        import mido
    except ImportError:
        logger.error("mido required for MIDI processing: pip install mido")
        return 0, 0
    
    from tqdm import tqdm
    
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    success_count = 0
    fail_count = 0
    duplicate_count = 0
    seen_content_hashes: set[str] = set()

    # Find all MIDI files
    midi_files = list(input_dir.rglob("*.mid")) + list(input_dir.rglob("*.midi"))

    logger.info(f"Found {len(midi_files)} MIDI files")

    for midi_path in tqdm(midi_files, desc="Processing MIDI"):
        try:
            source_hash = file_content_hash(midi_path)
        except Exception as e:
            logger.debug(f"Failed to hash {midi_path}: {e}")
            fail_count += 1
            continue
        if source_hash in seen_content_hashes:
            duplicate_count += 1
            continue

        try:
            mid = mido.MidiFile(str(midi_path))

            # Extract note sequences
            notes = []
            current_time = 0

            for track in mid.tracks:
                for msg in track:
                    current_time += msg.time
                    if msg.type == "note_on" and msg.velocity > 0:
                        notes.append({
                            "time": current_time,
                            "pitch": msg.note,
                            "velocity": msg.velocity,
                            "channel": msg.channel,
                        })

            if len(notes) < 10:
                fail_count += 1
                continue

            # Save processed notes
            output_path = processed_dir / f"{midi_path.stem}.json"
            with open(output_path, "w") as f:
                json.dump({
                    "notes": notes,
                    "ticks_per_beat": mid.ticks_per_beat,
                    "length": mid.length,
                }, f)

            metadata.append({
                "file": str(output_path.relative_to(output_dir)),
                "original_file": midi_path.name,
                "num_notes": len(notes),
                "duration": mid.length,
                "source_path": str(midi_path),
                "source_content_hash": source_hash,
            })
            seen_content_hashes.add(source_hash)
            success_count += 1

        except Exception as e:
            logger.debug(f"Failed to process {midi_path}: {e}")
            fail_count += 1

    metadata = dedupe_metadata(metadata)
    save_metadata_files(output_dir, metadata)
    if duplicate_count > 0:
        logger.info(f"Skipped duplicate MIDI files by content hash: {duplicate_count}")

    return success_count, fail_count


def preprocess_instrument_dataset(
    input_dir: Path,
    output_dir: Path,
    config: DatasetConfig,
) -> Tuple[int, int]:
    """Preprocess instrument dataset (e.g., NSynth)."""
    from tqdm import tqdm
    
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # NSynth specific metadata file
    nsynth_meta = input_dir / "examples.json"
    if nsynth_meta.exists():
        with open(nsynth_meta) as f:
            examples = json.load(f)
            
        metadata = []
        success_count = 0
        fail_count = 0
        
        for key, info in tqdm(examples.items(), desc="Processing NSynth"):
            audio_path = input_dir / "audio" / f"{key}.wav"
            if not audio_path.exists():
                fail_count += 1
                continue
                
            # Create output path
            instrument_family = info.get("instrument_family_str", "unknown")
            output_path = processed_dir / instrument_family / f"{key}.wav"
            
            if preprocess_audio_file(
                audio_path,
                output_path,
                target_sr=config.sample_rate,
                max_duration=config.max_duration,
            ):
                metadata.append({
                    "file": str(output_path.relative_to(output_dir)),
                    "instrument_family": instrument_family,
                    "pitch": info.get("pitch"),
                    "velocity": info.get("velocity"),
                    "source": info.get("instrument_source_str"),
                })
                success_count += 1
            else:
                fail_count += 1
                
        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({"samples": metadata}, f, indent=2)
            
        return success_count, fail_count
        
    return 0, 0


def preprocess_generic_dataset(
    input_dir: Path,
    output_dir: Path,
    config: DatasetConfig,
) -> Tuple[int, int]:
    """Generic preprocessor for any audio files."""
    from tqdm import tqdm
    
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    success_count = 0
    fail_count = 0
    duplicate_count = 0
    seen_content_hashes: set[str] = set()

    # Find all audio files
    extensions = [".wav", ".mp3", ".flac", ".ogg"]
    audio_files = []
    for ext in extensions:
        audio_files.extend(list(input_dir.rglob(f"*{ext}")))

    logger.info(f"Found {len(audio_files)} audio files for generic processing")

    for audio_path in tqdm(audio_files, desc="Processing"):
        try:
            source_hash = file_content_hash(audio_path)
        except Exception as e:
            logger.debug(f"Failed to hash {audio_path}: {e}")
            fail_count += 1
            continue
        if source_hash in seen_content_hashes:
            duplicate_count += 1
            continue

        # Create output path preserving some structure or just flattening
        # For local music, we preserve relative structure
        try:
            rel_path = audio_path.relative_to(input_dir)
            output_path = processed_dir / rel_path.with_suffix(".wav")
        except ValueError:
            output_path = processed_dir / f"{audio_path.stem}.wav"

        if preprocess_audio_file(
            audio_path,
            output_path,
            target_sr=config.sample_rate,
            max_duration=config.max_duration,
        ):
            metadata.append({
                "file": str(output_path.relative_to(output_dir)),
                "original_file": str(audio_path),
                "label": "generic",
                "source_content_hash": source_hash,
            })
            seen_content_hashes.add(source_hash)
            success_count += 1
        else:
            fail_count += 1

    metadata = dedupe_metadata(metadata)
    save_metadata_files(output_dir, metadata)
    if duplicate_count > 0:
        logger.info(f"Skipped duplicate generic files by content hash: {duplicate_count}")

    return success_count, fail_count


def preprocess_dataset(
    dataset_name: str,
    seed: int = 42,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> bool:
    """Preprocess a downloaded dataset."""
    if dataset_name not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_name}")
        return False

    config = DATASETS[dataset_name]

    input_dir = AUDIO_DATA_ROOT / "raw" / config.output_dir
    output_dir = AUDIO_DATA_ROOT / "processed" / config.output_dir

    if not input_dir.exists():
        logger.error(f"Dataset not downloaded: {input_dir}")
        return False

    logger.info(f"Preprocessing: {config.name}")

    if config.task == "emotion":
        success, fail = preprocess_emotion_dataset(input_dir, output_dir, config)
    elif config.task == "instrument":
        success, fail = preprocess_instrument_dataset(input_dir, output_dir, config)
    elif config.task in ["melody", "harmony", "groove"]:
        success, fail = preprocess_midi_dataset(input_dir, output_dir, config)
    elif config.task in ["all", "source_separation"]:
        success, fail = preprocess_generic_dataset(input_dir, output_dir, config)
    else:
        logger.error(f"Unknown task type: {config.task}")
        return False

    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f).get("samples", [])
        split_counts = create_split_manifests(
            output_dir=output_dir,
            samples=metadata,
            seed=seed,
            train_ratio=split_ratios[0],
            val_ratio=split_ratios[1],
            test_ratio=split_ratios[2],
        )
        logger.info(
            f"Generated splits: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}"
        )

    logger.info(f"Preprocessing complete: {success} success, {fail} failed")
    return success > 0


# =============================================================================
# Dataset Statistics
# =============================================================================

def compute_dataset_stats(dataset_name: str) -> Dict[str, Any]:
    """Compute statistics for a processed dataset."""
    if dataset_name not in DATASETS:
        return {}
    
    config = DATASETS[dataset_name]
    processed_dir = AUDIO_DATA_ROOT / "processed" / config.output_dir
    
    if not processed_dir.exists():
        return {"error": "Dataset not processed"}
    
    metadata_path = processed_dir / "metadata.json"
    if not metadata_path.exists():
        return {"error": "Metadata not found"}
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    samples = metadata.get("samples", [])
    
    stats = {
        "name": config.name,
        "task": config.task,
        "total_samples": len(samples),
    }
    
    if config.task == "emotion":
        # Count per emotion
        emotion_counts = {}
        for sample in samples:
            emotion = sample.get("emotion", "unknown")
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        stats["emotion_distribution"] = emotion_counts
    
    return stats


# =============================================================================
# Main
# =============================================================================

def list_datasets():
    """List available datasets."""
    print("\n" + "=" * 70)
    print("Available Datasets")
    print("=" * 70)
    
    for name, config in DATASETS.items():
        raw_dir = AUDIO_DATA_ROOT / "raw" / config.output_dir
        processed_dir = AUDIO_DATA_ROOT / "processed" / config.output_dir
        
        status = "❌ Not downloaded"
        if processed_dir.exists():
            status = "✅ Processed"
        elif raw_dir.exists():
            status = "📦 Downloaded (not processed)"
        
        print(f"\n  {name}")
        print(f"    Name: {config.name}")
        print(f"    Task: {config.task}")
        print(f"    Status: {status}")
        print(f"    Description: {config.description}")
    
    print("\n" + "=" * 70)


def main():
    global AUDIO_DATA_ROOT
    parser = argparse.ArgumentParser(
        description="Prepare datasets for Kelly ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/prepare_datasets.py --list
    python scripts/prepare_datasets.py --dataset emotion_ravdess --download
    python scripts/prepare_datasets.py --dataset emotion_ravdess --preprocess
    python scripts/prepare_datasets.py --dataset all --download --preprocess
        """,
    )
    
    parser.add_argument("--root", type=str, help=f"Override data root (default: {AUDIO_DATA_ROOT})")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--dataset", type=str, help="Dataset name (or 'all')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split generation")
    parser.add_argument(
        "--split-ratios",
        type=str,
        default="0.8,0.1,0.1",
        help="Train/val/test ratios as comma-separated floats",
    )
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess dataset")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    parser.add_argument("--sanitize", action="store_true", help="Sanitize dataset (check for silence/corruption)")
    parser.add_argument("--golden", action="store_true", help="Mark this dataset as the 'Golden' validation set")
    parser.add_argument("--pack", action="store_true", help="Pack dataset into LMDB for high-speed I/O")
    parser.add_argument(
        "--source-manifest", type=Path, default=None,
        help="Path to source_manifest.yaml (used with --list-from-manifest)",
    )
    parser.add_argument(
        "--list-from-manifest", action="store_true",
        help="List adopted dataset sources from config/source_manifest.yaml and exit",
    )
    
    args = parser.parse_args()

    if args.list_from_manifest:
        repo_root = ROOT.parent  # ROOT is scripts/, repo root is scripts/.parent
        manifest = args.source_manifest or (repo_root / "config" / "source_manifest.yaml")
        if not manifest.is_file():
            logger.error("Manifest not found: %s", manifest)
            sys.exit(1)
        if yaml is None:
            logger.error("yaml required for --list-from-manifest; pip install pyyaml")
            sys.exit(1)
        with manifest.open() as f:
            data = yaml.safe_load(f) or {}
        sources = data.get("sources") or []
        adopted = [
            s for s in sources
            if s.get("adoption_decision") == "adopted"
            and ("dataset" in (s.get("artifact_classes") or []) or "midi" in (s.get("integration_domain") or "").lower() or "emotion" in (s.get("integration_domain") or "").lower())
        ]
        if not adopted:
            print("No adopted dataset sources in manifest.")
            print("Set adoption_decision: adopted for dataset-like items in config/source_manifest.yaml.")
        else:
            for s in adopted:
                path = s.get("proposed_storage_path") or "(none)"
                env = s.get("storage_env_var") or "(none)"
                print(f"  {s.get('source_item', '?')}: proposed_storage_path={path} storage_env_var={env}")
        return
    
    # Allow overriding root after parsing
    if args.root:
        AUDIO_DATA_ROOT = Path(args.root).expanduser()

    try:
        split_ratios = tuple(float(x.strip()) for x in args.split_ratios.split(","))
    except ValueError:
        logger.error("Invalid --split-ratios format. Use e.g. 0.8,0.1,0.1")
        sys.exit(1)
    if len(split_ratios) != 3 or abs(sum(split_ratios) - 1.0) > 1e-6:
        logger.error("--split-ratios must contain 3 values summing to 1.0")
        sys.exit(1)

    # Check SSD is mounted
    if not AUDIO_DATA_ROOT.parent.exists():
        logger.error(f"External SSD not mounted: {AUDIO_DATA_ROOT.parent}")
        logger.info("Please connect the external SSD and try again")
        sys.exit(1)
    
    # Ensure directories exist
    AUDIO_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (AUDIO_DATA_ROOT / "raw").mkdir(exist_ok=True)
    (AUDIO_DATA_ROOT / "processed").mkdir(exist_ok=True)
    (AUDIO_DATA_ROOT / "downloads").mkdir(exist_ok=True)
    
    if args.list:
        list_datasets()
        return
    
    if not args.dataset:
        parser.print_help()
        return
    
    # Get datasets to process
    if args.dataset == "all":
        datasets = list(DATASETS.keys())
    else:
        if args.dataset not in DATASETS:
            logger.error(f"Unknown dataset: {args.dataset}")
            list_datasets()
            sys.exit(1)
        datasets = [args.dataset]
    
    # Process each dataset
    for dataset_name in datasets:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Processing: {dataset_name}")
        logger.info("=" * 50)
        
        if args.download:
            download_dataset(DATASETS[dataset_name])
        
        if args.preprocess:
            preprocess_dataset(
                dataset_name,
                seed=args.seed,
                split_ratios=split_ratios,
            )
        
        if args.sanitize:
            logger.info(f"Sanitizing: {dataset_name}")
            config = DATASETS[dataset_name]
            raw_dir = AUDIO_DATA_ROOT / "raw" / config.output_dir
            if raw_dir.exists():
                from scripts.sanitize_datasets import sanitize_directory
                sanitize_directory(raw_dir, quarantine_path=raw_dir.parent / "quarantine")
            else:
                logger.error(f"Raw directory not found for {dataset_name}")

        if args.golden:
            logger.info(f"Setting {dataset_name} as Golden Validation Set...")
            config = DATASETS[dataset_name]
            processed_dir = AUDIO_DATA_ROOT / "processed" / config.output_dir
            metadata_path = processed_dir / "metadata.json"
            if metadata_path.exists():
                golden_path = AUDIO_DATA_ROOT / "golden_manifest.json"
                shutil.copy(metadata_path, golden_path)
                logger.info(f"Golden manifest saved to {golden_path}")
            else:
                logger.error(f"Metadata not found for {dataset_name}. Process it first.")

        if args.pack:
            logger.info(f"Packing: {dataset_name} into LMDB...")
            config = DATASETS[dataset_name]
            raw_dir = AUDIO_DATA_ROOT / "raw" / config.output_dir
            # Generate a manifest if it doesn't exist
            manifest_path = raw_dir / "manifest.jsonl"
            if not manifest_path.exists():
                logger.info("Creating temporary manifest for packing...")
                raw_dir.mkdir(parents=True, exist_ok=True)
                with open(manifest_path, "w") as f:
                    for audio_file in raw_dir.rglob("*.wav"):
                        f.write(json.dumps({"audio": str(audio_file)}) + "\n")
            
            from scripts.pack_dataset import pack_dataset
            db_path = AUDIO_DATA_ROOT / "packed" / dataset_name
            db_path.mkdir(parents=True, exist_ok=True)
            pack_dataset(str(manifest_path), str(db_path))
        
        if args.stats:
            stats = compute_dataset_stats(dataset_name)
            print(f"\nStatistics for {dataset_name}:")
            print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
