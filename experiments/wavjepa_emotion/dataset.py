"""
Load CREMA-D and RAVDESS for emotion experiment.
Root = KMIDI_DATASETS_PATH | AUDIO_DATA_ROOT | ~/Datasets.
Subpaths: emotions/ravdess, emotions/cremad (match prepare_datasets.py).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Filename parsing aligned with scripts/utilities/prepare_datasets.py
RAVDESS_EMOTION = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fear", "07": "disgust", "08": "surprise",
}
CREMAD_EMOTION = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy",
    "NEU": "neutral", "SAD": "sad",
}


def _parse_ravdess(filename: str) -> str:
    # Format: 03-01-06-01-02-01-12.wav -> emotion code in parts[2]
    base = filename.replace(".wav", "")
    parts = base.split("-")
    if len(parts) != 7:
        return "unknown"
    return RAVDESS_EMOTION.get(parts[2], "unknown")


def _parse_cremad(filename: str) -> str:
    # Format: 1001_DFA_ANG_XX.wav -> emotion code in parts[2]
    parts = filename.split("_")
    if len(parts) < 3:
        return "unknown"
    return CREMAD_EMOTION.get(parts[2], "unknown")


def resolve_data_root(config: Dict[str, Any], config_path: Any = None) -> Path:
    """Root = KMIDI_DATASETS_PATH | AUDIO_DATA_ROOT | config dataset_root | ~/Datasets.
    If dataset_root is relative, it is resolved against the directory containing the config file.
    """
    root = (
        os.environ.get("KMIDI_DATASETS_PATH")
        or os.environ.get("AUDIO_DATA_ROOT")
        or None
    )
    if root:
        return Path(root).expanduser().resolve()
    cfg_root = config.get("dataset_root")
    if cfg_root:
        p = Path(cfg_root).expanduser()
        if not p.is_absolute() and config_path is not None:
            base = Path(config_path).resolve().parent
            p = (base / p).resolve()
        else:
            p = p.resolve()
        return p
    return Path.home() / "Datasets"


def load_emotion_tuples(
    config: Dict[str, Any],
    config_path: Any = None,
) -> List[Tuple[Path, str]]:
    """
    Return [(audio_path, emotion_label), ...] for all configured datasets.
    Skips files that do not parse to a known emotion.
    """
    root = resolve_data_root(config, config_path)
    subpaths = config.get("dataset_subpaths") or {
        "crema_d": "emotions/cremad",
        "ravdess": "emotions/ravdess",
    }
    dataset_names = config.get("datasets") or ["crema_d", "ravdess"]
    out: List[Tuple[Path, str]] = []
    for name in dataset_names:
        sub = subpaths.get(name)
        if not sub:
            continue
        # Try root/sub (e.g. emotions/ravdess) and root/raw/sub (prepare_datasets download layout)
        for dir_path in [root / sub, root / "raw" / sub]:
            if not dir_path.is_dir():
                continue
            for wav_path in dir_path.rglob("*.wav"):
                if "ravdess" in name.lower():
                    emotion = _parse_ravdess(wav_path.name)
                elif "crema" in name.lower():
                    emotion = _parse_cremad(wav_path.name)
                else:
                    emotion = "unknown"
                if emotion != "unknown":
                    out.append((wav_path.resolve(), emotion))
            break  # found this dataset
    return out


def get_label_set(tuples: List[Tuple[Path, str]]) -> List[str]:
    """Sorted unique emotion labels."""
    return sorted(set(t[1] for t in tuples))
