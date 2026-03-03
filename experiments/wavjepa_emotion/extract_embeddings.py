"""
Extract frozen encoder embeddings for all (path, label) samples; save to cache npz.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# Experiment dir on path so we can import dataset, encoders
_EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

from dataset import load_emotion_tuples
from encoders import load_encoder, extract


def _is_git_lfs_pointer(path: Path) -> bool:
    """Detect Git LFS pointer files (stub text, not real audio)."""
    try:
        with open(path, "rb") as f:
            head = f.read(50)
        return head.startswith(b"version https://git-lfs")
    except Exception:
        return False


def load_audio_16k(path: Path, sample_rate: int = 16000) -> np.ndarray:
    path_str = str(path.resolve())
    if _is_git_lfs_pointer(path):
        raise FileNotFoundError(
            f"File is a Git LFS pointer (stub), not real audio: {path}. "
            "In the dataset repo run: git lfs pull"
        )
    # Try soundfile first (handles most WAVs)
    try:
        import soundfile as sf
        data, sr = sf.read(path_str, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != sample_rate:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
        return data.astype(np.float32)
    except Exception:
        pass
    # Fallback: scipy.io.wavfile (no codec deps)
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(path_str)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != sample_rate:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
        return data.astype(np.float32)
    except Exception:
        pass
    # Last: librosa (may use audioread)
    import librosa
    data, sr = librosa.load(path_str, sr=sample_rate, mono=True)
    return data.astype(np.float32)


def run(
    config_path: Path,
    encoder_name: Optional[str] = None,
    limit: Optional[int] = None,
) -> Path:
    config_path = config_path.resolve()
    with open(config_path) as f:
        config = yaml.safe_load(f)
    encoder_name = encoder_name or config.get("encoder", "wavjepa")
    sample_rate = config.get("sample_rate", 16000)
    window_seconds = config.get("window_seconds", 2.0)
    cache_dir = Path(config.get("cache_dir", "cache"))
    if not cache_dir.is_absolute():
        cache_dir = (config_path.parent / cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    tuples = load_emotion_tuples(config, config_path)
    if not tuples:
        raise FileNotFoundError(
            "No (path, label) samples found. Set KMIDI_DATASETS_PATH or AUDIO_DATA_ROOT and ensure "
            "emotions/ravdess and emotions/cremad exist under the root."
        )
    if limit:
        tuples = tuples[:limit]

    encoder = load_encoder(encoder_name, config)
    embeddings_list = []
    labels_list = []
    paths_list = []
    skip_count = 0

    for i, (path, label) in enumerate(tuples):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  {i + 1}/{len(tuples)} {path.name} -> {label}")
        try:
            audio = load_audio_16k(path, sample_rate)
        except Exception as e:
            skip_count += 1
            if skip_count <= 3:
                print(f"  Skip (load) {path.name}: {type(e).__name__}: {e}")
            continue
        try:
            # Optional: 2s windows and mean over windows; for simplicity we use full-clip mean here
            if window_seconds > 0 and len(audio) > sample_rate * window_seconds:
                hop = int(sample_rate * window_seconds)
                vecs = []
                for start in range(0, len(audio) - hop + 1, hop):
                    chunk = audio[start : start + hop]
                    vec = extract(encoder, chunk, sample_rate=sample_rate, pool="mean")
                    vecs.append(vec)
                emb = np.mean(vecs, axis=0)
            else:
                emb = extract(encoder, audio, sample_rate=sample_rate, pool="mean")
        except Exception as e:
            skip_count += 1
            if skip_count <= 3:
                print(f"  Skip (extract) {path.name}: {type(e).__name__}: {e}")
            continue
        embeddings_list.append(emb)
        labels_list.append(label)
        paths_list.append(str(path))

    if not embeddings_list:
        raise RuntimeError(
            "No embeddings extracted (all files skipped). Check audio load and encoder output; "
            "print first few exceptions above."
        )
    embeddings = np.stack(embeddings_list)
    labels = np.array(labels_list, dtype=object)
    out_path = cache_dir / f"embeddings_{encoder_name}.npz"
    np.savez(out_path, embeddings=embeddings, labels=labels, paths=np.array(paths_list, dtype=object))
    print(f"Saved {embeddings.shape[0]} embeddings to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=_EXPERIMENT_DIR / "config.yaml")
    parser.add_argument("--encoder", type=str, default=None, choices=["wavjepa", "hubert_base", "wav2vec2_base"])
    parser.add_argument("--limit", type=int, default=None, help="Max samples (for quick test)")
    args = parser.parse_args()
    run(args.config, args.encoder, args.limit)
