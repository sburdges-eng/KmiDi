import json
from pathlib import Path

import numpy as np
import pytest

def test_audio_emotion_dataset_loads_and_maps_labels():
    librosa = pytest.importorskip("librosa")
    from training.train_integrated import AudioEmotionDataset

    manifest = Path("tests/fixtures/emotion_manifest.jsonl")
    ds = AudioEmotionDataset(str(manifest), sample_rate=16000, n_mels=32, use_augmentation=False)
    assert len(ds) == 1
    x, y = ds[0]
    assert x.shape[0] == 1  # channel
    assert x.shape[1] == 32  # n_mels
    assert y.item() == ds.label_to_idx["happy"]


def test_augmentation_pipeline_runs():
    pytest.importorskip("librosa")
    from training.data_augmentation import AugmentationConfig, build_augmentation_pipeline

    cfg = AugmentationConfig()
    augment = build_augmentation_pipeline(cfg, seed=42)
    sr = 16000
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
    audio = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    out = augment(audio, sr)
    assert out.size > 0
