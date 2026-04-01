#!/usr/bin/env python3
"""
Proof-of-life demo: feed audio through the JEPA encoder, map to emotion,
print emotion parameters evolving over time.

Exercises the same pipeline as the AU plugin (AudioEmotionRunner) but
from Python — mel spectrogram → ONNX/CoreML inference → latent pooling
→ emotion mapping.

Usage:
    python scripts/demo_proof_of_life.py
    python scripts/demo_proof_of_life.py --audio path/to/audio.wav
    python scripts/demo_proof_of_life.py --backend coreml
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import platform
import time

import numpy as np
import torch

from music_brain.jepa.audio_jepa import AudioJEPAEncoder
from music_brain.jepa.config import AudioJEPAConfig


def load_emotion_probe(checkpoint_path: str = "checkpoints/emotion_probe/best_probe.pt"):
    """Load trained emotion probe if available."""
    if not Path(checkpoint_path).exists():
        return None
    from music_brain.jepa.emotion_probe import EmotionProbe
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    probe = EmotionProbe(
        latent_dim=ckpt.get("latent_dim", 256),
        hidden_dim=ckpt.get("hidden_dim", 128),
    )
    probe.load_state_dict(ckpt["probe"])
    probe.eval()
    return probe


def load_encoder(checkpoint_path: str) -> AudioJEPAEncoder:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = AudioJEPAConfig(**ckpt["config"])
    encoder = AudioJEPAEncoder(config=config)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    return encoder


def generate_test_audio(duration_s: float = 6.0, sr: int = 22050) -> np.ndarray:
    """Generate a test signal: sweep from calm (low freq sine) to energetic (noise burst)."""
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)

    # Phase 1 (0-2s): calm low sine
    calm = 0.3 * np.sin(2 * np.pi * 220 * t)
    # Phase 2 (2-4s): rising energy — higher freq + harmonics
    energy = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    # Phase 3 (4-6s): intense — distorted + noise
    intense = 0.7 * np.tanh(3 * np.sin(2 * np.pi * 330 * t)) + 0.2 * np.random.randn(len(t)).astype(np.float32)

    # Crossfade between phases
    audio = np.zeros_like(t)
    for i, sample_t in enumerate(t):
        if sample_t < 2.0:
            audio[i] = calm[i]
        elif sample_t < 4.0:
            blend = (sample_t - 2.0) / 2.0
            audio[i] = (1 - blend) * calm[i] + blend * energy[i]
        else:
            blend = (sample_t - 4.0) / 2.0
            audio[i] = (1 - blend) * energy[i] + blend * intense[i]

    return audio


def load_audio_file(path: str, sr: int = 22050) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def audio_to_mel(audio: np.ndarray, sr: int = 22050, n_mels: int = 128,
                 hop_length: int = 512, n_fft: int = 2048) -> np.ndarray:
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db


def latent_to_emotion(latent: np.ndarray, probe=None) -> dict:
    """Map latent to emotion using trained probe or heuristic fallback."""
    pooled = latent[0].mean(axis=0)  # (256,)

    if probe is not None:
        with torch.no_grad():
            inp = torch.from_numpy(pooled).unsqueeze(0).float()
            out = probe(inp).squeeze(0).numpy()
        valence = float(out[0])
        arousal = float((out[1] + 1.0) * 0.5)  # [-1,1] → [0,1]
    else:
        # Heuristic fallback (same as before)
        mean_val = float(pooled.mean())
        std_val = float(pooled.std())
        energy = float(np.abs(pooled).mean())
        skew = float(np.mean((pooled - mean_val) ** 3) / (std_val ** 3 + 1e-8))
        valence = float(np.tanh(skew * 2))
        arousal = float(np.clip(energy * 3, 0, 1))

    dominance = float(np.clip(0.5 + 0.3 * arousal + 0.2 * abs(valence), 0, 1))
    confidence = 0.8 if probe else 0.3

    return {
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "dominance": round(dominance, 3),
        "confidence": round(confidence, 3),
    }


def run_onnx_inference(mel_input: np.ndarray, model_path: str) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path)
    return sess.run(None, {"mel": mel_input})[0]


def run_coreml_inference(mel_input: np.ndarray, model_path: str) -> np.ndarray:
    import coremltools as ct
    model = ct.models.MLModel(model_path)
    pred = model.predict({"mel": mel_input})
    return list(pred.values())[0]


def emotion_bar(label: str, value: float, lo: float = -1.0, hi: float = 1.0,
                width: int = 30) -> str:
    """Render a simple ASCII bar for an emotion value."""
    norm = (value - lo) / (hi - lo)
    norm = max(0.0, min(1.0, norm))
    filled = int(norm * width)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"  {label:>10s}: [{bar}] {value:+.3f}"


def main():
    parser = argparse.ArgumentParser(description="Proof-of-life: audio → emotion → display")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (default: generate test signal)")
    parser.add_argument("--backend", choices=["onnx", "coreml", "pytorch"], default="onnx")
    parser.add_argument("--checkpoint", default="checkpoints/audio_jepa/best_model.pt")
    parser.add_argument("--window-sec", type=float, default=1.0, help="Analysis window in seconds")
    args = parser.parse_args()

    sr = 22050
    hop_length = 512
    max_frames = 512
    window_samples = int(args.window_sec * sr)

    print("=" * 60)
    print("  KmiDi Demo: Audio JEPA → Emotion Detection")
    print("=" * 60)
    probe = load_emotion_probe()
    if probe:
        print("Emotion probe: TRAINED (using learned mapping)")
    else:
        print("Emotion probe: not found (using heuristic fallback)")
    print()

    # Load audio
    if args.audio:
        print(f"Loading audio: {args.audio}")
        audio = load_audio_file(args.audio, sr=sr)
    else:
        print("Generating test signal (calm → energetic → intense)")
        audio = generate_test_audio(duration_s=6.0, sr=sr)

    duration = len(audio) / sr
    print(f"Duration: {duration:.1f}s, Sample rate: {sr}Hz")
    print()

    # Set up inference
    if args.backend == "pytorch":
        print(f"Backend: PyTorch (checkpoint: {args.checkpoint})")
        encoder = load_encoder(args.checkpoint)
        infer = lambda mel: encoder(torch.from_numpy(mel)).detach().numpy()
    elif args.backend == "coreml":
        model_path = "models/audio_jepa_v01.mlpackage"
        print(f"Backend: Core ML (ANE-preferred) — {model_path}")
        infer = lambda mel: run_coreml_inference(mel, model_path)
    else:
        model_path = "models/audio_jepa_v01.onnx"
        print(f"Backend: ONNX Runtime — {model_path}")
        infer = lambda mel: run_onnx_inference(mel, model_path)

    print()
    print("Processing audio windows...")
    print("-" * 60)

    # Slide through audio in windows
    step_samples = window_samples // 2  # 50% overlap
    window_idx = 0

    for start in range(0, len(audio) - window_samples, step_samples):
        end = start + window_samples
        chunk = audio[start:end]
        time_s = start / sr

        # Compute mel spectrogram
        mel = audio_to_mel(chunk, sr=sr, n_mels=128, hop_length=hop_length)

        # Pad/truncate to max_frames
        if mel.shape[1] < max_frames:
            mel = np.pad(mel, ((0, 0), (0, max_frames - mel.shape[1])))
        else:
            mel = mel[:, :max_frames]

        # Shape: (1, 1, 128, 512)
        mel_input = mel[np.newaxis, np.newaxis, :, :].astype(np.float32)

        # Run inference
        t0 = time.perf_counter_ns()
        latent = infer(mel_input)
        latency_ms = (time.perf_counter_ns() - t0) / 1e6

        # Map to emotion
        emotion = latent_to_emotion(latent, probe=probe)

        # Display
        print(f"\n  [{time_s:5.1f}s - {(end/sr):5.1f}s]  (inference: {latency_ms:.1f}ms)")
        print(emotion_bar("Valence", emotion["valence"], -1.0, 1.0))
        print(emotion_bar("Arousal", emotion["arousal"], 0.0, 1.0))
        print(emotion_bar("Dominance", emotion["dominance"], 0.0, 1.0))
        print(f"  {'Confidence':>10s}: {emotion['confidence']:.3f}")

        window_idx += 1

    print()
    print("-" * 60)
    print(f"Processed {window_idx} windows across {duration:.1f}s of audio")
    print()
    print("Pipeline: Audio → Mel Spectrogram → JEPA Encoder → Latent Pooling → Emotion")
    print("This is the same pipeline the AU plugin runs in processBlock().")
    print()


if __name__ == "__main__":
    main()
