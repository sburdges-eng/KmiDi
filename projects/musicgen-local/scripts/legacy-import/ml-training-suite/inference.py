#!/usr/bin/env python3
"""Inference script with LoRA support."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
import json

from src.models import AudioClassifier
from src.models.lora import apply_lora_to_model, load_lora_weights, merge_lora_weights
from src.utils.audio import load_audio, extract_mel_spectrogram, normalize_audio, pad_or_trim


def load_model_with_lora(
    checkpoint_path: str,
    lora_path: str = None,
    num_classes: int = 10,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    device: str = "cpu",
    merge: bool = False,
) -> torch.nn.Module:
    """
    Load model with optional LoRA weights.

    Args:
        checkpoint_path: Path to base model checkpoint
        lora_path: Path to LoRA weights (optional)
        num_classes: Number of output classes
        lora_rank: LoRA rank (must match training)
        lora_alpha: LoRA alpha (must match training)
        device: Device to load model on
        merge: If True, merge LoRA weights into base model

    Returns:
        Loaded model
    """
    # Create base model
    model = AudioClassifier(
        num_classes=num_classes,
        n_mels=128,
        channels=[32, 64, 128, 256],
        dropout=0.0,  # No dropout at inference
    )

    # Load base checkpoint
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            # Filter out LoRA keys if present
            state_dict = {
                k: v for k, v in checkpoint["model_state_dict"].items()
                if "lora_" not in k
            }
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded base model from: {checkpoint_path}")

    # Apply LoRA if path provided
    if lora_path and Path(lora_path).exists():
        model = apply_lora_to_model(
            model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=0.0,
            target_modules=["fc"],
        )
        model = load_lora_weights(model, lora_path, device)
        print(f"Loaded LoRA weights from: {lora_path}")

        if merge:
            model = merge_lora_weights(model)
            print("LoRA weights merged into base model")

    model = model.to(device)
    model.eval()

    return model


def preprocess_audio(
    audio_path: str,
    sample_rate: int = 22050,
    duration: float = 5.0,
    n_mels: int = 128,
) -> torch.Tensor:
    """
    Preprocess an audio file for inference.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        duration: Duration to extract
        n_mels: Number of mel bands

    Returns:
        Preprocessed mel spectrogram tensor [1, 1, n_mels, time]
    """
    # Load audio
    waveform, sr = load_audio(
        audio_path,
        target_sr=sample_rate,
        mono=True,
        duration=duration,
    )

    # Normalize
    waveform = normalize_audio(waveform, method="peak")

    # Pad/trim to exact length
    target_samples = int(sample_rate * duration)
    waveform = pad_or_trim(waveform, target_samples)

    # Extract mel spectrogram
    mel_spec = extract_mel_spectrogram(
        waveform,
        sample_rate=sample_rate,
        n_mels=n_mels,
    )

    # Add batch dimension
    return mel_spec.unsqueeze(0)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    audio_path: str,
    device: str = "cpu",
    top_k: int = 5,
    class_names: List[str] = None,
) -> Dict:
    """
    Run inference on an audio file.

    Args:
        model: Trained model
        audio_path: Path to audio file
        device: Device to run on
        top_k: Number of top predictions to return
        class_names: Optional list of class names

    Returns:
        Dictionary with predictions
    """
    # Preprocess
    mel_spec = preprocess_audio(audio_path).to(device)

    # Forward pass
    logits = model(mel_spec)
    probs = F.softmax(logits, dim=1)

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probs, min(top_k, probs.size(1)))
    top_probs = top_probs[0].cpu().numpy()
    top_indices = top_indices[0].cpu().numpy()

    # Build results
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        pred = {
            "class_id": int(idx),
            "probability": float(prob),
        }
        if class_names and idx < len(class_names):
            pred["class_name"] = class_names[idx]
        predictions.append(pred)

    return {
        "file": str(audio_path),
        "predictions": predictions,
        "top_class": int(top_indices[0]),
        "confidence": float(top_probs[0]),
    }


@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    audio_paths: List[str],
    device: str = "cpu",
    class_names: List[str] = None,
) -> List[Dict]:
    """Run inference on multiple audio files."""
    results = []
    for path in audio_paths:
        try:
            result = predict(model, path, device, class_names=class_names)
            results.append(result)
        except Exception as e:
            results.append({
                "file": str(path),
                "error": str(e),
            })
    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference with LoRA model")
    parser.add_argument("audio", nargs="+", help="Audio file(s) to process")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--lora", type=str, default=None,
                       help="Path to LoRA weights")
    parser.add_argument("--lora-rank", type=int, default=8,
                       help="LoRA rank (must match training)")
    parser.add_argument("--num-classes", type=int, default=10,
                       help="Number of classes")
    parser.add_argument("--class-names", type=str, default=None,
                       help="Path to JSON file with class names")
    parser.add_argument("--merge-lora", action="store_true",
                       help="Merge LoRA weights for faster inference")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of top predictions to show")
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load class names if provided
    class_names = None
    if args.class_names and Path(args.class_names).exists():
        with open(args.class_names) as f:
            class_names = json.load(f)

    # Load model
    model = load_model_with_lora(
        checkpoint_path=args.checkpoint,
        lora_path=args.lora,
        num_classes=args.num_classes,
        lora_rank=args.lora_rank,
        device=device,
        merge=args.merge_lora,
    )

    # Run inference
    print(f"\nProcessing {len(args.audio)} file(s)...")
    results = []

    for audio_path in args.audio:
        if not Path(audio_path).exists():
            print(f"  [SKIP] {audio_path} - file not found")
            continue

        result = predict(
            model,
            audio_path,
            device=device,
            top_k=args.top_k,
            class_names=class_names,
        )
        results.append(result)

        # Print result
        print(f"\n{Path(audio_path).name}:")
        for i, pred in enumerate(result["predictions"][:3]):
            class_str = pred.get("class_name", f"Class {pred['class_id']}")
            print(f"  {i+1}. {class_str}: {pred['probability']*100:.1f}%")

    # Save results if output specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
