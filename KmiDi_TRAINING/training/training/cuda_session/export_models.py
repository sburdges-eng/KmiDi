#!/usr/bin/env python3
"""
Export trained models to ONNX and CoreML formats.

Usage:
    python export_models.py --spectocloud checkpoints/spectocloud/best.pt
    python export_models.py --midi-generator checkpoints/midi_generator/best.pt
    python export_models.py --all  # Export both
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn


def export_spectocloud_to_onnx(checkpoint_path: str, output_path: str):
    """Export Spectocloud model to ONNX."""
    from train_spectocloud import SpectocloudViT

    print(f"Loading Spectocloud from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})

    model_cfg = config.get('model', {})
    model = SpectocloudViT(
        n_mels=config.get('data', {}).get('n_mels', 128),
        emotion_dim=64,
        midi_dim=32,
        hidden_dim=model_cfg.get('encoder', {}).get('hidden_dim', 256),
        num_points=model_cfg.get('decoder', {}).get('output_points', 1200),
        num_layers=model_cfg.get('encoder', {}).get('num_layers', 6),
        num_heads=model_cfg.get('encoder', {}).get('num_heads', 8),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    class SpectocloudONNXWrapper(nn.Module):
        def __init__(self, wrapped: nn.Module):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, spectrogram, emotion, midi_features):
            outputs = self.wrapped(spectrogram, emotion, midi_features)
            return outputs["positions"], outputs["colors"], outputs["properties"]

    export_model = SpectocloudONNXWrapper(model)

    # Create dummy inputs
    batch_size = 1
    dummy_spectrogram = torch.randn(batch_size, 128, 64)
    dummy_emotion = torch.randn(batch_size, 64)
    dummy_midi = torch.randn(batch_size, 32)

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        export_model,
        (dummy_spectrogram, dummy_emotion, dummy_midi),
        output_path,
        input_names=['spectrogram', 'emotion', 'midi_features'],
        output_names=['positions', 'colors', 'properties'],
        dynamic_axes={
            'spectrogram': {0: 'batch'},
            'emotion': {0: 'batch'},
            'midi_features': {0: 'batch'},
            'positions': {0: 'batch'},
            'colors': {0: 'batch'},
            'properties': {0: 'batch'},
        },
        opset_version=17,
    )

    print(f"ONNX export complete: {output_path}")

    # Verify
    try:
        import onnx
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print("ONNX model verified ✓")
    except ImportError:
        print("(Install onnx to verify export)")


def export_midi_generator_to_onnx(checkpoint_path: str, output_path: str):
    """Export MIDI Generator to ONNX."""
    from train_midi_generator import EmotionMIDITransformer

    print(f"Loading MIDI Generator from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})

    arch = config.get('model', {}).get('architecture', {})
    model = EmotionMIDITransformer(
        vocab_size=arch.get('vocab_size', 388),
        max_seq_len=arch.get('max_seq_length', 1024),
        hidden_dim=arch.get('hidden_dim', 384),
        num_layers=arch.get('num_layers', 8),
        num_heads=arch.get('num_heads', 6),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy inputs
    batch_size = 1
    seq_len = 256
    dummy_input_ids = torch.randint(0, 388, (batch_size, seq_len))
    dummy_emotion = torch.randn(batch_size, 64)

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_emotion),
        output_path,
        input_names=['input_ids', 'emotion'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'emotion': {0: 'batch'},
            'logits': {0: 'batch', 1: 'sequence'},
        },
        opset_version=17,
    )

    print(f"ONNX export complete: {output_path}")


def export_to_coreml(onnx_path: str, output_path: str, model_type: str = 'spectocloud'):
    """Convert ONNX to CoreML."""
    try:
        import coremltools as ct
    except ImportError:
        print("CoreML tools not available. Install with: pip install coremltools")
        return

    print(f"Converting {onnx_path} to CoreML")

    # Load ONNX
    import onnx
    onnx_model = onnx.load(onnx_path)

    # Convert to CoreML
    if model_type == 'spectocloud':
        ml_model = ct.convert(
            onnx_model,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS14,
        )
    else:  # midi_generator
        ml_model = ct.convert(
            onnx_model,
            compute_units=ct.ComputeUnit.CPU_AND_GPU,  # ANE doesn't work well with transformers
            minimum_deployment_target=ct.target.macOS14,
        )

    # Save
    ml_model.save(output_path)
    print(f"CoreML export complete: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Export trained models')
    parser.add_argument('--spectocloud', type=str, help='Spectocloud checkpoint path')
    parser.add_argument('--midi-generator', type=str, help='MIDI Generator checkpoint path')
    parser.add_argument('--all', action='store_true', help='Export all models')
    parser.add_argument('--output-dir', type=str, default='exports', help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        args.spectocloud = 'checkpoints/spectocloud/best.pt'
        args.midi_generator = 'checkpoints/midi_generator/best.pt'

    if args.spectocloud and os.path.exists(args.spectocloud):
        onnx_path = output_dir / 'spectocloud.onnx'
        export_spectocloud_to_onnx(args.spectocloud, str(onnx_path))

        coreml_path = output_dir / 'spectocloud.mlpackage'
        export_to_coreml(str(onnx_path), str(coreml_path), 'spectocloud')

    if args.midi_generator and os.path.exists(args.midi_generator):
        onnx_path = output_dir / 'midi_generator.onnx'
        export_midi_generator_to_onnx(args.midi_generator, str(onnx_path))

        coreml_path = output_dir / 'midi_generator.mlpackage'
        export_to_coreml(str(onnx_path), str(coreml_path), 'midi_generator')

    print("\nExport complete!")
    print(f"Files saved to: {output_dir}")


if __name__ == '__main__':
    main()
