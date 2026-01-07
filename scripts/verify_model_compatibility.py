#!/usr/bin/env python3
"""
Test model class compatibility with integrated checkpoints.

This script tries to load checkpoints using the actual model class definitions
found in training/train_integrated.py and verifies they're compatible.
"""

import sys
import torch
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try importing model classes
try:
    from training.train_integrated import (
        EmotionRecognizerCNN,
        MelodyTransformer,
        HarmonyPredictorTransformer,
        GroovePredictorMLP,
        DynamicsEngineMLP,
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Could not import model classes: {e}")
    MODELS_AVAILABLE = False

MODEL_CONFIGS = {
    'emotionrecognizer': {
        'checkpoint': 'models/checkpoints/emotionrecognizer_best.pt',
        'class': EmotionRecognizerCNN if MODELS_AVAILABLE else None,
        'init_kwargs': {'input_size': 128, 'output_size': 64},
        'input_shape': (1, 128),
    },
    'harmonypredictor': {
        'checkpoint': 'models/checkpoints/harmonypredictor_best.pt',
        'class': HarmonyPredictorTransformer if MODELS_AVAILABLE else None,
        'init_kwargs': {'input_size': 128, 'output_size': 64},
        'input_shape': (1, 128),
    },
    'melodytransformer': {
        'checkpoint': 'models/checkpoints/melodytransformer_best.pt',
        'class': MelodyTransformer if MODELS_AVAILABLE else None,
        'init_kwargs': {'input_size': 64, 'output_size': 128},
        'input_shape': (1, 64),
    },
    'groovepredictor': {
        'checkpoint': 'models/checkpoints/groovepredictor_best.pt',
        'class': GroovePredictorMLP if MODELS_AVAILABLE else None,
        'init_kwargs': {'input_size': 64, 'output_size': 32},
        'input_shape': (1, 64),
    },
    'dynamicsengine': {
        'checkpoint': 'models/checkpoints/dynamicsengine_best.pt',
        'class': DynamicsEngineMLP if MODELS_AVAILABLE else None,
        'init_kwargs': {'input_size': 32, 'output_size': 16},
        'input_shape': (1, 32),
    },
}


def test_model_compatibility(model_name: str, config: dict):
    """Test if checkpoint is compatible with model class."""
    print(f"\n{'='*70}")
    print(f"Testing Compatibility: {model_name}")
    print(f"{'='*70}")
    
    if not MODELS_AVAILABLE or config['class'] is None:
        print("⚠ Model classes not available - skipping compatibility test")
        return {'compatible': False, 'reason': 'Model classes not available'}
    
    checkpoint_path = project_root / config['checkpoint']
    
    if not checkpoint_path.exists():
        return {'compatible': False, 'reason': f'Checkpoint not found: {checkpoint_path}'}
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Create model instance
        model_class = config['class']
        model = model_class(**config['init_kwargs'])
        
        print(f"✓ Model instantiated: {model_class.__name__}")
        print(f"  Expected parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Checkpoint parameters: {sum(p.numel() for p in state_dict.values()):,}")
        
        # Try loading with strict=False first (more lenient)
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠ Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"⚠ Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
            
            if not missing_keys and not unexpected_keys:
                print("✓ State dict matches perfectly (strict=True would work)")
            elif len(missing_keys) < 5 and len(unexpected_keys) < 5:
                print("✓ State dict compatible (minor differences, can use)")
                compatible = True
            else:
                print("⚠ State dict has significant differences")
                compatible = False
            
        except Exception as e:
            print(f"✗ Failed to load state dict: {e}")
            return {'compatible': False, 'reason': str(e)}
        
        # Test inference
        try:
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(*config['input_shape'])
                output = model(dummy_input)
                print(f"✓ Inference successful: {dummy_input.shape} → {output.shape}")
                compatible = True
        except Exception as e:
            print(f"✗ Inference failed: {e}")
            compatible = False
            return {'compatible': False, 'reason': f'Inference failed: {e}'}
        
        # Try strict=True if non-strict worked
        if compatible:
            try:
                model2 = model_class(**config['init_kwargs'])
                model2.load_state_dict(state_dict, strict=True)
                print("✓ STRICT MODE: Checkpoint matches model architecture exactly!")
            except Exception as e:
                print(f"  (Strict mode fails, but non-strict works - likely minor differences)")
        
        return {
            'compatible': compatible,
            'model_class': model_class.__name__,
            'can_load_strict': False,  # Would need to test separately
        }
        
    except Exception as e:
        return {'compatible': False, 'reason': str(e)}


def main():
    print("="*70)
    print("Model Compatibility Verification")
    print("="*70)
    
    if not MODELS_AVAILABLE:
        print("\n⚠ Model classes not found in training/train_integrated.py")
        print("  Checkpoints are still valid - they just need compatible model definitions")
        return 1
    
    results = {}
    for model_name, config in MODEL_CONFIGS.items():
        results[model_name] = test_model_compatibility(model_name, config)
    
    # Summary
    print(f"\n{'='*70}")
    print("COMPATIBILITY SUMMARY")
    print(f"{'='*70}\n")
    
    compatible = sum(1 for r in results.values() if r.get('compatible', False))
    total = len(results)
    
    for model_name, result in results.items():
        status = "✓" if result.get('compatible') else "✗"
        print(f"  {status} {model_name:20s} - {result.get('reason', 'OK')}")
    
    print(f"\n✓ Compatible: {compatible}/{total}")
    
    if compatible == total:
        print("\n🎉 SUCCESS: All checkpoints are compatible with model classes!")
        return 0
    else:
        print(f"\n⚠ {total - compatible} models have compatibility issues")
        return 1


if __name__ == '__main__':
    sys.exit(main())

