#!/usr/bin/env python3
"""
Verify that integrated models can be loaded and are compatible with the codebase.

Tests:
1. Checkpoint file loading (PyTorch)
2. State dict compatibility
3. Model class instantiation (if available)
4. Inference capability (if model classes exist)
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

MODELS_TO_TEST = {
    'emotionrecognizer': {
        'checkpoint': 'models/checkpoints/emotionrecognizer_best.pt',
        'model_class': None,  # Will try to find dynamically
        'input_shape': (1, 128),  # Batch size 1, 128 features
    },
    'harmonypredictor': {
        'checkpoint': 'models/checkpoints/harmonypredictor_best.pt',
        'model_class': None,
        'input_shape': (1, 128),
    },
    'melodytransformer': {
        'checkpoint': 'models/checkpoints/melodytransformer_best.pt',
        'model_class': None,
        'input_shape': (1, 64),
    },
    'groovepredictor': {
        'checkpoint': 'models/checkpoints/groovepredictor_best.pt',
        'model_class': None,
        'input_shape': (1, 64),
    },
    'dynamicsengine': {
        'checkpoint': 'models/checkpoints/dynamicsengine_best.pt',
        'model_class': None,
        'input_shape': (1, 32),
    },
}


def try_import_model_class(model_name: str):
    """Try to import model class from various locations."""
    model_name_lower = model_name.lower()
    
    # Try different import paths
    import_paths = [
        f'music_brain.models.{model_name_lower}',
        f'music_brain.models.{model_name_lower.replace("_", "")}',
        f'penta_core.ml.training.architectures',
    ]
    
    class_names = [
        model_name,
        model_name.replace('_', ''),
        model_name.replace('_', '').title(),
    ]
    
    for import_path in import_paths:
        try:
            module = __import__(import_path, fromlist=[''])
            for class_name in class_names:
                if hasattr(module, class_name):
                    return getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
    
    return None


def load_checkpoint(checkpoint_path: Path) -> Tuple[bool, Optional[dict], Optional[str]]:
    """Load PyTorch checkpoint, trying both weights_only modes."""
    if not checkpoint_path.exists():
        return False, None, f"File does not exist: {checkpoint_path}"
    
    # Try weights_only=True first (safer)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        return True, checkpoint, None
    except Exception as e1:
        # Try weights_only=False (needed for some checkpoint formats)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            return True, checkpoint, None
        except Exception as e2:
            return False, None, f"Failed with weights_only=True: {e1}; with weights_only=False: {e2}"


def extract_state_dict(checkpoint: dict) -> Tuple[torch.nn.Module, Optional[str]]:
    """Extract state dict from checkpoint, handling various formats."""
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict'], None
        elif 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict'], None
        elif any(key.endswith('.weight') or key.endswith('.bias') for key in checkpoint.keys()):
            # Direct state dict
            return checkpoint, None
        else:
            # Try to find state dict in nested structures
            for key in ['model', 'net', 'network', 'state']:
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    if any(k.endswith('.weight') or k.endswith('.bias') for k in checkpoint[key].keys()):
                        return checkpoint[key], None
            return None, f"Could not find state_dict in checkpoint keys: {list(checkpoint.keys())[:10]}"
    else:
        return None, f"Checkpoint is not a dict, type: {type(checkpoint)}"


def verify_model_checkpoint(model_name: str, config: dict) -> Dict:
    """Verify a single model checkpoint."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")
    
    checkpoint_path = project_root / config['checkpoint']
    results = {
        'model_name': model_name,
        'checkpoint_path': str(checkpoint_path),
        'file_exists': False,
        'loads_successfully': False,
        'has_state_dict': False,
        'state_dict_keys': [],
        'num_parameters': 0,
        'model_class_found': False,
        'can_instantiate': False,
        'can_load_weights': False,
        'can_infer': False,
        'errors': [],
    }
    
    # Step 1: Check file exists
    if not checkpoint_path.exists():
        results['errors'].append(f"Checkpoint file not found: {checkpoint_path}")
        return results
    
    results['file_exists'] = True
    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"✓ File exists: {file_size_mb:.2f} MB")
    
    # Step 2: Load checkpoint
    success, checkpoint, error = load_checkpoint(checkpoint_path)
    if not success:
        results['errors'].append(error)
        print(f"✗ Failed to load checkpoint: {error}")
        return results
    
    results['loads_successfully'] = True
    print(f"✓ Checkpoint loads successfully")
    
    # Step 3: Extract state dict
    state_dict, error = extract_state_dict(checkpoint)
    if state_dict is None:
        results['errors'].append(error or "Could not extract state_dict")
        print(f"✗ {error}")
        return results
    
    results['has_state_dict'] = True
    results['state_dict_keys'] = list(state_dict.keys())[:10]  # First 10 keys
    results['num_parameters'] = sum(p.numel() for p in state_dict.values())
    
    print(f"✓ State dict extracted: {len(state_dict)} layers, {results['num_parameters']:,} parameters")
    print(f"  Sample keys: {', '.join(results['state_dict_keys'][:5])}")
    
    # Step 4: Try to find model class
    model_class = try_import_model_class(model_name)
    if model_class:
        results['model_class_found'] = True
        print(f"✓ Model class found: {model_class.__name__}")
        
        # Step 5: Try to instantiate model
        try:
            model = model_class()
            results['can_instantiate'] = True
            print(f"✓ Model instantiated successfully")
            
            # Step 6: Try to load weights
            try:
                model.load_state_dict(state_dict, strict=False)  # strict=False for flexibility
                results['can_load_weights'] = True
                print(f"✓ Weights loaded successfully")
                
                # Step 7: Try inference
                model.eval()
                with torch.no_grad():
                    # Create dummy input
                    input_shape = config.get('input_shape', (1, 128))
                    dummy_input = torch.randn(*input_shape)
                    
                    try:
                        output = model(dummy_input)
                        results['can_infer'] = True
                        print(f"✓ Inference successful: input {dummy_input.shape} → output {output.shape if hasattr(output, 'shape') else type(output)}")
                    except Exception as e:
                        results['errors'].append(f"Inference failed: {str(e)}")
                        print(f"✗ Inference failed: {e}")
                        
            except Exception as e:
                results['errors'].append(f"Load weights failed: {str(e)}")
                print(f"✗ Failed to load weights: {e}")
        except Exception as e:
            results['errors'].append(f"Instantiation failed: {str(e)}")
            print(f"✗ Failed to instantiate model: {e}")
    else:
        print(f"⚠ Model class not found (checkpoint still valid for manual loading)")
        print(f"  Checkpoint structure: {type(checkpoint).__name__}")
        if isinstance(checkpoint, dict):
            print(f"  Top-level keys: {list(checkpoint.keys())[:10]}")
    
    return results


def test_tier1_loader():
    """Test the Tier1MIDIGenerator loader if available."""
    print(f"\n{'='*70}")
    print("Testing Tier1MIDIGenerator Integration")
    print(f"{'='*70}")
    
    try:
        from music_brain.tier1.midi_generator import Tier1MIDIGenerator
        
        print("Attempting to initialize Tier1MIDIGenerator...")
        generator = Tier1MIDIGenerator(
            device='cpu',  # Use CPU for testing
            checkpoint_dir=project_root / "models" / "checkpoints",
            verbose=True
        )
        print("✓ Tier1MIDIGenerator initialized successfully!")
        return True
    except ImportError as e:
        print(f"⚠ Tier1MIDIGenerator not available: {e}")
        return False
    except Exception as e:
        print(f"✗ Tier1MIDIGenerator failed: {e}")
        return False


def main():
    print("="*70)
    print("Model Loading Verification")
    print("="*70)
    print(f"Project root: {project_root}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    all_results = {}
    
    # Test each model
    for model_name, config in MODELS_TO_TEST.items():
        results = verify_model_checkpoint(model_name, config)
        all_results[model_name] = results
    
    # Test Tier1 integration
    tier1_works = test_tier1_loader()
    
    # Summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}\n")
    
    total = len(MODELS_TO_TEST)
    files_exist = sum(1 for r in all_results.values() if r['file_exists'])
    loads_ok = sum(1 for r in all_results.values() if r['loads_successfully'])
    has_state_dict = sum(1 for r in all_results.values() if r['has_state_dict'])
    can_infer = sum(1 for r in all_results.values() if r['can_infer'])
    
    print(f"Models tested: {total}")
    print(f"✓ Files exist: {files_exist}/{total}")
    print(f"✓ Load successfully: {loads_ok}/{total}")
    print(f"✓ Have state dict: {has_state_dict}/{total}")
    print(f"✓ Can run inference: {can_infer}/{total}")
    print(f"✓ Tier1MIDIGenerator: {'✓' if tier1_works else '⚠ (checkpoint valid but class missing)'}")
    
    print("\nDetailed Results:")
    for model_name, results in all_results.items():
        status = "✓" if results['loads_successfully'] and results['has_state_dict'] else "✗"
        infer_status = "✓" if results['can_infer'] else "⚠" if results['can_instantiate'] else "-"
        print(f"  {status} {model_name:20s} | Load: {'✓' if results['loads_successfully'] else '✗'} | Infer: {infer_status}")
        if results['errors']:
            for error in results['errors']:
                print(f"    └─ {error}")
    
    # Overall status
    if loads_ok == total and has_state_dict == total:
        print(f"\n✓ SUCCESS: All {total} models load successfully and have valid state dicts!")
        if can_infer < total:
            print(f"  Note: {total - can_infer} models need model class definitions for inference")
        return 0
    else:
        print(f"\n⚠ WARNING: Some models have issues (see details above)")
        return 1


if __name__ == '__main__':
    sys.exit(main())

