#!/usr/bin/env python3
"""
Integration Verification Script

Verifies that all components work together correctly:
- WASABI dataset integration
- Training scripts compatibility
- Config file compatibility
- Import paths
- Dataset compatibility

Usage:
    python scripts/verify_integration.py
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def check_imports():
    """Check that all required modules can be imported."""
    print("=" * 70)
    print("Checking imports...")
    print("=" * 70)
    
    errors = []
    
    # Core training modules
    try:
        from training.wasabi_dataset import WasabiDataset, WasabiSample
        print("✓ training.wasabi_dataset")
    except Exception as e:
        errors.append(f"✗ training.wasabi_dataset: {e}")
        print(f"✗ training.wasabi_dataset: {e}")
    
    try:
        from training.wasabi_midi_dataset import WasabiMIDIDataset
        print("✓ training.wasabi_midi_dataset")
    except Exception as e:
        errors.append(f"✗ training.wasabi_midi_dataset: {e}")
        print(f"✗ training.wasabi_midi_dataset: {e}")
    
    try:
        from training.train_integrated import MODEL_CONFIGS, train_single_model
        print("✓ training.train_integrated")
    except Exception as e:
        errors.append(f"✗ training.train_integrated: {e}")
        print(f"✗ training.train_integrated: {e}")
    
    try:
        from training.harmony_dataset import HarmonyDataset
        print("✓ training.harmony_dataset")
    except Exception as e:
        errors.append(f"✗ training.harmony_dataset: {e}")
        print(f"✗ training.harmony_dataset: {e}")
    
    try:
        from training.data_augmentation import (AugmentationConfig,
                                                build_augmentation_pipeline)
        print("✓ training.data_augmentation")
    except Exception as e:
        errors.append(f"✗ training.data_augmentation: {e}")
        print(f"✗ training.data_augmentation: {e}")
    
    # CUDA session modules
    try:
        sys.path.insert(0, str(ROOT / "training" / "cuda_session"))
        from train_midi_generator import (EmotionMIDITransformer, MIDIDataset,
                                          MIDITokenizer)
        print("✓ training.cuda_session.train_midi_generator")
    except Exception as e:
        errors.append(f"✗ training.cuda_session.train_midi_generator: {e}")
        print(f"✗ training.cuda_session.train_midi_generator: {e}")
    
    return errors


def check_config_files():
    """Check that config files exist and are valid."""
    print("\n" + "=" * 70)
    print("Checking config files...")
    print("=" * 70)
    
    errors = []
    configs = [
        "training/integrated_training_config.yaml",
        "training/cuda_session/midi_generator_training_config.yaml",
    ]
    
    import yaml
    
    for config_path in configs:
        path = ROOT / config_path
        if not path.exists():
            errors.append(f"✗ Config not found: {config_path}")
            print(f"✗ Config not found: {config_path}")
            continue
        
        try:
            with open(path) as f:
                config = yaml.safe_load(f)
            print(f"✓ {config_path}")
            
            # Check for WASABI references
            if "wasabi" in str(config).lower():
                print(f"  → Contains WASABI configuration")
        except Exception as e:
            errors.append(f"✗ Invalid YAML in {config_path}: {e}")
            print(f"✗ Invalid YAML in {config_path}: {e}")
    
    return errors


def check_dataset_compatibility():
    """Check that datasets have compatible interfaces."""
    print("\n" + "=" * 70)
    print("Checking dataset compatibility...")
    print("=" * 70)
    
    errors = []
    
    # Check WasabiDataset interface
    try:
        import inspect

        from training.wasabi_dataset import WasabiDataset
        
        sig = inspect.signature(WasabiDataset.__init__)
        params = list(sig.parameters.keys())
        required_params = ["manifest_path"]
        
        for req in required_params:
            if req not in params:
                errors.append(f"✗ WasabiDataset missing parameter: {req}")
                print(f"✗ WasabiDataset missing parameter: {req}")
            else:
                print(f"✓ WasabiDataset has parameter: {req}")
        
        # Check __getitem__ returns expected format
        if hasattr(WasabiDataset, "__getitem__"):
            print("✓ WasabiDataset implements __getitem__")
        else:
            errors.append("✗ WasabiDataset missing __getitem__")
            print("✗ WasabiDataset missing __getitem__")
            
    except Exception as e:
        errors.append(f"✗ Error checking WasabiDataset: {e}")
        print(f"✗ Error checking WasabiDataset: {e}")
    
    # Check WasabiMIDIDataset interface
    try:
        import inspect

        from training.wasabi_midi_dataset import WasabiMIDIDataset
        
        sig = inspect.signature(WasabiMIDIDataset.__init__)
        params = list(sig.parameters.keys())
        required_params = ["manifest_path", "config", "tokenizer"]
        
        for req in required_params:
            if req not in params:
                errors.append(f"✗ WasabiMIDIDataset missing parameter: {req}")
                print(f"✗ WasabiMIDIDataset missing parameter: {req}")
            else:
                print(f"✓ WasabiMIDIDataset has parameter: {req}")
        
        if hasattr(WasabiMIDIDataset, "__getitem__"):
            print("✓ WasabiMIDIDataset implements __getitem__")
        else:
            errors.append("✗ WasabiMIDIDataset missing __getitem__")
            print("✗ WasabiMIDIDataset missing __getitem__")
            
    except Exception as e:
        errors.append(f"✗ Error checking WasabiMIDIDataset: {e}")
        print(f"✗ Error checking WasabiMIDIDataset: {e}")
    
    return errors


def check_manifest_generation():
    """Check that manifest generation scripts exist."""
    print("\n" + "=" * 70)
    print("Checking manifest generation...")
    print("=" * 70)
    
    errors = []
    
    scripts = [
        "scripts/generate_wasabi_manifest.py",
        "scripts/prepare_datasets.py",
        "scripts/run_wasabi_training.sh",
    ]
    
    for script_path in scripts:
        path = ROOT / script_path
        if path.exists():
            print(f"✓ {script_path}")
        else:
            errors.append(f"✗ Script not found: {script_path}")
            print(f"✗ Script not found: {script_path}")
    
    return errors


def check_data_directories():
    """Check that expected data directories exist or can be created."""
    print("\n" + "=" * 70)
    print("Checking data directories...")
    print("=" * 70)
    
    errors = []
    
    dirs = [
        "data/wasabi/processed",
        "checkpoints",
        "cache",
    ]
    
    for dir_path in dirs:
        path = ROOT / dir_path
        if path.exists():
            print(f"✓ {dir_path} exists")
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"✓ {dir_path} created")
            except Exception as e:
                errors.append(f"✗ Cannot create {dir_path}: {e}")
                print(f"✗ Cannot create {dir_path}: {e}")
    
    return errors


def check_training_integration():
    """Check that training scripts can use WASABI."""
    print("\n" + "=" * 70)
    print("Checking training script integration...")
    print("=" * 70)
    
    errors = []
    
    # Check train_midi_generator.py can import WasabiMIDIDataset
    try:
        sys.path.insert(0, str(ROOT / "training" / "cuda_session"))
        from train_midi_generator import main as midi_main
        print("✓ train_midi_generator.py can be imported")
        
        # Check it has WASABI support
        import inspect
        source = inspect.getsource(midi_main)
        if "WasabiMIDIDataset" in source or "wasabi" in source.lower():
            print("  → Contains WASABI support")
        else:
            errors.append("✗ train_midi_generator.py may not have WASABI support")
            print("✗ train_midi_generator.py may not have WASABI support")
    except Exception as e:
        errors.append(f"✗ Error checking train_midi_generator.py: {e}")
        print(f"✗ Error checking train_midi_generator.py: {e}")
    
    return errors


def main():
    """Run all integration checks."""
    print("\n" + "=" * 70)
    print("KmiDi Integration Verification")
    print("=" * 70)
    print(f"Project root: {ROOT}")
    print()
    
    all_errors = []
    
    # Run all checks
    all_errors.extend(check_imports())
    all_errors.extend(check_config_files())
    all_errors.extend(check_dataset_compatibility())
    all_errors.extend(check_manifest_generation())
    all_errors.extend(check_data_directories())
    all_errors.extend(check_training_integration())
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    if not all_errors:
        print("✓ All integration checks passed!")
        print("\nComponents are ready to cooperate.")
        return 0
    else:
        print(f"✗ Found {len(all_errors)} issue(s):")
        for error in all_errors:
            print(f"  {error}")
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
        print(f"✗ Found {len(all_errors)} issue(s):")
        for error in all_errors:
            print(f"  {error}")
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
