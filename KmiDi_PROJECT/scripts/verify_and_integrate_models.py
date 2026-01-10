#!/usr/bin/env python3
"""
Verify and integrate trained ML models from RECOVERY_OPS directory.

This script:
1. Scans RECOVERY_OPS for trained model checkpoints
2. Validates model files can be loaded
3. Checks model architecture compatibility
4. Copies validated models to the project's models/checkpoints/ directory
5. Generates a detailed report

Usage:
    python scripts/verify_and_integrate_models.py [--dry-run] [--models-dir PATH]
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. Model validation will be limited.")


@dataclass
class ModelInfo:
    """Information about a discovered model checkpoint."""
    model_name: str
    file_path: Path
    file_size_mb: float
    is_valid: bool = False
    error_message: Optional[str] = None
    architecture: Optional[str] = None
    num_parameters: Optional[int] = None
    checkpoint_epoch: Optional[int] = None


@dataclass
class IntegrationReport:
    """Report of model integration results."""
    models_found: int = 0
    models_valid: int = 0
    models_copied: int = 0
    models_skipped: int = 0
    errors: List[str] = None
    details: List[Dict] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.details is None:
            self.details = []


class ModelValidator:
    """Validates PyTorch model checkpoints."""

    def __init__(self):
        self.supported_models = {
            'emotionrecognizer': ['EmotionRecognizer', 'EmotionClassifier'],
            'harmonypredictor': ['HarmonyPredictor', 'HarmonyModel'],
            'melodytransformer': ['MelodyTransformer', 'TransformerModel'],
            'groovepredictor': ['GroovePredictor', 'GrooveModel'],
            'dynamicsengine': ['DynamicsEngine', 'DynamicsModel'],
            'instrumentrecognizer': ['InstrumentRecognizer', 'InstrumentModel'],
            'emotionnodeclassifier': ['EmotionNodeClassifier', 'NodeClassifier'],
        }

    def validate_model(self, model_path: Path, model_name: str) -> Tuple[bool, Optional[str], Dict]:
        """
        Validate a model checkpoint file.

        Returns:
            (is_valid, error_message, model_info_dict)
        """
        if not TORCH_AVAILABLE:
            # Basic file validation only
            if model_path.exists() and model_path.stat().st_size > 0:
                return True, None, {'validated': False, 'reason': 'PyTorch not available'}
            return False, "File does not exist or is empty", {}

        if not model_path.exists():
            return False, "File does not exist", {}

        if model_path.stat().st_size == 0:
            return False, "File is empty", {}

        try:
            # Try loading the checkpoint
            # First attempt: with weights_only=True (safe, PyTorch 2.6+ default)
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            except Exception as e:
                # If that fails, try with weights_only=False (needed for TorchScript and older formats)
                # This is safe since we're loading from RECOVERY_OPS (trusted source)
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            info = {
                'validated': True,
                'checkpoint_keys': list(checkpoint.keys()) if isinstance(checkpoint, dict) else None,
            }

            # Check for common checkpoint structures
            if isinstance(checkpoint, dict):
                # Standard PyTorch checkpoint format
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    info['num_parameters'] = sum(p.numel() for p in state_dict.values())
                    info['architecture'] = checkpoint.get('arch', checkpoint.get('model_name', 'unknown'))
                    info['epoch'] = checkpoint.get('epoch')
                    info['checkpoint_type'] = 'state_dict'
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    info['num_parameters'] = sum(p.numel() for p in state_dict.values())
                    info['architecture'] = checkpoint.get('model_type', 'unknown')
                    info['epoch'] = checkpoint.get('epoch')
                    info['checkpoint_type'] = 'model_state_dict'
                elif any(key.endswith('.weight') or key.endswith('.bias') for key in checkpoint.keys()):
                    # Direct state dict
                    info['num_parameters'] = sum(p.numel() for p in checkpoint.values())
                    info['checkpoint_type'] = 'direct_state_dict'
                else:
                    # Unknown structure, but file loaded successfully
                    info['checkpoint_type'] = 'unknown'
                    info['keys'] = list(checkpoint.keys())[:10]  # First 10 keys

            return True, None, info

        except Exception as e:
            return False, f"Failed to load checkpoint: {str(e)}", {}


class ModelScanner:
    """Scans RECOVERY_OPS directory for model checkpoints."""

    def __init__(self, recovery_ops_path: Path):
        self.recovery_ops_path = Path(recovery_ops_path)
        self.validator = ModelValidator()

    def find_model_files(self) -> List[ModelInfo]:
        """Scan for model checkpoint files."""
        model_files = []
        
        # Common patterns for model files
        patterns = [
            '**/*_best.pt',
            '**/*_latest.pt',
            '**/best_model.pt',
            '**/latest_checkpoint.pt',
            '**/checkpoint_epoch_*.pt',
            '**/emotionrecognizer*.pt',
            '**/harmonypredictor*.pt',
            '**/melodytransformer*.pt',
            '**/groovepredictor*.pt',
            '**/dynamicsengine*.pt',
            '**/instrumentrecognizer*.pt',
            '**/emotionnodeclassifier*.pt',
        ]

        if not self.recovery_ops_path.exists():
            print(f"ERROR: RECOVERY_OPS path does not exist: {self.recovery_ops_path}")
            return model_files

        # Primary search location
        ml_models_dir = self.recovery_ops_path / "ML_TRAINED_MODELS"
        
        if ml_models_dir.exists():
            print(f"Scanning {ml_models_dir}...")
            search_paths = [ml_models_dir]
        else:
            print(f"ML_TRAINED_MODELS not found, scanning entire RECOVERY_OPS...")
            search_paths = [self.recovery_ops_path]

        seen_files = set()

        for search_path in search_paths:
            for pattern in patterns:
                for file_path in search_path.rglob(pattern):
                    # Skip if we've already seen this file (by resolved path)
                    resolved = file_path.resolve()
                    if resolved in seen_files:
                        continue
                    seen_files.add(resolved)

                    # Extract model name from filename
                    model_name = self._extract_model_name(file_path)
                    
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    
                    model_info = ModelInfo(
                        model_name=model_name,
                        file_path=file_path,
                        file_size_mb=file_size_mb
                    )
                    model_files.append(model_info)

        return sorted(model_files, key=lambda m: (m.model_name, m.file_path))

    def _extract_model_name(self, file_path: Path) -> str:
        """Extract model name from file path."""
        filename = file_path.name.lower()
        
        # Try to match known model names
        for model_name in self.validator.supported_models.keys():
            if model_name in filename:
                return model_name
        
        # Extract from parent directories or filename
        if 'emotion' in filename:
            return 'emotionrecognizer'
        elif 'harmony' in filename:
            return 'harmonypredictor'
        elif 'melody' in filename:
            return 'melodytransformer'
        elif 'groove' in filename:
            return 'groovepredictor'
        elif 'dynamics' in filename:
            return 'dynamicsengine'
        elif 'instrument' in filename:
            return 'instrumentrecognizer'
        elif 'node' in filename and 'emotion' in filename:
            return 'emotionnodeclassifier'
        
        # Default: use filename without extension
        return file_path.stem.lower().replace('_', '').replace('-', '')

    def validate_models(self, models: List[ModelInfo]) -> List[ModelInfo]:
        """Validate all discovered models."""
        validated_models = []
        
        for model in models:
            print(f"  Validating {model.file_path.name}...")
            is_valid, error, info = self.validator.validate_model(
                model.file_path, model.model_name
            )
            
            model.is_valid = is_valid
            model.error_message = error
            if info:
                model.architecture = info.get('architecture')
                model.num_parameters = info.get('num_parameters')
                model.checkpoint_epoch = info.get('epoch')
            
            validated_models.append(model)
            
            status = "✓" if is_valid else "✗"
            print(f"    {status} {model.model_name}: {error or 'Valid'}")

        return validated_models


class ModelIntegrator:
    """Integrates validated models into the project."""

    def __init__(self, project_root: Path, models_dir: Optional[Path] = None):
        self.project_root = Path(project_root)
        if models_dir:
            self.models_dir = Path(models_dir)
        else:
            self.models_dir = self.project_root / "models" / "checkpoints"
        
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def integrate_model(self, model: ModelInfo, dry_run: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Copy a validated model to the project models directory.

        Returns:
            (success, error_message)
        """
        # Determine target filename
        if model.file_path.name.endswith('_best.pt') or 'best' in model.file_path.name.lower():
            target_name = f"{model.model_name}_best.pt"
        elif 'latest' in model.file_path.name.lower():
            target_name = f"{model.model_name}_latest.pt"
        else:
            target_name = model.file_path.name

        target_path = self.models_dir / target_name

        # Check if target already exists
        if target_path.exists():
            existing_size = target_path.stat().st_size
            new_size = model.file_path.stat().st_size
            
            # If sizes match, skip (might be same file)
            if existing_size == new_size:
                return False, f"Target already exists with same size ({existing_size} bytes)"
            
            # If new is larger or different, we might want to overwrite
            # For now, create a versioned name
            base_name = target_path.stem
            target_name = f"{base_name}_recovered.pt"
            target_path = self.models_dir / target_name

        if dry_run:
            print(f"  [DRY RUN] Would copy: {model.file_path} -> {target_path}")
            return True, None

        try:
            shutil.copy2(model.file_path, target_path)
            print(f"  ✓ Copied: {target_name}")
            return True, None
        except Exception as e:
            return False, f"Copy failed: {str(e)}"

    def generate_model_registry(self, integrated_models: List[ModelInfo]) -> None:
        """Generate/update model registry JSON file."""
        registry_path = self.project_root / "models" / "registry.json"
        
        # Load existing registry if it exists
        registry = {}
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
            except Exception:
                registry = {}

        # Update with integrated models
        for model in integrated_models:
            if model.model_name not in registry:
                registry[model.model_name] = {
                    'status': 'integrated',
                    'source': 'RECOVERY_OPS',
                    'architecture': model.architecture,
                    'num_parameters': model.num_parameters,
                    'checkpoint_epoch': model.checkpoint_epoch,
                    'file_size_mb': round(model.file_size_mb, 2),
                    'validated': model.is_valid,
                }

        # Write updated registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

        print(f"\n✓ Updated model registry: {registry_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Verify and integrate ML models from RECOVERY_OPS'
    )
    parser.add_argument(
        '--recovery-ops',
        type=str,
        default='/Users/seanburdges/RECOVERY_OPS',
        help='Path to RECOVERY_OPS directory'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        help='Target directory for models (default: models/checkpoints/)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually copying files'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip PyTorch model validation (faster but less safe)'
    )
    parser.add_argument(
        '--output-report',
        type=str,
        help='Path to save JSON report file'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Model Integration from RECOVERY_OPS")
    print("=" * 70)
    print()

    # Initialize components
    recovery_ops_path = Path(args.recovery_ops)
    scanner = ModelScanner(recovery_ops_path)
    integrator = ModelIntegrator(project_root, args.models_dir)

    # Step 1: Scan for models
    print("Step 1: Scanning for model files...")
    models = scanner.find_model_files()
    print(f"  Found {len(models)} model file(s)\n")

    if not models:
        print("ERROR: No model files found in RECOVERY_OPS")
        print(f"  Searched in: {recovery_ops_path}")
        return 1

    # Step 2: Validate models
    if not args.skip_validation:
        print("Step 2: Validating models...")
        models = scanner.validate_models(models)
        print()
    else:
        print("Step 2: Skipping validation (--skip-validation)\n")

    # Step 3: Filter valid models
    valid_models = [m for m in models if m.is_valid]
    invalid_models = [m for m in models if not m.is_valid]

    print(f"Step 3: Summary")
    print(f"  Total found: {len(models)}")
    print(f"  Valid: {len(valid_models)}")
    print(f"  Invalid: {len(invalid_models)}")
    print()

    if invalid_models:
        print("Invalid models (skipped):")
        for model in invalid_models:
            print(f"  ✗ {model.file_path.name}: {model.error_message}")
        print()

    # Step 4: Integrate valid models
    if valid_models:
        print("Step 4: Integrating models...")
        integrated = []
        skipped = []

        # Group by model name, prefer _best.pt files
        model_groups = {}
        for model in valid_models:
            if model.model_name not in model_groups:
                model_groups[model.model_name] = []
            model_groups[model.model_name].append(model)

        # Select best file for each model (prefer _best.pt, then largest file)
        for model_name, group in model_groups.items():
            # Prefer files with 'best' in name
            best_files = [m for m in group if 'best' in m.file_path.name.lower()]
            if best_files:
                selected = max(best_files, key=lambda m: m.file_size_mb)
            else:
                selected = max(group, key=lambda m: m.file_size_mb)

            success, error = integrator.integrate_model(selected, args.dry_run)
            if success:
                integrated.append(selected)
            else:
                skipped.append((selected, error))

        print(f"\n  Integrated: {len(integrated)}")
        print(f"  Skipped: {len(skipped)}")

        if skipped:
            print("\nSkipped models:")
            for model, error in skipped:
                print(f"  - {model.model_name}: {error}")

        # Step 5: Generate registry
        if integrated and not args.dry_run:
            print("\nStep 5: Updating model registry...")
            integrator.generate_model_registry(integrated)

    # Generate report
    report = IntegrationReport(
        models_found=len(models),
        models_valid=len(valid_models),
        models_copied=len([m for m in valid_models if m in integrated]) if 'integrated' in locals() else 0,
        models_skipped=len(invalid_models) + (len(skipped) if 'skipped' in locals() else 0),
        errors=[m.error_message for m in invalid_models if m.error_message],
        details=[asdict(m) for m in models]
    )

    if args.output_report:
        with open(args.output_report, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"\n✓ Report saved to: {args.output_report}")

    print("\n" + "=" * 70)
    print("Integration Complete")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())

