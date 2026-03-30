#!/usr/bin/env python3
"""
Model Verification Script

Verifies all models in the system are properly configured and available.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

from penta_core.ml.model_registry import get_registry, list_models, get_model
from typing import Dict, List, Tuple


def check_model_file(model_info) -> Tuple[bool, str]:
    """Check if model file exists"""
    if not model_info.path:
        return False, "No path specified"

    model_path = Path(model_info.path)
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        return True, f"Exists ({size_mb:.2f} MB)"
    else:
        return False, f"Missing: {model_path}"


def verify_trained_models() -> Dict[str, Dict]:
    """Verify all trained models"""
    registry = get_registry()
    registry.discover()  # Discover models from directories

    models = list_models()
    results = {"trained": [], "stub": [], "missing": [], "optional": []}

    for model in models:
        status = {
            "name": model.name,
            "task": model.task.value,
            "backend": model.backend.value,
            "path": model.path,
            "version": model.version,
            "file_exists": False,
            "file_status": "",
            "integration": {"python_api": False, "cpp_mlinterface": False, "tauri_ui": False},
        }

        # Check file existence
        file_exists, file_status = check_model_file(model)
        status["file_exists"] = file_exists
        status["file_status"] = file_status

        # Categorize
        if model.name in [
            "emotionrecognizer",
            "melodytransformer",
            "harmonypredictor",
            "dynamicsengine",
            "groovepredictor",
        ]:
            if file_exists:
                results["trained"].append(status)
            else:
                results["missing"].append(status)
        elif model.name in ["instrumentrecognizer", "emotionnodeclassifier"]:
            results["stub"].append(status)
        elif model.name == "llama_onnx":
            results["optional"].append(status)
        else:
            results["missing"].append(status)

    return results


def verify_prrot_models() -> Dict[str, bool]:
    """Verify PRROT system models"""
    prrot_dir = project_root / "python" / "prrot" / "models"
    results = {}

    # Check phoneme aligner
    phoneme_path = prrot_dir / "phoneme_aligner_q4.bin"
    results["phoneme_aligner"] = {
        "exists": phoneme_path.exists(),
        "path": str(phoneme_path),
        "size_mb": phoneme_path.stat().st_size / (1024 * 1024) if phoneme_path.exists() else 0,
    }

    return results


def print_results(results: Dict, prrot_results: Dict):
    """Print verification results"""
    print("=" * 80)
    print("MODEL VERIFICATION REPORT")
    print("=" * 80)
    print()

    # Trained models
    print("✅ TRAINED MODELS:")
    if results["trained"]:
        for model in results["trained"]:
            status_icon = "✅" if model["file_exists"] else "❌"
            print(f"  {status_icon} {model['name']:25s} {model['task']:20s} {model['file_status']}")
    else:
        print("  (none)")
    print()

    # Stub models
    print("⚠️  STUB MODELS (Need Training):")
    if results["stub"]:
        for model in results["stub"]:
            status_icon = "⚠️" if not model["file_exists"] else "✅"
            print(f"  {status_icon} {model['name']:25s} {model['task']:20s} {model['file_status']}")
    else:
        print("  (none)")
    print()

    # Missing models
    print("❌ MISSING MODELS:")
    if results["missing"]:
        for model in results["missing"]:
            print(f"  ❌ {model['name']:25s} {model['task']:20s} {model['file_status']}")
    else:
        print("  (none)")
    print()

    # Optional models
    print("🔵 OPTIONAL MODELS:")
    if results["optional"]:
        for model in results["optional"]:
            status_icon = "🔵" if not model["file_exists"] else "✅"
            print(f"  {status_icon} {model['name']:25s} {model['task']:20s} {model['file_status']}")
    else:
        print("  (none)")
    print()

    # PRROT models
    print("PRROT SYSTEM MODELS:")
    for name, info in prrot_results.items():
        status_icon = "✅" if info["exists"] else "❌"
        if info["exists"]:
            print(f"  {status_icon} {name:25s} {info['size_mb']:.2f} MB - {info['path']}")
        else:
            print(f"  {status_icon} {name:25s} Missing - {info['path']}")
    print()

    # Summary
    total_trained = len(results["trained"])
    total_stub = len(results["stub"])
    total_missing = len(results["missing"])
    total_optional = len(results["optional"])

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Trained Models:     {total_trained}")
    print(f"Stub Models:        {total_stub}")
    print(f"Missing Models:     {total_missing}")
    print(f"Optional Models:    {total_optional}")
    print()

    # Check timbre extractor integration
    print("TIMBRE EXTRACTOR:")
    try:
        from prrot.timbre_embeddings import TimbreEmbeddingExtractor

        extractor = TimbreEmbeddingExtractor()
        # Check if it's using placeholder
        import inspect

        source = inspect.getsource(extractor.extract_embedding)
        if "random.randn" in source or "placeholder" in source.lower():
            print("  ⚠️  Using placeholder implementation")
            print("     See docs/TIMBRE_EXTRACTOR_INTEGRATION.md for integration guide")
        else:
            print("  ✅ Integrated with ML model")
    except Exception as e:
        print(f"  ❌ Error checking: {e}")
    print()

    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if total_missing > 0:
        print("1. Missing models need to be obtained or trained")
        print("   See docs/MODEL_SETUP_GUIDE.md for details")

    if total_stub > 0:
        print("2. Stub models need training")
        print("   Configs available in configs/ directory")
        print("   See docs/MODEL_SETUP_GUIDE.md for training instructions")

    if not prrot_results.get("phoneme_aligner", {}).get("exists", False):
        print("3. Phoneme aligner model missing")
        print("   See docs/PHONEME_ALIGNER_INTEGRATION.md")

    print()
    print("For detailed information, see:")
    print("  - docs/MODEL_INVENTORY.md")
    print("  - docs/MODEL_SETUP_GUIDE.md")
    print("  - docs/MODEL_IMPLEMENTATION_SUMMARY.md")


def main():
    """Main verification function"""
    try:
        results = verify_trained_models()
        prrot_results = verify_prrot_models()
        print_results(results, prrot_results)

        # Exit code based on status
        total_issues = (
            len([m for m in results["trained"] if not m["file_exists"]])
            + len(results["stub"])
            + len(results["missing"])
        )

        if total_issues == 0:
            print("\n✅ All models verified successfully!")
            return 0
        else:
            print(f"\n⚠️  Found {total_issues} model(s) that need attention")
            return 1

    except Exception as e:
        print(f"❌ Error during verification: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
