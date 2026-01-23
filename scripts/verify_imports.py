#!/usr/bin/env python3
"""
Import Verification Script for KmiDi-1 Migration

This script verifies that all key imports work correctly after the migration.
"""

import sys
import importlib
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test cases for key imports
IMPORT_TESTS = [
    # Core music_brain imports
    ("music_brain", "music_brain package"),
    ("music_brain.session", "Session module"),
    ("music_brain.session.intent_schema", "Intent schema"),
    ("music_brain.session.intent_processor", "Intent processor"),
    ("music_brain.kelly_companion", "Kelly Companion"),
    ("music_brain.kelly_companion.core.emotion_thesaurus", "Emotion thesaurus"),
    ("music_brain.kelly_companion.engines", "Engines module"),
    ("music_brain.kelly_companion.engines.bass_engine", "Bass engine"),
    ("music_brain.kelly_companion.engines.melody_engine", "Melody engine"),
    ("music_brain.kelly_companion.session", "Kelly Companion session"),
    ("music_brain.kelly_companion.utils.harmony_deps", "Harmony dependencies"),
    ("music_brain.penta_core", "Penta-core module"),
    ("music_brain.orchestrator", "Orchestrator"),
    ("music_brain.intelligence", "Intelligence module"),
    ("music_brain.learning", "Learning module"),
]

def test_import(module_name, description):
    """Test if a module can be imported."""
    try:
        module = importlib.import_module(module_name)
        print(f"✅ {description:50} ({module_name})")
        return True
    except ImportError as e:
        print(f"❌ {description:50} ({module_name})")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {description:50} ({module_name})")
        print(f"   Warning: {e}")
        return False

def main():
    """Run all import tests."""
    print("=" * 80)
    print("KmiDi-1 Import Verification")
    print("=" * 80)
    print()
    
    results = []
    for module_name, description in IMPORT_TESTS:
        success = test_import(module_name, description)
        results.append((module_name, success))
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All imports successful!")
        return 0
    else:
        print(f"❌ {total - passed} import(s) failed")
        print("\nFailed imports:")
        for module_name, success in results:
            if not success:
                print(f"  - {module_name}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
