#!/usr/bin/env python3
"""Test Qt GUI components without launching full GUI.

This script verifies that:
1. PySide6 is installed
2. GUI components can be imported
3. Core engine works
4. GUI widgets can be instantiated (headless)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_pyside6_import():
    """Test that PySide6 is installed."""
    print("Testing PySide6 import...")
    try:
        import PySide6
        print(f"  ✓ PySide6 {PySide6.__version__} installed")
        return True
    except ImportError as e:
        print(f"  ✗ PySide6 not installed: {e}")
        print("  Install with: pip install PySide6")
        return False

def test_gui_imports():
    """Test that GUI components can be imported."""
    print("\nTesting GUI component imports...")
    errors = []
    
    try:
        from kmidi_gui.gui.main_window import MainWindow
        print("  ✓ MainWindow imported")
    except Exception as e:
        errors.append(f"MainWindow: {e}")
        print(f"  ✗ MainWindow import failed: {e}")
    
    try:
        from kmidi_gui.controllers.actions import ActionController
        print("  ✓ ActionController imported")
    except Exception as e:
        errors.append(f"ActionController: {e}")
        print(f"  ✗ ActionController import failed: {e}")
    
    try:
        from kmidi_gui.themes.palette import apply_audio_palette, AudioColors
        print("  ✓ Theme components imported")
    except Exception as e:
        errors.append(f"Theme: {e}")
        print(f"  ✗ Theme import failed: {e}")
    
    return len(errors) == 0

def test_core_engine():
    """Test that core engine works."""
    print("\nTesting core engine...")
    try:
        from kmidi_gui.core.engine import MusicEngine
        from kmidi_gui.core.models import EmotionIntent
        
        # Create engine
        engine = MusicEngine(music_brain_api_url="http://127.0.0.1:8000")
        print("  ✓ MusicEngine created")
        
        # Create intent
        intent = EmotionIntent(
            core_event="test emotion",
            mood_primary="sad",
            vulnerability_scale=0.5
        )
        print("  ✓ EmotionIntent created")
        print(f"    Intent: {intent.to_dict()}")
        
        return True
    except Exception as e:
        print(f"  ✗ Core engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_headless_gui():
    """Test GUI widgets can be instantiated without display."""
    print("\nTesting headless GUI instantiation...")
    try:
        from PySide6.QtWidgets import QApplication
        from kmidi_gui.gui.main_window import MainWindow
        from kmidi_gui.controllers.actions import ActionController
        from kmidi_gui.themes.palette import apply_audio_palette
        
        # Create QApplication (headless - no display needed)
        # Set QT_QPA_PLATFORM=offscreen to avoid needing display
        if not QApplication.instance():
            app = QApplication([])  # Empty args for headless
        
        # Apply theme
        app = QApplication.instance()
        apply_audio_palette(app)
        print("  ✓ QApplication created and themed")
        
        # Try to create MainWindow (without showing)
        # This will test that all UI setup code runs
        window = MainWindow()
        print("  ✓ MainWindow instantiated")
        
        # Create controller
        controller = ActionController(window)
        print("  ✓ ActionController instantiated")
        
        # Verify signals are connected
        if hasattr(window, 'generate_requested'):
            print("  ✓ Window signals available")
        
        print("\n  ⚠ Note: Full GUI test requires display (run: python -m kmidi_gui.main)")
        
        return True
    except Exception as e:
        print(f"  ✗ Headless GUI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("KmiDi Qt GUI Component Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: PySide6 import
    results.append(("PySide6 Import", test_pyside6_import()))
    
    # Test 2: GUI imports (only if PySide6 works)
    if results[0][1]:
        results.append(("GUI Imports", test_gui_imports()))
        
        # Test 3: Core engine
        results.append(("Core Engine", test_core_engine()))
        
        # Test 4: Headless GUI
        results.append(("Headless GUI", test_headless_gui()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All Qt GUI component tests passed!")
        print("\nTo launch full GUI, run:")
        print("  python -m kmidi_gui.main")
        return 0
    else:
        print("\n❌ Some tests failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
