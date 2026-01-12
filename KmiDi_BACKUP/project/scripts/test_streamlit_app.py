#!/usr/bin/env python3
"""Test Streamlit app without launching full server.

This script verifies that:
1. Streamlit is installed
2. Streamlit app can be imported/parsed
3. Music Brain API integration works
4. App components are functional
"""

import sys
import ast
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_streamlit_import():
    """Test that Streamlit is installed."""
    print("Testing Streamlit import...")
    try:
        import streamlit
        print(f"  ✓ Streamlit {streamlit.__version__} installed")
        return True
    except ImportError as e:
        print(f"  ✗ Streamlit not installed: {e}")
        print("  Install with: pip install streamlit")
        return False

def test_streamlit_app_imports():
    """Test that Streamlit app can be imported."""
    print("\nTesting Streamlit app imports...")
    try:
        # Try to import the app module (not run it)
        streamlit_app_path = project_root / "streamlit_app.py"
        
        if not streamlit_app_path.exists():
            print(f"  ✗ streamlit_app.py not found: {streamlit_app_path}")
            return False
        
        # Parse the file to check syntax
        with open(streamlit_app_path, "r") as f:
            code = f.read()
        
        try:
            ast.parse(code)
            print("  ✓ streamlit_app.py syntax is valid")
        except SyntaxError as e:
            print(f"  ✗ streamlit_app.py has syntax errors: {e}")
            return False
        
        # Try to import dependencies
        try:
            from music_brain.api import api as music_api, EMOTIONAL_PRESETS
            print("  ✓ Music Brain API imported")
            
            if EMOTIONAL_PRESETS:
                print(f"  ✓ Found {len(EMOTIONAL_PRESETS)} emotional presets")
            else:
                print("  ⚠ No emotional presets found")
            
            return True
        except ImportError as e:
            print(f"  ✗ Music Brain API import failed: {e}")
            return False
            
    except Exception as e:
        print(f"  ✗ Streamlit app import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_music_api_integration():
    """Test that Music Brain API can be called."""
    print("\nTesting Music Brain API integration...")
    try:
        from music_brain.api import api as music_api
        
        # Test a simple API call
        # Use a safe test emotion
        test_emotion = "calm"
        
        print(f"  Testing API with emotion: {test_emotion}")
        result = music_api.therapy_session(
            text=test_emotion,
            motivation=5,
            chaos_tolerance=0.5,
            output_midi=None,
        )
        
        if result and isinstance(result, dict):
            print("  ✓ API call succeeded")
            if 'chords' in result:
                print(f"    Generated {len(result['chords'])} chords")
            return True
        else:
            print("  ⚠ API call returned unexpected format")
            return False
            
    except Exception as e:
        print(f"  ✗ API integration test failed: {e}")
        print("  ⚠ This may be expected if API requires specific setup")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_components():
    """Test that Streamlit components are available."""
    print("\nTesting Streamlit components...")
    try:
        import streamlit as st
        
        # Check that key Streamlit functions are available
        components = [
            'st.title', 'st.markdown', 'st.header', 'st.subheader',
            'st.text_input', 'st.text_area', 'st.selectbox', 'st.slider',
            'st.button', 'st.spinner', 'st.success', 'st.error',
            'st.download_button', 'st.expander', 'st.json', 'st.code'
        ]
        
        missing = []
        for comp in components:
            try:
                eval(comp)  # Check if attribute exists
            except AttributeError:
                missing.append(comp)
        
        if missing:
            print(f"  ✗ Missing components: {', '.join(missing)}")
            return False
        else:
            print(f"  ✓ All {len(components)} components available")
            return True
            
    except Exception as e:
        print(f"  ✗ Streamlit components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("KmiDi Streamlit App Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: Streamlit import
    results.append(("Streamlit Import", test_streamlit_import()))
    
    # Test 2: Streamlit app imports (only if Streamlit works)
    if results[0][1]:
        results.append(("Streamlit App Imports", test_streamlit_app_imports()))
        
        # Test 3: Music API integration (only if imports work)
        if results[1][1]:
            results.append(("Music API Integration", test_music_api_integration()))
        
        # Test 4: Streamlit components
        results.append(("Streamlit Components", test_streamlit_components()))
    
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
        print("\n✅ All Streamlit app tests passed!")
        print("\nTo launch the Streamlit app, run:")
        print("  streamlit run streamlit_app.py")
        return 0
    else:
        print("\n⚠ Some tests failed or were skipped.")
        print("\nTo launch the Streamlit app anyway, run:")
        print("  streamlit run streamlit_app.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
