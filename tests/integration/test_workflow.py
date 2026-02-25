"""Integration tests for complete workflows."""

import pytest
import sys
from pathlib import Path

# Add build/python to path for C++ bindings
build_python = Path(__file__).parent.parent.parent / "build" / "python"
if build_python.exists():
    sys.path.insert(0, str(build_python))


@pytest.mark.integration
class TestWorkflow:
    """Test complete workflows."""
    
    def test_emotion_to_intent_workflow(self, sample_intent):
        """Test emotion to intent conversion."""
        from music_brain.session.intent_processor import IntentProcessor
        
        processor = IntentProcessor(sample_intent)
        assert processor is not None
        assert processor.intent == sample_intent
    
    @pytest.mark.cpp
    def test_cpp_bridge_import(self):
        """Test C++ Python bridge import."""
        try:
            import penta_core_native
            assert hasattr(penta_core_native, '__version__')
        except ImportError:
            pytest.skip("C++ bindings not available")
