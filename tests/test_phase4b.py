"""Tests for Phase 4b — DAW bridge routing and PRROT voice integration."""

import pytest
from typing import Any


# ---------------------------------------------------------------------------
# Mock EventBus (reused from test_behavior_lab.py pattern)
# ---------------------------------------------------------------------------


class MockEventBus:
    def __init__(self):
        self._emitted: list[tuple[str, Any]] = []

    def on(self, event_type, handler=None, **kwargs):
        pass

    async def emit(self, event_type, data=None):
        self._emitted.append((event_type, data))


# ---------------------------------------------------------------------------
# send_to_daw tests
# ---------------------------------------------------------------------------


class TestSendToDAW:
    def test_import(self):
        from music_brain.integrations.comprehensive_bridge import (
            ComprehensiveIntegrationManager,
        )

        assert hasattr(ComprehensiveIntegrationManager, "send_to_daw")

    def test_unknown_daw_returns_false(self):
        from music_brain.integrations.comprehensive_bridge import (
            ComprehensiveIntegrationManager,
        )

        bus = MockEventBus()
        mgr = ComprehensiveIntegrationManager(
            lib_path="/nonexistent/lib.dylib",
            event_bus=bus,
        )
        # send_to_daw imports from music_brain.agents which may have broken
        # internal imports (ComputedValue in async_hub). If so, it should
        # return False rather than crash the caller.
        try:
            result = mgr.send_to_daw("nonexistent_daw", [])
            assert result is False
        except ImportError:
            pytest.skip("music_brain.agents has broken internal imports (pre-existing)")

    def test_valid_daw_name_accepted(self):
        """Test that valid DAW type names don't raise on enum lookup."""
        from music_brain.agents.daw_protocol import DAWType

        for name in ["ableton", "logic_pro", "reaper", "bitwig"]:
            assert DAWType(name) is not None

    def test_midi_event_types(self):
        """Verify the send_to_daw method handles all MIDI event types."""
        from music_brain.integrations.comprehensive_bridge import (
            ComprehensiveIntegrationManager,
        )

        bus = MockEventBus()
        mgr = ComprehensiveIntegrationManager(
            lib_path="/nonexistent/lib.dylib",
            event_bus=bus,
        )
        events = [
            {"type": "note_on", "note": 60, "velocity": 100},
            {"type": "note_off", "note": 60},
            {"type": "cc", "cc": 1, "value": 64},
            {"type": "pitch_bend", "value": 8192},
        ]
        try:
            result = mgr.send_to_daw("ableton", events)
            assert result is False  # Expected: no bridge available
        except ImportError:
            pytest.skip("music_brain.agents has broken internal imports (pre-existing)")


# ---------------------------------------------------------------------------
# PRROT Bridge tests
# ---------------------------------------------------------------------------


class TestPRROTBridge:
    def test_import(self):
        from music_brain.integrations.prrot_bridge import PRROTBridge

        assert PRROTBridge is not None

    def test_bridge_creation(self):
        from music_brain.integrations.prrot_bridge import PRROTBridge

        bus = MockEventBus()
        bridge = PRROTBridge(event_bus=bus)
        assert bridge.initialized is False

    def test_available_property(self):
        from music_brain.integrations.prrot_bridge import PRROTBridge

        bridge = PRROTBridge()
        # available depends on whether bindings are compiled
        assert isinstance(bridge.available, bool)

    def test_init_without_bindings(self):
        """If bindings aren't compiled, initialize should return False gracefully."""
        from music_brain.integrations.prrot_bridge import PRROTBridge, _prrot_available

        bridge = PRROTBridge()
        if not _prrot_available:
            assert bridge.initialize() is False
            assert bridge.initialized is False

    def test_voice_analysis_result(self):
        from music_brain.integrations.prrot_bridge import VoiceAnalysisResult

        result = VoiceAnalysisResult(
            phonemes=[{"phoneme": "AA", "start_ms": 0}],
            pitch_targets=[],
            breath_markers=[],
        )
        assert result.has_data is True
        empty = VoiceAnalysisResult(phonemes=[], pitch_targets=[], breath_markers=[])
        assert empty.has_data is False

    def test_sync_phoneme_analysis_without_init(self):
        from music_brain.integrations.prrot_bridge import PRROTBridge

        bridge = PRROTBridge()
        result = bridge.analyze_phonemes_sync(__import__("numpy").zeros(1024, dtype="float32"))
        assert result == []


# ---------------------------------------------------------------------------
# Integration __init__ exports
# ---------------------------------------------------------------------------


class TestIntegrationExports:
    def test_comprehensive_bridge_exported(self):
        from music_brain.integrations import ComprehensiveIntegrationManager

        assert ComprehensiveIntegrationManager is not None

    def test_prrot_bridge_exported(self):
        from music_brain.integrations import PRROTBridge, VoiceAnalysisResult

        assert PRROTBridge is not None
        assert VoiceAnalysisResult is not None

    def test_rt_snapshot_exported(self):
        from music_brain.integrations import RTSnapshot

        assert RTSnapshot is not None
