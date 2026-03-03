"""Unit tests for music_brain.agents.events."""
import importlib.util
import sys
import pytest
from datetime import datetime


# Load events module directly to bypass pre-existing async_hub import issue
def _load_events():
    spec = importlib.util.spec_from_file_location(
        "music_brain.agents.events",
        "music_brain/agents/events.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["music_brain.agents.events"] = mod
    spec.loader.exec_module(mod)
    return mod


_events = _load_events()
Event = _events.Event
EventBus = _events.EventBus
EventPriority = _events.EventPriority


@pytest.mark.unit
class TestEventFromDict:
    """Tests for Event.from_dict deserialization."""

    def test_source_deserializes_correctly(self):
        """source= must survive a round-trip through to_dict / from_dict."""
        event = Event(type="test.event", data={"x": 1}, source="my_source")
        d = event.to_dict()
        restored = Event.from_dict(d)
        assert restored.source == "my_source"

    def test_source_none_when_absent(self):
        """source defaults to None when not present in the dict."""
        d = {"type": "test.event", "timestamp": datetime.now().isoformat()}
        restored = Event.from_dict(d)
        assert restored.source is None

    def test_priority_deserializes_correctly(self):
        """priority must survive a round-trip through to_dict / from_dict."""
        event = Event(type="test.event", priority=EventPriority.HIGH)
        d = event.to_dict()
        restored = Event.from_dict(d)
        assert restored.priority == EventPriority.HIGH

    def test_priority_defaults_to_normal_when_absent(self):
        """priority defaults to NORMAL when not present in the dict."""
        d = {"type": "test.event", "timestamp": datetime.now().isoformat()}
        restored = Event.from_dict(d)
        assert restored.priority == EventPriority.NORMAL

    def test_timestamp_uses_isoformat_when_present(self):
        """A timestamp string in the dict is parsed back to a datetime."""
        ts = datetime(2024, 6, 15, 12, 0, 0)
        d = {
            "type": "test.event",
            "timestamp": ts.isoformat(),
        }
        restored = Event.from_dict(d)
        assert restored.timestamp == ts

    def test_timestamp_defaults_to_now_when_absent(self):
        """timestamp defaults to approximately now() when not in the dict."""
        before = datetime.now()
        d = {"type": "test.event"}
        restored = Event.from_dict(d)
        after = datetime.now()
        assert before <= restored.timestamp <= after

    def test_full_roundtrip(self):
        """All fields survive a to_dict / from_dict round-trip."""
        original = Event(
            type="daw.play",
            data={"tempo": 120},
            source="transport",
            priority=EventPriority.CRITICAL,
        )
        restored = Event.from_dict(original.to_dict())
        assert restored.type == original.type
        assert restored.data == original.data
        assert restored.id == original.id
        assert restored.source == original.source
        assert restored.priority == original.priority


@pytest.mark.unit
class TestEventBusRepr:
    """Tests for EventBus.__repr__."""

    def test_repr_is_str(self):
        bus = EventBus()
        r = repr(bus)
        assert isinstance(r, str)

    def test_repr_contains_handler_count(self):
        bus = EventBus()
        assert "handlers=0" in repr(bus)

    def test_repr_contains_events_emitted(self):
        bus = EventBus()
        assert "events=0" in repr(bus)

    def test_repr_line_length(self):
        """repr must not exceed 100 chars for typical values."""
        bus = EventBus()
        assert len(repr(bus)) <= 100
