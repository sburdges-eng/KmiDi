"""Tests for kmidi_gui.core.export — MIDI, Intent, and Annotation exporters."""

import json
import tempfile
from pathlib import Path

import pytest

# Conftest adds KmiDi_CANON/brain to path
from kmidi_gui.core.export import MIDIExporter, IntentExporter, AnnotationExporter


class TestMIDIExporter:
    """MIDI export: copy from path, pipeline dict, placeholder."""

    def test_export_from_dict_file_path(self, tmp_path):
        """Dict with file_path copies source to path."""
        source = tmp_path / "source.mid"
        source.write_bytes(b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60")
        dest = tmp_path / "dest.mid"
        exporter = MIDIExporter()
        exporter.export(dest, {"file_path": str(source)})
        assert dest.exists()
        assert dest.read_bytes() == source.read_bytes()

    def test_export_from_dict_midi_path(self, tmp_path):
        """Dict with midi_path (pipeline result) copies source."""
        source = tmp_path / "generated.mid"
        source.write_bytes(b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60")
        dest = tmp_path / "out.mid"
        exporter = MIDIExporter()
        exporter.export(dest, {"midi_path": str(source)})
        assert dest.exists()
        assert dest.read_bytes() == source.read_bytes()

    def test_export_from_str_path(self, tmp_path):
        """Str path copies source to path."""
        source = tmp_path / "a.mid"
        source.write_bytes(b"MThd")
        dest = tmp_path / "b.mid"
        exporter = MIDIExporter()
        exporter.export(dest, str(source))
        assert dest.exists()
        assert dest.read_bytes() == source.read_bytes()

    def test_export_placeholder_when_no_source(self, tmp_path):
        """When midi_data has no path/MidiFile, writes placeholder file."""
        dest = tmp_path / "placeholder.mid"
        exporter = MIDIExporter()
        exporter.export(dest, None)
        assert dest.exists()
        assert dest.read_bytes() == b""

    def test_export_placeholder_empty_dict(self, tmp_path):
        """Empty dict or dict without path keys writes placeholder."""
        dest = tmp_path / "empty.mid"
        exporter = MIDIExporter()
        exporter.export(dest, {})
        assert dest.exists()
        assert dest.read_bytes() == b""


class TestIntentExporter:
    """Intent schema JSON export."""

    def test_export_intent_to_json(self, tmp_path):
        """IntentExporter writes valid JSON from CompleteSongIntent."""
        from music_brain.session.intent_schema import CompleteSongIntent, TechnicalConstraints
        intent = CompleteSongIntent(technical_constraints=TechnicalConstraints(technical_key="C", technical_mode="major"))
        path = tmp_path / "intent.json"
        IntentExporter().export(path, intent)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data.get("technical_constraints", {}).get("technical_key") == "C"


class TestAnnotationExporter:
    """ML annotation JSON export."""

    def test_export_annotations_to_json(self, tmp_path):
        """AnnotationExporter writes valid JSON."""
        path = tmp_path / "annotations.json"
        annotations = {"version": "1.0", "labels": ["verse", "chorus"]}
        AnnotationExporter().export(path, annotations)
        assert path.exists()
        assert json.loads(path.read_text(encoding="utf-8")) == annotations
