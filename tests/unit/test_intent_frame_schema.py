import pytest
from music_brain.engine_api.schema import (
    IntentMetaSchema,
    MusicalIntentSchema,
    MusicHintsSchema,
    DSPTargetsSchema,
    TimeScopeSchema,
    IntentConstraintsSchema,
    IntentProvenanceSchema,
    IntentFrameSchema,
)


def test_default_frame():
    f = IntentFrameSchema()
    assert f.meta.schema_version == 1
    assert f.timestamp_ms == 0
    assert f.emotion.valence == 0.0
    assert f.music.tempo_bias == 0.0
    assert f.music_hints.key == ""
    assert f.dsp_targets.filter_cutoff == 0.5
    assert f.dsp_targets.stale is True
    assert f.time.start_bar == -1
    assert f.constraints.max_cpu_cost == 1.0
    assert f.provenance.source == 0
    assert f.latency_budget_ms == 10.0


def test_full_frame():
    f = IntentFrameSchema(
        meta=IntentMetaSchema(schema_version=1, intent_id=42, session_id=100),
        timestamp_ms=5000,
        music=MusicalIntentSchema(tempo_bias=0.5, rhythmic_density=0.8),
        music_hints=MusicHintsSchema(key="C", tempo_bpm=120.0, section_role="chorus"),
        dsp_targets=DSPTargetsSchema(
            filter_cutoff=0.8, filter_cutoff_confidence=0.9,
            reverb_send=0.4, reverb_send_confidence=0.8,
            drive=0.3, drive_confidence=0.7, stale=False,
        ),
        time=TimeScopeSchema(start_bar=1, end_bar=8),
        provenance=IntentProvenanceSchema(source=3, user_override_weight=0.7),
        latency_budget_ms=5.0,
    )
    assert f.meta.intent_id == 42
    assert f.timestamp_ms == 5000
    assert f.dsp_targets.stale is False
    assert f.dsp_targets.filter_cutoff_confidence == 0.9
    assert f.music_hints.section_role == "chorus"
    assert f.latency_budget_ms == 5.0


def test_dsp_safe_defaults():
    d = DSPTargetsSchema()
    assert d.filter_cutoff == 0.5
    assert d.reverb_send == 0.2
    assert d.drive == 0.0
    assert d.stale is True
    assert d.filter_cutoff_confidence == 0.0
    assert d.reverb_send_confidence == 0.0
    assert d.drive_confidence == 0.0


def test_invalid_tempo_bias_oob():
    with pytest.raises(Exception):
        MusicalIntentSchema(tempo_bias=5.0)


def test_invalid_mode_preference():
    with pytest.raises(Exception):
        MusicalIntentSchema(mode_preference=2)


def test_invalid_time_scope():
    with pytest.raises(Exception):
        TimeScopeSchema(start_bar=5, end_bar=2)


def test_invalid_source_oob():
    with pytest.raises(Exception):
        IntentProvenanceSchema(source=99)


def test_invalid_extra_field():
    with pytest.raises(Exception):
        IntentFrameSchema(unknown_field="bad")


def test_invalid_version():
    with pytest.raises(Exception):
        IntentMetaSchema(schema_version=99)


def test_invalid_section_role():
    with pytest.raises(Exception):
        MusicHintsSchema(section_role="invalid_section")


def test_invalid_dsp_cutoff_oob():
    with pytest.raises(Exception):
        DSPTargetsSchema(filter_cutoff=2.0)
