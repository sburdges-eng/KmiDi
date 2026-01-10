"""
Unit tests for emotion_production.py
"""

from music_brain.emotion.emotion_production import EmotionProductionMapper, ProductionPreset
from music_brain.emotion.emotion_thesaurus import EmotionMatch

DYN_ORDER = ["pp", "p", "mp", "mf", "f", "ff", "fff"]


def _dyn_idx(level: str) -> int:
    try:
        return DYN_ORDER.index(level)
    except ValueError:
        return DYN_ORDER.index("mf")


def make_match(
    base: str,
    sub: str,
    tier: int,
    sub_sub: str = "detail",
) -> EmotionMatch:
    return EmotionMatch(
        base_emotion=base,
        sub_emotion=sub,
        sub_sub_emotion=sub_sub,
        intensity_tier=tier,
        matched_synonym=sub.lower(),
        all_tier_synonyms=[sub.lower()],
        emotion_id=f"{base}-{sub}",
        description=f"{base} - {sub}",
    )


def test_preset_includes_section_maps_and_fx():
    mapper = EmotionProductionMapper()
    happy = make_match("happy", "joy", tier=4, sub_sub="ecstatic")

    preset = mapper.get_production_preset(happy)

    assert isinstance(preset, ProductionPreset)
    assert preset.drum_style == "pop"
    assert preset.tempo_range == (108, 128)  # 105–125 shifted by tier 4

    # Section mappings should show lift in chorus
    assert _dyn_idx(preset.section_dynamics["chorus"]) > _dyn_idx(preset.section_dynamics["verse"])
    assert preset.section_density["chorus"] > preset.section_density["verse"]

    # FX hints come through from base profile
    assert "reverb" in preset.fx
    assert "delay" in preset.fx


def test_genre_override_adjusts_drum_style_and_swing():
    mapper = EmotionProductionMapper()
    happy = make_match("happy", "joy", tier=3, sub_sub="bright")

    preset = mapper.get_production_preset(happy, genre="hip-hop")

    assert preset.drum_style == "hip-hop"
    assert preset.swing >= 0.12  # boosted by genre hint
    assert preset.groove_motif == "boom_bap"
    assert preset.feel in {"swing", "straight"}  # feel stays valid


def test_sub_emotion_grief_uses_brush_profile():
    mapper = EmotionProductionMapper()
    grief = make_match("sad", "grief", tier=2, sub_sub="bereaved")

    preset = mapper.get_production_preset(grief)

    assert preset.drum_style == "brushes"
    assert preset.feel == "swing"
    assert preset.swing >= 0.1
    assert preset.tempo_range[0] < 70  # pulled down for grief
    assert "brush" in preset.kit_hint.lower()


def test_intensity_scales_density_and_tempo_range():
    mapper = EmotionProductionMapper()
    fear_low = make_match("fear", "anxiety", tier=2)
    fear_high = make_match("fear", "anxiety", tier=6)

    low_preset = mapper.get_production_preset(fear_low)
    high_preset = mapper.get_production_preset(fear_high)

    assert low_preset.arrangement_density < high_preset.arrangement_density
    assert low_preset.tempo_range[0] < high_preset.tempo_range[0]
    assert low_preset.section_density["chorus"] < high_preset.section_density["chorus"]


def test_transition_notes_present_and_emphatic_for_high_energy():
    mapper = EmotionProductionMapper()
    rage = make_match("angry", "rage", tier=5, sub_sub="furious")

    preset = mapper.get_production_preset(rage)

    assert preset.drum_style == "metal"
    assert "into_chorus" in preset.transitions
    assert "fill" in preset.transitions["into_chorus"].lower() or "hat" in preset.transitions["into_chorus"].lower()
    assert _dyn_idx(preset.section_dynamics["chorus"]) >= _dyn_idx("f")
