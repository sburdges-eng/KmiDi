"""Tests for example-based music learning and orchestrator hooks."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from music_brain.learning.music_learning_manager import MusicLearningManager
from music_brain.learning.melody_learning import MelodyExample
from music_brain.learning.harmony_learning import ChordExample
from music_brain.learning.groove_learning import GrooveExample
from music_brain.learning.bass_learning import BassExample
from music_brain.learning.arrangement_learning import ArrangementExample
from music_brain.learning.expression_learning import ExpressionExample
from music_brain.learning.rulebreak_learning import RuleBreakExample
from music_brain.orchestrator.interfaces import ExecutionContext, ProcessorConfig
from music_brain.orchestrator.processors.harmony import HarmonyProcessor, HarmonyInput
from music_brain.orchestrator.processors.groove import GrooveProcessor, GrooveInput
from music_brain.orchestrator.processors.melody import MelodyProcessor, MelodyInput
from music_brain.integrations.dynamics_integration import DynamicsIntegration
from music_brain.arrangement.bass_generator import generate_bass_line, BassPattern
from music_brain.arrangement.generator import ArrangementGenerator
from music_brain.session.ml_melody_generator import MLMelodyGenerator, MelodyConfig


@pytest.fixture()
def learning_root(tmp_path: Path) -> Path:
    return tmp_path / "music_learning"


def _seed_manager(root: Path) -> MusicLearningManager:
    m = MusicLearningManager(storage_root=root)
    m.add_melody_example(
        MelodyExample(melody=[60, 62, 64, 65, 67, 69, 71, 72], emotion="joy", tempo=120),
        name="m1",
    )
    m.learn_melody_profile("p_melody", ["m1"])
    m.add_chord_example(
        ChordExample(
            progression=["C", "G", "Am", "F"],
            roman_numerals=["I", "V", "vi", "IV"],
            emotion="joy",
            key="C",
            mode="major",
        ),
        name="h1",
    )
    m.learn_harmony_profile("p_harmony", ["h1"])
    m.add_groove_example(
        GrooveExample(
            emotion="joy",
            timing_offsets_16th=[0.0] * 16,
            velocity_curve=[90] * 16,
            tempo_bpm=120,
        ),
        name="g1",
    )
    m.learn_groove_profile("p_groove", ["g1"])
    m.add_bass_example(
        BassExample(notes=[36, 38, 40, 41], emotion="joy", pattern="walking"),
        name="b1",
    )
    m.learn_bass_profile("p_bass", ["b1"])
    m.add_arrangement_example(
        ArrangementExample(sections=["verse", "chorus", "verse"], emotion="joy"),
        name="a1",
    )
    m.learn_arrangement_profile("p_arr", ["a1"])
    m.add_expression_example(
        ExpressionExample(velocity_curve=[70] * 16, emotion="joy"),
        name="e1",
    )
    m.learn_expression_profile("p_expr", ["e1"])
    m.add_rulebreak_example(
        RuleBreakExample(rule_break="HARMONY_ModalInterchange", emotion="joy", accepted=True),
        name="r1",
    )
    m.learn_rulebreak_profile("p_rb", ["r1"])
    return m


def test_harmony_processor_uses_learning_profile(learning_root: Path):
    _seed_manager(learning_root)
    ctx = ExecutionContext(execution_id="1", pipeline_id="p")
    ctx.set_shared("learning_profiles", {"harmony": "p_harmony"})
    ctx.set_shared("learning_storage_root", str(learning_root))

    proc = HarmonyProcessor()

    async def _run():
        return await proc.process(
            HarmonyInput(emotion="joy", key="C", mode="major", progression_length=4),
            ctx,
        )

    res = asyncio.run(_run())
    assert res.success
    assert res.data.chords == ["C", "G", "Am", "F"]


def test_groove_processor_uses_learning_profile(learning_root: Path):
    _seed_manager(learning_root)
    ctx = ExecutionContext(execution_id="1", pipeline_id="p")
    ctx.set_shared("learning_profiles", {"groove": "p_groove"})
    ctx.set_shared("learning_storage_root", str(learning_root))

    proc = GrooveProcessor()

    async def _run():
        return await proc.process(GrooveInput(tempo=120, genre="pop", emotion="joy"), ctx)

    res = asyncio.run(_run())
    assert res.success
    assert len(res.data.velocity_curve) == 16


def test_melody_processor_runs(learning_root: Path):
    _seed_manager(learning_root)
    ctx = ExecutionContext(execution_id="1", pipeline_id="p")
    ctx.set_shared("learning_profiles", {"melody": "p_melody"})
    ctx.set_shared("learning_storage_root", str(learning_root))

    proc = MelodyProcessor(ProcessorConfig(name="melody"))

    async def _run():
        return await proc.process(MelodyInput(emotion="joy", length=8), ctx)

    res = asyncio.run(_run())
    assert res.success
    assert len(res.data.notes) == 8


def test_ml_melody_generator_learning_profile(learning_root: Path):
    _seed_manager(learning_root)
    cfg = MelodyConfig(
        learning_profile="p_melody",
        learning_storage=learning_root / "melodies",
        use_ml_model=False,
    )
    gen = MLMelodyGenerator(cfg)
    out = gen.generate("joy", length=8, key="C", mode="major")
    assert len(out.notes) == 8
    assert out.method == "learned_plus_ml_fallback"


def test_bass_generator_learning(learning_root: Path):
    _seed_manager(learning_root)
    line = generate_bass_line(
        ["C", "G"],
        pattern=BassPattern.ROOT_FIFTH,
        emotion="joy",
        learning_profile="p_bass",
        learning_storage=learning_root,
    )
    assert line.notes


def test_arrangement_generator_learning_metadata(learning_root: Path):
    _seed_manager(learning_root)
    gen = ArrangementGenerator()
    arr = gen.generate(
        "pop",
        "joy",
        learning_profile="p_arr",
        learning_storage=learning_root,
    )
    assert "sections" in arr.learning_metadata


def test_dynamics_expression_learning(learning_root: Path):
    _seed_manager(learning_root)
    dyn = DynamicsIntegration()
    curve = dyn.apply_expression_learning(
        "joy",
        learning_profile="p_expr",
        learning_storage=learning_root,
    )
    assert curve is not None
    assert len(curve) == 16
