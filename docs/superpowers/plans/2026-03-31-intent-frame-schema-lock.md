# IntentFrame Schema Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish `shared_schemas/intent_frame_schema.json` as the single source-of-truth IntentFrame contract with all roadmap fields (timestamp, DSP targets with confidence, music hints, latency budget), codegen to TS/Rust, and parity tests.

**Architecture:** Pydantic models for 8 sub-structs + 2 top-level fields → `sync_entities.py` generates JSON Schema + TypeScript + Rust. Golden fixtures validate Python and Rust agree. Existing `intent_ir` FFI types untouched — new fields exist only in the schema contract.

**Tech Stack:** Pydantic v2, JSON Schema, sync_entities.py codegen, pytest, cargo test

---

### Task 1: Pydantic Models for IntentFrame

**Files:**
- Modify: `music_brain/engine_api/schema.py`
- Create: `tests/unit/test_intent_frame_schema.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_intent_frame_schema.py`:

```python
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
    """DSP defaults must be deterministic safe values."""
    d = DSPTargetsSchema()
    assert d.filter_cutoff == 0.5    # mid-open
    assert d.reverb_send == 0.2      # subtle
    assert d.drive == 0.0            # off
    assert d.stale is True           # not yet valid
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/seanburdges/Dev/KmiDi && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest tests/unit/test_intent_frame_schema.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the Pydantic models**

Add to `music_brain/engine_api/schema.py` AFTER `EmotionStateSchema` and BEFORE `TrackIntent`:

```python
class IntentMetaSchema(BaseModel):
    """Intent metadata — version and routing IDs."""
    model_config = {"extra": "forbid"}

    schema_version: int = Field(default=1, description="Schema version")
    intent_id: int = Field(default=0, ge=0, description="Monotonic intent ID")
    session_id: int = Field(default=0, ge=0, description="Session ID")

    @field_validator("schema_version")
    @classmethod
    def validate_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Unsupported schema version: {v}")
        return v


class MusicalIntentSchema(BaseModel):
    """Musical intent — biases and tendencies, no notes or MIDI."""
    model_config = {"extra": "forbid"}

    tempo_bias: float = Field(default=0.0, ge=-1.0, le=1.0)
    rhythmic_density: float = Field(default=0.5, ge=0.0, le=1.0)
    groove_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    harmonic_tension: float = Field(default=0.5, ge=0.0, le=1.0)
    harmonic_motion: float = Field(default=0.5, ge=0.0, le=1.0)
    mode_preference: int = Field(default=0, ge=-1, le=1)
    melodic_activity: float = Field(default=0.5, ge=0.0, le=1.0)
    contour_variance: float = Field(default=0.5, ge=0.0, le=1.0)
    dynamic_range: float = Field(default=0.5, ge=0.0, le=1.0)
    texture_density: float = Field(default=0.5, ge=0.0, le=1.0)


class SectionRole(str, Enum):
    INTRO = "intro"
    VERSE = "verse"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    OUTRO = "outro"
    BUILD = "build"
    DROP = "drop"
    UNSPECIFIED = ""


class MusicHintsSchema(BaseModel):
    """Music hints — key, tempo, chord bias, section role."""
    model_config = {"extra": "forbid"}

    key: str = Field(default="", max_length=3, description="Key (e.g. 'C', 'F#', '' = unspecified)")
    tempo_bpm: float = Field(default=0.0, ge=0.0, description="Tempo BPM (0 = unspecified)")
    chord_bias: str = Field(default="", max_length=32, description="Chord bias (e.g. 'minor7')")
    section_role: SectionRole = Field(default=SectionRole.UNSPECIFIED, description="Section role")


class DSPTargetsSchema(BaseModel):
    """DSP targets with per-parameter confidence and stale flag.

    Safe defaults: filter mid-open, reverb subtle, drive off, stale=True.
    When stale=True or all confidences are 0, consumers must hold
    last-known-good or use these defaults.
    """
    model_config = {"extra": "forbid"}

    filter_cutoff: float = Field(default=0.5, ge=0.0, le=1.0)
    filter_cutoff_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reverb_send: float = Field(default=0.2, ge=0.0, le=1.0)
    reverb_send_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    drive: float = Field(default=0.0, ge=0.0, le=1.0)
    drive_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    stale: bool = Field(default=True, description="True if DSP values are not yet valid")


class TimeScopeSchema(BaseModel):
    """Time scope — intent without time is noise."""
    model_config = {"extra": "forbid"}

    start_bar: int = Field(default=-1, description="Start bar (-1 = immediate)")
    end_bar: int = Field(default=-1, description="End bar (-1 = open-ended)")
    fade_in_beats: float = Field(default=0.0, ge=0.0)
    fade_out_beats: float = Field(default=0.0, ge=0.0)

    @field_validator("end_bar")
    @classmethod
    def validate_time_scope(cls, end_bar: int, info) -> int:
        start_bar = info.data.get("start_bar", -1)
        if end_bar != -1 and start_bar != -1 and end_bar <= start_bar:
            raise ValueError(f"end_bar ({end_bar}) must be > start_bar ({start_bar})")
        return end_bar


class IntentConstraintsSchema(BaseModel):
    """Intent constraints — limit generation, not force it."""
    model_config = {"extra": "forbid"}

    allowed_engines_mask: int = Field(default=0xFFFFFFFF, ge=0)
    forbidden_engines_mask: int = Field(default=0, ge=0)
    max_cpu_cost: float = Field(default=1.0, ge=0.0)
    max_event_rate: float = Field(default=1000.0, ge=0.0)


class IntentProvenanceSchema(BaseModel):
    """Intent provenance — debugging and trust."""
    model_config = {"extra": "forbid"}

    source: int = Field(default=0, ge=0, le=5)
    user_override_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class IntentFrameSchema(BaseModel):
    """IntentFrame — top-level unit representing one musical intention.

    Canonical contract v1. Source of truth for all language bindings.
    JSON is for validation/codegen/fixtures only — RT hot path uses
    C structs directly (no JSON parsing on audio thread).
    """
    model_config = {"extra": "forbid"}

    meta: IntentMetaSchema = Field(default_factory=IntentMetaSchema)
    timestamp_ms: int = Field(default=0, ge=0, description="Monotonic ms since session start")
    emotion: EmotionStateSchema = Field(default_factory=EmotionStateSchema)
    music: MusicalIntentSchema = Field(default_factory=MusicalIntentSchema)
    music_hints: MusicHintsSchema = Field(default_factory=MusicHintsSchema)
    dsp_targets: DSPTargetsSchema = Field(default_factory=DSPTargetsSchema)
    time: TimeScopeSchema = Field(default_factory=TimeScopeSchema)
    constraints: IntentConstraintsSchema = Field(default_factory=IntentConstraintsSchema)
    provenance: IntentProvenanceSchema = Field(default_factory=IntentProvenanceSchema)
    latency_budget_ms: float = Field(default=10.0, ge=0.0, description="Max ms for RT engine")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/seanburdges/Dev/KmiDi && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest tests/unit/test_intent_frame_schema.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/seanburdges/Dev/KmiDi
git add music_brain/engine_api/schema.py tests/unit/test_intent_frame_schema.py
git commit -m "feat: add IntentFrame Pydantic models with DSP targets, music hints, latency budget

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Codegen — sync_entities.py for IntentFrame

**Files:**
- Modify: `scripts/sync_entities.py`
- Create: `shared_schemas/intent_frame_schema.json` (generated)
- Create: `src/types/IntentFrame.ts` (generated)
- Create: `src-tauri/src/generated/intent_frame.rs` (generated)
- Modify: `src-tauri/src/generated/mod.rs`
- Create: `tests/unit/test_sync_intent_frame.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_sync_intent_frame.py`:

```python
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_intent_frame_schema_json_exists_after_sync():
    import subprocess
    result = subprocess.run(
        ["/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13",
         "scripts/sync_entities.py"],
        cwd=str(ROOT), capture_output=True, text=True
    )
    assert result.returncode == 0, f"sync_entities.py failed: {result.stderr}"

    schema_path = ROOT / "shared_schemas" / "intent_frame_schema.json"
    assert schema_path.exists(), "intent_frame_schema.json not generated"

    schema = json.loads(schema_path.read_text())
    assert schema["title"] == "IntentFrameSchema"
    props = schema.get("properties", {})
    assert "meta" in props
    assert "timestamp_ms" in props
    assert "emotion" in props
    assert "music" in props
    assert "music_hints" in props
    assert "dsp_targets" in props
    assert "time" in props
    assert "constraints" in props
    assert "provenance" in props
    assert "latency_budget_ms" in props


def test_intent_frame_ts_exists_after_sync():
    ts_path = ROOT / "src" / "types" / "IntentFrame.ts"
    assert ts_path.exists(), "IntentFrame.ts not generated"
    content = ts_path.read_text()
    assert "IntentFrame" in content
    assert "DSPTargets" in content
    assert "MusicHints" in content
    assert "latency_budget_ms" in content


def test_intent_frame_rust_exists_after_sync():
    rs_path = ROOT / "src-tauri" / "src" / "generated" / "intent_frame.rs"
    assert rs_path.exists(), "intent_frame.rs not generated"
    content = rs_path.read_text()
    assert "IntentFrame" in content
    assert "DSPTargets" in content
    assert "MusicHints" in content
    assert "latency_budget_ms" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/seanburdges/Dev/KmiDi && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest tests/unit/test_sync_intent_frame.py -v`
Expected: FAIL

- [ ] **Step 3: Extend sync_entities.py**

Read `scripts/sync_entities.py` fully first. Add import, paths, rendering functions, and `sync_intent_frame()` following the same pattern as `sync_emotion()`. The rendering functions must:

1. Handle nested `$ref` to sub-structs in `$defs`
2. Skip `EmotionStateSchema` and `EmotionTag` (already generated, imported)
3. Strip "Schema" suffix from type names in TS/Rust output
4. Add `#[serde(deny_unknown_fields)]` to all Rust structs
5. Add `validate()` impl to the root `IntentFrame` struct checking: schema_version==1, tempo_bias range, source range, user_override_weight range, time scope ordering
6. Handle `SectionRole` enum generation
7. Add `pub mod intent_frame;` to `src-tauri/src/generated/mod.rs`
8. Call `sync_intent_frame()` from `__main__`

For the TypeScript output, import `EmotionState` and `EmotionTag` from `"./EmotionState"`.
For the Rust output, import from `super::emotion::{EmotionState, EmotionTag}`.

- [ ] **Step 4: Run sync and verify**

Run: `cd /Users/seanburdges/Dev/KmiDi && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 scripts/sync_entities.py`
Then: `cd /Users/seanburdges/Dev/KmiDi && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest tests/unit/test_sync_intent_frame.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/seanburdges/Dev/KmiDi
git add scripts/sync_entities.py shared_schemas/intent_frame_schema.json \
    src/types/IntentFrame.ts src-tauri/src/generated/intent_frame.rs \
    src-tauri/src/generated/mod.rs tests/unit/test_sync_intent_frame.py
git commit -m "feat: extend sync_entities.py for IntentFrame codegen (JSON, TS, Rust)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Golden Fixtures

**Files:**
- Create: 7 files in `tests/fixtures/intent/`

- [ ] **Step 1: Create valid fixtures**

`tests/fixtures/intent/frame_valid_default.json`:
```json
{
  "meta": {"schema_version": 1, "intent_id": 0, "session_id": 0},
  "timestamp_ms": 0,
  "emotion": {"valence": 0.0, "arousal": 0.5, "dominance": 0.5, "confidence": 0.0},
  "music": {"tempo_bias": 0.0, "rhythmic_density": 0.5, "groove_strength": 0.5, "harmonic_tension": 0.5, "harmonic_motion": 0.5, "mode_preference": 0, "melodic_activity": 0.5, "contour_variance": 0.5, "dynamic_range": 0.5, "texture_density": 0.5},
  "music_hints": {"key": "", "tempo_bpm": 0.0, "chord_bias": "", "section_role": ""},
  "dsp_targets": {"filter_cutoff": 0.5, "filter_cutoff_confidence": 0.0, "reverb_send": 0.2, "reverb_send_confidence": 0.0, "drive": 0.0, "drive_confidence": 0.0, "stale": true},
  "time": {"start_bar": -1, "end_bar": -1, "fade_in_beats": 0.0, "fade_out_beats": 0.0},
  "constraints": {"allowed_engines_mask": 4294967295, "forbidden_engines_mask": 0, "max_cpu_cost": 1.0, "max_event_rate": 1000.0},
  "provenance": {"source": 0, "user_override_weight": 0.5},
  "latency_budget_ms": 10.0
}
```

`tests/fixtures/intent/frame_valid_full.json`:
```json
{
  "meta": {"schema_version": 1, "intent_id": 42, "session_id": 100},
  "timestamp_ms": 5000,
  "emotion": {"valence": 0.8, "arousal": 0.9, "dominance": 0.7, "tags": ["bright", "drive"], "confidence": 0.9},
  "music": {"tempo_bias": 0.3, "rhythmic_density": 0.8, "groove_strength": 0.7, "harmonic_tension": 0.6, "harmonic_motion": 0.4, "mode_preference": 1, "melodic_activity": 0.9, "contour_variance": 0.3, "dynamic_range": 0.7, "texture_density": 0.6},
  "music_hints": {"key": "C", "tempo_bpm": 120.0, "chord_bias": "minor7", "section_role": "chorus"},
  "dsp_targets": {"filter_cutoff": 0.8, "filter_cutoff_confidence": 0.9, "reverb_send": 0.4, "reverb_send_confidence": 0.85, "drive": 0.3, "drive_confidence": 0.7, "stale": false},
  "time": {"start_bar": 1, "end_bar": 16, "fade_in_beats": 2.0, "fade_out_beats": 4.0},
  "constraints": {"allowed_engines_mask": 15, "forbidden_engines_mask": 0, "max_cpu_cost": 0.8, "max_event_rate": 500.0},
  "provenance": {"source": 1, "user_override_weight": 0.7},
  "latency_budget_ms": 5.0
}
```

`tests/fixtures/intent/frame_valid_ml_audio.json`:
```json
{
  "meta": {"schema_version": 1, "intent_id": 7, "session_id": 3},
  "timestamp_ms": 12500,
  "emotion": {"valence": -0.3, "arousal": 0.8, "dominance": 0.4, "tags": ["tension"], "confidence": 0.3},
  "music": {"tempo_bias": -0.2, "rhythmic_density": 0.6, "groove_strength": 0.5, "harmonic_tension": 0.7, "harmonic_motion": 0.5, "mode_preference": -1, "melodic_activity": 0.4, "contour_variance": 0.5, "dynamic_range": 0.5, "texture_density": 0.5},
  "music_hints": {"key": "Am", "tempo_bpm": 90.0, "chord_bias": "", "section_role": "verse"},
  "dsp_targets": {"filter_cutoff": 0.5, "filter_cutoff_confidence": 0.0, "reverb_send": 0.2, "reverb_send_confidence": 0.0, "drive": 0.0, "drive_confidence": 0.0, "stale": true},
  "time": {"start_bar": -1, "end_bar": -1, "fade_in_beats": 0.0, "fade_out_beats": 0.0},
  "constraints": {"allowed_engines_mask": 4294967295, "forbidden_engines_mask": 0, "max_cpu_cost": 1.0, "max_event_rate": 1000.0},
  "provenance": {"source": 3, "user_override_weight": 0.2},
  "latency_budget_ms": 8.0
}
```

- [ ] **Step 2: Create invalid fixtures**

`tests/fixtures/intent/frame_invalid_version.json` — same as default but `"schema_version": 99`

`tests/fixtures/intent/frame_invalid_tempo_oob.json` — same as default but `"tempo_bias": 5.0`

`tests/fixtures/intent/frame_invalid_time_scope.json` — same as default but `"start_bar": 5, "end_bar": 2`

`tests/fixtures/intent/frame_invalid_extra_field.json` — same as default plus `"unknown_field": "bad"` at top level

- [ ] **Step 3: Commit**

```bash
cd /Users/seanburdges/Dev/KmiDi
git add tests/fixtures/intent/frame_*.json
git commit -m "feat: add 7 golden IntentFrame fixtures (3 valid, 4 invalid)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Python Parity Tests Against Fixtures

**Files:**
- Modify: `tests/unit/test_intent_frame_schema.py`

- [ ] **Step 1: Append fixture-based tests**

Append to `tests/unit/test_intent_frame_schema.py`:

```python
import json
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "intent"

VALID_FRAME_FIXTURES = [
    "frame_valid_default.json",
    "frame_valid_full.json",
    "frame_valid_ml_audio.json",
]

INVALID_FRAME_FIXTURES = [
    "frame_invalid_version.json",
    "frame_invalid_tempo_oob.json",
    "frame_invalid_time_scope.json",
    "frame_invalid_extra_field.json",
]


@pytest.mark.parametrize("fixture_name", VALID_FRAME_FIXTURES)
def test_valid_frame_fixture_accepted(fixture_name):
    data = json.loads((FIXTURE_DIR / fixture_name).read_text())
    f = IntentFrameSchema(**data)
    assert f.meta.schema_version == 1
    assert f.timestamp_ms >= 0
    assert -1.0 <= f.music.tempo_bias <= 1.0
    assert 0.0 <= f.dsp_targets.filter_cutoff <= 1.0
    assert 0.0 <= f.latency_budget_ms


@pytest.mark.parametrize("fixture_name", INVALID_FRAME_FIXTURES)
def test_invalid_frame_fixture_rejected(fixture_name):
    data = json.loads((FIXTURE_DIR / fixture_name).read_text())
    with pytest.raises(Exception):
        IntentFrameSchema(**data)
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/seanburdges/Dev/KmiDi && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest tests/unit/test_intent_frame_schema.py -v`
Expected: All 19 tests PASS (12 unit + 3 valid fixtures + 4 invalid fixtures)

- [ ] **Step 3: Commit**

```bash
cd /Users/seanburdges/Dev/KmiDi
git add tests/unit/test_intent_frame_schema.py
git commit -m "test: add fixture-based parity tests for IntentFrame schema (Python)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Rust Parity Tests

**Files:**
- Create: `src-tauri/tests/test_intent_frame_schema.rs`

- [ ] **Step 1: Write the Rust parity test**

Create `src-tauri/tests/test_intent_frame_schema.rs`:

```rust
use std::fs;
use std::path::PathBuf;

use idaw_lib::generated::intent_frame::IntentFrame;

fn fixture_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.push("tests/fixtures/intent");
    p
}

fn try_parse(name: &str) -> Result<IntentFrame, serde_json::Error> {
    let path = fixture_dir().join(name);
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read fixture {}: {}", name, e));
    serde_json::from_str(&text)
}

#[test]
fn valid_default() {
    let f = try_parse("frame_valid_default.json").expect("Should parse");
    assert_eq!(f.meta.schema_version, 1);
    assert_eq!(f.timestamp_ms, 0);
    assert!(f.dsp_targets.stale);
    assert!(f.validate().is_ok());
}

#[test]
fn valid_full() {
    let f = try_parse("frame_valid_full.json").expect("Should parse");
    assert_eq!(f.meta.intent_id, 42);
    assert_eq!(f.timestamp_ms, 5000);
    assert!(!f.dsp_targets.stale);
    assert!((f.dsp_targets.filter_cutoff_confidence - 0.9).abs() < 0.01);
    assert!(f.validate().is_ok());
}

#[test]
fn valid_ml_audio() {
    let f = try_parse("frame_valid_ml_audio.json").expect("Should parse");
    assert_eq!(f.provenance.source, 3);
    assert!(f.dsp_targets.stale);
    assert!(f.validate().is_ok());
}

#[test]
fn invalid_version_rejected_by_validate() {
    let f = try_parse("frame_invalid_version.json").expect("serde parses it");
    assert!(f.validate().is_err());
}

#[test]
fn invalid_tempo_oob_rejected_by_validate() {
    let result = try_parse("frame_invalid_tempo_oob.json");
    if let Ok(f) = result {
        assert!(f.validate().is_err());
    }
}

#[test]
fn invalid_time_scope_rejected_by_validate() {
    let f = try_parse("frame_invalid_time_scope.json").expect("serde parses it");
    assert!(f.validate().is_err());
}

#[test]
fn invalid_extra_field() {
    let result = try_parse("frame_invalid_extra_field.json");
    assert!(result.is_err(), "deny_unknown_fields should reject");
}
```

- [ ] **Step 2: Run the Rust tests**

Run: `cd /Users/seanburdges/Dev/KmiDi/src-tauri && cargo test --test test_intent_frame_schema -- --nocapture`
Expected: All 7 tests PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/seanburdges/Dev/KmiDi
git add src-tauri/tests/test_intent_frame_schema.rs
git commit -m "test: add Rust parity tests for IntentFrame schema fixtures

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Final Verification

**Files:** None new

- [ ] **Step 1: Run full Python test suite**

Run: `cd /Users/seanburdges/Dev/KmiDi && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest tests/unit/test_intent_frame_schema.py tests/unit/test_sync_intent_frame.py tests/unit/test_emotion_schema.py tests/unit/test_sync_emotion.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run Rust tests**

Run: `cd /Users/seanburdges/Dev/KmiDi/src-tauri && cargo test --test test_intent_frame_schema --test test_emotion_schema -- --nocapture`
Expected: All PASS

- [ ] **Step 3: Verify fixture count and schema**

Run: `ls /Users/seanburdges/Dev/KmiDi/tests/fixtures/intent/frame_*.json | wc -l`
Expected: 7

Run: `cd /Users/seanburdges/Dev/KmiDi && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -c "import json; s=json.load(open('shared_schemas/intent_frame_schema.json')); print('Props:', list(s['properties'].keys()))"`
Expected: `['meta', 'timestamp_ms', 'emotion', 'music', 'music_hints', 'dsp_targets', 'time', 'constraints', 'provenance', 'latency_budget_ms']`
