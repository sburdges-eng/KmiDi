# Emotion Schema Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish `shared_schemas/emotion_schema.json` as the single source-of-truth emotion contract, with codegen to TS/Rust and parity tests across Python, Rust, and C++.

**Architecture:** Pydantic model → JSON Schema → sync_entities.py generates TypeScript interface and Rust serde struct. Golden fixtures validate all languages agree. The `intent_ir` Rust crate is `no_std`/`repr(C)` so generated Rust targets `engine/intent_ir/` instead.

**Tech Stack:** Pydantic v2, JSON Schema draft 2020-12, sync_entities.py codegen, pytest, Catch2, cargo test

---

### Task 1: Pydantic EmotionState Model

**Files:**
- Modify: `music_brain/engine_api/schema.py`
- Test: `tests/unit/test_emotion_schema.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_emotion_schema.py`:

```python
import pytest
from music_brain.engine_api.schema import EmotionStateSchema


def test_valid_neutral():
    e = EmotionStateSchema(valence=0.0, arousal=0.5, dominance=0.5, confidence=0.5)
    assert e.valence == 0.0
    assert e.arousal == 0.5
    assert e.dominance == 0.5
    assert e.tags == []
    assert e.confidence == 0.5


def test_valid_with_tags():
    e = EmotionStateSchema(
        valence=0.8, arousal=0.9, dominance=0.7,
        tags=["bright", "drive"], confidence=0.9
    )
    assert len(e.tags) == 2


def test_invalid_valence_out_of_range():
    with pytest.raises(Exception):
        EmotionStateSchema(valence=2.0, arousal=0.5, dominance=0.5, confidence=0.5)


def test_invalid_unknown_tag():
    with pytest.raises(Exception):
        EmotionStateSchema(
            valence=0.0, arousal=0.5, dominance=0.5,
            tags=["angry"], confidence=0.5
        )


def test_invalid_too_many_tags():
    with pytest.raises(Exception):
        EmotionStateSchema(
            valence=0.0, arousal=0.5, dominance=0.5,
            tags=["tension", "release", "warm", "cold"], confidence=0.5
        )


def test_invalid_extra_field():
    with pytest.raises(Exception):
        EmotionStateSchema(
            valence=0.0, arousal=0.5, dominance=0.5,
            confidence=0.5, intensity=0.5
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_emotion_schema.py -v`
Expected: FAIL with `ImportError: cannot import name 'EmotionStateSchema'`

- [ ] **Step 3: Write the Pydantic model**

Add to `music_brain/engine_api/schema.py` (after existing imports, before `TrackIntent`):

```python
from enum import Enum
from typing import List

class EmotionTag(str, Enum):
    TENSION = "tension"
    RELEASE = "release"
    WARM = "warm"
    COLD = "cold"
    BRIGHT = "bright"
    DARK = "dark"
    DRIVE = "drive"
    FLOAT = "float"


class EmotionStateSchema(BaseModel):
    """Canonical emotion contract v1. Source of truth for all language bindings."""
    model_config = {"extra": "forbid"}

    valence: float = Field(default=0.0, ge=-1.0, le=1.0, description="Negative to positive [-1, 1]")
    arousal: float = Field(default=0.5, ge=0.0, le=1.0, description="Calm to excited [0, 1]")
    dominance: float = Field(default=0.5, ge=0.0, le=1.0, description="Submissive to dominant [0, 1]")
    tags: List[EmotionTag] = Field(
        default_factory=list,
        max_length=3,
        description="Max 3 tags from controlled vocabulary",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Inference quality gate [0, 1]")

    @field_validator("tags")
    @classmethod
    def validate_unique_tags(cls, v: List[EmotionTag]) -> List[EmotionTag]:
        if len(v) != len(set(v)):
            raise ValueError("Tags must be unique")
        return v
```

Note: The class is named `EmotionStateSchema` (not `EmotionState`) to avoid colliding with the existing dataclass in `music_brain/intent_ir/__init__.py`. The `Enum` import may already be present — check before adding a duplicate.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_emotion_schema.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add music_brain/engine_api/schema.py tests/unit/test_emotion_schema.py
git commit -m "feat: add EmotionStateSchema Pydantic model with validation"
```

---

### Task 2: JSON Schema Generation via sync_entities.py

**Files:**
- Modify: `scripts/sync_entities.py`
- Create: `shared_schemas/emotion_schema.json` (generated)
- Create: `src/types/EmotionState.ts` (generated)
- Create: `engine/intent_ir/src/generated/emotion.rs` (generated)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_sync_emotion.py`:

```python
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_emotion_schema_json_exists_after_sync():
    """After sync, emotion_schema.json must exist and be valid JSON Schema."""
    import subprocess
    result = subprocess.run(
        ["python3", "scripts/sync_entities.py"],
        cwd=str(ROOT), capture_output=True, text=True
    )
    assert result.returncode == 0, f"sync_entities.py failed: {result.stderr}"

    schema_path = ROOT / "shared_schemas" / "emotion_schema.json"
    assert schema_path.exists(), "emotion_schema.json not generated"

    schema = json.loads(schema_path.read_text())
    assert schema["title"] == "EmotionStateSchema"
    assert "valence" in schema.get("properties", {})
    assert "arousal" in schema.get("properties", {})
    assert "dominance" in schema.get("properties", {})
    assert "confidence" in schema.get("properties", {})
    assert "tags" in schema.get("properties", {})


def test_emotion_ts_exists_after_sync():
    ts_path = ROOT / "src" / "types" / "EmotionState.ts"
    assert ts_path.exists(), "EmotionState.ts not generated"
    content = ts_path.read_text()
    assert "valence" in content
    assert "arousal" in content


def test_emotion_rust_exists_after_sync():
    rs_path = ROOT / "engine/intent_ir" / "src" / "generated" / "emotion.rs"
    assert rs_path.exists(), "emotion.rs not generated"
    content = rs_path.read_text()
    assert "valence" in content
    assert "arousal" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_sync_emotion.py -v`
Expected: FAIL — `emotion_schema.json not generated`

- [ ] **Step 3: Extend sync_entities.py**

Add these lines to `scripts/sync_entities.py`:

After the existing import on line 15, add:

```python
from music_brain.engine_api.schema import EmotionStateSchema
```

After the existing path definitions (line 22), add:

```python
EMOTION_SCHEMA_PATH = SCHEMA_DIR / "emotion_schema.json"
EMOTION_TS_OUT = ROOT / "src" / "types" / "EmotionState.ts"
EMOTION_RUST_OUT = ROOT / "engine/intent_ir" / "src" / "generated" / "emotion.rs"
```

After the existing `_render_rust` function (line 156), add:

```python
def _emotion_schema() -> dict:
    if hasattr(EmotionStateSchema, "model_json_schema"):
        return EmotionStateSchema.model_json_schema()
    return EmotionStateSchema.schema()


def _render_emotion_typescript(schema: dict) -> str:
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines = [
        "/* AUTO-GENERATED by scripts/sync_entities.py. Do not edit manually. */",
        "",
    ]

    # Emit the tag enum
    tag_items = props.get("tags", {}).get("items", {})
    tag_ref = tag_items.get("$ref", "")
    tag_enum_values = None

    # Pydantic v2 puts enums in $defs
    defs = schema.get("$defs", schema.get("definitions", {}))
    for def_name, def_node in defs.items():
        if "enum" in def_node:
            tag_enum_values = def_node["enum"]
            lines.append(f"export type {def_name} = {' | '.join(repr(v) for v in tag_enum_values)};")
            lines.append("")

    lines.append("export interface EmotionState {")
    for key, value in props.items():
        optional = "" if key in required else "?"
        desc = value.get("description", "")
        if desc:
            lines.append(f"  /** {desc} */")
        if key == "tags":
            lines.append(f"  {key}{optional}: EmotionTag[];")
        elif value.get("type") == "number":
            lines.append(f"  {key}{optional}: number;")
        else:
            lines.append(f"  {key}{optional}: {_json_to_ts(key, value, schema)};")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _render_emotion_rust(schema: dict) -> str:
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines = [
        "// AUTO-GENERATED by scripts/sync_entities.py. Do not edit manually.",
        "use serde::{Deserialize, Serialize};",
        "",
    ]

    # Emit tag enum
    defs = schema.get("$defs", schema.get("definitions", {}))
    for def_name, def_node in defs.items():
        if "enum" in def_node:
            lines.append("#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]")
            lines.append(f"pub enum {def_name} {{")
            for val in def_node["enum"]:
                variant = val.capitalize()
                lines.append(f'    #[serde(rename = "{val}")]')
                lines.append(f"    {variant},")
            lines.append("}")
            lines.append("")

    lines.append("#[derive(Debug, Clone, Serialize, Deserialize)]")
    lines.append("#[serde(deny_unknown_fields)]")
    lines.append("pub struct EmotionState {")
    for key, value in props.items():
        is_required = key in required
        if key == "tags":
            lines.append("    #[serde(default)]")
            lines.append("    pub tags: Vec<EmotionTag>,")
        else:
            field_type = _json_to_rust_type(value, is_required)
            if not is_required:
                lines.append("    #[serde(default)]")
            lines.append(f"    pub {key}: {field_type},")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def sync_emotion() -> None:
    schema = _emotion_schema()
    schema_payload = json.dumps(schema, indent=2) + "\n"
    EMOTION_SCHEMA_PATH.write_text(schema_payload, encoding="utf-8")
    EMOTION_TS_OUT.write_text(_render_emotion_typescript(schema), encoding="utf-8")
    EMOTION_RUST_OUT.write_text(_render_emotion_rust(schema), encoding="utf-8")

    print("Emotion schema exported to:", EMOTION_SCHEMA_PATH)
    print("Emotion TypeScript contract written to:", EMOTION_TS_OUT)
    print("Emotion Rust contract written to:", EMOTION_RUST_OUT)
```

Update the `__main__` block (line 180-181) to:

```python
if __name__ == "__main__":
    sync_boundaries()
    sync_emotion()
```

- [ ] **Step 4: Run sync and verify outputs**

Run: `python3 scripts/sync_entities.py`
Expected: Prints paths for emotion schema, TS, and Rust files

Run: `python3 -m pytest tests/unit/test_sync_emotion.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Verify generated files look correct**

Manually inspect:
- `shared_schemas/emotion_schema.json` — should have all 5 fields with correct ranges
- `src/types/EmotionState.ts` — should export `EmotionState` interface and `EmotionTag` type
- `engine/intent_ir/src/generated/emotion.rs` — should have `EmotionState` struct and `EmotionTag` enum

- [ ] **Step 6: Commit**

```bash
git add scripts/sync_entities.py shared_schemas/emotion_schema.json \
    src/types/EmotionState.ts engine/intent_ir/src/generated/emotion.rs \
    tests/unit/test_sync_emotion.py
git commit -m "feat: extend sync_entities.py to generate emotion schema, TS, and Rust"
```

---

### Task 3: Golden Fixtures

**Files:**
- Create: `tests/fixtures/intent/emotion_valid_neutral.json`
- Create: `tests/fixtures/intent/emotion_valid_excited.json`
- Create: `tests/fixtures/intent/emotion_valid_sad.json`
- Create: `tests/fixtures/intent/emotion_valid_max_tags.json`
- Create: `tests/fixtures/intent/emotion_valid_no_tags.json`
- Create: `tests/fixtures/intent/emotion_invalid_valence_oob.json`
- Create: `tests/fixtures/intent/emotion_invalid_tag_unknown.json`
- Create: `tests/fixtures/intent/emotion_invalid_too_many_tags.json`
- Create: `tests/fixtures/intent/emotion_invalid_extra_field.json`

- [ ] **Step 1: Create valid fixtures**

`tests/fixtures/intent/emotion_valid_neutral.json`:
```json
{"valence": 0.0, "arousal": 0.5, "dominance": 0.5, "confidence": 0.5}
```

`tests/fixtures/intent/emotion_valid_excited.json`:
```json
{"valence": 0.8, "arousal": 0.9, "dominance": 0.7, "tags": ["bright", "drive"], "confidence": 0.9}
```

`tests/fixtures/intent/emotion_valid_sad.json`:
```json
{"valence": -0.7, "arousal": 0.2, "dominance": 0.3, "tags": ["cold", "dark"], "confidence": 0.8}
```

`tests/fixtures/intent/emotion_valid_max_tags.json`:
```json
{"valence": 0.0, "arousal": 0.5, "dominance": 0.5, "tags": ["tension", "warm", "drive"], "confidence": 0.6}
```

`tests/fixtures/intent/emotion_valid_no_tags.json`:
```json
{"valence": 0.0, "arousal": 0.5, "dominance": 0.5, "confidence": 0.5}
```

- [ ] **Step 2: Create invalid fixtures**

`tests/fixtures/intent/emotion_invalid_valence_oob.json`:
```json
{"valence": 2.0, "arousal": 0.5, "dominance": 0.5, "confidence": 0.5}
```

`tests/fixtures/intent/emotion_invalid_tag_unknown.json`:
```json
{"valence": 0.0, "arousal": 0.5, "dominance": 0.5, "tags": ["angry"], "confidence": 0.5}
```

`tests/fixtures/intent/emotion_invalid_too_many_tags.json`:
```json
{"valence": 0.0, "arousal": 0.5, "dominance": 0.5, "tags": ["tension", "release", "warm", "cold"], "confidence": 0.5}
```

`tests/fixtures/intent/emotion_invalid_extra_field.json`:
```json
{"valence": 0.0, "arousal": 0.5, "dominance": 0.5, "confidence": 0.5, "intensity": 0.5}
```

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/intent/
git commit -m "feat: add 9 golden emotion schema fixtures (5 valid, 4 invalid)"
```

---

### Task 4: Python Parity Tests Against Fixtures

**Files:**
- Modify: `tests/unit/test_emotion_schema.py`

- [ ] **Step 1: Add fixture-based parity tests**

Append to `tests/unit/test_emotion_schema.py`:

```python
import json
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "intent"

VALID_FIXTURES = [
    "emotion_valid_neutral.json",
    "emotion_valid_excited.json",
    "emotion_valid_sad.json",
    "emotion_valid_max_tags.json",
    "emotion_valid_no_tags.json",
]

INVALID_FIXTURES = [
    "emotion_invalid_valence_oob.json",
    "emotion_invalid_tag_unknown.json",
    "emotion_invalid_too_many_tags.json",
    "emotion_invalid_extra_field.json",
]


@pytest.mark.parametrize("fixture_name", VALID_FIXTURES)
def test_valid_fixture_accepted(fixture_name):
    data = json.loads((FIXTURE_DIR / fixture_name).read_text())
    e = EmotionStateSchema(**data)
    assert -1.0 <= e.valence <= 1.0
    assert 0.0 <= e.arousal <= 1.0
    assert 0.0 <= e.dominance <= 1.0
    assert 0.0 <= e.confidence <= 1.0
    assert len(e.tags) <= 3


@pytest.mark.parametrize("fixture_name", INVALID_FIXTURES)
def test_invalid_fixture_rejected(fixture_name):
    data = json.loads((FIXTURE_DIR / fixture_name).read_text())
    with pytest.raises(Exception):
        EmotionStateSchema(**data)
```

- [ ] **Step 2: Run tests**

Run: `python3 -m pytest tests/unit/test_emotion_schema.py -v`
Expected: All 15 tests PASS (6 original + 5 valid fixtures + 4 invalid fixtures)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_emotion_schema.py
git commit -m "test: add fixture-based parity tests for emotion schema (Python)"
```

---

### Task 5: Rust Parity Tests

**Files:**
- Modify: `engine/intent_ir/src/generated/mod.rs` (add `pub mod emotion;`)
- Create: `engine/intent_ir/tests/test_emotion_schema.rs`

- [ ] **Step 1: Wire up the generated module**

Read `engine/intent_ir/src/generated/mod.rs` and add:

```rust
pub mod emotion;
```

- [ ] **Step 2: Write the Rust parity test**

Create `engine/intent_ir/tests/test_emotion_schema.rs`:

```rust
use serde_json;
use std::fs;
use std::path::PathBuf;

// Import the generated type
use kmidi_app::generated::emotion::{EmotionState, EmotionTag};

fn fixture_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // engine/intent_ir -> project root
    p.push("tests/fixtures/intent");
    p
}

fn load_fixture(name: &str) -> serde_json::Value {
    let path = fixture_dir().join(name);
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read fixture {}: {}", name, e));
    serde_json::from_str(&text).unwrap()
}

fn try_parse(name: &str) -> Result<EmotionState, serde_json::Error> {
    let path = fixture_dir().join(name);
    let text = fs::read_to_string(&path).unwrap();
    serde_json::from_str(&text)
}

#[test]
fn valid_neutral() {
    let e = try_parse("emotion_valid_neutral.json").expect("Should parse");
    assert!((e.valence - 0.0).abs() < f64::EPSILON);
    assert!((e.arousal - 0.5).abs() < f64::EPSILON);
    assert!(e.tags.is_empty());
}

#[test]
fn valid_excited() {
    let e = try_parse("emotion_valid_excited.json").expect("Should parse");
    assert_eq!(e.tags.len(), 2);
    assert!(e.tags.contains(&EmotionTag::Bright));
    assert!(e.tags.contains(&EmotionTag::Drive));
}

#[test]
fn valid_sad() {
    let e = try_parse("emotion_valid_sad.json").expect("Should parse");
    assert!(e.valence < 0.0);
    assert!(e.tags.contains(&EmotionTag::Cold));
}

#[test]
fn valid_max_tags() {
    let e = try_parse("emotion_valid_max_tags.json").expect("Should parse");
    assert_eq!(e.tags.len(), 3);
}

#[test]
fn valid_no_tags() {
    let e = try_parse("emotion_valid_no_tags.json").expect("Should parse");
    assert!(e.tags.is_empty());
}

#[test]
fn invalid_tag_unknown() {
    // serde will fail to deserialize unknown enum variant
    assert!(try_parse("emotion_invalid_tag_unknown.json").is_err());
}

#[test]
fn invalid_extra_field() {
    // Note: serde by default ignores unknown fields.
    // To enforce additionalProperties:false, we use #[serde(deny_unknown_fields)]
    // on the generated struct. If the generated code doesn't have this,
    // this test documents the gap.
    let result = try_parse("emotion_invalid_extra_field.json");
    assert!(result.is_err(), "Should reject extra fields");
}
```

Note: The generated Rust struct needs `#[serde(deny_unknown_fields)]` to reject extra fields. After running the test, if `invalid_extra_field` fails, update the `_render_emotion_rust` function in `sync_entities.py` to add this attribute.

- [ ] **Step 3: Run the Rust tests**

Run: `cd engine/intent_ir && cargo test test_emotion_schema -- --nocapture`
Expected: All 7 tests PASS (5 valid + 2 invalid that serde catches)

Note: `emotion_invalid_valence_oob.json` and `emotion_invalid_too_many_tags.json` will parse successfully via serde because serde doesn't enforce JSON Schema range constraints. These are documented gaps — range validation happens at the application layer (Rust `validate()` function), not at deserialization. The Python Pydantic model catches these at parse time. This is an acceptable divergence: serde parses, then `validate()` rejects.

- [ ] **Step 4: Add range validation function to generated Rust**

Update `_render_emotion_rust` in `sync_entities.py` to append a `validate()` impl after the struct:

```rust
impl EmotionState {
    pub fn validate(&self) -> Result<(), String> {
        if self.valence < -1.0 || self.valence > 1.0 {
            return Err(format!("valence {} out of range [-1.0, 1.0]", self.valence));
        }
        if self.arousal < 0.0 || self.arousal > 1.0 {
            return Err(format!("arousal {} out of range [0.0, 1.0]", self.arousal));
        }
        if self.dominance < 0.0 || self.dominance > 1.0 {
            return Err(format!("dominance {} out of range [0.0, 1.0]", self.dominance));
        }
        if self.confidence < 0.0 || self.confidence > 1.0 {
            return Err(format!("confidence {} out of range [0.0, 1.0]", self.confidence));
        }
        if self.tags.len() > 3 {
            return Err(format!("tags count {} exceeds max 3", self.tags.len()));
        }
        Ok(())
    }
}
```

Then add these tests to the test file:

```rust
#[test]
fn invalid_valence_oob_rejected_by_validate() {
    let e = try_parse("emotion_invalid_valence_oob.json").expect("serde parses it");
    assert!(e.validate().is_err());
}

#[test]
fn invalid_too_many_tags_rejected_by_validate() {
    let e = try_parse("emotion_invalid_too_many_tags.json").expect("serde parses it");
    assert!(e.validate().is_err());
}
```

- [ ] **Step 5: Re-run sync and Rust tests**

Run: `python3 scripts/sync_entities.py`
Run: `cd engine/intent_ir && cargo test test_emotion_schema -- --nocapture`
Expected: All 9 tests PASS

- [ ] **Step 6: Commit**

```bash
git add engine/intent_ir/src/generated/mod.rs engine/intent_ir/src/generated/emotion.rs \
    engine/intent_ir/tests/test_emotion_schema.rs scripts/sync_entities.py
git commit -m "test: add Rust parity tests for emotion schema fixtures"
```

---

### Task 6: C++ Parity Test

**Files:**
- Create: `tests/cpp/test_emotion_schema.cpp`
- Modify: `CMakeLists.txt` (add test target)

- [ ] **Step 1: Write the C++ test**

Create `tests/cpp/test_emotion_schema.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>

#include "penta/ml/AudioEmotionRunner.h"
#include "penta/common/RTState.h"

using namespace penta::ml;

// ─── EmotionResult field parity with schema ─────────────────────────────────
// These tests verify that the C++ RT-safe structs have fields matching
// the canonical emotion_schema.json contract.

TEST_CASE("EmotionResult has valence in [-1, 1]", "[EmotionSchema][parity]") {
    EmotionResult e;
    // Default should be in range
    REQUIRE(e.valence >= -1.0f);
    REQUIRE(e.valence <= 1.0f);

    // Boundary values
    e.valence = -1.0f;
    REQUIRE(e.valence == -1.0f);
    e.valence = 1.0f;
    REQUIRE(e.valence == 1.0f);
}

TEST_CASE("EmotionResult has arousal in [0, 1]", "[EmotionSchema][parity]") {
    EmotionResult e;
    REQUIRE(e.arousal >= 0.0f);
    REQUIRE(e.arousal <= 1.0f);
}

TEST_CASE("EmotionResult has dominance in [0, 1]", "[EmotionSchema][parity]") {
    EmotionResult e;
    REQUIRE(e.dominance >= 0.0f);
    REQUIRE(e.dominance <= 1.0f);
}

TEST_CASE("EmotionResult has confidence in [0, 1]", "[EmotionSchema][parity]") {
    EmotionResult e;
    REQUIRE(e.confidence >= 0.0f);
    REQUIRE(e.confidence <= 1.0f);
}

TEST_CASE("RTState emotion fields match schema", "[EmotionSchema][parity]") {
    penta::RTState state;

    // Verify all schema fields exist as atomics in RTState
    float v = state.valence.load();
    float a = state.arousal.load();
    float d = state.dominance.load();
    float c = state.emotionConfidence.load();

    // Defaults should be in valid ranges
    REQUIRE(v >= -1.0f);
    REQUIRE(v <= 1.0f);
    REQUIRE(a >= 0.0f);
    REQUIRE(a <= 1.0f);
    REQUIRE(d >= 0.0f);
    REQUIRE(d <= 1.0f);
    REQUIRE(c >= 0.0f);
    REQUIRE(c <= 1.0f);
}

TEST_CASE("EmotionResult defaults match schema defaults", "[EmotionSchema][parity]") {
    EmotionResult e;
    REQUIRE(e.valence == 0.0f);       // schema default: 0.0
    REQUIRE(e.arousal == 0.5f);       // schema default: 0.5
    REQUIRE(e.dominance == 0.5f);     // schema default: 0.5
    REQUIRE(e.confidence == 0.0f);    // schema default: 0.0
}
```

- [ ] **Step 2: Add CMake test target**

Add to `CMakeLists.txt` near the existing `AudioEmotionRunnerTests` target:

```cmake
if(BUILD_TESTS)
    add_executable(EmotionSchemaTests
        tests/cpp/test_emotion_schema.cpp
    )
    target_link_libraries(EmotionSchemaTests PRIVATE
        KellyCore
        Catch2::Catch2WithMain
    )
    target_include_directories(EmotionSchemaTests PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    catch_discover_tests(EmotionSchemaTests)
endif()
```

- [ ] **Step 3: Build and run**

Run:
```bash
cmake --build build --target EmotionSchemaTests -j8
ctest --test-dir build -R EmotionSchema --output-on-failure
```
Expected: All 6 tests PASS

Note: If Catch2 is not yet installed locally, this step will fail at build time. The test is written and ready — it will pass once `external/Catch2/` is populated.

- [ ] **Step 4: Commit**

```bash
git add tests/cpp/test_emotion_schema.cpp CMakeLists.txt
git commit -m "test: add C++ parity tests for emotion schema fields and ranges"
```

---

### Task 7: Deprecate Old Fields in Rust and Python

**Files:**
- Modify: `engine/intent_ir/src/types.rs`
- Modify: `music_brain/intent_ir/__init__.py`

- [ ] **Step 1: Add deprecation comments to Rust EmotionState**

In `engine/intent_ir/src/types.rs`, update the `EmotionState` struct (lines 24-33):

```rust
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EmotionState {
    pub valence: f32,      // [-1.0, 1.0]
    pub arousal: f32,      // [0.0, 1.0]
    pub dominance: f32,    // [0.0, 1.0]
    /// DEPRECATED: Not part of canonical emotion_schema.json v1. Will be removed.
    pub discrete_id: i16,  // -1 if unused
    /// DEPRECATED: Not part of canonical emotion_schema.json v1. Will be removed.
    pub intensity: f32,    // [0.0, 1.0]
    pub confidence: f32,    // [0.0, 1.0]
}
```

Note: We cannot use `#[deprecated]` on individual fields in Rust (it's not supported for struct fields). Doc comments are the correct approach for `#[repr(C)]` FFI structs. The fields stay for now to preserve ABI compatibility.

- [ ] **Step 2: Add deprecation warnings to Python EmotionState**

In `music_brain/intent_ir/__init__.py`, update the `EmotionState` dataclass (lines 34-42):

```python
import warnings

@dataclass
class EmotionState:
    """Emotion State - VAD coordinates with optional discrete mapping.

    NOTE: discrete_id and intensity are deprecated. They are not part of
    the canonical emotion_schema.json v1 contract. Use EmotionStateSchema
    from music_brain.engine_api.schema for new code.
    """
    valence: float = 0.0  # [-1.0, 1.0]
    arousal: float = 0.5  # [0.0, 1.0]
    dominance: float = 0.5  # [0.0, 1.0]
    discrete_id: int = -1  # DEPRECATED - not in emotion_schema.json v1
    intensity: float = 0.0  # DEPRECATED - not in emotion_schema.json v1
    confidence: float = 0.0  # [0.0, 1.0]

    def __post_init__(self):
        if self.discrete_id != -1:
            warnings.warn(
                "EmotionState.discrete_id is deprecated and will be removed. "
                "Use EmotionStateSchema from music_brain.engine_api.schema.",
                DeprecationWarning, stacklevel=2,
            )
        if self.intensity != 0.0:
            warnings.warn(
                "EmotionState.intensity is deprecated and will be removed. "
                "Use EmotionStateSchema from music_brain.engine_api.schema.",
                DeprecationWarning, stacklevel=2,
            )
```

- [ ] **Step 3: Verify existing tests still pass**

Run: `python3 -m pytest tests/ -x -q --ignore=tests/cpp 2>&1 | head -30`
Expected: No failures from the deprecation changes (default values don't trigger warnings)

- [ ] **Step 4: Commit**

```bash
git add engine/intent_ir/src/types.rs music_brain/intent_ir/__init__.py
git commit -m "deprecate: mark discrete_id and intensity as deprecated in EmotionState"
```

---

### Task 8: Final Verification & Cleanup

**Files:** None new — verification only

- [ ] **Step 1: Run full Python test suite for emotion schema**

Run: `python3 -m pytest tests/unit/test_emotion_schema.py tests/unit/test_sync_emotion.py -v`
Expected: All tests PASS

- [ ] **Step 2: Verify JSON Schema is valid**

Run:
```python
python3 -c "
import json
schema = json.loads(open('shared_schemas/emotion_schema.json').read())
print('Title:', schema.get('title'))
print('Properties:', list(schema.get('properties', {}).keys()))
print('Required:', schema.get('required'))
props = schema['properties']
print('Valence range:', props['valence'].get('minimum'), 'to', props['valence'].get('maximum'))
print('Tags maxItems:', props['tags'].get('maxItems'))
"
```

Expected output:
```
Title: EmotionStateSchema
Properties: ['valence', 'arousal', 'dominance', 'tags', 'confidence']
Required: ['valence', 'arousal', 'dominance', 'confidence']
Valence range: -1.0 to 1.0
Tags maxItems: 3
```

- [ ] **Step 3: Run Rust tests**

Run: `cd engine/intent_ir && cargo test test_emotion_schema -v`
Expected: All tests PASS

- [ ] **Step 4: Verify fixture count**

Run: `ls tests/fixtures/intent/emotion_*.json | wc -l`
Expected: 9

- [ ] **Step 5: Commit any final adjustments**

If any files needed tweaking during verification:

```bash
git add -u
git commit -m "fix: final adjustments from emotion schema verification"
```
