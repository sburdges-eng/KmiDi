import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_emotion_schema_json_exists_after_sync():
    """After sync, emotion_schema.json must exist and be valid JSON Schema."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/sync_entities.py"],
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
    rs_path = ROOT / "src-tauri" / "src" / "generated" / "emotion.rs"
    assert rs_path.exists(), "emotion.rs not generated"
    content = rs_path.read_text()
    assert "valence" in content
    assert "arousal" in content
