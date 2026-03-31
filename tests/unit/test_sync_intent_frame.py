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
    assert "latency_budget_ms" in props


def test_intent_frame_ts_exists_after_sync():
    ts_path = ROOT / "src" / "types" / "IntentFrame.ts"
    assert ts_path.exists(), "IntentFrame.ts not generated"
    content = ts_path.read_text()
    assert "IntentFrame" in content
    assert "DSPTargets" in content
    assert "MusicHints" in content


def test_intent_frame_rust_exists_after_sync():
    rs_path = ROOT / "src-tauri" / "src" / "generated" / "intent_frame.rs"
    assert rs_path.exists(), "intent_frame.rs not generated"
    content = rs_path.read_text()
    assert "IntentFrame" in content
    assert "DSPTargets" in content
    assert "MusicHints" in content
