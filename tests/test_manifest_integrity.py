from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.package_dataset import package_task
from scripts.validate_package import validate_task_package


class ManifestIntegrityTest(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
                handle.write("\n")

    def _prepare_splits(self, split_root: Path) -> None:
        for task in ("intent_router", "axis_proposer"):
            for split in ("train", "val", "test"):
                rows = [
                    {
                        "sessionId": f"{task}-session-{split}",
                        "sampleId": f"{task}-sample-{split}-001",
                        "sourceLine": 1,
                        "text": "minimal request",
                        "label": "VALID",
                        "axes": {"valence": 0.1, "energy": 0.2} if task == "axis_proposer" else None,
                    }
                ]
                if task != "axis_proposer":
                    rows[0].pop("axes", None)
                self._write_jsonl(split_root / task / f"{split}.jsonl", rows)

    def test_validate_package_passes_for_fresh_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_root = root / "splits"
            package_root = root / "PACKAGES"
            schema_path = Path("config/dataset_manifest_schema.json")

            self._prepare_splits(split_root)
            package_dir = package_root / "pkg-unit"
            package_dir.mkdir(parents=True, exist_ok=True)

            package_task("intent_router", "pkg-unit", split_root, package_dir, records_per_shard=2, deterministic=False)
            package_task("axis_proposer", "pkg-unit", split_root, package_dir, records_per_shard=2, deterministic=False)

            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            for task in ("intent_router", "axis_proposer"):
                errors = validate_task_package(package_dir / task, schema)
                self.assertEqual(errors, [], f"expected no validation errors for {task}")

    def test_validate_package_detects_tampered_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_root = root / "splits"
            package_root = root / "PACKAGES"
            schema = json.loads(Path("config/dataset_manifest_schema.json").read_text(encoding="utf-8"))

            self._prepare_splits(split_root)
            package_dir = package_root / "pkg-unit"
            package_dir.mkdir(parents=True, exist_ok=True)
            package_task("intent_router", "pkg-unit", split_root, package_dir, records_per_shard=2, deterministic=False)

            manifest = json.loads((package_dir / "intent_router" / "manifest.json").read_text(encoding="utf-8"))
            shard_name = manifest["splits"]["train"]["shards"][0]["file"]
            shard_path = package_dir / "intent_router" / shard_name
            shard_path.write_bytes(shard_path.read_bytes() + b"tamper")

            errors = validate_task_package(package_dir / "intent_router", schema)
            self.assertTrue(errors)
            joined = "\n".join(errors)
            self.assertIn("sha256 mismatch", joined)


if __name__ == "__main__":
    unittest.main()
