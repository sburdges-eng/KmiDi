from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.sync_package_to_s3 import collect_upload_targets, plan_sync_actions
from scripts.training_common import sha256_file


class IncrementalSyncTest(unittest.TestCase):
    def _create_package(self, root: Path) -> Path:
        package_dir = root / "PACKAGES" / "pkg-sync"
        task_dir = package_dir / "intent_router"
        task_dir.mkdir(parents=True, exist_ok=True)

        shard_path = task_dir / "train-0000.jsonl.zst"
        shard_path.write_bytes(b'{"label":"VALID","sessionId":"s1"}\n')

        manifest = {
            "schemaVersion": "1.0",
            "packageId": "pkg-sync",
            "task": "intent_router",
            "createdAt": "2026-02-16T00:00:00Z",
            "hashAlgorithm": "sha256",
            "compression": "identity",
            "splits": {
                "train": {
                    "recordCount": 1,
                    "shards": [
                        {
                            "file": "train-0000.jsonl.zst",
                            "split": "train",
                            "recordCount": 1,
                            "sha256": sha256_file(shard_path),
                            "sizeBytes": shard_path.stat().st_size,
                            "compression": "identity",
                        }
                    ],
                },
                "val": {"recordCount": 0, "shards": []},
                "test": {"recordCount": 0, "shards": []},
            },
            "totals": {"recordCount": 1, "shardCount": 1},
            "labelSpace": ["VALID", "ABSTAIN", "INVALID"],
        }
        (task_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return package_dir

    def test_plan_skips_when_remote_sha_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = self._create_package(Path(tmpdir))
            targets = collect_upload_targets(package_dir)

            local_by_key = {
                f"intent_router/{target.local_path.name}": sha256_file(target.local_path)
                for target in targets
            }

            def lookup(key: str) -> str | None:
                return local_by_key.get(key)

            actions = plan_sync_actions(targets, key_prefix="", remote_sha_lookup=lookup)
            self.assertTrue(actions)
            self.assertTrue(all(action.action == "skip" for action in actions))

    def test_plan_uploads_when_remote_missing_or_changed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = self._create_package(Path(tmpdir))
            targets = collect_upload_targets(package_dir)

            def lookup(key: str) -> str | None:
                if key.endswith("manifest.json"):
                    return "different"
                return None

            actions = plan_sync_actions(targets, key_prefix="pkg-sync", remote_sha_lookup=lookup)
            self.assertEqual(len(actions), 2)
            self.assertTrue(all(action.action == "upload" for action in actions))
            self.assertTrue(all(action.s3_key.startswith("pkg-sync/") for action in actions))


if __name__ == "__main__":
    unittest.main()
