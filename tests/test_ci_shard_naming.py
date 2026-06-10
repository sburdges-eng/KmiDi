"""CI Gate G2: Shard naming convention + checksums.txt presence.

Validates against the fixture package in tests/fixtures/ci_package/.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

FIXTURE_ROOT = Path("tests/fixtures/ci_package")
SHARD_PATTERN = re.compile(r"^(train|val|test)-\d{4}\.jsonl(?:\.zst)?$")
EXPECTED_SUFFIX = {"zstd": ".jsonl.zst", "identity": ".jsonl"}


class ShardNamingTest(unittest.TestCase):
    def test_fixture_package_exists(self) -> None:
        self.assertTrue(FIXTURE_ROOT.exists(), f"Missing CI fixture: {FIXTURE_ROOT}")

    def test_each_task_has_checksums_txt(self) -> None:
        for task_dir in sorted(FIXTURE_ROOT.iterdir()):
            if not task_dir.is_dir():
                continue
            checksums = task_dir / "checksums.txt"
            self.assertTrue(
                checksums.exists(),
                f"{task_dir.name}: missing checksums.txt",
            )

    def test_each_task_has_manifest(self) -> None:
        for task_dir in sorted(FIXTURE_ROOT.iterdir()):
            if not task_dir.is_dir():
                continue
            manifest = task_dir / "manifest.json"
            self.assertTrue(
                manifest.exists(),
                f"{task_dir.name}: missing manifest.json",
            )

    def test_shard_filenames_match_convention(self) -> None:
        for task_dir in sorted(FIXTURE_ROOT.iterdir()):
            if not task_dir.is_dir():
                continue
            manifest_path = task_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            splits = manifest.get("splits", {})
            for split_name, split_data in splits.items():
                for shard in split_data.get("shards", []):
                    filename = shard.get("file", "")
                    self.assertTrue(
                        SHARD_PATTERN.match(filename),
                        f"{task_dir.name}/{filename}: does not match {SHARD_PATTERN.pattern}",
                    )
                    self.assertTrue(
                        filename.startswith(f"{split_name}-"),
                        f"{task_dir.name}/{filename}: shard prefix must match split '{split_name}'",
                    )
                    compression = shard.get("compression", "")
                    self.assertIn(
                        compression,
                        EXPECTED_SUFFIX,
                        f"{task_dir.name}/{filename}: unknown compression '{compression}'",
                    )
                    self.assertTrue(
                        filename.endswith(EXPECTED_SUFFIX[compression]),
                        f"{task_dir.name}/{filename}: extension must match compression "
                        f"'{compression}'",
                    )

    def test_checksums_covers_all_shards(self) -> None:
        for task_dir in sorted(FIXTURE_ROOT.iterdir()):
            if not task_dir.is_dir():
                continue
            checksums_path = task_dir / "checksums.txt"
            manifest_path = task_dir / "manifest.json"
            if not checksums_path.exists() or not manifest_path.exists():
                continue

            checksum_entries: set[str] = set()
            for line in checksums_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    checksum_entries.add(parts[-1])

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for split_data in manifest.get("splits", {}).values():
                for shard in split_data.get("shards", []):
                    filename = shard.get("file", "")
                    self.assertIn(
                        filename,
                        checksum_entries,
                        f"{task_dir.name}: checksums.txt missing entry for {filename}",
                    )
            self.assertIn("manifest.json", checksum_entries)


if __name__ == "__main__":
    unittest.main()
