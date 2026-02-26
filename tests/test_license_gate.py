from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.license_gate import (
    LicenseGateError,
    ThirdPartyNotice,
    ensure_allowlisted_license,
    generate_third_party_notices,
)


class LicenseGateTest(unittest.TestCase):
    def test_allowlisted_license_passes(self) -> None:
        metadata = {"license": "MIT"}
        license_id = ensure_allowlisted_license("org/model-a", metadata)
        self.assertEqual(license_id, "mit")

    def test_tag_license_passes(self) -> None:
        metadata = {"tags": ["text-classification", "license:apache-2.0"]}
        license_id = ensure_allowlisted_license("org/model-b", metadata)
        self.assertEqual(license_id, "apache-2.0")

    def test_disallowed_license_fails(self) -> None:
        metadata = {"cardData": {"license": "cc-by-nc-4.0"}}
        with self.assertRaises(LicenseGateError):
            ensure_allowlisted_license("org/model-c", metadata)

    def test_generate_notices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "THIRD_PARTY_NOTICES.md"
            notices = [
                ThirdPartyNotice(name="org/model-a", source="https://huggingface.co/org/model-a", license_id="mit")
            ]
            generate_third_party_notices(notices, path)
            content = path.read_text(encoding="utf-8")
            self.assertIn("THIRD_PARTY_NOTICES", content)
            self.assertIn("org/model-a", content)
            self.assertIn("mit", content)


if __name__ == "__main__":
    unittest.main()
