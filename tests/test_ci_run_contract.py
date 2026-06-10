"""CI Gate G5: run_contract.yaml required keys and fail-closed rules.

This gate validates the minimal run contract required by the AWS training
scripts in `scripts/`:
  - S3 buckets/prefix conventions
  - budget cap invariants
  - checkpoint and early-stop policy bounds
  - fail-closed teacher artifact fetch policy
"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from scripts.training_common import load_run_contract, validate_run_contract

CONTRACT_PATH = Path("config/run_contract.yaml")


def _load_contract() -> dict:
    """Load and semantically validate run contract."""
    return load_run_contract(CONTRACT_PATH)


class RunContractRequiredKeysTest(unittest.TestCase):
    """Validate all required keys exist and fail-closed rules hold."""

    @classmethod
    def setUpClass(cls) -> None:
        if not CONTRACT_PATH.exists():
            raise unittest.SkipTest(
                f"Run contract not found at {CONTRACT_PATH}; "
                "create config/run_contract.yaml to enable this gate"
            )
        cls.contract = _load_contract()

    def test_schema_version_present(self) -> None:
        self.assertIn("schemaVersion", self.contract)
        self.assertIsInstance(self.contract["schemaVersion"], str)

    def test_s3_bucket_present(self) -> None:
        s3 = self.contract.get("s3", {})
        self.assertIsInstance(s3, dict, "s3 section must be a dict")
        self.assertIn("packageBucket", s3, "s3.packageBucket is required")
        self.assertIn("runBucket", s3, "s3.runBucket is required")
        self.assertIsInstance(s3["packageBucket"], str)
        self.assertIsInstance(s3["runBucket"], str)

    def test_s3_prefix_conventions(self) -> None:
        s3 = self.contract.get("s3", {})
        package_prefix = s3.get("packagePrefix", "")
        runs_prefix = s3.get("runPrefix", "")
        self.assertIsInstance(package_prefix, str)
        self.assertIsInstance(runs_prefix, str)
        self.assertTrue(
            package_prefix.startswith("training/"),
            f"s3.packagePrefix should start with 'training/': {package_prefix}",
        )
        self.assertTrue(
            runs_prefix.startswith("training/"),
            f"s3.runPrefix should start with 'training/': {runs_prefix}",
        )

    def test_budget_cap_within_bounds(self) -> None:
        """FC-6: training.budgetCapUsd must be > 0 and <= 100."""
        training = self.contract.get("training", {})
        self.assertIsInstance(training, dict, "training section must be a dict")
        cap = training.get("budgetCapUsd", 0)
        self.assertGreater(cap, 0, "budget_cap_usd must be > 0")
        self.assertLessEqual(cap, 100, "budget_cap_usd cannot exceed $100")

    def test_checkpoint_policy_bounds(self) -> None:
        """Checkpoint cadence and retention policy must be bounded (fail-closed)."""
        training = self.contract.get("training", {})
        checkpoints = training.get("checkpoints", {})
        self.assertIsInstance(checkpoints, dict, "training.checkpoints must be a dict")

        default_cadence = int(checkpoints.get("defaultCadenceSteps", 0) or 0)
        min_cadence = int(checkpoints.get("minCadenceSteps", 0) or 0)
        max_cadence = int(checkpoints.get("maxCadenceSteps", 0) or 0)
        self.assertGreaterEqual(min_cadence, 1)
        self.assertGreaterEqual(max_cadence, min_cadence)
        self.assertGreaterEqual(default_cadence, min_cadence)
        self.assertLessEqual(default_cadence, max_cadence)

        default_max = int(checkpoints.get("defaultMaxCheckpoints", 0) or 0)
        max_keep = int(checkpoints.get("maxCheckpoints", 0) or 0)
        max_total = int(checkpoints.get("maxTotalCheckpoints", 0) or 0)
        self.assertGreaterEqual(default_max, 1)
        self.assertGreaterEqual(max_keep, default_max)
        self.assertGreaterEqual(
            max_total, max_keep + 1, "maxTotalCheckpoints must allow keep-last-N + best"
        )

    def test_preflight_is_configured(self) -> None:
        """Launch should be gated by a preflight command (fail-closed)."""
        training = self.contract.get("training", {})
        ops = training.get("ops", {})
        self.assertIsInstance(ops, dict, "training.ops must be a dict")
        cmd = ops.get("preflightCommand", "")
        self.assertIsInstance(cmd, str)
        self.assertTrue(cmd.strip(), "training.ops.preflightCommand must be non-empty")

    def test_teacher_fetch_is_fail_closed(self) -> None:
        """Teacher artifact fetch must be break-glass only by default."""
        security = self.contract.get("security", {})
        self.assertIsInstance(security, dict, "security section must be a dict")
        teacher_fetch = security.get("teacherFetch", {})
        self.assertIsInstance(teacher_fetch, dict, "security.teacherFetch must be a dict")

        self.assertIs(
            teacher_fetch.get("allowByDefault"),
            False,
            "security.teacherFetch.allowByDefault must be false",
        )
        required_profile = teacher_fetch.get("requiredProfile", "")
        self.assertIsInstance(required_profile, str)
        self.assertTrue(
            required_profile.strip(), "security.teacherFetch.requiredProfile must be non-empty"
        )

        role_arn = teacher_fetch.get("requiredRoleArn", "")
        self.assertIsInstance(
            role_arn, str, "security.teacherFetch.requiredRoleArn must be a string (may be empty)"
        )

        aws = self.contract.get("aws", {})
        self.assertIsInstance(aws, dict, "aws section must be a dict")
        break_glass_profile = aws.get("breakGlassProfile", "")
        self.assertIsInstance(break_glass_profile, str)
        if break_glass_profile.strip():
            self.assertEqual(
                break_glass_profile,
                required_profile,
                "aws.breakGlassProfile must match security.teacherFetch.requiredProfile when set",
            )

    def test_rejects_unknown_top_level_key(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["unexpected"] = {"enabled": True}
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            validate_run_contract(mutated)

    def test_rejects_unknown_nested_key(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["training"]["unexpectedControl"] = True
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            validate_run_contract(mutated)

    def test_rejects_unknown_teacher_fetch_key(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["security"]["teacherFetch"]["extra"] = "x"
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            validate_run_contract(mutated)


if __name__ == "__main__":
    unittest.main()
