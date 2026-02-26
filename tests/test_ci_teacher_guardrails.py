"""CI Gate G3: fetch_aws_artifacts teacher-fetch guardrails.

Validates:
  - default scope stays student
  - teacher/all scopes exist but require break-glass policy checks
  - key_is_local_safe enforces per-scope filtering
  - main() always calls enforce_teacher_fetch_policy before listing/download
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

FETCH_SCRIPT = Path("scripts/fetch_aws_artifacts.py")


class TeacherFetchGuardrailTest(unittest.TestCase):
    """Static analysis of fetch_aws_artifacts.py to verify teacher isolation."""

    def setUp(self) -> None:
        self.source = FETCH_SCRIPT.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source, filename=str(FETCH_SCRIPT))

    def test_scope_choices_include_breakglass_paths(self) -> None:
        """--scope choices must include student default plus teacher/all gated paths."""
        # Walk AST for add_argument("--scope", ..., choices=[...])
        found_choices = False
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            # Look for add_argument calls with "--scope"
            for arg in node.args:
                if isinstance(arg, ast.Constant) and arg.value == "--scope":
                    for kw in node.keywords:
                        if kw.arg == "choices":
                            if isinstance(kw.value, ast.List):
                                values = [
                                    elt.value
                                    for elt in kw.value.elts
                                    if isinstance(elt, ast.Constant)
                                ]
                                self.assertEqual(
                                    values,
                                    ["student", "teacher", "all"],
                                    "fetch_aws_artifacts --scope choices must be ['student','teacher','all']",
                                )
                                found_choices = True
        self.assertTrue(found_choices, "--scope choices not found in argparse definition")

    def test_scope_default_is_student(self) -> None:
        """--scope default must be 'student'."""
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and arg.value == "--scope":
                    for kw in node.keywords:
                        if kw.arg == "default":
                            self.assertIsInstance(kw.value, ast.Constant)
                            self.assertEqual(
                                kw.value.value,
                                "student",
                                "--scope default must be 'student'",
                            )

    def test_key_is_local_safe_scope_filters(self) -> None:
        """key_is_local_safe must filter keys by requested scope."""
        from scripts.fetch_aws_artifacts import key_is_local_safe

        # Student paths: allowed
        self.assertTrue(key_is_local_safe("artifacts/student/model_bundle.json"))
        self.assertTrue(key_is_local_safe("metrics.json"))

        # Teacher paths: blocked in student scope
        self.assertFalse(key_is_local_safe("artifacts/teacher/model_bundle.json"))
        self.assertFalse(key_is_local_safe("artifacts/teacher/weights.bin"))
        self.assertFalse(key_is_local_safe("teacher/anything"))
        # Teacher paths: allowed in teacher scope
        self.assertTrue(key_is_local_safe("artifacts/teacher/model_bundle.json", "teacher"))
        self.assertFalse(key_is_local_safe("artifacts/student/model_bundle.json", "teacher"))
        # All scope includes both
        self.assertTrue(key_is_local_safe("artifacts/teacher/model_bundle.json", "all"))
        self.assertTrue(key_is_local_safe("artifacts/student/model_bundle.json", "all"))

    def test_main_enforces_breakglass_policy(self) -> None:
        """main() must call enforce_teacher_fetch_policy before object listing."""
        self.assertIn(
            "enforce_teacher_fetch_policy(args, contract, default_profile)",
            self.source,
            "main() must enforce teacher break-glass policy",
        )
        self.assertIn(
            "if not args.break_glass",
            self.source,
            "teacher policy must require explicit --break-glass",
        )

    def test_teacher_path_requires_role_and_separate_profile(self) -> None:
        """Teacher fetch flow must require role ARN and non-default profile."""
        self.assertIn("Teacher artifact fetch requires explicit --profile", self.source)
        self.assertIn("Teacher artifact fetch requires --break-glass-role-arn", self.source)
        self.assertIn("requires a separate break-glass profile", self.source)


if __name__ == "__main__":
    unittest.main()
