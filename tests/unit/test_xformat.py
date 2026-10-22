"""Unit tests for the xformat pre-merge verification gate (scripts/xformat.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import Mock

import pytest

_XFORMAT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "xformat.py"


def _load_xformat():
    spec = importlib.util.spec_from_file_location("xformat", _XFORMAT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    # Register before exec: @dataclass resolution looks the module up in
    # sys.modules (Python 3.12+/3.14), which fails for an unregistered module.
    sys.modules["xformat"] = mod
    spec.loader.exec_module(mod)
    return mod


xf = _load_xformat()


# --------------------------- scope_violations --------------------------- #
def test_scope_no_patterns_means_no_violations():
    assert xf.scope_violations(["music_brain/anything.py"], []) == []


def test_scope_all_within():
    files = ["music_brain/latent/projection.py", "tests/unit/test_projection.py"]
    patterns = ["music_brain/latent/**", "tests/unit/**"]
    assert xf.scope_violations(files, patterns) == []


def test_scope_flags_out_of_scope_file():
    files = ["music_brain/latent/projection.py", "engine/intent_ir/src/lib.rs"]
    patterns = ["music_brain/latent/**"]
    assert xf.scope_violations(files, patterns) == ["engine/intent_ir/src/lib.rs"]


def test_check_scope_skips_without_patterns():
    res = xf.check_scope(["a.py"], [])
    assert res.status == xf.SKIP


def test_check_scope_fails_on_violation():
    res = xf.check_scope(["src/main.tsx"], ["music_brain/**"])
    assert res.status == xf.FAIL
    assert "src/main.tsx" in res.detail


# ------------------------------ classify -------------------------------- #
def test_classify_buckets_by_extension():
    files = [
        "music_brain/latent/projection.py",
        "src/types/Intent.ts",
        "src/App.tsx",
        "engine/intent_ir/src/lib.rs",
        "engine/kelly_core.cpp",
        "shared_schemas/CompleteSongIntentRequest.json",
        "tests/unit/test_projection.py",
    ]
    cls = xf.classify(files)
    assert "music_brain/latent/projection.py" in cls.py
    assert set(cls.ts) == {"src/types/Intent.ts", "src/App.tsx"}
    assert cls.rust == ["engine/intent_ir/src/lib.rs"]
    assert cls.cpp == ["engine/kelly_core.cpp"]
    assert "shared_schemas/CompleteSongIntentRequest.json" in cls.schema
    assert cls.test_files == ["tests/unit/test_projection.py"]


def test_classify_rust_only_counts_intent_ir():
    cls = xf.classify(["engine/other/foo.rs", "engine/intent_ir/src/lib.rs"])
    assert cls.rust == ["engine/intent_ir/src/lib.rs"]


def test_classify_affected_tests_maps_existing_only(tmp_path, monkeypatch):
    # This test module itself exists, so a changed music_brain/xformat-like
    # module name that has no matching test maps to nothing.
    cls = xf.classify(["music_brain/does_not_exist_module.py"])
    assert cls.affected_tests == []


# ------------------------------ rendering ------------------------------- #
def test_render_reports_pass_when_no_failures():
    out = xf.render([xf.Result("lint", xf.PASS, "clean"), xf.Result("tests", xf.SKIP)])
    assert "RESULT: PASS" in out


def test_render_reports_failed_check_names():
    out = xf.render([xf.Result("lint", xf.FAIL, "2 findings"), xf.Result("tests", xf.PASS)])
    assert "RESULT: FAIL" in out
    assert "lint" in out


# --------------------------- check behaviours --------------------------- #
def test_check_lint_skips_without_python(monkeypatch):
    cls = xf.classify(["src/App.tsx"])
    assert xf.check_lint(cls).status == xf.SKIP


def test_check_schema_drift_skips_without_schema_changes():
    cls = xf.classify(["music_brain/latent/projection.py"])
    assert xf.check_schema_drift(cls).status == xf.SKIP


def test_check_tests_skips_when_no_targets():
    cls = xf.classify(["music_brain/no_matching_test.py"])
    assert xf.check_tests(cls, full=False).status == xf.SKIP


def test_main_returns_int(monkeypatch):
    # Drive main() with a base that yields no diff -> all SKIP -> exit 0.
    monkeypatch.setattr(xf, "changed_files", lambda base, include_dirty: ([], True))
    monkeypatch.setattr(xf, "check_gitnexus", lambda: xf.Result("gitnexus", xf.SKIP))
    assert xf.main(["--base", "HEAD"]) == 0


@pytest.mark.parametrize("directory", [False, True])
def test_gitnexus_missing_cli_warns_without_running_any_command(tmp_path, monkeypatch, directory):
    monkeypatch.setattr(xf, "REPO_ROOT", tmp_path)
    executable = "gitnexus.cmd" if sys.platform == "win32" else "gitnexus"
    if directory:
        (tmp_path / "node_modules" / ".bin" / executable).mkdir(parents=True)
    run = Mock(side_effect=AssertionError("missing CLI must not trigger a package runner"))
    monkeypatch.setattr(xf, "_run", run)
    result = xf.check_gitnexus()
    assert result.status == xf.WARN
    assert "local" in result.detail.lower()
    run.assert_not_called()


@pytest.mark.parametrize("platform,executable", [("darwin", "gitnexus"), ("win32", "gitnexus.cmd")])
@pytest.mark.parametrize(
    "returncode,stdout,stderr,expected",
    [
        (0, "Index is fresh", "", xf.PASS),
        (0, "Index is STALE", "", xf.WARN),
        (0, "Index is out of date", "", xf.WARN),
        (0, "", "Index is out-of-date", xf.WARN),
        (1, "", "Index unavailable", xf.WARN),
    ],
)
def test_gitnexus_uses_only_local_cli(
    tmp_path, monkeypatch, platform, executable, returncode, stdout, stderr, expected
):
    monkeypatch.setattr(xf, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(xf.sys, "platform", platform)
    cli = tmp_path / "node_modules" / ".bin" / executable
    cli.parent.mkdir(parents=True)
    cli.touch()
    run = Mock(return_value=CompletedProcess([], returncode, stdout, stderr))
    monkeypatch.setattr(xf, "_run", run)
    result = xf.check_gitnexus()
    assert result.status == expected
    run.assert_called_once_with([str(cli), "status"], cwd=tmp_path)
    assert "npx" not in result.detail
    if returncode == 0 and expected == xf.WARN:
        assert f"node_modules/.bin/{executable} analyze" in result.detail


@pytest.mark.parametrize("error", [FileNotFoundError("removed"), PermissionError("not executable")])
def test_gitnexus_unusable_local_cli_warns(tmp_path, monkeypatch, error):
    monkeypatch.setattr(xf, "REPO_ROOT", tmp_path)
    executable = "gitnexus.cmd" if sys.platform == "win32" else "gitnexus"
    cli = tmp_path / "node_modules" / ".bin" / executable
    cli.parent.mkdir(parents=True)
    cli.touch()
    run = Mock(side_effect=error)
    monkeypatch.setattr(xf, "_run", run)
    assert xf.check_gitnexus().status == xf.WARN
    run.assert_called_once_with([str(cli), "status"], cwd=tmp_path)
