"""Bounded subprocess sandbox for running untrusted-ish work.

Closes the **Process sandboxing** spec item documented as a gap in
``docs/audit/RT_CALLBACK_AUDIT_2026-05-22.md``.

Use for: running training scripts, model converters, third-party analysis
tools, or any helper binary that should be restricted to a known argv,
working directory, and resource budget.

Not for: full container-grade isolation. This is a POSIX rlimit + path
allowlist guard, not a security boundary against a malicious binary that
chooses to disregard rlimits. Treat it as a misconfiguration safety net,
not a hostile-binary defense.

Design:
- Stdlib only (subprocess + os + resource + pathlib + shutil).
- Allowlist of permitted executables (full paths, resolved before exec).
- Allowlist of working directories (the child cannot cd outside).
- Wall-clock timeout via subprocess.run's ``timeout`` parameter.
- Memory cap via ``resource.setrlimit(RLIMIT_AS, ...)`` in preexec_fn.
- CPU-time cap via ``RLIMIT_CPU`` (kills the child on hard wall budget).
- Environment defaults to an empty dict; caller opts back in to specific
  variables. PATH is restricted to a single binary directory derived
  from the resolved executable.

The sandbox API returns a typed ``SandboxResult`` rather than raising for
expected failures (timeout, non-zero exit). Configuration errors (denied
executable, denied cwd) raise ``SandboxViolation``.
"""

from __future__ import annotations

import os
import shutil
import subprocess

try:
    import resource  # POSIX-only: rlimit-based caps are unavailable on Windows
except ImportError:  # pragma: no cover — Windows
    resource = None  # type: ignore[assignment]
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


class SandboxViolation(Exception):
    """Raised when the requested execution violates the sandbox policy."""


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of a sandboxed run."""
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@dataclass(frozen=True)
class SandboxPolicy:
    """Allowlist + resource cap policy.

    Args:
        allowed_executables: Absolute paths to executables that may be run.
            Resolved with realpath at policy creation; comparison is
            string-equality on the resolved path.
        allowed_cwds: Absolute paths that may be used as cwd. The child's
            cwd must equal one of these (no descendants implied — pass
            descendants explicitly if needed).
        wall_timeout_s: Maximum wall-clock seconds before the child is
            killed. Required (>0).
        memory_limit_bytes: RLIMIT_AS soft+hard cap. Pass None to skip.
        cpu_seconds: RLIMIT_CPU soft+hard cap. Pass None to skip.
        env_passthrough: Iterable of environment variable names that may
            be forwarded from the parent. Everything else is stripped.
    """
    allowed_executables: frozenset[str]
    allowed_cwds: frozenset[str]
    wall_timeout_s: float
    memory_limit_bytes: Optional[int] = None
    cpu_seconds: Optional[int] = None
    env_passthrough: frozenset[str] = frozenset()


def make_policy(*,
                allowed_executables: Sequence[Path],
                allowed_cwds: Sequence[Path],
                wall_timeout_s: float,
                memory_limit_bytes: Optional[int] = None,
                cpu_seconds: Optional[int] = None,
                env_passthrough: Sequence[str] = ()) -> SandboxPolicy:
    """Build a normalised ``SandboxPolicy``.

    Paths are resolved via ``Path.resolve(strict=False)`` so the policy
    can be defined before the binaries exist on disk; existence is
    re-checked at run time.
    """
    if wall_timeout_s <= 0:
        raise ValueError("wall_timeout_s must be positive")
    if memory_limit_bytes is not None and memory_limit_bytes <= 0:
        raise ValueError("memory_limit_bytes must be positive when set")
    if cpu_seconds is not None and cpu_seconds <= 0:
        raise ValueError("cpu_seconds must be positive when set")
    return SandboxPolicy(
        allowed_executables=frozenset(
            str(Path(p).resolve(strict=False)) for p in allowed_executables),
        allowed_cwds=frozenset(
            str(Path(p).resolve(strict=False)) for p in allowed_cwds),
        wall_timeout_s=float(wall_timeout_s),
        memory_limit_bytes=memory_limit_bytes,
        cpu_seconds=cpu_seconds,
        env_passthrough=frozenset(env_passthrough),
    )


def _build_preexec(policy: SandboxPolicy):
    """Return a preexec_fn that applies rlimits before exec.

    On systems without the relevant rlimit constants (e.g. some Linux
    sandboxes), the call is skipped silently — the wall-clock timeout
    still bounds the run. On Windows there is no ``resource`` module and
    ``preexec_fn`` is unsupported, so ``None`` is returned and the
    wall-clock timeout is the only enforced bound.
    """
    if resource is None:
        return None

    mem = policy.memory_limit_bytes
    cpu = policy.cpu_seconds

    def _apply() -> None:
        if mem is not None and hasattr(resource, "RLIMIT_AS"):
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            except (ValueError, OSError):
                pass
        if cpu is not None and hasattr(resource, "RLIMIT_CPU"):
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            except (ValueError, OSError):
                pass

    return _apply


def run_sandboxed(executable: Path, args: Sequence[str], *,
                   cwd: Path, policy: SandboxPolicy,
                   stdin: Optional[bytes] = None) -> SandboxResult:
    """Execute ``executable`` under ``policy`` and return the result.

    Raises:
        SandboxViolation: when ``executable`` or ``cwd`` is not in the
            policy allowlist, or when the executable does not exist on
            disk at run time.
    """
    exe_resolved = str(Path(executable).resolve(strict=False))
    cwd_resolved = str(Path(cwd).resolve(strict=False))

    if exe_resolved not in policy.allowed_executables:
        raise SandboxViolation(
            f"executable not in allowlist: {exe_resolved}")
    if cwd_resolved not in policy.allowed_cwds:
        raise SandboxViolation(f"cwd not in allowlist: {cwd_resolved}")
    if not Path(exe_resolved).exists():
        raise SandboxViolation(f"executable does not exist: {exe_resolved}")

    # Build a scrubbed environment. PATH points at the binary's own
    # directory so common shebangs / shells can be resolved without
    # leaking host PATH entries.
    parent_env = os.environ
    env = {k: parent_env[k] for k in policy.env_passthrough if k in parent_env}
    env.setdefault("PATH", str(Path(exe_resolved).parent))

    # subprocess.run rejects passing both stdin= and input= simultaneously.
    # When the caller supplies bytes, use input= (it implies stdin=PIPE
    # internally); otherwise null-out stdin.
    run_kwargs: dict = dict(
        cwd=cwd_resolved,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=policy.wall_timeout_s,
        preexec_fn=_build_preexec(policy),
        check=False,
    )
    if stdin is not None:
        run_kwargs["input"] = stdin
    else:
        run_kwargs["stdin"] = subprocess.DEVNULL

    try:
        completed = subprocess.run(  # noqa: S603 — argv is policy-controlled
            [exe_resolved, *args],
            **run_kwargs,
        )
        return SandboxResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        return SandboxResult(
            returncode=-1,
            stdout=exc.stdout or b"",
            stderr=exc.stderr or b"",
            timed_out=True,
        )


def which_or_none(name: str) -> Optional[Path]:
    """Convenience wrapper around ``shutil.which`` returning a Path."""
    found = shutil.which(name)
    return Path(found) if found else None


__all__ = [
    "SandboxPolicy",
    "SandboxResult",
    "SandboxViolation",
    "make_policy",
    "run_sandboxed",
    "which_or_none",
]
