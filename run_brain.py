#!/usr/bin/env python3
"""
run_brain.py – Deterministic Brain boot entry point.

Modes
-----
  penta        Start the penta_core ML server (music_brain.penta_core).
  orchestrator Start the AI orchestrator (music_brain.orchestrator).
  gui          Launch the KmiDi GUI (Music Brain FastAPI + Tauri/React).
  check        Pre-flight import check – verifies all required modules are
               importable without starting any server.  Safe to run in CI.

Usage
-----
  python run_brain.py check
  python run_brain.py penta
  python run_brain.py orchestrator
  python run_brain.py gui
"""
from __future__ import annotations

import argparse
import importlib
import sys


# ---------------------------------------------------------------------------
# Required modules verified by `check` mode.
# ---------------------------------------------------------------------------
_REQUIRED_MODULES: list[tuple[str, str]] = [
    ("music_brain", "music_brain core package"),
    ("music_brain.penta_core", "penta_core ML layer"),
    ("music_brain.orchestrator", "AI orchestrator"),
    ("music_brain.session", "session / intent schema"),
    ("music_brain.emotion", "emotion mapping"),
    ("music_brain.kelly_companion", "Kelly Companion (session / engines)"),
]


def _check() -> int:
    """Import-check all required modules; return 0 on success, 1 on failure."""
    print("Brain pre-flight check")
    print("=" * 40)
    passed = True
    for module_name, description in _REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
            print(f"  ok  {description} ({module_name})")
        except ImportError as exc:
            print(f"  FAIL {description} ({module_name}): {exc}")
            passed = False
    print("=" * 40)
    if passed:
        print("All modules OK – brain is bootable.")
        return 0
    print("One or more modules are missing. Fix imports before starting.")
    return 1


def _penta() -> int:
    """Start the penta_core server."""
    try:
        from music_brain.penta_core import server
        server.mcp.run()
    except ImportError as exc:
        print(f"Cannot start penta_core server: {exc}", file=sys.stderr)
        return 1
    return 0


def _orchestrator() -> int:
    """Start the AI orchestrator pipeline."""
    try:
        import asyncio
        from music_brain.orchestrator.orchestrator import AIOrchestrator

        async def _run() -> None:
            async with AIOrchestrator() as orch:
                print(f"Orchestrator ready: {orch}")
                # Keep alive until interrupted.
                await asyncio.Event().wait()

        asyncio.run(_run())
    except ImportError as exc:
        print(f"Cannot start orchestrator: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOrchestrator stopped.")
    return 0


def _gui() -> int:
    """Launch the Music Brain API (and optionally the Tauri GUI)."""
    try:
        import subprocess
        import os

        # Start the Music Brain FastAPI server.
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "music_brain.api:app",
                "--reload",
                "--port",
                "8000",
                "--host",
                "127.0.0.1",
            ],
            env=os.environ.copy(),
        )
        print("Music Brain API started on http://127.0.0.1:8000")
        print("Press Ctrl-C to stop.")
        proc.wait()
    except FileNotFoundError:
        print("uvicorn not found. Install with: pip install uvicorn", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nGUI stopped.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_brain.py",
        description="Deterministic Brain boot entry point.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "mode",
        choices=["penta", "orchestrator", "gui", "check"],
        help="Boot mode to run.",
    )
    args = parser.parse_args()

    dispatch = {
        "check": _check,
        "penta": _penta,
        "orchestrator": _orchestrator,
        "gui": _gui,
    }
    return dispatch[args.mode]()


if __name__ == "__main__":
    sys.exit(main())
