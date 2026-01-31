#!/usr/bin/env python3
"""
KmiDi Brain launcher.

Entry point for running the brain stack:
- penta_core ML (inference, training)
- mcp_workstation Orchestrator (when music_brain modules restored)
- kmidi_gui (control surface)

Reference: If music_brain is missing, restore from sburdges-eng/KmiDi forensic
or rebuild. Not in online branches = cannot recover easily.
"""

import argparse
import sys
from pathlib import Path

# Project root and brain path (so penta_core.ml.* resolves when importing from KmiDi_CANON.brain)
ROOT = Path(__file__).resolve().parent
BRAIN = ROOT / "KmiDi_CANON" / "brain"
for p in (ROOT, BRAIN):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)


def main():
    parser = argparse.ArgumentParser(description="KmiDi Brain launcher")
    parser.add_argument(
        "mode",
        nargs="?",
        default="penta",
        choices=["penta", "orchestrator", "gui", "check"],
        help="Run mode: penta (ML), orchestrator, gui, or check (module status)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Restart on exit: run mode in a loop (use with tmux for long-running Brain)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        metavar="SEC",
        help="Seconds to wait before restart when using --loop (default: 2)",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=0,
        metavar="N",
        help="Max restarts when using --loop (0 = infinite, default: 0)",
    )
    args = parser.parse_args()

    if args.mode == "check":
        _check_modules()
        return 0

    if args.loop:
        return _run_loop(args)
    if args.mode == "penta":
        return _run_penta()
    if args.mode == "orchestrator":
        return _run_orchestrator()
    if args.mode == "gui":
        return _run_gui()

    return 1


def _run_loop(args):
    """Run chosen mode in a restart loop. Restarts only on non-zero exit or exception. Use with tmux."""
    import time
    import traceback

    restarts = 0
    while True:
        if restarts > 0:
            print(f"[brain] restart #{restarts} in {args.delay}s ...")
            time.sleep(args.delay)
        if args.max_restarts and restarts >= args.max_restarts:
            print(f"[brain] max restarts ({args.max_restarts}) reached, exiting")
            return 1
        try:
            code = _run_mode(args.mode)
        except Exception as e:
            print(f"[brain] exception: {e}")
            traceback.print_exc()
            code = 1
        if code == 0:
            return 0
        restarts += 1
    return 1


def _run_mode(mode: str) -> int:
    """Run a single mode; return exit code."""
    if mode == "penta":
        return _run_penta()
    if mode == "orchestrator":
        return _run_orchestrator()
    if mode == "gui":
        return _run_gui()
    return 1


def _check_modules():
    """Report module availability."""
    print("KmiDi Brain module check:")
    brain = ROOT / "KmiDi_CANON" / "brain"
    checks = [
        ("penta_core/ml", (brain / "penta_core" / "ml" / "inference.py").exists()),
        ("mcp_workstation", (brain / "mcp_workstation" / "orchestrator.py").exists()),
        ("kmidi_gui", (brain / "kmidi_gui" / "core" / "control_surface.py").exists()),
        (
            "music_brain/session",
            (brain / "music_brain" / "session").exists(),
        ),
    ]
    for name, ok in checks:
        status = "OK" if ok else "MISSING (restore or rebuild)"
        print(f"  {name}: {status}")


def _run_penta():
    """Run penta_core ML inference check."""
    try:
        from KmiDi_CANON.brain.penta_core.ml.inference import create_engine_by_name

        print("penta_core ML loaded. Use model_registry for inference.")
        return 0
    except ImportError as e:
        print(f"penta_core import error: {e}")
        return 1


def _run_orchestrator():
    """Run mcp_workstation Orchestrator."""
    try:
        from KmiDi_CANON.brain.mcp_workstation.orchestrator import main as orch_main

        return orch_main()
    except ImportError as e:
        print(f"Orchestrator import error: {e}")
        print("  music_brain.session or music_brain.tier1 may be missing.")
        print("  Restore from forensic or rebuild.")
        return 1


def _run_gui():
    """Run kmidi_gui (stub — GUI typically via Tauri)."""
    print("kmidi_gui runs via Tauri app. Use: cargo tauri dev")
    return 0


if __name__ == "__main__":
    sys.exit(main())
