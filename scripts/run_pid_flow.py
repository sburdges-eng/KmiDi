#!/usr/bin/env python3
"""
Run PID Flow diagnostic and print a layer-wise report
(Redundant / Text Unique / Audio Unique / Synergy).

Use after fine-tuning to check for modality collapse: if late layers show 0% Audio/MIDI Unique,
the model is effectively deaf to the music.

Example (with a model and get_activations provided by your code):
  from penta_core.ml.diagnostics.pid_flow import run_pid_flow_report, pid_report_to_string

  def get_activations(model):
      # Run a batch through model, collect (t, a, b) per layer from hooks or forward returns
      return [(t_np, a_np, b_np), ...]  # one tuple per layer

  report = run_pid_flow_report(model, get_activations, layer_names=["layer_0", "layer_1", ...])
  print(pid_report_to_string(report))

This script runs a dummy check (random activations) to validate the report shape.
For real use, call run_pid_flow_report from your training/eval code with real activations.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PID Flow diagnostic (dummy run or with custom get_activations)."
    )
    parser.add_argument(
        "--dummy", action="store_true", help="Run on random activations to validate report shape."
    )
    parser.add_argument("--json", action="store_true", help="Output report as JSON.")
    args = parser.parse_args()

    if not args.dummy:
        print(
            "Provide --dummy to run a quick validation, "
            "or call run_pid_flow_report from your code. "
            "See script docstring for usage.",
            file=sys.stderr,
        )
        return 0

    import numpy as np

    try:
        from penta_core.ml.diagnostics.pid_flow import (
            check_modality_collapse,
            run_pid_flow_report,
            pid_report_to_string,
            pid_report_to_dict,
        )
    except ModuleNotFoundError:
        from music_brain.penta_core.ml.diagnostics.pid_flow import (
            check_modality_collapse,
            run_pid_flow_report,
            pid_report_to_string,
            pid_report_to_dict,
        )

    n_samples, dim = 64, 128
    layer_names = ["layer_0", "layer_4", "layer_8", "layer_12"]

    def get_activations(_model: None) -> list:
        out = []
        for _ in layer_names:
            t = np.random.randn(n_samples, dim).astype(np.float32)
            a = np.random.randn(n_samples, dim).astype(np.float32)
            b = np.random.randn(n_samples, dim).astype(np.float32)
            out.append((t, a, b))
        return out

    report = run_pid_flow_report(None, get_activations, layer_names)
    collapse_warnings = check_modality_collapse(report)

    if args.json:
        import json as _json

        out = {
            "report": pid_report_to_dict(report),
            "collapse_warnings": [
                {"layer": layer, "message": msg} for layer, msg in collapse_warnings
            ],
        }
        print(_json.dumps(out, indent=2))
    else:
        print(pid_report_to_string(report))
        for _layer, msg in collapse_warnings:
            print(msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
