#!/usr/bin/env python3
"""
Export LLaMA (or compatible) LLM to Core ML with stateful KV-cache for iOS18/macOS15.

Wraps ExecuTorch's export_llama_lib.py with the exact flag matrix for:
- Stateful prediction (--coreml-enable-state) for KV cache across steps
- Preserved SDPA (--coreml-preserve-sdpa) to use iOS18 SDPA op
- Blockwise 4-bit weight quantization (--coreml-quantize b4w)
- Static shapes (--disable_dynamic_shape) required for Core ML/MPS/QNN
- Explicit KV cache export (--use_kv_cache, --use_sdpa_with_kv_cache)

ExecuTorch is NOT bundled in this repo. You must:
  1. Clone or install ExecuTorch (e.g. pip install executorch, or clone
     https://github.com/pytorch/executorch and use its examples/portable/scripts).
  2. Set EXECUTORCH_DIR or pass --executorch-dir to the path containing
     export_llama_lib.py (often examples/portable/scripts/ or similar).
  3. Run from a Python env that has the export script's dependencies.

Usage:
  python scripts/export_llm_coreml.py --model-path /path/to/model --output-dir ./mlpackage_out
  EXECUTORCH_DIR=/path/to/executorch python scripts/export_llm_coreml.py --model-path ./model

Output: .mlpackage in output-dir. Compile to mlmodelc with Xcode or:
  xcrun coremlcompiler compile Model.mlpackage build/
Then point the Swift runner (tools/coreml_llm_runner) at the compiled mlmodelc.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export LLM to Core ML with stateful KV-cache (iOS18/macOS15)."
    )
    parser.add_argument(
        "--executorch-dir",
        type=Path,
        default=os.environ.get("EXECUTORCH_DIR"),
        help="Path to ExecuTorch repo (or set EXECUTORCH_DIR). Must contain export_llama_lib.py.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the PyTorch/LLaMA model (directory or checkpoint).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./coreml_llm_export"),
        help="Output directory for .mlpackage (default: ./coreml_llm_export).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Path to tokenizer (optional; export script may infer from model-path).",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=2048,
        help="Max context length for static shapes (default 2048). Required for Core ML stateful "
        "backend.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be run without executing.",
    )
    args = parser.parse_args()

    if not args.executorch_dir or not args.executorch_dir.is_dir():
        print(
            "export_llm_coreml: EXECUTORCH_DIR not set or not a directory. "
            "Set EXECUTORCH_DIR or pass --executorch-dir to the ExecuTorch repo root.",
            file=sys.stderr,
        )
        return 1

    # Common locations for the export script inside ExecuTorch
    candidates = [
        args.executorch_dir / "examples/portable/scripts/export_llama_lib.py",
        args.executorch_dir / "examples/llama/scripts/export_llama_lib.py",
        args.executorch_dir / "export_llama_lib.py",
    ]
    export_script = None
    for c in candidates:
        if c.exists():
            export_script = c
            break
    if not export_script:
        print(
            f"export_llm_coreml: export_llama_lib.py not found under {args.executorch_dir}. "
            "Tried: " + ", ".join(str(p) for p in candidates),
            file=sys.stderr,
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Flag names match ExecuTorch export_llama_lib.py (dash or underscore form;
    # adjust if your ExecuTorch version differs).
    cmd: list[str] = [
        sys.executable,
        str(export_script),
        "--model_path",
        str(args.model_path.resolve()),
        "--coreml-enable-state",
        "--coreml-preserve-sdpa",
        "--coreml-quantize",
        "b4w",
        "--disable_dynamic_shape",
        "--use_kv_cache",
        "--use_sdpa_with_kv_cache",
        "--output_dir",
        str(args.output_dir.resolve()),
    ]
    # Static context length required for Core ML stateful backend
    # (ExecuTorch may accept one or both)
    cmd.extend(["--max_context_length", str(args.max_context_length)])
    cmd.extend(["--max_seq_length", str(args.max_context_length)])
    if args.tokenizer_path is not None:
        cmd.extend(["--tokenizer_path", str(args.tokenizer_path.resolve())])

    if args.dry_run:
        print(" ".join(cmd))
        return 0

    try:
        subprocess.run(cmd, check=True, cwd=args.executorch_dir)
    except subprocess.CalledProcessError as e:
        print(f"export_llm_coreml: export failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    except FileNotFoundError:
        print("export_llm_coreml: could not run export script.", file=sys.stderr)
        return 1

    print(f"Export complete. mlpackage is in {args.output_dir}.")
    print("Compile to mlmodelc: xcrun coremlcompiler compile <Model.mlpackage> <output_dir>")
    print("Then run the Swift runner: tools/coreml_llm_runner with the mlmodelc path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
