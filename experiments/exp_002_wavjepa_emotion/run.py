#!/usr/bin/env python3
"""
Run WavJEPA emotion experiment: extract embeddings (if needed) then evaluate.
Usage:
  python run.py
  python run.py --encoder hubert_base
  python run.py --skip-extract   # use existing cache
  python run.py --limit 200      # quick test (extract only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

_EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

from extract_embeddings import run as run_extract
from evaluate import run as run_evaluate


def main():
    parser = argparse.ArgumentParser(description="WavJEPA emotion: extract + evaluate")
    parser.add_argument("--config", type=Path, default=_EXPERIMENT_DIR / "config.yaml")
    parser.add_argument("--encoder", type=str, default=None, choices=["wavjepa", "hubert_base", "wav2vec2_base"])
    parser.add_argument("--skip-extract", action="store_true", help="Use existing cache, only evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for extraction (quick test)")
    parser.add_argument("--no-classifier", action="store_true", help="Skip linear classifier in evaluation")
    args = parser.parse_args()

    config_path = args.config.resolve()
    with open(config_path) as f:
        config = yaml.safe_load(f)
    encoder = args.encoder or config.get("encoder", "wavjepa")
    cache_dir = Path(config.get("cache_dir", "cache"))
    output_dir = Path(config.get("output_dir", "output"))
    base = config_path.parent
    if not cache_dir.is_absolute():
        cache_dir = (base / cache_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (base / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = cache_dir / f"embeddings_{encoder}.npz"
    if not args.skip_extract:
        print(f"Extract embeddings ({encoder})...")
        npz_path = run_extract(args.config, encoder_name=encoder, limit=args.limit)
    elif not npz_path.exists():
        print(f"Cache not found: {npz_path}. Run without --skip-extract first.")
        sys.exit(1)

    print(f"Evaluate {npz_path}...")
    results = run_evaluate(npz_path, args.config, encoder, use_classifier=not args.no_classifier)

    out_json = output_dir / f"results_{encoder}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_json}")

    print("\nResults:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
