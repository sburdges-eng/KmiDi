#!/usr/bin/env python3
"""
KmiDi CLI: train, generate, perform, release.
Run from repo root with PYTHONPATH=.
"""

import os
import sys
import argparse

# Ensure repo root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.storage_paths import (
    DATASETS_DIR,
    CHECKPOINTS_DIR,
    EXPORT_AUDIO_ONNX,
    EXPERIMENTS_DIR,
    GENERATED_DIR,
    PROCESSED_DATASET_FILE,
    RELEASE_DIR,
    ensure_storage_dirs,
)


def cmd_train(args):
    import torch
    if args.batch_size is None:
        args.batch_size = 12 if torch.cuda.is_available() else 6
    from training.train_unified import train
    train(args)


def cmd_generate(args):
    # Stub: load model, generate from prompt/seed
    print("Generate: load model from", args.checkpoint or CHECKPOINTS_DIR, "and generate.")
    if args.checkpoint and os.path.isfile(args.checkpoint):
        import torch
        try:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(args.checkpoint, map_location="cpu")
        print("Checkpoint keys:", list(ckpt.keys()))
    else:
        print("No checkpoint found. Run: kmidi_cli.py train --epochs 1 first.")


def cmd_perform(args):
    # Stub: live performance mode (KellyBrain + performer loop or singer)
    print("Perform: live performance mode.")
    try:
        from Brain.performer import run_performer_loop
        from Brain.kellybrain import KellyBrain
        print("KellyBrain + run_performer_loop available. Use --duration N for stub run.")
    except ImportError:
        print("Brain not on path. Set PYTHONPATH=.", file=sys.stderr)


def cmd_release(args):
    from tools.artist_stubs import DeploymentPack
    pack = DeploymentPack()
    out_dir = args.out or RELEASE_DIR
    pack.build(out_dir=out_dir)
    print("Release build triggered. Output dir:", out_dir)


def main():
    ensure_storage_dirs()

    p = argparse.ArgumentParser(prog="kmidi", description="KmiDi CLI: train, generate, perform, release")
    sub = p.add_subparsers(dest="command", required=True)

    # train
    t = sub.add_parser("train", help="Train Audio-JEPA (unified CUDA/MPS/CPU)")
    t.add_argument("--dataset-file", default=PROCESSED_DATASET_FILE, help="Single preprocessed dataset file (.npy)")
    t.add_argument("--datasets-dir", default=DATASETS_DIR, help="Fallback folder of raw audio files")
    t.add_argument("--files", nargs="*")
    t.add_argument("--epochs", type=int, default=120)
    t.add_argument("--batch-size", type=int, default=None)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--latent-dim", type=int, default=256)
    t.add_argument("--tier", default="medium", choices=("small", "medium", "large"))
    t.add_argument("--save-every", type=int, default=5)
    t.add_argument("--checkpoint-dir", default=CHECKPOINTS_DIR)
    t.add_argument("--export-path", default=EXPORT_AUDIO_ONNX)
    t.add_argument("--sample-rate", type=int, default=22050)
    t.add_argument("--experiment-name", default=None, help="Log run to experiment tracker")
    t.add_argument("--experiment-dir", default=EXPERIMENTS_DIR, help="Experiment runs dir")
    t.add_argument("--cpu", action="store_true", help="Force CPU (avoids MPS OOM on Apple Silicon)")
    t.set_defaults(func=cmd_train)

    # generate
    g = sub.add_parser("generate", help="Generate from checkpoint / prompt")
    g.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    g.add_argument("--prompt", default="", help="Text or seed (stub)")
    g.add_argument("--out", default=GENERATED_DIR)
    g.set_defaults(func=lambda a: cmd_generate(a))

    # perform
    pf = sub.add_parser("perform", help="Live performance mode")
    pf.add_argument("--duration", type=int, default=10, help="Stub duration (sec)")
    pf.add_argument("--singer", action="store_true", help="Use singer model (stub)")
    pf.set_defaults(func=lambda a: cmd_perform(a))

    # release
    r = sub.add_parser("release", help="Build release package (models, persona, plugins)")
    r.add_argument("--out", default=RELEASE_DIR)
    r.set_defaults(func=lambda a: cmd_release(a))

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
