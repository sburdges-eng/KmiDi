"""
Model manager + auto-updater (stub).
Check for updates, download models, list local checkpoints.
"""

import os
import json

from training.storage_paths import CHECKPOINTS_DIR, ensure_storage_dirs

DEFAULT_CHECKPOINT_DIR = CHECKPOINTS_DIR
DEFAULT_MANIFEST = "models_manifest.json"


def list_local_models(checkpoint_dir=DEFAULT_CHECKPOINT_DIR):
    """List .pt files in checkpoint dir."""
    if not os.path.isdir(checkpoint_dir):
        return []
    return [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt")
    ]


def get_manifest_path():
    return os.path.join(os.path.dirname(__file__), "..", DEFAULT_MANIFEST)


def check_for_updates(manifest_url=None):
    """Stub: check remote manifest for newer model versions."""
    return {"available": False, "message": "Auto-updater stub. Set manifest_url for real check."}


def download_model(model_id, out_dir=DEFAULT_CHECKPOINT_DIR):
    """Stub: download model by id to out_dir."""
    print("Download stub: would download", model_id, "to", out_dir)
    return None


def ensure_model(name, checkpoint_dir=DEFAULT_CHECKPOINT_DIR):
    """Ensure model is present: list local or stub download."""
    local = list_local_models(checkpoint_dir)
    for path in local:
        if name in os.path.basename(path):
            return path
    return None


if __name__ == "__main__":
    import argparse
    ensure_storage_dirs()
    p = argparse.ArgumentParser(description="KmiDi model manager")
    p.add_argument("--list", action="store_true", help="List local checkpoints")
    p.add_argument("--check-dir", default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--check-updates", action="store_true")
    args = p.parse_args()
    if args.list:
        for path in list_local_models(args.check_dir):
            print(path)
    if args.check_updates:
        print(check_for_updates())
