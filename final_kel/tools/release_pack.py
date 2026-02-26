"""
Release packager: zip checkpoints, ONNX, persona manifest, and scripts for a versioned release.
Run: PYTHONPATH=. python tools/release_pack.py --version 1.0.0 [--checkpoints checkpoints] [--onnx Export] [--out release]
"""

import os
import sys
import json
import argparse
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.storage_paths import CHECKPOINTS_DIR, EXPORT_DIR, RELEASE_DIR, ensure_storage_dirs


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def main():
    ensure_storage_dirs()

    p = argparse.ArgumentParser(description="KmiDi release packager")
    p.add_argument("--version", default="1.0.0", help="Version string (e.g. 1.0.0)")
    p.add_argument("--checkpoints", default=CHECKPOINTS_DIR, help="Checkpoint dir to include")
    p.add_argument("--onnx", default=EXPORT_DIR, help="ONNX/Export dir to include")
    p.add_argument("--out", default=RELEASE_DIR, help="Output dir for pack")
    p.add_argument("--zip", action="store_true", help="Create zip of pack")
    args = p.parse_args()

    version = args.version.replace(" ", "_")
    pack_name = f"kmidi-v{version}"
    out_dir = resolve_path(args.out)
    pack_dir = os.path.join(out_dir, pack_name)
    os.makedirs(pack_dir, exist_ok=True)

    manifest = {"version": args.version, "files": []}

    # Copy checkpoints
    ckpt_src = resolve_path(args.checkpoints)
    if os.path.isdir(ckpt_src):
        ckpt_dst = os.path.join(pack_dir, "checkpoints")
        os.makedirs(ckpt_dst, exist_ok=True)
        for f in os.listdir(ckpt_src):
            if f.endswith(".pt"):
                src = os.path.join(ckpt_src, f)
                dst = os.path.join(ckpt_dst, f)
                shutil.copy2(src, dst)
                manifest["files"].append(f"checkpoints/{f}")
    # Copy ONNX
    onnx_src = resolve_path(args.onnx)
    if os.path.isdir(onnx_src):
        onnx_dst = os.path.join(pack_dir, "Export")
        os.makedirs(onnx_dst, exist_ok=True)
        for f in os.listdir(onnx_src):
            if f.endswith(".onnx") or f.endswith(".pt"):
                src = os.path.join(onnx_src, f)
                dst = os.path.join(onnx_dst, f)
                shutil.copy2(src, dst)
                manifest["files"].append(f"Export/{f}")

    manifest_path = os.path.join(pack_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    manifest["files"].append("manifest.json")

    print("Pack dir:", pack_dir)
    print("Manifest:", manifest)

    if args.zip:
        zip_path = os.path.join(out_dir, f"{pack_name}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(pack_dir):
                for f in files:
                    path = os.path.join(root, f)
                    arc = os.path.relpath(path, pack_dir)
                    zf.write(path, os.path.join(pack_name, arc))
        print("Zip:", zip_path)


if __name__ == "__main__":
    main()
