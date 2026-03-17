#!/usr/bin/env python3
"""
Resolve storage paths for source_manifest items and list or download what would be acquired.

Cooperates with existing dataset layout (docs/DATASETS_LAYOUT.md):
- KMIDI_DATASETS_PATH → by_source/<source_item>/downloads for dataset-like items
- KELLY_MODELS_PATH → proposed_storage_path for weights/models

Use --list, --resolve-paths, --dry-run to inspect. Use --download to fetch adopted
items with a valid primary_url (adoption_decision=adopted). Items with license=UNKNOWN
are skipped unless --accept-unknown-license.

Usage:
    python scripts/acquire/acquire_from_manifest.py --list
    python scripts/acquire/acquire_from_manifest.py --resolve-paths
    python scripts/acquire/acquire_from_manifest.py --dry-run [--approve]
    python scripts/acquire/acquire_from_manifest.py --download [--dry-run] [--accept-unknown-license]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# Repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_SOURCE_MANIFEST = REPO_ROOT / "config" / "source_manifest.yaml"


def _load_manifest(path: Path) -> list[dict]:
    try:
        import yaml
    except ImportError:
        sys.exit("pyyaml required; pip install pyyaml")
    if not path.exists():
        sys.exit(f"Manifest not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("sources") or []


def _resolve_root(env_var: str, fallback_subdir: str | None = None) -> Path | None:
    val = os.environ.get(env_var)
    if not val:
        if env_var == "KMIDI_DATASETS_PATH" and os.environ.get("KMIDI_DATA_ROOT"):
            # Align with prepare_datasets: KMIDI_DATA_ROOT/Datasets
            val = str(Path(os.environ["KMIDI_DATA_ROOT"]).expanduser() / "Datasets")
        else:
            return None
    p = Path(val).expanduser()
    return p if p.exists() else None


def resolve_storage_path(item: dict) -> Path | None:
    """
    Resolve storage path for one manifest item per docs/DATASETS_LAYOUT.md.
    - KMIDI_DATASETS_PATH → root/by_source/<source_item>/downloads
    - KELLY_MODELS_PATH → root/<proposed_storage_path>
    """
    env_var = item.get("storage_env_var")
    if not env_var or env_var == "null":
        return None
    root = _resolve_root(env_var)
    if not root:
        return None
    source_item = item.get("source_item") or ""
    proposed = item.get("proposed_storage_path") or ""

    if env_var == "KMIDI_DATASETS_PATH":
        # Dataset layout: by_source/<source_item>/downloads
        return root / "by_source" / source_item / "downloads"
    if env_var == "KELLY_MODELS_PATH":
        return root / proposed if proposed else root
    # Generic: root / proposed or root / source_item
    return root / (proposed or source_item) if (proposed or source_item) else root


def _filename_from_url(url: str, source_item: str) -> str:
    """Derive a safe filename from URL or source_item."""
    parsed = urlparse(url)
    name = (Path(parsed.path).name or "").strip()
    if name and re.match(r"^[\w.\-]+$", name):
        return name
    return f"{source_item}.bin"


def _download_one(
    url: str,
    dest_path: Path,
    *,
    timeout: int = 300,
    chunk_kb: int = 1024,
    user_agent: str = "KmiDi-acquire/1.0",
) -> bool:
    """Download url to dest_path (file). Returns True on success."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url)
    req.add_header("User-Agent", user_agent)
    try:
        with urlopen(req, timeout=timeout) as resp:
            with open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_kb * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"  download failed: {e}", file=sys.stderr)
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="List, resolve paths, or download from source_manifest.")
    ap.add_argument("--list", action="store_true", help="List all manifest items with status and path.")
    ap.add_argument("--resolve-paths", action="store_true", help="Print resolved storage path per downloadable item.")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be fetched; with --download no bytes written.")
    ap.add_argument("--approve", action="store_true", help="Include manual_review_required items in dry-run.")
    ap.add_argument("--download", action="store_true", help="Download adopted items with valid primary_url.")
    ap.add_argument("--accept-unknown-license", action="store_true", help="Allow download when license=UNKNOWN (override red flag).")
    ap.add_argument("--manifest", type=Path, default=CONFIG_SOURCE_MANIFEST, help="Path to source_manifest.yaml")
    ap.add_argument("--timeout", type=int, default=300, help="Download timeout per URL in seconds (default 300).")
    args = ap.parse_args()

    if not args.list and not args.resolve_paths and not args.dry_run and not args.download:
        args.list = True

    sources = _load_manifest(args.manifest)

    # Download path: only adopted + valid URL + resolved path
    if args.download:
        to_fetch = []
        for item in sources:
            if (item.get("adoption_decision") or "").lower() != "adopted":
                continue
            url = (item.get("primary_url") or "").strip()
            if not url or url.upper() in ("UNKNOWN", "N/A", "NULL") or not url.startswith(("http://", "https://")):
                continue
            path = resolve_storage_path(item)
            if not path:
                print(f"  skip {item.get('source_item', '?')}: no storage path (set storage_env_var and proposed_storage_path)", file=sys.stderr)
                continue
            license_val = (item.get("license") or "").strip()
            if license_val.upper() == "UNKNOWN" and not args.accept_unknown_license:
                print(f"  skip {item.get('source_item', '?')}: license UNKNOWN (use --accept-unknown-license to override)", file=sys.stderr)
                continue
            to_fetch.append((item, url, path))

        if not to_fetch:
            print("No adopted items with valid primary_url to download.", file=sys.stderr)
            print("Set adoption_decision: adopted and primary_url in config/source_manifest.yaml.", file=sys.stderr)
        else:
            for item, url, base_path in to_fetch:
                source_item = item.get("source_item") or "?"
                filename = _filename_from_url(url, source_item)
                dest = base_path / filename
                if args.dry_run:
                    print(f"  would download {source_item} -> {dest}")
                    continue
                print(f"  downloading {source_item} -> {dest} ...")
                if _download_one(url, dest, timeout=args.timeout):
                    print(f"  ok {dest}")
                else:
                    sys.exit(1)
        return

    for item in sources:
        source_item = item.get("source_item") or "?"
        dl = item.get("downloadable")
        downloadable = dl is True or (isinstance(dl, str) and dl.lower() == "yes")
        manual = item.get("manual_review_required") is True
        path = resolve_storage_path(item)

        if args.list:
            path_str = str(path) if path else "(no path)"
            manual_flag = " [manual_review]" if manual else ""
            adopted = (item.get("adoption_decision") or "").lower() == "adopted"
            adopted_flag = " [adopted]" if adopted else ""
            print(f"  {source_item}: downloadable={downloadable}, path={path_str}{manual_flag}{adopted_flag}")

        if args.resolve_paths and path and downloadable:
            print(path)

        if args.dry_run and downloadable:
            if manual and not args.approve:
                continue
            path_str = str(path) if path else "(path not resolved)"
            url = item.get("primary_url") or "UNKNOWN"
            print(f"  would fetch {source_item} -> {path_str}  (primary_url={url})")


if __name__ == "__main__":
    main()
