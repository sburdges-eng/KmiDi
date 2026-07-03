#!/usr/bin/env python3
"""
Prepare (and on-demand download) the expressive subset of
Metacreation/GigaMIDI v2.0.0 — rows with nomml_score > THRESHOLD.

Dry-run by default: probes HF, reports expected sizes and auth status,
does NOT pull parquet shards. Pass --download to actually fetch.

  # Inspect only (safe, ~1 s of API calls):
  python3 scripts/prepare_gigamidi_expressive.py

  # Full pull + filter + extraction to disk:
  python3 scripts/prepare_gigamidi_expressive.py --download --extract

Cache lives on the external SSD so the boot drive isn't filled.

Prereqs:
  - Python 3.10+ with `datasets`, `huggingface_hub`, `pretty_midi`, `mido`
    (all present in /Library/Frameworks/Python.framework/Versions/3.14 on
    this machine; the shebang uses /usr/bin/env so pick an interpreter
    that has them).
  - `hf auth login` once, or export HUGGINGFACE_HUB_TOKEN.
  - GigaMIDI is gated (auto-approve); the HF account must have accepted
    the dataset's terms on the repo page before `load_dataset` will work.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# --- Configuration pinned on the external SSD ---------------------------
SSD_ROOT = Path("/Volumes/Sean's SSD/Datasets")
HF_CACHE = SSD_ROOT / "hf_cache"
OUT_DIR = SSD_ROOT / "midi" / "gigamidi" / "expressive"

REPO_ID = "Metacreation/GigaMIDI"
CONFIG = "v2.0.0"
DEFAULT_SPLIT = "train"
DEFAULT_THRESHOLD = 0.5

# Hoist HF cache env vars BEFORE importing datasets / huggingface_hub so
# the libraries pick up the SSD path instead of ~/.cache.
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE / "hub"))


def bytes_human(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    f = float(n)
    for u in units:
        if f < 1024:
            return f"{f:,.2f} {u}"
        f /= 1024
    return f"{f:,.2f} PiB"


def check_ssd_mounted() -> tuple[bool, str]:
    if not SSD_ROOT.exists():
        return False, f"SSD not mounted at {SSD_ROOT}"
    stat = shutil.disk_usage(SSD_ROOT)
    return True, f"{bytes_human(stat.free)} free of {bytes_human(stat.total)}"


def check_auth() -> tuple[bool, str]:
    try:
        from huggingface_hub import whoami  # type: ignore
    except ImportError:
        return False, "huggingface_hub not installed (pip install huggingface_hub)"
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    try:
        info = whoami(token=token) if token else whoami()
        return True, f"logged in as {info.get('name', '?')}"
    except Exception as e:
        return False, (
            f"not logged in ({e.__class__.__name__}). "
            "Run `hf auth login` or export HUGGINGFACE_HUB_TOKEN."
        )


def check_deps() -> list[str]:
    missing = []
    for mod in ("datasets", "huggingface_hub", "pretty_midi", "mido"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return missing


def probe_repo() -> dict:
    from huggingface_hub import HfApi

    api = HfApi()
    tree = api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True, path_in_repo=CONFIG)
    per_split: dict[str, dict] = {}
    for item in tree:
        if not item.path.endswith(".parquet"):
            continue
        # v2.0.0/<split>/<shard>.parquet
        parts = item.path.split("/")
        if len(parts) < 3:
            continue
        split = parts[1]
        s = per_split.setdefault(split, {"shards": 0, "bytes": 0})
        s["shards"] += 1
        s["bytes"] += item.size
    return per_split


def cmd_dry_run(args: argparse.Namespace) -> int:
    print(f"Repo:      https://hf.co/datasets/{REPO_ID}  (config {CONFIG})")
    print(f"Split:     {args.split}")
    print(f"Threshold: nomml_score > {args.threshold}")
    print(f"Cache:     {HF_CACHE}")
    print(f"Output:    {OUT_DIR}")
    print()

    ok, msg = check_ssd_mounted()
    print(f"SSD:       {'OK' if ok else 'MISSING'} — {msg}")
    if not ok:
        print("  (Plug it in before --download.)")

    missing = check_deps()
    if missing:
        print(f"Deps:      MISSING: {', '.join(missing)}")
        print(f"           -> pip install {' '.join(missing)}")
    else:
        print("Deps:      OK (datasets, huggingface_hub, pretty_midi, mido)")

    ok, msg = check_auth()
    print(f"Auth:      {'OK' if ok else 'MISSING'} — {msg}")

    print()
    print("Parquet shards on HF (v2.0.0):")
    try:
        splits = probe_repo()
    except Exception as e:
        print(f"  failed to probe: {e}")
        return 2
    total_bytes = 0
    for split, info in sorted(splits.items()):
        print(f"  {split:12s}  {info['shards']:2d} shards  {bytes_human(info['bytes'])}")
        total_bytes += info["bytes"]
    print(f"  {'ALL':12s}                 {bytes_human(total_bytes)}")

    target_bytes = splits.get(args.split, {}).get("bytes", 0)
    print()
    print(
        f"On --download, {args.split} will cache "
        f"~{bytes_human(target_bytes)} of parquet to {HF_CACHE}."
    )
    print("Filter is applied in-memory after the parquet shards are in cache;")
    print("it does NOT reduce network transfer (all rows must be read to evaluate).")
    print()
    print("Next steps:")
    print("  1. Ensure auth: `hf auth login` (token from https://hf.co/settings/tokens).")
    print(f"  2. Accept the dataset terms at https://hf.co/datasets/{REPO_ID} if not done.")
    print(f"  3. Run: python3 {Path(__file__).resolve()} --download")
    print(f"  4. Optional: add --extract to dump filtered MIDI bytes to {OUT_DIR}.")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    if not SSD_ROOT.exists():
        print(f"ERROR: SSD not mounted at {SSD_ROOT}", file=sys.stderr)
        return 2

    from datasets import load_dataset  # type: ignore

    print(f"Loading {REPO_ID} config={CONFIG} split={args.split} (cache -> {HF_CACHE}) ...")
    ds = load_dataset(REPO_ID, CONFIG, split=args.split)
    print(f"  {len(ds):,} rows, columns: {list(ds.column_names)}")

    # Validate nomml_score presence before filtering.
    if "nomml_score" not in ds.column_names:
        print("ERROR: column `nomml_score` not found in this split.", file=sys.stderr)
        print(f"Available columns: {ds.column_names}", file=sys.stderr)
        return 2

    thr = args.threshold
    expressive = ds.filter(lambda x: (x.get("nomml_score") or 0) > thr)
    print(
        f"  expressive (nomml_score > {thr}):  {len(expressive):,} rows "
        f"({100 * len(expressive) / max(len(ds), 1):.1f}% of split)"
    )

    if args.extract:
        return _extract(expressive, args)

    print()
    print("Done (parquet cached, subset materialised).")
    print(f"Pass --extract to write per-row MIDI bytes to {OUT_DIR}.")
    return 0


def _extract(expressive, args: argparse.Namespace) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figure out which column holds the MIDI bytes. GigaMIDI is tabular;
    # expected candidates are 'bytes', 'midi', 'audio', or a dict column.
    candidates = [c for c in expressive.column_names if c in ("bytes", "midi", "audio", "data")]
    midi_col = candidates[0] if candidates else None
    if midi_col is None:
        print(
            "ERROR: could not locate a MIDI bytes column. Columns were:",
            expressive.column_names,
            file=sys.stderr,
        )
        return 2
    print(f"Extracting column `{midi_col}` to {OUT_DIR} ...")

    written = 0
    for i, row in enumerate(expressive):
        raw = row.get(midi_col)
        if isinstance(raw, dict):  # datasets audio-like object
            raw = raw.get("bytes")
        if not raw:
            continue
        # Derive a filename; prefer `path` if present, fall back to the index.
        stem = Path(str(row.get("path") or f"row_{i:08d}")).stem
        out_path = OUT_DIR / f"{stem}.mid"
        out_path.write_bytes(raw)
        written += 1
        if written % 5000 == 0:
            print(f"  {written:,} written")
    print(f"Done: {written:,} MIDI files in {OUT_DIR}.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Actually download parquet shards and materialise the filter " "(NOT run by default).",
    )
    p.add_argument(
        "--extract",
        action="store_true",
        help="After downloading, extract per-row MIDI bytes to OUT_DIR. "
        "Ignored without --download.",
    )
    p.add_argument("--split", default=DEFAULT_SPLIT, choices=["train", "validation", "test"])
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="nomml_score > THRESHOLD (default 0.5).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.download:
        return cmd_download(args)
    return cmd_dry_run(args)


if __name__ == "__main__":
    sys.exit(main())
