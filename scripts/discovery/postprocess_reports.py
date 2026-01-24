#!/usr/bin/env python3
"""
Post-process discovery outputs without re-scanning the filesystem.

Fixes / improvements:
- Populate `git_roots` (generate_reports.py's scan prunes `.git` dirs)
- Regenerate `project_directories.md`, `DISCOVERY_SUMMARY.md`,
  and integration reports with better vendor/venv filtering.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from discovery_common import HOME_DEFAULT, KMIDI_KEYWORDS, ML_KEYWORDS, MUSIC_KEYWORDS, now_iso, read_text_sample, value_bucket, value_score


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def out_dir() -> Path:
    return repo_root() / "docs" / "DISCOVERY"


def _load_db(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _find_git_roots(home: Path) -> List[str]:
    prunes = [
        str(home / "System"),
        str(home / ".Trash"),
        str(home / ".cache"),
        str(home / ".cursor"),
        str(home / ".npm"),
        str(home / ".cargo"),
        str(home / ".rustup"),
        str(home / ".gradle"),
    ]

    def run_find(root: Path, *, extra_prunes: Sequence[str] = ()) -> List[str]:
        cmd: List[str] = ["find", str(root)]
        for p in list(prunes) + list(extra_prunes):
            # Prune the directory and everything under it.
            cmd += ["-path", p, "-prune", "-o", "-path", p + "/*", "-prune", "-o"]
        cmd += ["-type", "d", "-name", ".git", "-print"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        # `find` often returns non-zero when it hits permission-denied paths; still use stdout.
        roots = []
        for line in result.stdout.splitlines():
            p = Path(line.strip())
            if p.name == ".git":
                roots.append(str(p.parent))
        return roots

    # Avoid traversing full ~/Library; do a dedicated scan of CloudStorage.
    git_roots = set(run_find(home, extra_prunes=[str(home / "Library")]))
    cloud = home / "Library" / "CloudStorage"
    if cloud.exists():
        git_roots |= set(run_find(cloud))
    return sorted(git_roots)


def _is_vendor_path(rel_path: str) -> bool:
    rp = rel_path.replace("\\", "/").lower()
    vendor_markers = [
        "site-packages/",
        "dist-packages/",
        "node_modules/",
        ".cargo/",
        ".rustup/",
        ".gradle/",
        ".npm/",
        ".cursor/",
        ".cache/",
        "venv/",
        ".venv/",
    ]
    return any(m in rp for m in vendor_markers)


def _parse_mtime(iso: str) -> datetime:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _rank_subset(items: Sequence[Dict[str, Any]], *, max_total: int = 1200) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for it in items:
        rp = it.get("rel_path") or ""
        if _is_vendor_path(rp):
            continue
        text = read_text_sample(Path(it["path"]), max_bytes=256_000)
        kmidi = []
        ml = []
        music = []
        has_todos = False
        if text:
            lowered = text.lower()
            kmidi = [k for k in KMIDI_KEYWORDS if k.lower() in lowered]
            ml = [k for k in ML_KEYWORDS if k.lower() in lowered]
            music = [k for k in MUSIC_KEYWORDS if k.lower() in lowered]
            has_todos = bool(re.search(r"\b(TODO|FIXME|HACK|XXX)\b", text, re.IGNORECASE))

        mtime = _parse_mtime(it.get("modified") or "1970-01-01T00:00:00Z")
        score = value_score(
            size_bytes=int(it.get("size") or 0),
            mtime=mtime,
            kmidi_hits=len(kmidi),
            ml_hits=len(ml),
            music_hits=len(music),
            has_todos=has_todos,
        )
        ranked.append({**it, "value_score": round(score, 2), "value_bucket": value_bucket(score)})
    ranked.sort(key=lambda x: x["value_score"], reverse=True)
    return ranked[:max_total]


def main() -> int:
    home = HOME_DEFAULT
    db_path = out_dir() / "discovery_database.json"
    if not db_path.exists():
        print("Missing discovery database. Run generate_reports.py first.")
        return 2

    db = _load_db(db_path)
    git_roots = _find_git_roots(home)
    db["git_roots"] = [p.replace(str(home) + os.sep, "") if p.startswith(str(home) + os.sep) else p for p in git_roots]
    _write_json(db_path, db)

    # Regenerate project_directories.md with git repos populated.
    projects = db.get("projects") or {}
    lines = ["# Project-Like Directories", "", f"**Generated:** {now_iso()}", "", "## Git Repositories (detected)", ""]
    for p in db["git_roots"][:10_000]:
        lines.append(f"- `{p}`")
    lines += ["", "## Project Markers", ""]
    for marker, dirs in sorted(projects.items()):
        lines += [f"### {marker}", ""]
        for p in (dirs or [])[:10_000]:
            lines.append(f"- `{p}`")
        lines.append("")
    _write(out_dir() / "project_directories.md", "\n".join(lines).rstrip() + "\n")

    # Regenerate summary + integration candidates with vendor filtering.
    code_inventory = db["inventories"]["code"]
    docs_inventory = db["inventories"]["docs"]

    # Subset selection: take top-by-size lists (already sorted) and a small recent sample.
    def subset_from_groups(groups: Dict[str, List[Dict[str, Any]]], per_group: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for items in groups.values():
            out.extend(items[:per_group])
            out.extend(sorted(items[: min(len(items), per_group * 5)], key=lambda x: x.get("modified") or "", reverse=True)[:per_group])
        # Deduplicate by path
        by_path = {}
        for it in out:
            by_path[it["path"]] = it
        return list(by_path.values())

    code_subset = subset_from_groups(code_inventory, per_group=250)
    docs_subset = subset_from_groups(docs_inventory, per_group=250)

    ranked_code = _rank_subset(code_subset, max_total=800)
    ranked_docs = _rank_subset(docs_subset, max_total=400)

    def outside_kmidi1(rel_path: str) -> bool:
        rp = rel_path.replace("\\", "/")
        return not (rp.startswith("KmiDi-1/") or rp.startswith("KmiDi-1" + "/"))

    candidates = [it for it in ranked_code if outside_kmidi1(it.get("rel_path") or "") and it.get("value_bucket") in {"Critical", "High"}]
    candidates = candidates[:20]

    summary_lines = [
        "# Discovery Summary",
        "",
        f"**Generated:** {now_iso()}",
        "",
        "## Executive Summary",
        "",
        f"- Detected {len(db.get('git_roots', [])):,} Git repositories.",
        f"- Inventoried code/docs/models/large files in `docs/DISCOVERY/discovery_database.json`.",
        "",
        "## Top Integration Candidates (outside KmiDi-1, vendor-filtered)",
        "",
    ]
    for it in candidates:
        summary_lines.append(f"- `{it.get('rel_path')}` (score={it.get('value_score')})")
    _write(out_dir() / "DISCOVERY_SUMMARY.md", "\n".join(summary_lines).rstrip() + "\n")

    # Integration opportunities/recommendations (vendor-filtered)
    opps = [it for it in (ranked_code + ranked_docs) if outside_kmidi1(it.get("rel_path") or "") and it.get("value_bucket") in {"Critical", "High"}]
    opps.sort(key=lambda x: x.get("value_score", 0), reverse=True)

    opp_lines = ["# Integration Opportunities", "", f"**Generated:** {now_iso()}", "", "High-value files outside `KmiDi-1/` (vendor-filtered).", ""]
    for it in opps[:300]:
        kind = it.get("language") or it.get("type") or "file"
        opp_lines.append(f"- `{it.get('rel_path')}` ({kind}, {it.get('value_bucket')}, score={it.get('value_score')})")
    _write(out_dir() / "integration_opportunities.md", "\n".join(opp_lines).rstrip() + "\n")

    rec_lines = ["# Integration Recommendations", "", f"**Generated:** {now_iso()}", ""]
    rec_lines += ["## Immediate (review + copy/adapt)", ""]
    for it in opps[:50]:
        rec_lines.append(f"- `{it.get('rel_path')}`")
    rec_lines += ["", "## High Value (needs adaptation)", ""]
    for it in opps[50:150]:
        rec_lines.append(f"- `{it.get('rel_path')}`")
    rec_lines += ["", "## Archive/Reference", ""]
    for it in opps[150:250]:
        rec_lines.append(f"- `{it.get('rel_path')}`")
    _write(out_dir() / "integration_recommendations.md", "\n".join(rec_lines).rstrip() + "\n")

    print("✅ Post-processed:")
    print(f"  - {db_path}")
    print(f"  - {out_dir() / 'project_directories.md'}")
    print(f"  - {out_dir() / 'DISCOVERY_SUMMARY.md'}")
    print(f"  - {out_dir() / 'integration_opportunities.md'}")
    print(f"  - {out_dir() / 'integration_recommendations.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
