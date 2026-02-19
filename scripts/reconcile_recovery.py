#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# In-scope: source/config/docs/build metadata.
INCLUDE_EXTS = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx", ".m", ".mm", ".swift", ".rs",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".kts", ".go", ".rb", ".sh", ".bash", ".zsh",
    ".cmake", ".gradle", ".toml", ".yaml", ".yml", ".json", ".jsonc", ".ini", ".cfg", ".conf", ".plist", ".xml",
    ".md", ".txt", ".rst", ".adoc", ".markdown",
    ".css", ".scss", ".sass", ".html", ".htm", ".sql", ".proto",
    ".lock", ".gitignore", ".gitattributes",
}

EXCLUDE_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".icns", ".webp", ".svg",
    ".wav", ".mp3", ".aiff", ".flac", ".m4a", ".ogg", ".mid", ".midi",
    ".mp4", ".mov", ".avi", ".mkv",
    ".zip", ".tar", ".gz", ".7z", ".rar", ".dmg", ".pkg",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".so", ".dylib", ".dll", ".a", ".o", ".obj", ".class", ".jar", ".pyc",
    ".ttf", ".otf", ".woff", ".woff2",
}

DEFAULT_EXCLUDES = [
    "/forensics_histories/",
    "/_QUARANTINE_",
    "/trash/",
    "/.Trash/",
    "/logs/",
]

# Scoring weights for heuristic matching (sum to 1.0)
# Path similarity is weighted highest as directory structure is most stable
PATH_WEIGHT = 0.45  # Weight for directory path similarity
NAME_WEIGHT = 0.30  # Weight for filename similarity
EXT_WEIGHT = 0.20   # Weight for extension match
SIZE_WEIGHT = 0.05  # Weight for file size similarity

# Confidence thresholds for reconciliation decisions
CANDIDATE_MIN_SCORE = 0.60      # Minimum score to consider a file as a candidate match
HIGH_CONFIDENCE_SCORE = 0.92    # Score threshold for high-confidence heuristic matches
MIN_GAP_TO_SECOND = 0.03        # Minimum score gap between top and second candidate to avoid ambiguity
CONFLICT_MIN_SCORE = 0.75       # Minimum score to treat as potential conflict vs new file

@dataclass(frozen=True)
class CanonicalFile:
    repo_rel_path: str
    sha256: str
    size: int
    tracked: bool = True


@dataclass(frozen=True)
class PrecomputedCanonicalFile:
    """Canonical file with precomputed Path fields for efficient heuristic matching."""
    canonical: CanonicalFile
    normalized_path: str
    name: str
    dir: str
    ext: str



def run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    return subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, text=True).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(s: str) -> str:
    s = s.replace("\\", "/").lower()
    s = re.sub(r"[^a-z0-9/_\.-]", "", s)
    return s


def classify_hint(path: Path) -> str:
    p = normalize(str(path))
    ext = path.suffix.lower()
    if "/docs/" in p or ext in {".md", ".rst", ".txt", ".adoc", ".markdown"}:
        return "doc"
    if ext in {".cmake", ".toml", ".yaml", ".yml", ".json", ".jsonc", ".ini", ".cfg", ".conf", ".plist", ".xml"}:
        return "config"
    if path.name in {"CMakeLists.txt", "Makefile", "Dockerfile"}:
        return "build_meta"
    return "source"


def is_in_scope(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext in EXCLUDE_EXTS:
        return False
    if path.name in {"CMakeLists.txt", "Makefile", "Dockerfile"}:
        return True
    if ext in INCLUDE_EXTS:
        return True
    # Include extensionless text-like build scripts/config names.
    if ext == "" and path.name.lower() in {"readme", "license", "notice", "changelog"}:
        return True
    return False


def path_excluded(path_str: str, exclude_patterns: List[str]) -> bool:
    p = path_str.replace("\\", "/")
    return any(pat in p for pat in exclude_patterns)


def seq_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def size_score(a: int, b: int) -> float:
    if a == 0 and b == 0:
        return 1.0
    m = max(a, b, 1)
    return max(0.0, 1.0 - abs(a - b) / m)


def precompute_canonical(canonical: CanonicalFile) -> PrecomputedCanonicalFile:
    """Precompute normalized path components for efficient heuristic matching."""
    norm = normalize(canonical.repo_rel_path)
    p = Path(norm)
    return PrecomputedCanonicalFile(
        canonical=canonical,
        normalized_path=norm,
        name=p.name,
        dir=str(p.parent),
        ext=p.suffix,
    )


def heuristic_score_precomputed(
    rec_name: str, rec_dir: str, rec_ext: str, rec_size: int,
    precomp: PrecomputedCanonicalFile
) -> float:
    """Compute heuristic score using precomputed canonical file data."""
    path_sim = seq_sim(rec_dir, precomp.dir)
    name_sim = seq_sim(rec_name, precomp.name)
    ext_match = 1.0 if rec_ext == precomp.ext and rec_ext != "" else 0.0
    sscore = size_score(rec_size, precomp.canonical.size)
    return PATH_WEIGHT * path_sim + NAME_WEIGHT * name_sim + EXT_WEIGHT * ext_match + SIZE_WEIGHT * sscore


def heuristic_score(recovered_rel: str, recovered_size: int, cand_rel: str, cand_size: int) -> float:
    """Legacy heuristic score function for backward compatibility."""
    rec_norm = normalize(recovered_rel)
    cand_norm = normalize(cand_rel)

    rec_name = Path(rec_norm).name
    cand_name = Path(cand_norm).name
    rec_dir = str(Path(rec_norm).parent)
    cand_dir = str(Path(cand_norm).parent)
    rec_ext = Path(rec_norm).suffix
    cand_ext = Path(cand_norm).suffix

    path_sim = seq_sim(rec_dir, cand_dir)
    name_sim = seq_sim(rec_name, cand_name)
    ext_match = 1.0 if rec_ext == cand_ext and rec_ext != "" else 0.0
    sscore = size_score(recovered_size, cand_size)
    return PATH_WEIGHT * path_sim + NAME_WEIGHT * name_sim + EXT_WEIGHT * ext_match + SIZE_WEIGHT * sscore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--source-root", action="append", required=True)
    parser.add_argument("--recovered-list-file", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--matcher-version", default="v1.0.0")
    parser.add_argument("--remote-url", required=True)
    parser.add_argument("--baseline-ref", default="origin/main")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    report_dir = workspace / "recovery_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    exclude_patterns = list(DEFAULT_EXCLUDES)
    for e in args.exclude:
        if e not in exclude_patterns:
            exclude_patterns.append(e)

    baseline_sha = run(["git", "rev-parse", args.baseline_ref], cwd=workspace)

    # Canonical inventory from git ls-files.
    tracked_files = run(["git", "ls-files"], cwd=workspace).splitlines()
    canonical: List[CanonicalFile] = []
    canonical_by_rel: Dict[str, CanonicalFile] = {}
    canonical_by_hash: Dict[str, List[CanonicalFile]] = {}

    inventory_canonical_path = report_dir / "inventory_canonical.jsonl"
    with inventory_canonical_path.open("w", encoding="utf-8") as out:
        for rel in sorted(tracked_files):
            p = workspace / rel
            if not p.is_file():
                continue
            if not is_in_scope(p):
                continue
            sha = sha256_file(p)
            size = p.stat().st_size
            c = CanonicalFile(repo_rel_path=rel, sha256=sha, size=size, tracked=True)
            canonical.append(c)
            canonical_by_rel[rel] = c
            canonical_by_hash.setdefault(sha, []).append(c)
            out.write(json.dumps({
                "repo_rel_path": rel,
                "sha256": sha,
                "size": size,
                "tracked": True,
                "branch_ref": args.baseline_ref,
            }, sort_keys=True) + "\n")

    # Recovered inventory.
    recovered_records = []
    inventory_recovered_path = report_dir / "inventory_recovered.jsonl"

    manual_recovered_paths: List[Path] = []
    if args.recovered_list_file:
        seen_manual = set()
        for list_file in args.recovered_list_file:
            lf = Path(list_file)
            if not lf.exists():
                continue
            for raw in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                s = raw.strip()
                if not s:
                    continue
                p = Path(s).resolve()
                if str(p) not in seen_manual:
                    seen_manual.add(str(p))
                    manual_recovered_paths.append(p)

    with inventory_recovered_path.open("w", encoding="utf-8") as out:
        if manual_recovered_paths:
            roots = [Path(s).resolve() for s in sorted(set(args.source_root))]
            for p in sorted(manual_recovered_paths):
                if not p.is_file():
                    continue
                p_str = str(p)
                if path_excluded(p_str, exclude_patterns):
                    continue
                if not is_in_scope(p):
                    continue

                source_root = ""
                rel_hint = p.name
                for root in roots:
                    if p.is_relative_to(root):
                        rel_hint = str(p.relative_to(root)).replace("\\", "/")
                        source_root = str(root)
                        break
                if not source_root:
                    source_root = str(p.parent)

                try:
                    sha = sha256_file(p)
                    size = p.stat().st_size
                except (OSError, PermissionError):
                    continue

                rec = {
                    "abs_path": p_str,
                    "source_root": source_root,
                    "rel_hint": rel_hint,
                    "ext": p.suffix.lower(),
                    "size": size,
                    "sha256": sha,
                    "mtime": int(p.stat().st_mtime),
                    "class_hint": classify_hint(p),
                }
                recovered_records.append(rec)
                out.write(json.dumps(rec, sort_keys=True) + "\n")
        else:
            for src_root in sorted(set(args.source_root)):
                root = Path(src_root).resolve()
                if not root.exists():
                    continue
                for p in sorted(root.rglob("*")):
                    if not p.is_file():
                        continue
                    p_str = str(p)
                    if path_excluded(p_str, exclude_patterns):
                        continue
                    if not is_in_scope(p):
                        continue

                    if p.is_relative_to(root):
                        rel_hint = str(p.relative_to(root)).replace("\\", "/")
                    else:
                        rel_hint = p.name

                    try:
                        sha = sha256_file(p)
                        size = p.stat().st_size
                    except (OSError, PermissionError):
                        continue

                    rec = {
                        "abs_path": p_str,
                        "source_root": str(root),
                        "rel_hint": rel_hint,
                        "ext": p.suffix.lower(),
                        "size": size,
                        "sha256": sha,
                        "mtime": int(p.stat().st_mtime),
                        "class_hint": classify_hint(p),
                    }
                    recovered_records.append(rec)
                    out.write(json.dumps(rec, sort_keys=True) + "\n")

    recovered_records.sort(key=lambda r: (r["source_root"], r["rel_hint"], r["abs_path"]))

    matches_path = report_dir / "matches.jsonl"
    decisions_path = report_dir / "decisions.csv"
    review_queue_path = report_dir / "review_queue.csv"

    counts = {
        "exact_match": 0,
        "duplicate_or_move": 0,
        "candidate_match": 0,
        "conflict": 0,
        "new_file": 0,
        "excluded": 0,
    }

    with matches_path.open("w", encoding="utf-8") as m_out, \
        decisions_path.open("w", encoding="utf-8", newline="") as d_out, \
        review_queue_path.open("w", encoding="utf-8", newline="") as q_out:

        d_writer = csv.DictWriter(d_out, fieldnames=[
            "status", "recovered_abs_path", "destination_path", "action", "confidence", "reviewer_note"
        ])
        d_writer.writeheader()

        q_writer = csv.DictWriter(q_out, fieldnames=[
            "status", "recovered_abs_path", "top_candidate", "top_score", "second_candidate", "second_score", "reason"
        ])
        q_writer.writeheader()

        # Precompute canonical file data and index by extension for efficient lookup
        precomputed_canonical: List[PrecomputedCanonicalFile] = [
            precompute_canonical(c) for c in canonical
        ]
        canonical_by_ext: Dict[str, List[PrecomputedCanonicalFile]] = {}
        for pc in precomputed_canonical:
            ext = pc.ext
            canonical_by_ext.setdefault(ext, []).append(pc)

        for rec in recovered_records:
            rel_hint = rec["rel_hint"]
            sha = rec["sha256"]
            rec_size = rec["size"]
            abs_path = rec["abs_path"]

            status = ""
            action = ""
            confidence = 0.0
            reason = ""
            destination = ""
            top_candidate = ""
            top_score = 0.0
            second_candidate = ""
            second_score = 0.0

            c_rel = canonical_by_rel.get(rel_hint)
            if c_rel and c_rel.sha256 == sha:
                status = "exact_match"
                action = "auto_apply_pending_remote_gate"
                confidence = 1.0
                destination = c_rel.repo_rel_path
                reason = "same_rel_path_and_hash"
            else:
                hash_hits = canonical_by_hash.get(sha, [])
                if hash_hits:
                    chosen = sorted(hash_hits, key=lambda x: x.repo_rel_path)[0]
                    status = "duplicate_or_move"
                    action = "auto_apply_pending_remote_gate"
                    confidence = 0.99
                    destination = chosen.repo_rel_path
                    reason = "same_hash_different_path"
                else:
                    # Heuristic search using precomputed canonical data, indexed by extension
                    rec_norm = normalize(rel_hint)
                    rec_p = Path(rec_norm)
                    rec_name = rec_p.name
                    rec_dir = str(rec_p.parent)
                    rec_ext = rec_p.suffix
                    
                    # Get candidates with matching extension, or all if no extension
                    candidates = canonical_by_ext.get(rec_ext, []) if rec_ext else precomputed_canonical
                    scored: List[Tuple[float, PrecomputedCanonicalFile]] = []
                    for pc in candidates:
                        score = heuristic_score_precomputed(rec_name, rec_dir, rec_ext, rec_size, pc)
                        if score >= CANDIDATE_MIN_SCORE:
                            scored.append((score, pc))
                    scored.sort(key=lambda t: (-t[0], t[1].canonical.repo_rel_path))

                    if scored:
                        top_score, top_pc = scored[0]
                        top_candidate = top_pc.canonical.repo_rel_path
                        if len(scored) > 1:
                            second_score, second_pc = scored[1]
                            second_candidate = second_pc.canonical.repo_rel_path

                        if top_score >= HIGH_CONFIDENCE_SCORE and (top_score - second_score) > MIN_GAP_TO_SECOND:
                            status = "candidate_match"
                            action = "manual_review_required"
                            confidence = round(top_score, 4)
                            destination = top_candidate
                            reason = "high_confidence_heuristic"
                        elif top_score >= CONFLICT_MIN_SCORE:
                            status = "conflict"
                            action = "manual_review_required"
                            confidence = round(top_score, 4)
                            destination = top_candidate
                            reason = "ambiguous_or_medium_confidence_heuristic"
                        else:
                            status = "new_file"
                            action = "manual_review_required"
                            confidence = round(top_score, 4)
                            destination = ""
                            reason = "no_valid_candidate_above_threshold"
                    else:
                        status = "new_file"
                        action = "manual_review_required"
                        confidence = 0.0
                        destination = ""
                        reason = "no_candidate"

            counts[status] += 1
            m_out.write(json.dumps({
                "recovered_abs_path": abs_path,
                "recovered_rel_hint": rel_hint,
                "candidate_repo_path": destination,
                "match_type": status,
                "confidence": confidence,
                "reason": reason,
                "source_priority": "KmiDi-remote-first",
                "sha_equal": status in {"exact_match", "duplicate_or_move"},
                "name_equal": Path(rel_hint).name == Path(destination).name if destination else False,
                "top_candidate": top_candidate,
                "top_score": round(top_score, 4),
                "second_candidate": second_candidate,
                "second_score": round(second_score, 4),
            }, sort_keys=True) + "\n")

            d_writer.writerow({
                "status": status,
                "recovered_abs_path": abs_path,
                "destination_path": destination,
                "action": action,
                "confidence": f"{confidence:.4f}",
                "reviewer_note": reason,
            })

            if status in {"candidate_match", "conflict", "new_file"}:
                q_writer.writerow({
                    "status": status,
                    "recovered_abs_path": abs_path,
                    "top_candidate": top_candidate,
                    "top_score": f"{top_score:.4f}",
                    "second_candidate": second_candidate,
                    "second_score": f"{second_score:.4f}",
                    "reason": reason,
                })

    # Run metadata
    run_meta_path = report_dir / "run_meta.json"
    run_meta = {
        "repo_url": args.remote_url,
        "baseline_ref": args.baseline_ref,
        "baseline_sha": baseline_sha,
        "branch_name": run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=workspace),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selected_source_roots": sorted(set(args.source_root)),
        "exclusion_patterns": exclude_patterns,
        "matcher_version": args.matcher_version,
    }
    run_meta_path.write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Patchset plan markdown
    patchset_plan_path = report_dir / "patchset_plan.md"

    # Derive remote gate status from an actual remote check.
    remote_gate_lines: List[str] = ["## Remote Gate\n"]
    try:
        # Use a non-destructive dry-run fetch to validate connectivity / ref.
        fetch_cmd = ["git", "fetch", "--dry-run", args.remote_url, args.baseline_ref]
        fetch_result = subprocess.run(
            fetch_cmd,
            cwd=workspace,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if fetch_result.returncode == 0:
            remote_gate_lines.append(
                "- Online source-of-truth refresh check succeeded for this run.\n"
            )
        else:
            remote_gate_lines.append(
                "- Online source-of-truth refresh check failed or was not performed in this run.\n"
            )
    except Exception:
        remote_gate_lines.append(
            "- Online source-of-truth refresh check failed or was not performed in this run.\n"
        )
    # This script does not auto-apply any reconciled file changes.
    remote_gate_lines.append(
        "- No reconciled file changes were auto-applied by this script.\n\n"
    )
    remote_gate_section = "".join(remote_gate_lines)

    patchset_plan_path.write_text(
        "# Patchset Plan (Safe Staged Apply)\n\n"
        f"{remote_gate_section}"
        "## Batch A (Exact)\n"
        f"- exact_match count: {counts['exact_match']}\n"
        "- action: auto-apply when remote gate is cleared.\n\n"
        "## Batch B (Duplicate/Move)\n"
        f"- duplicate_or_move count: {counts['duplicate_or_move']}\n"
        "- action: apply only deterministic destination mappings.\n\n"
        "## Batch C (Heuristic Candidates)\n"
        f"- candidate_match count: {counts['candidate_match']}\n"
        "- action: manual approval required.\n\n"
        "## Batch D (Conflicts / New Files)\n"
        f"- conflict count: {counts['conflict']}\n"
        f"- new_file count: {counts['new_file']}\n"
        "- action: resolve via review_queue.csv before apply.\n\n"
        "## Rollback\n"
        "- Revert staged recovery commits only (no history rewrite).\n"
        "- Keep `recovery_reports/` for audit.\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "canonical_files": len(canonical),
        "recovered_files": len(recovered_records),
        "counts": counts,
        "report_dir": str(report_dir),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
