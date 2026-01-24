#!/usr/bin/env python3
"""
Integration-focused review (prioritized):

Priority 1: Review 20 ML models for integration
Priority 2: Evaluate 30 high-complexity code modules
Priority 3: Review 20 KmiDi-related documentation files

Outputs (docs/DISCOVERY):
- INTEGRATION_SUMMARY.md
- integration_recommendations.csv
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from discovery_common import (
    HOME_DEFAULT,
    KMIDI_KEYWORDS,
    ML_KEYWORDS,
    MUSIC_KEYWORDS,
    now_iso,
    read_text_sample,
    value_bucket,
    value_score,
)


RE_TODO = re.compile(r"\b(TODO|FIXME|HACK|XXX|IMPORTANT|NOTE)\b", re.IGNORECASE)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def out_dir() -> Path:
    return repo_root() / "docs" / "DISCOVERY"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_iso(ts: str) -> datetime:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def is_vendor_path(rel_path: str) -> bool:
    rp = rel_path.replace("\\", "/").lower()
    markers = [
        "site-packages/",
        "dist-packages/",
        "node_modules/",
        ".cache/",
        ".cursor/",
        ".npm/",
        ".cargo/",
        ".rustup/",
        ".gradle/",
        "venv/",
        ".venv/",
        "/build/",
        "/dist/",
        "/target/",
    ]
    return any(m in rp for m in markers)


def keyword_hits(text: str, keywords: Sequence[str]) -> List[str]:
    lowered = text.lower()
    return sorted({k for k in keywords if k.lower() in lowered}, key=str.lower)


def infer_task_from_name(name: str) -> str:
    n = name.lower()
    if "melody" in n:
        return "melody_generation"
    if "harmony" in n or "chord" in n:
        return "harmony_prediction"
    if "groove" in n or "beat" in n:
        return "groove_prediction"
    if "dynamic" in n:
        return "dynamics_mapping"
    if "emotion" in n:
        # Often either embedding or classification; keep generic
        return "emotion_embedding"
    if "intent" in n:
        return "intent_mapping"
    if "key" in n:
        return "key_detection"
    if "instrument" in n:
        return "custom"
    return "custom"


def read_rtneural_header(path: Path) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    RTNeural JSON exports store model_name/input_size/output_size at the top.
    Avoid full json.load on large files; just regex the first chunk.
    """
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            head = f.read(16 * 1024)
    except Exception:
        return None, None, None
    name = None
    ins = None
    outs = None
    m = re.search(r"\"model_name\"\s*:\s*\"([^\"]+)\"", head)
    if m:
        name = m.group(1)
    m = re.search(r"\"input_size\"\s*:\s*(\d+)", head)
    if m:
        ins = int(m.group(1))
    m = re.search(r"\"output_size\"\s*:\s*(\d+)", head)
    if m:
        outs = int(m.group(1))
    return name, ins, outs


@dataclass
class Recommendation:
    priority: int
    category: str  # model|code|doc
    id: str
    task: str
    format: str
    source_path: str
    target_path: str
    size_mb: float
    modified: str
    readiness: str
    notes: str


def select_models(repo: Path) -> Tuple[List[Dict[str, Any]], List[Recommendation]]:
    home = HOME_DEFAULT
    selected: List[Dict[str, Any]] = []
    recs: List[Recommendation] = []

    # 1) RTNeural JSON models that match C++ pipeline sizes.
    rt_dir = repo / "KmiDi_FINAL" / "ml" / "models"
    pipeline_map = [
        ("emotion", "emotionrecognizer.json"),
        ("melody", "melodytransformer.json"),
        ("harmony", "harmonypredictor.json"),
        ("dynamics", "dynamicsengine.json"),
        ("groove", "groovepredictor.json"),
    ]
    for short, fname in pipeline_map:
        src = rt_dir / fname
        if not src.exists():
            continue
        st = src.stat()
        model_name, ins, outs = read_rtneural_header(src)
        model_id = model_name or src.stem
        task = infer_task_from_name(model_id)
        target = f"models/{short}.json"
        selected.append(
            {
                "id": model_id,
                "task": task,
                "format": "rtneural-json",
                "source_path": str(src),
                "target_path": target,
                "size_mb": st.st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                "input_size": ins,
                "output_size": outs,
                "readiness": "ready (C++ RTNeural)",
                "notes": f"Matches KellyMLPipeline::loadModels sizes ({ins}→{outs}).",
            }
        )
        recs.append(
            Recommendation(
                priority=1,
                category="model",
                id=model_id,
                task=task,
                format="rtneural-json",
                source_path=str(src),
                target_path=target,
                size_mb=round(st.st_size / (1024 * 1024), 3),
                modified=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                readiness="ready",
                notes="Symlink into models/*.json; requires ENABLE_RTNEURAL at build time.",
            )
        )

    # 2) Additional RTNeural models in KmiDi_FINAL (not directly wired).
    extra_rt = []
    if rt_dir.exists():
        for p in rt_dir.glob("*.json"):
            if p.name in {x[1] for x in pipeline_map}:
                continue
            if p.name in {"registry.json", "registry.schema.json", "validation_results.json", "llama_onnx.json", "README.md"}:
                continue
            st = p.stat()
            model_name, ins, outs = read_rtneural_header(p)
            model_id = model_name or p.stem
            extra_rt.append(
                {
                    "id": model_id,
                    "task": infer_task_from_name(model_id),
                    "format": "rtneural-json",
                    "source_path": str(p),
                    "target_path": f"models/rtneural/{p.name}",
                    "size_mb": st.st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                    "input_size": ins,
                    "output_size": outs,
                    "readiness": "needs mapping",
                    "notes": "Useful RTNeural export; not currently used by KellyMLPipeline.",
                }
            )
    extra_rt.sort(key=lambda x: x["size_mb"], reverse=True)
    for it in extra_rt[:5]:
        selected.append(it)
        recs.append(
            Recommendation(
                priority=1,
                category="model",
                id=it["id"],
                task=it["task"],
                format="rtneural-json",
                source_path=it["source_path"],
                target_path=it["target_path"],
                size_mb=round(it["size_mb"], 3),
                modified=it["modified"],
                readiness="review",
                notes="Consider wiring to a feature path or adding a dedicated RTNeural loader integration.",
            )
        )

    # 3) Select PyTorch checkpoints under repo (exclude venv/vendor).
    def walk_for_models(base: Path) -> List[Path]:
        out: List[Path] = []
        if not base.exists():
            return out
        for dirpath, dirnames, filenames in os.walk(base):
            dp = Path(dirpath)
            rel = str(dp.relative_to(repo)).replace("\\", "/")
            if is_vendor_path(rel):
                dirnames[:] = []
                continue
            for name in filenames:
                if name.endswith((".pt", ".pth", ".onnx", ".tflite", ".mlmodel", ".pkl", ".pickle", ".npy", ".npz")):
                    out.append(dp / name)
        return out

    pytorch_candidates: List[Dict[str, Any]] = []
    for base in [
        repo / "models",
        repo / "python" / "penta_core" / "ml" / "checkpoints",
        repo / "KmiDi_TRAINING",
        repo / "KmiDi_BACKUP",
        repo / "training",
    ]:
        for p in walk_for_models(base):
            st = p.stat()
            model_id = p.stem
            task = infer_task_from_name(model_id)
            fmt = p.suffix.lower().lstrip(".")
            if p.suffix.lower() in {".pt", ".pth"}:
                fmt = "pytorch"
            pytorch_candidates.append(
                {
                    "id": model_id,
                    "task": task,
                    "format": fmt,
                    "source_path": str(p),
                    "target_path": f"Data_Files/models/external_candidates/{p.name}",
                    "size_mb": st.st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                    "readiness": "needs verification",
                    "notes": "Verify whether this is TorchScript (torch.jit.load) or state_dict; consider exporting to ONNX.",
                }
            )

    # Prefer *_best.pt and latest checkpoints
    def model_rank(it: Dict[str, Any]) -> Tuple[int, float, str]:
        name = Path(it["source_path"]).name.lower()
        preferred = 0
        if "best" in name:
            preferred = 3
        elif "latest" in name:
            preferred = 2
        elif "checkpoint_epoch" in name:
            preferred = 1
        return (preferred, float(it["size_mb"]), it["modified"])

    pytorch_candidates.sort(key=model_rank, reverse=True)

    # Deduplicate by basename
    seen_names: set[str] = set()
    for it in pytorch_candidates:
        base = Path(it["source_path"]).name.lower()
        if base in seen_names:
            continue
        seen_names.add(base)
        selected.append(it)
        recs.append(
            Recommendation(
                priority=1,
                category="model",
                id=it["id"],
                task=it["task"],
                format=it["format"],
                source_path=it["source_path"],
                target_path=it["target_path"],
                size_mb=round(it["size_mb"], 3),
                modified=it["modified"],
                readiness="review",
                notes=it["notes"],
            )
        )
        if len(selected) >= 20:
            break

    return selected[:20], recs[:20]


def normalize_for_dedupe(rel_path: str) -> str:
    rp = rel_path.replace("\\", "/")
    # Strip common archive prefixes to reduce duplicates.
    for prefix in [
        "RECOVERY_OPS/",
        "RECOVERY_OPS/ARCHIVE/",
        "RECOVERY_OPS/CANONICAL/",
        "RECOVERY_OPS/sbdrive/",
        "RECOVERY_OPS/KmiDi/",
        "RECOVERY_OPS/KmiDi/ARCHIVE/",
        "RECOVERY_OPS/KmiDi/CANONICAL/",
        "RECOVERY_OPS/KmiDi/_sorted/",
        "_sorted/",
        "My Mac/",
        "Downloads/",
        "Desktop/",
    ]:
        if rp.startswith(prefix):
            rp = rp[len(prefix) :]
            break
    parts = rp.split("/")
    # Keep tail segments for stability.
    return "/".join(parts[-5:]).lower()


def select_code_modules() -> Tuple[List[Dict[str, Any]], List[Recommendation]]:
    db = load_json(out_dir() / "discovery_database.json")
    code_groups = db["inventories"]["code"]

    candidates: List[Dict[str, Any]] = []
    for lang, items in code_groups.items():
        for it in items:
            rp = it.get("rel_path") or ""
            if is_vendor_path(rp):
                continue
            # Prefer likely project code
            rp_l = rp.lower()
            if not any(k in rp_l for k in ["kmidi", "music_brain", "penta_core", "kelly", "training", "src/", "python/"]):
                continue
            candidates.append(it)

    # Simple complexity: size + counts of defs/imports (sampled) + keyword boosts
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for it in candidates[:8000]:
        path = Path(it["path"])
        text = read_text_sample(path, max_bytes=256_000) or ""
        if not text:
            continue
        rp = it.get("rel_path") or ""
        defs = len(re.findall(r"\b(def|class)\b", text))
        imports = len(re.findall(r"^(\s*(from\s+\S+\s+import|import\s+\S+))", text, re.MULTILINE))
        braces = len(re.findall(r"[{}]", text))
        templates = len(re.findall(r"\btemplate\b", text))
        todo = 1 if RE_TODO.search(text) else 0
        kmidi = len(keyword_hits(text, KMIDI_KEYWORDS))
        ml = len(keyword_hits(text, ML_KEYWORDS))
        music = len(keyword_hits(text, MUSIC_KEYWORDS))
        base = float(it.get("size") or 0) / 1024.0
        score = base + (defs * 40) + (imports * 15) + (braces * 0.1) + (templates * 25) + (todo * 50)
        score += (kmidi * 100) + (ml * 40) + (music * 30)
        scored.append((score, {**it, "complexity_score": round(score, 2), "defs": defs, "imports": imports}))

    scored.sort(key=lambda x: x[0], reverse=True)
    # Deduplicate near-identical copies across RECOVERY_OPS/_sorted/etc.
    seen: set[str] = set()
    top: List[Dict[str, Any]] = []
    for _, it in scored:
        key = normalize_for_dedupe(it.get("rel_path") or it["path"])
        if key in seen:
            continue
        seen.add(key)
        top.append(it)
        if len(top) >= 30:
            break

    recs: List[Recommendation] = []
    for it in top:
        recs.append(
            Recommendation(
                priority=2,
                category="code",
                id=Path(it["path"]).stem,
                task=it.get("language") or "code",
                format=it.get("language") or "code",
                source_path=it["path"],
                target_path="(review)",
                size_mb=round((it.get("size") or 0) / (1024 * 1024), 3),
                modified=it.get("modified") or "",
                readiness="review",
                notes=f"complexity_score={it.get('complexity_score')}, defs/classes={it.get('defs')}, imports={it.get('imports')}",
            )
        )
    return top, recs


def select_docs() -> Tuple[List[Dict[str, Any]], List[Recommendation]]:
    db = load_json(out_dir() / "discovery_database.json")
    docs_groups = db["inventories"]["docs"]
    candidates: List[Dict[str, Any]] = []
    for doc_type, items in docs_groups.items():
        for it in items:
            rp = it.get("rel_path") or ""
            if is_vendor_path(rp):
                continue
            rp_l = rp.lower()
            # Skip vendored docs and build dependencies.
            if any(x in rp_l for x in ["/_deps/", "/node_modules/", "/site-packages/"]):
                continue
            rp_l = rp.lower()
            if any(k in rp_l for k in ["kmidi", "music_brain", "penta_core", "kelly", "midi", "music"]):
                candidates.append(it)

    scored: List[Tuple[float, Dict[str, Any], List[str], List[str]]] = []
    for it in candidates[:6000]:
        text = read_text_sample(Path(it["path"]), max_bytes=256_000)
        if text is None:
            continue
        kmidi = keyword_hits(text, KMIDI_KEYWORDS)
        music = keyword_hits(text, MUSIC_KEYWORDS)
        ml = keyword_hits(text, ML_KEYWORDS)
        if not (kmidi or music or ml):
            continue
        todos = [ln.strip() for ln in text.splitlines() if RE_TODO.search(ln)][:8]
        score = value_score(
            size_bytes=int(it.get("size") or 0),
            mtime=parse_iso(it.get("modified") or "1970-01-01T00:00:00Z"),
            kmidi_hits=len(kmidi),
            ml_hits=len(ml),
            music_hits=len(music),
            has_todos=bool(todos),
        )
        scored.append((score, {**it, "value_score": round(score, 2), "value_bucket": value_bucket(score)}, kmidi + music, todos))

    scored.sort(key=lambda x: x[0], reverse=True)
    # Deduplicate near-identical copies across RECOVERY_OPS/_sorted/etc.
    seen: set[str] = set()
    top: List[Dict[str, Any]] = []
    for _, it, _, _ in scored:
        key = normalize_for_dedupe(it.get("rel_path") or it["path"])
        if key in seen:
            continue
        seen.add(key)
        top.append(it)
        if len(top) >= 20:
            break
    recs: List[Recommendation] = []
    for it in top:
        recs.append(
            Recommendation(
                priority=3,
                category="doc",
                id=Path(it["path"]).stem,
                task=it.get("type") or "doc",
                format=it.get("type") or "doc",
                source_path=it["path"],
                target_path="(review)",
                size_mb=round((it.get("size") or 0) / (1024 * 1024), 3),
                modified=it.get("modified") or "",
                readiness="review",
                notes=f"{it.get('value_bucket')} score={it.get('value_score')}",
            )
        )
    return top, recs


def write_csv(path: Path, recs: Sequence[Recommendation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "priority",
                "category",
                "id",
                "task",
                "format",
                "source_path",
                "target_path",
                "size_mb",
                "modified",
                "readiness",
                "notes",
            ]
        )
        for r in recs:
            w.writerow(
                [
                    r.priority,
                    r.category,
                    r.id,
                    r.task,
                    r.format,
                    r.source_path,
                    r.target_path,
                    f"{r.size_mb:.3f}",
                    r.modified,
                    r.readiness,
                    r.notes,
                ]
            )


def write_summary(
    path: Path,
    *,
    models: Sequence[Dict[str, Any]],
    code: Sequence[Dict[str, Any]],
    docs: Sequence[Dict[str, Any]],
    csv_path: Path,
) -> None:
    lines: List[str] = []
    lines += ["# Integration Summary", "", f"**Generated:** {now_iso()}", ""]
    lines += [
        "This report follows the requested priorities:",
        "",
        "- Priority 1: 20 ML models reviewed for integration",
        "- Priority 2: 30 high-complexity code modules evaluated",
        "- Priority 3: 20 KmiDi-related documentation files reviewed",
        "",
        f"Open `{csv_path.relative_to(repo_root())}` for the prioritized list.",
        "",
    ]

    lines += ["## Priority 1 — ML Models (20)", ""]
    for it in models[:20]:
        sp = it["source_path"]
        tp = it.get("target_path", "")
        dims = ""
        if it.get("input_size") is not None and it.get("output_size") is not None:
            dims = f", io={it['input_size']}→{it['output_size']}"
        lines.append(
            f"- `{sp}` ({it.get('format')}, {it.get('task')}, {it.get('size_mb'):.2f} MB{dims})"
            + (f" -> `{tp}`" if tp else "")
            + (f" — {it.get('notes')}" if it.get("notes") else "")
        )

    lines += ["", "## Priority 2 — High-Complexity Code Modules (30)", ""]
    for it in code[:30]:
        rel = it.get("rel_path") or it.get("path")
        lines.append(
            f"- `{rel}` ({it.get('language')}, {it.get('size',0)/1024:.1f} KB, complexity={it.get('complexity_score')}, defs={it.get('defs')}, imports={it.get('imports')})"
        )

    lines += ["", "## Priority 3 — KmiDi-Related Documentation (20)", ""]
    for it in docs[:20]:
        rel = it.get("rel_path") or it.get("path")
        lines.append(f"- `{rel}` ({it.get('type')}, {it.get('size',0)/1024:.1f} KB, {it.get('modified','')[:10]})")

    lines += [
        "",
        "## Start Integration (Priority 1)",
        "",
        "Immediate wiring for C++ RTNeural pipeline:",
        "",
        "- `models/emotion.json` -> `KmiDi_FINAL/ml/models/emotionrecognizer.json`",
        "- `models/melody.json` -> `KmiDi_FINAL/ml/models/melodytransformer.json`",
        "- `models/harmony.json` -> `KmiDi_FINAL/ml/models/harmonypredictor.json`",
        "- `models/dynamics.json` -> `KmiDi_FINAL/ml/models/dynamicsengine.json`",
        "- `models/groove.json` -> `KmiDi_FINAL/ml/models/groovepredictor.json`",
        "",
        "Notes:",
        "- Requires `ENABLE_RTNEURAL` at build time (otherwise loader returns false).",
        "- The pipeline already has rule-based fallbacks if models fail to load.",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    repo = repo_root()
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)

    models, model_recs = select_models(repo)
    code, code_recs = select_code_modules()
    docs, doc_recs = select_docs()

    all_recs = list(model_recs) + list(code_recs) + list(doc_recs)
    # Stable ordering: priority then category then score-ish size
    all_recs.sort(key=lambda r: (r.priority, r.category, -r.size_mb))

    csv_path = out / "integration_recommendations.csv"
    write_csv(csv_path, all_recs)

    summary_path = out / "INTEGRATION_SUMMARY.md"
    write_summary(summary_path, models=models, code=code, docs=docs, csv_path=csv_path)

    print("✅ Wrote:")
    print(f"  - {summary_path}")
    print(f"  - {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
