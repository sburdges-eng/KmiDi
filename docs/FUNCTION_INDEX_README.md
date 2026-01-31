# Function & path index — future search and restore

**Purpose:** Repo-wide index of function/class/symbol paths so agents and tools can find where code lives without recreating it. Use with `docs/GIT_RESTORE_PATHWAYS.md` for git history restore.

**References:** `docs/GIT_RESTORE_PATHWAYS.md`, `scripts/build_function_index.py`

---

## 1. What is indexed

| Kind | Languages | Contents |
|------|-----------|----------|
| **def / class / async def** | Python (`.py`) | Function and class names with file path and line number |
| **fn / struct / enum / impl / trait** | Rust (`.rs`) | Same |
| **function / class / struct** | C/C++ (`.cpp`, `.c`, `.h`, `.hpp`, etc.) | Declarations and definitions |

Excluded: `node_modules`, `__pycache__`, `.git`, `venv`, `build`, `dist`, `target`, binary suffixes (`.pyc`, `.gguf`, etc.).

---

## 2. Index files

| File | Scope | Size (approx) | Committed? |
|------|--------|----------------|------------|
| `docs/.index/symbol_index_canon.tsv` | KmiDi_CANON, run_brain.py, tests/, src/ | ~6k symbols, ~400 KB | Yes (recommended) |
| `docs/.index/symbol_index.tsv` | Full repo | ~410k symbols, ~48 MB | No (regenerate locally) |
| `docs/.index/symbol_index.jsonl.gz` | Full repo | ~3.6 MB gzipped | No (regenerate locally) |

Format:

- **TSV:** `path\tline\tkind\tsymbol` (header on first line). Use `grep`, `awk`, or any tab tool.
- **JSONL.gz:** One JSON object per line: `{"path":"...","line":N,"kind":"def","symbol":"..."}`. Use `zgrep` or decompress and stream.

---

## 3. How to regenerate

From repo root:

```bash
# Canon-only (KmiDi_CANON + run_brain + tests + src) — small, committable
python3 scripts/build_function_index.py --canon-only

# Full repo (~410k symbols, 20k+ files) — for local “future use”
python3 scripts/build_function_index.py
```

Output paths: `docs/.index/symbol_index.tsv`, `symbol_index_canon.tsv`, `symbol_index.jsonl.gz` (full only).

---

## 4. How to search the index

**Find where a symbol is defined:**

```bash
# TSV (canon or full)
grep -F "parse_user_intent" docs/.index/symbol_index_canon.tsv
grep -F "CompleteSongIntent" docs/.index/symbol_index_canon.tsv

# Full index (if generated)
grep -F "IntentPipeline" docs/.index/symbol_index.tsv
```

**Find all symbols in a path prefix:**

```bash
awk -F'\t' '$1 ~ /^KmiDi_CANON\/brain\/music_brain\/session\//' docs/.index/symbol_index_canon.tsv
```

**Search compressed JSONL (full index):**

```bash
zgrep -F '"parse_user_intent"' docs/.index/symbol_index.jsonl.gz
```

---

## 5. Step 1 — Recovery/reimplementation: find in git before recreate

**Use this first** when recovering or reimplementing; find in history then restore or diff.

### 5.1 Commands (this repo)

```bash
cd "/Users/seanburdges/Dev/KmiDi MIDI Companion"

# Pickaxe: commits where a symbol was added/removed
git log -S "parse_user_intent" --oneline -- "*.py"
git log -S "LLMReasoningEngine" --oneline -- "*.py"
git log -S "create_engine_by_name" --oneline -- "*.py"
git log -S "JEPA" --oneline -- "**/*"
git log -S "stem_jepa" --oneline -- "**/*"
git log -S "IntentPipeline" --oneline -- "**/*"

# Which commits touched a path
git log --all --oneline -- "KmiDi_CANON/brain/mcp_workstation/llm_reasoning_engine.py"
git log --all --oneline -- "KmiDi_CANON/brain/penta_core/ml/*"
git log --all --oneline -- "KmiDi_CANON/body/engine/*"
git log --all --oneline -- "src/**/*"

# Show file at a commit (restore or diff)
git show <commit>:KmiDi_CANON/brain/mcp_workstation/llm_reasoning_engine.py
git show <commit>:KmiDi_CANON/brain/penta_core/ml/inference.py
git show <commit>:src/kelly/integrations/stem_jepa_integration.py
```

### 5.2 Target symbols/paths for recovery

| Domain | Symbol or path | Repo |
|--------|----------------|------|
| **Brain / LLM reasoning** | `parse_user_intent`, `LLMReasoningEngine`, `CompleteSongIntent` | This repo, forensic |
| **ML / inference** | `create_engine_by_name`, `InferenceEngine`, `penta_core/ml` | This repo |
| **JEPA / neural** | `JEPA`, `stem_jepa`, `magenta_integration` | This repo, `src/kelly/integrations/` |
| **Engine (C++)** | `IntentPipeline`, `RuleBreakEngine`, `body/engine/` | This repo |
| **Session / intent** | `process_intent`, `intent_schema`, `music_brain/session/` | This repo, forensic |

### 5.3 Forensic (DAiW-Music-Brain)

```bash
FORENSIC="/Users/seanburdges/Dev/_FORENSIC_READONLY_KMIDI/iDAWComp/DAiW-Music-Brain"
git -C "$FORENSIC" log --all -S "process_intent" --oneline -- "*.py"
git -C "$FORENSIC" log --all --oneline -- music_brain/session/
git -C "$FORENSIC" show <commit>:music_brain/session/intent_processor.py
```

---

## 6. Workflow for “find or restore, don’t recreate”

1. **Step 1 (recovery/reimplementation):** Use §5 (and `GIT_RESTORE_PATHWAYS.md` §4): `git log -S "symbol"`, `git show <commit>:path`.
2. **Find path:** `grep -F "SymbolName" docs/.index/symbol_index_canon.tsv` (or full index if generated).
3. **Open file** at the path (and line) from the index.
4. **If file is missing or stub:** Search git history (this repo and forensic) per §5 and `GIT_RESTORE_PATHWAYS.md`; restore with `git show <commit>:path > file` then wire into canon.
5. **If not in index:** Run full index (`python3 scripts/build_function_index.py`) and search again, or search with `rg`/`grep` in source.

---

## 7. Maintenance

- **After large refactors or restores:** Run `scripts/build_function_index.py --canon-only` and commit updated `docs/.index/symbol_index_canon.tsv`.
- **Full index:** Regenerate locally when needed for deep search; do not commit the 48 MB TSV unless the repo policy allows it.

---

*Last updated: 2026-01-31. Index script: `scripts/build_function_index.py`.*
