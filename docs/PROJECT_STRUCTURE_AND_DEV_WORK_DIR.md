# Project Structure & Dev Work Dir — Comparison and Recommendations

**Purpose:** Compare project structure and content (including experiments), and determine what is appropriate in the dev work directory. Aligns with PRIME LAW, EXPERIMENT LAW, DATA LAW, and ARCHIVE LAW.

**Date:** 2026-01-31

---

## 1. Dev work dir (PRIME LAW)

- **Dev work dir** = `~/Dev`. All active engineering lives here only.
- **Active canonical repo:** `~/Dev/KmiDi MIDI Companion`. This is the single source of truth for KmiDi/Kelly brain work.
- **Archive (read-only):** `~/Dev/_FORENSIC_READONLY_KMIDI` — extraction from forensic recovery; reference only; not for active edits. Do not add new work here.

**Nothing else needs to be "added" to ~/Dev** in the sense of bringing new projects in; the rule is that all active work stays under ~/Dev and the canonical repo is this one. The question is what *within* the repo (and immediately under ~/Dev) is appropriate to keep, relocate, or archive.

---

## 2. Project structure summary

| Area | Location | Content | Appropriate in dev work dir? |
|------|----------|---------|-------------------------------|
| **Spine / core** | `run_brain.py`, `KmiDi_CANON/`, `configs/`, `scripts/`, `tests/`, `src/kelly/` | Boot, brain, contracts, integrations | **Yes** — keep; this is the load-bearing core. |
| **Docs** | `docs/` | BOOT, CONTRACTS, roadmaps, DATA_AND_TRAINING, etc. | **Yes** — keep; single source of truth for process and contracts. |
| **Data (manifests)** | `data/manifests/` | Stub + README; symlink to `~/Datasets` per DATA LAW | **Yes** — keep; no large datasets in repo. |
| **Experiments** | `experiments/` | See §3 below | **Yes** — structure and policy are correct. |
| **ML training code** | `ML_TRAINED_MODELS/` | Training suite (scripts, API, CLI); model configs | **Yes** — code belongs in repo. Checkpoints/logs → `~/Models/checkpoints` (not in repo) per DATA LAW. |
| **Tauri app** | `tauri-app/` | package.json (skeleton) | **Yes** — keep; GUI path per BOOT. |
| **Documents** | `Documents/` | Cursor prompts, Dev_Notes, Kelly_Business (decks, pricing, strategy) | **Optional** — keep if actively used for product/business; else consider move to ~/Documents or archive. |
| **Music** | `Music/` | AudioVault, iDAW_Output, Kelly_Song_Project, Logic (chord charts, scripts) | **Clarify** — product-related (e.g. test kits, demos) keep; purely personal/one-off → consider outside repo or archive. |
| **GOOGLE KELLY INFO** | `GOOGLE KELLY INFO/TEST UPLOADS/` | Uploads/copies: .mid, .tsx, kelly-midi-companion copy, schemas | **Review** — duplicates of repo content and test assets. Prefer: canonical assets in repo or `~/Datasets`; upload dumps → archive (e.g. COLD_STORAGE) or delete after verification. |
| **lariat-bible** | `lariat-bible/` | 172 files (Python, MD, JSON) — separate product? | **Clarify** — if same product family, keep under ~/Dev (here or sibling repo). If unrelated, consider separate repo under ~/Dev or move out. |
| **CLEANUP_GUIDE.md** | repo root | References ~/RECOVERY_OPS, excess-duplicate cleanup | **Optional** — recovery ops are external; keep for reference or move to COLD_STORAGE if recovery is done. |
| **local-ai-feature-factory.html** | repo root | Standalone HTML | **Optional** — keep if used for demos; else archive or remove. |
| **logs/** | `logs/shadow/` | Shadow JEPA logs (e.g. midi_understanding.jsonl) | **Yes** — keep; per BOOT use `~/Models/logs/shadow` or similar for production; repo dir for dev. |
| **Forensic** | `~/Dev/_FORENSIC_READONLY_KMIDI` | Read-only extracted archive | **Do not add to** — reference only; no new work here. |

---

## 3. Experiments — current state and what to add

### 3.1 Current state

- **`experiments/`** exists with:
  - **`experiments/README.md`** — policy: promote to core only after validation; naming `exp_NNN_short_description`; layout and promotion checklist.
  - **`experiments/research/`** — design/strategy only (no code):
    - `README.md`, `TEMPLATE_proposed_model.md`, `ai_types_project_mapping.md`, `local_vs_cloud_deployment.md`.
- **No `exp_001_*`, `exp_002_*` directories yet** — no numbered experiment runs in repo.

### 3.2 What is appropriate to add to dev work dir (experiments)

- **Keep in repo (already appropriate):**
  - `experiments/` directory and `experiments/README.md`.
  - `experiments/research/` and all current research docs.
- **When starting new experimental work:**
  - Create **`experiments/exp_NNN_short_description/`** (e.g. `exp_001_emotion_encoder`, `exp_002_groove_ablation`).
  - Put experiment-specific code/notebooks/config there; optional `config.yaml`; README with goal, setup, results summary, and reference to research doc if applicable.
  - **Do not** add experimental code to `src/` or `KmiDi_CANON/` until promoted per EXPERIMENT LAW.
- **Run artifacts (not in repo):**
  - Checkpoints → `~/Models/checkpoints/<exp_name>`.
  - Run manifests → `experiments/exp_NNN_*/manifest_run_*.json` (in repo) for traceability; dataset path `~/Datasets`, checkpoint path `~/Models/checkpoints`.

So: **nothing to "add" to dev work dir** for experiments except following this layout when you create new `exp_NNN_*` dirs. Research docs and policy are already in the right place.

---

## 4. Recommendations (actionable)

### 4.1 Keep as-is (core and policy)

- Repo root: `run_brain.py`, `KmiDi_CANON/`, `configs/`, `data/`, `scripts/`, `tests/`, `src/`, `docs/`, `experiments/` (structure + research), `tauri-app/`, `ML_TRAINED_MODELS/` (code only; outputs outside repo), `logs/`, `.github/`, `pyrightconfig.json`, `README.macos-metal.md`, `INSTALL_ALL.md`, `TODO.md`, `DISTRIBUTE.md`.

### 4.2 Optional / clarify

- **Documents/** — Keep in repo if actively used for KmiDi/Kelly business and dev notes; otherwise move to ~/Documents or archive.
- **Music/** — Keep product-related assets (kits, demos, Kelly_Song_Project if part of product); move or archive purely personal/one-off material.
- **lariat-bible/** — Confirm whether it’s same product family (then keep under ~/Dev) or separate (then separate repo or path under ~/Dev).
- **CLEANUP_GUIDE.md** — Keep for reference to recovery ops or move to COLD_STORAGE if recovery is finished.
- **local-ai-feature-factory.html** — Keep if used; else archive or remove.

### 4.3 Relocate or archive (governance-friendly)

- **GOOGLE KELLY INFO/TEST UPLOADS/** — Treat as upload dump. Keep only what’s not already in repo or in `~/Datasets`; archive the rest (e.g. COLD_STORAGE) or remove after verification. Prefer canonical locations: repo for schemas/small fixtures, `~/Datasets` for larger assets.
- **Checkpoints / large outputs** — Never in repo. Ensure training scripts and docs point to `~/Models/checkpoints` and `~/Datasets` (DATA LAW, TRAINING SAFETY LAW).

### 4.4 Do not add to dev work dir

- No new active work inside `_FORENSIC_READONLY_KMIDI`; it stays read-only reference.
- No parallel clone of the canonical repo for active engineering; single tree in `KmiDi MIDI Companion`.

---

## 5. Summary

| Question | Answer |
|----------|--------|
| **What is the dev work dir?** | `~/Dev`; active repo = `KmiDi MIDI Companion`. |
| **What to add to dev work dir?** | No new top-level projects required. Within repo: add only `experiments/exp_NNN_*/` when starting new experiments; keep research and policy in `experiments/` as-is. |
| **Experiments** | Structure and research docs are appropriate. Create `exp_001_*`, `exp_002_*` under `experiments/` when needed; keep run artifacts (checkpoints, large data) in `~/Models` and `~/Datasets`. |
| **What to relocate/archive?** | GOOGLE KELLY INFO uploads (canonicalize or archive); optional: Documents/Music/CLEANUP_GUIDE/local-ai-feature-factory per §4.2–4.3. |

**Stability > novelty. Clarity > expansion. Systems > fragments.**
