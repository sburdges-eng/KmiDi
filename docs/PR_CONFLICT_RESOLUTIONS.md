# KmiDi GitHub PR conflict resolutions

Summary of open PRs with merge conflicts (as of 2026-03-10) and how to resolve them.

**Legend:** `main` = current default branch; “our” = base branch; “their” = PR branch.

---

## PRs targeting `main`

### [#126](https://github.com/sburdges-eng/KmiDi/pull/126) — Core: CompleteSongIntent UI-to-Engine mapping

- **Base:** `main` ← **Head:** `fix/ui-to-engine-mapping-83-953699174610195667`
- **Status:** CONFLICTING (branch behind main)

**Conflicts / context:**

- `main` has **removed** `.cursor/rules/engineering-governance.mdc`.
- `main` has **changed** `.github/workflows/ci-preflight.yml` (e.g. removed Brain Boot gate, dependency changes).

**Resolution:**

1. On the PR branch: `git fetch origin main && git merge origin/main`.
2. **Accept main’s version** of:
   - `.cursor/rules/engineering-governance.mdc` (deleted).
   - `.github/workflows/ci-preflight.yml` (keep main’s current jobs and deps).
3. Resolve any other modified files (e.g. `music_brain/api.py`, `shared_schemas/`, `scripts/sync_entities.py`) by keeping the PR’s **feature changes** and main’s **unrelated** changes. No same-region conflicts were seen in the PR’s core files from merge-tree; if any appear, prefer the PR’s intent-mapping logic.

---

### [#129](https://github.com/sburdges-eng/KmiDi/pull/129) — Use executemany for SQL inserts in audio_feel_extractor

- **Base:** `main` ← **Head:** `optimize-audio-feel-extractor-db-inserts-14479352300256985225`
- **Status:** CONFLICTING

**Conflicting files:**

| File | Resolution |
|------|------------|
| `.github/workflows/pre-training-hardening.yml` | Keep **main**’s step name and content (e.g. “Install dependencies”, `qt6-base-dev`). The PR’s version used “Install Qt6”; main is the source of truth for this workflow. |
| `.gitignore` | Keep **main**’s version (e.g. `*.lock` / `package-lock.json` comment). Do not re-add PR-only ignores unless needed. |
| `music_brain/agents/events.py` | Keep **main**’s version, then re-apply only the PR’s **audio_feel_extractor**-related changes elsewhere (this file is unrelated to the PR’s goal). |
| `music_brain/agents/unified_hub.py` | Keep **main**’s version (F1 clamp + docstring). No re-apply needed for #129. |
| `music_brain/api.py` | Keep **main**’s version. |
| `music_brain/misc_code/audio_feel_extractor.py` | **Keep PR’s version** (this is the only file that must carry the executemany change). If merge produces conflicts, resolve by taking main’s surrounding context and re-applying the PR’s `executemany` / batch-insert logic. |
| `music_brain/video/video_generator.py` | Keep **main**’s version. |
| `src/types/Intent.ts` | Keep **main**’s version. |
| `tests/unit/test_video_generation.py` | Keep **main**’s version. |

**PR-added files:** `benchmark.py`, `benchmark2.py`, `benchmark_result.txt`, `patch.py` — either add to `.gitignore` (if throwaway) or move under `scripts/`/`experiments/` and keep out of root.

**Steps:**

1. Merge main into the PR branch.
2. For every conflicting file **except** `music_brain/misc_code/audio_feel_extractor.py`, choose **main**.
3. In `audio_feel_extractor.py`, resolve by keeping main’s structure and re-applying the PR’s executemany/batch-insert changes.
4. Optionally add `benchmark*.py`, `benchmark_result.txt`, `patch.py` to `.gitignore` or relocate.

---

### [#124](https://github.com/sburdges-eng/KmiDi/pull/124) — Revert “feat: deterministic Brain boot path …”

- **Base:** `main` ← **Head:** `revert-107-copilot/document-boot-path-sequence`
- **Status:** CONFLICTING

**Context:**

- PR removes: `run_brain.py`, `docs/BOOT.md`, and the Brain Boot Check job from `.github/workflows/ci-preflight.yml`.
- **main** has already changed `ci-preflight.yml` (e.g. removed Brain Boot gate and/or pytest from install).

**Resolution:**

1. Merge main into the PR branch.
2. **If main no longer has a “Brain Boot” job or run_brain.py check:** the revert is largely done; keep main’s `ci-preflight.yml` and decide whether to keep or drop the PR’s deletions of `run_brain.py` and `docs/BOOT.md` per product decision.
3. **If main still has the boot path:** keep the PR’s removal of the Brain Boot job and of `run_brain.py` / `docs/BOOT.md`; for `ci-preflight.yml` take main’s other jobs/deps and only remove the brain-boot-check step.
4. Accept main’s removal of `.cursor/rules/engineering-governance.mdc` if that appears.

---

### [#100](https://github.com/sburdges-eng/KmiDi/pull/100) — Fix narrative_arc source in _convert_request_to_complete_intent

- **Base:** `main` ← **Head:** `copilot/fix-narrative-arc-references`
- **Status:** CONFLICTING

**Conflicting files:**

| File | Resolution |
|------|------------|
| `.gitignore` | Keep **main**’s version. |
| `music_brain/agents/events.py` | Keep **main**’s version (timestamp/from_dict). PR #100 does not touch this. |
| `music_brain/agents/unified_hub.py` | Keep **main**’s version (F1 clamp + docstring). |
| `music_brain/api.py` | **Re-apply only the narrative_arc/vulnerability_scale fix:** use `request.intent.narrative_arc` and `request.intent.vulnerability_scale` (with fallbacks), not `tech.get("narrative_arc")` or `getattr(request.intent, "vulnerability_scale", 0.5)`. Prefer PR’s pattern: `request.intent.narrative_arc or "Climb-to-Climax"`, and explicit `vulnerability_scale` from `request.intent` with `None` check and default `0.5`. |
| `music_brain/video/video_generator.py` | Keep **main**’s version. |
| `tests/unit/test_video_generation.py` | Keep **main**’s version. |

**Steps:**

1. Merge main into the PR branch.
2. For all conflicts except `music_brain/api.py`, take **main**.
3. In `api.py`, keep main’s structure and imports, then in `_convert_request_to_complete_intent` set:
   - `narrative_arc = request.intent.narrative_arc or "Climb-to-Climax"`
   - `vulnerability_scale = request.intent.vulnerability_scale if request.intent.vulnerability_scale is not None else 0.5`
   - Pass these into `CompleteSongIntent(...)` and remove any `tech.get("narrative_arc")` / `getattr(..., "narrative_arc", "")` for this field.

---

### [#92](https://github.com/sburdges-eng/KmiDi/pull/92) — Limit forbidden-path scan to vetted source dirs; fix /Users/ leaks

- **Base:** `main` ← **Head:** `copilot/limit-forbidden-path-scan`
- **Status:** CONFLICTING (many files)

**Conflicts (high level):**

- **Removed on main:** `.cursor/rules/engineering-governance.mdc` — accept deletion.
- **Changed on both:** `.github/workflows/ci-preflight.yml`, `bootstrap.sh`, `ci_listening_guardrails.sh`, `music_brain/api.py`, `music_brain/emotion/audio_emotion_classifier.py`, `pyproject.toml`, `run_brain.py`, `scripts/bootstrap.sh`, `scripts/dev-setup.sh`, `scripts/verify_build.py`, `KmiDi_FINAL/CMakeLists.txt`, docs, configs.

**Resolution strategy:**

1. Merge main into the PR branch.
2. **Accept main’s version** for: `.cursor/rules/engineering-governance.mdc` (gone), `ci-preflight.yml`, `run_brain.py`, `pyproject.toml`, `KmiDi_FINAL/CMakeLists.txt`, `docs/`, `configs/`, and any file where the PR did not intend a change.
3. **Re-apply only the PR’s guardrail logic:**
   - In `ci_listening_guardrails.sh`: keep main’s script structure, then re-apply the PR’s “vetted source dirs” and “no /Users/ leaks” logic (limit scan to vetted dirs, fix path leaks).
   - In `scripts/verify_build.py` and `music_brain/emotion/audio_emotion_classifier.py`: take main’s version, then re-apply the minimal changes from the PR (e.g. single-line fixes in the PR description).
4. For `bootstrap.sh` and `scripts/bootstrap.sh`, `scripts/dev-setup.sh`: take main’s versions unless the PR explicitly added guardrails there; if so, re-apply only those guardrail snippets.
5. For `music_brain/api.py`: keep main’s version unless #92 intentionally changed it for path checks; usually it does not.

---

## PRs targeting a non-main branch

### [#131–#136] — Sub-PRs for CompleteSongIntent (base: `fix/ui-to-engine-mapping-83-953699174610195667`)

- These branches conflict with **PR #126’s branch**, not main.
- **Resolution:** First merge `fix/ui-to-engine-mapping-83-953699174610195667` into each sub-PR branch (`copilot/sub-pr-126`, `copilot/sub-pr-126-again`, etc.). Then resolve conflicts in:
  - `scripts/sync_entities.py`
  - `shared_schemas/CompleteSongIntentRequest.json`
  - `src/types/Intent.ts`
  - `music_brain/session/intent_schema.py`
  - `tests/test_generate_mapping.py`
  by keeping the **sub-PR’s focused change** (e.g. JSDoc, anyOf→string|null, test naming) and the parent’s other edits. After #126 is merged to main, rebase these onto main and resolve again if needed.

### [#117](https://github.com/sburdges-eng/KmiDi/pull/117) — VideoGenerator cleanup guardrails

- **Base:** `fix-videogenerator-cleanup-11321490845064029542` ← **Head:** `copilot/sub-pr-104-please-work`
- **Status:** CONFLICTING (base is not main)

**Resolution:**

1. Rebase the PR onto **main** (or onto the branch that contains the current VideoGenerator cleanup, if that’s not yet main).
2. Resolve conflicts in `music_brain/video/video_generator.py` and related tests by keeping main’s current cleanup/guardrails and adding only the PR’s extra guardrails (e.g. scope temp dir deletion, logging).

---

## Quick reference: “take main” vs “re-apply PR”

- **Take main:** Use main’s version as-is (e.g. workflows, .gitignore, unrelated agents/video/api).
- **Re-apply PR:** After merging main, re-apply only the PR’s intended change in the relevant file(s) so that the branch contains both main’s latest and the PR’s fix.

After resolving, run:

- `npx tsc --noEmit`
- `python3 -m flake8 music_brain/ --max-line-length 100`
- `python3 -m pytest tests/` (and any relevant subset)

then push the resolution branch and re-run CI.
