# Code Review: Post-Merge Changes on feature/universal-music-input

## Context

We completed a 3-stream additive merge from KmiDi_FINAL archives into the current project, plus 20 bug/security fixes across 3 review rounds. All changes are on `feature/universal-music-input` branch. There are 229 changed files (192 staged, 21 unstaged modified, 16 untracked). CodeRabbit CLI is installed and authenticated but has a **150-file limit per review**.

**Goal:** Run a comprehensive CodeRabbit review covering all meaningful code changes before committing.

---

## Step 1: Stage All Outstanding Changes

Stage the 21 unstaged modified files and 16 untracked `music_brain/` files so the full changeset is visible to the reviewer.

```bash
# Stage unstaged modifications
git add -u music_brain/

# Stage new untracked voice/data files
git add music_brain/voice/neural_backend.py music_brain/voice/phoneme_processor.py \
  music_brain/voice/pitch_controller.py music_brain/voice/singing_synthesizer.py \
  music_brain/voice/singing_voice.py music_brain/voice/singing_voice_dev.py \
  music_brain/voice/voice_input.py music_brain/voice/voice_learning.py \
  music_brain/voice/instrument_synth.py music_brain/data/ \
  music_brain/intelligence/brain_controller.py
```

---

## Step 2: Commit in Two Batches to Enable Scoped Reviews

Since CodeRabbit reviews committed changes and has a 150-file limit, split into two commits:

### Commit A: Core Code (~100 files)
- `music_brain/` — all Python modules (voice, data, intelligence, session, structure, etc.)
- `src/` — TypeScript/C++ (UniversalMusicInput, hooks, plugins)
- `tests/` — new/modified tests

### Commit B: Infrastructure & Cleanup (~130 files)
- `experiments/` deletions (44 files)
- `scripts/`, `docs/`, config files
- `.github/workflows/`, cmake, `.pre-commit-config`
- `_archive/` changelogs
- `kmidi-docs-archive/`

---

## Step 3: Run CodeRabbit Reviews

```bash
# Review Commit A (core code)
coderabbit review --plain -t committed

# After reviewing and fixing Commit A issues, review Commit B
coderabbit review --plain -t committed
```

---

## Step 4: Fix Issues

For each review round:
1. Triage findings by severity (Critical > Important > Suggestion)
2. Fix Critical and Important issues immediately
3. Re-run review if Critical issues were found
4. Document suggestions as deferred

---

## Step 5: Verification

1. `python3 -c "import music_brain"` — no import crashes
2. `python3 -m py_compile` on all changed `.py` files
3. `python3 -m json.tool` on all `.json` data files
4. `npx vitest run tests/unit/` — unit tests pass
5. `npx vitest run` — check for new test regressions

---

## Critical Files

### Stream 1 (Voice — already reviewed 3x, 20 fixes applied):
- `music_brain/voice/neural_backend.py`
- `music_brain/voice/phoneme_processor.py`
- `music_brain/voice/pitch_controller.py`
- `music_brain/voice/singing_synthesizer.py`
- `music_brain/voice/singing_voice.py`
- `music_brain/voice/voice_input.py`
- `music_brain/voice/voice_learning.py`
- `music_brain/voice/instrument_synth.py`
- `music_brain/emotion_api.py`
- `music_brain/common.py`
- `music_brain/vernacular.py`

### Stream 3 (Additive merges — reviewed 3x):
- `music_brain/kelly_companion/groove/groove_engine.py`
- `music_brain/kelly_companion/groove/templates.py`
- `music_brain/session/intent_schema.py`
- `music_brain/structure/progression.py`
- `music_brain/intelligence/brain_controller.py`

### Frontend (reviewed once in initial code review):
- `src/components/UniversalMusicInput/*.tsx`
- `src/hooks/useTextParse.ts`
- `src/hooks/useMusicBrain.ts`
- `src/data/interpretationEngine.ts`
- `src/data/taxonomyTree.ts`

### C++ (reviewed once):
- `src/plugin/PluginProcessor.cpp`
- `src/plugin/PluginState.cpp`
- `src/project/ProjectManager.cpp`
- `src/common/PersistenceTimestamp.h`
