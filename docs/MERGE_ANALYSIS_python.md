# Python Feature-Branch Merge Analysis

**Scope:** Python stack only (`music_brain/` + `tests/`).
**Repo inspected:** `/Users/seanburdges/Dev/KmiDi` (refs identical via shared `origin`).
**Method:** `git diff --stat origin/main..origin/<branch>` plus `git diff <merge-base>..origin/<branch>` to separate real changes from rebase artifacts.
**Date:** 2026-05-23

---

## TL;DR

- **11 feature branches**, all Python-only, all sharing one merge-base: `676581c5`.
- That merge-base is exactly `origin/main~1`. Main HEAD `abf90773` is the **latent control core (#196)** commit. Every branch is **one commit behind main**.
- Because they predate #196, every branch's `git diff` against `main` shows **phantom deletions** of the entire `music_brain/latent/` tree (+ phantom edits to `emitter.py`, `jepa/*`, `world_model.py`, and doc deletions). **None of these are real** — they are #196's additions appearing inverted. They vanish on rebase onto main.
- **Only three real conflict clusters** exist; everything else is non-overlapping.
- **Recommended order:** independent branches → `generation/` cluster → `latent/` cluster (manual `__init__.py` accumulation) → `ttg-energy-gating` last.

---

## 1. Merge base & the phantom-deletion artifact

| Fact | Value |
|------|-------|
| Shared merge-base (all 11 branches) | `676581c5ed59ea96a0bd5a8479f88d671a6eec9c` |
| `origin/main` HEAD | `abf90773b7c0b861687a90f28f2f962e539409e8` |
| `origin/main~1` | `676581c5…` (== merge-base) |
| Commit #196 added | `music_brain/latent/` (latent control core, 5 waves) |

**Verification.** `git merge-base origin/main origin/feat/<branch>` returns `676581c5` for all 11 branches; `git rev-parse origin/main~1` returns the same SHA.

Because the branches were cut **before** #196 landed, `origin/main..origin/<branch>` reports the `latent/` tree (28 source files + ~24 latent test files), the `docs/MERGE_FOLLOWUPS_LATENT_CORE.md` / `docs/research/MULTIMODAL_REPRESENTATIONS_2026.md` edits, and modifications to `intent_ir/emitter.py`, `jepa/audio_jepa.py`, `jepa/chord_jepa.py`, `world_model.py` as **deletions/reversions**. Spot-checked against the merge-base:

```
git diff --stat 676581c5..origin/feat/constrained-decoding -- \
    music_brain/intent_ir/emitter.py music_brain/jepa/*.py music_brain/world_model.py
# → (empty)  ← branch never touched these files; the diff vs main is pure #196 artifact
git diff --stat 676581c5..origin/feat/ttg-energy-gating -- \
    music_brain/intent_ir/emitter.py music_brain/jepa/audio_jepa.py music_brain/world_model.py
# → (empty)
```

**Action:** rebase every branch onto `origin/main` (or merge `main` in) before integrating. All phantom deletions resolve automatically. **Do not** review the `latent/*` deletions as if they were intentional removals.

---

## 2. Real change footprint per branch (vs merge-base, phantom stripped)

| Branch | Commits | Real files touched | New / Modified |
|--------|:---:|--------------------|----------------|
| `feat/dual-representation` | 1 | `music_brain/audio/dual_clip.py`, `tests/unit/test_dual_clip.py` | new only |
| `feat/stem-bus` | 1 | `music_brain/audio/stems.py`, `tests/unit/test_stem_bus.py` | new only |
| `feat/symbolic-realization` | 1 | `music_brain/symbolic/__init__.py`, `music_brain/symbolic/realize.py`, `tests/unit/test_symbolic_realize.py` | new module |
| `feat/transition-model` | 1 | `music_brain/prediction/__init__.py`, `music_brain/prediction/transition.py`, `tests/unit/test_transition_model.py` | new module |
| `feat/world-state` | 1 | `music_brain/session/world_state.py`, `tests/unit/test_world_state.py` | new only |
| `feat/context-window` | — | `music_brain/generation/__init__.py`, `music_brain/generation/context_window.py`, `tests/unit/test_context_window.py` | new pkg + shared `__init__` |
| `feat/generation-scope` | — | `music_brain/generation/{__init__,scope,latency_budget}.py`, `tests/unit/{test_generation_scope,test_latency_budget}.py` | new pkg + shared `__init__` |
| `feat/constrained-decoding` | 3 | `music_brain/latent/{__init__,normalization}.py`, `music_brain/audio/chunking.py`, `tests/unit/{test_constrained_decoding,test_audio_chunking,test_latent_normalization}.py` | new files + shared `latent/__init__` |
| `feat/linear-projection` | — | `music_brain/latent/{__init__,projection}.py`, `tests/unit/test_linear_projection.py` | new files + shared `latent/__init__` |
| `feat/multimodal-fusion` | 1 | `music_brain/latent/{__init__,fusion}.py`, `tests/unit/test_fusion.py` | shared `latent/__init__` + **collides with #196 `fusion.py`** |
| `feat/ttg-energy-gating` | 4 | `music_brain/api_schemas/ttg_adapter.py`, `music_brain/pipeline/intent_pipeline.py`, `tests/unit/{test_ttg_energy_gating,test_ttg_motif_tracking,test_ttg_phrase_boundaries}.py` | **modifies 2 existing files** |

---

## 3. Conflict clusters (the only real merge work)

### Cluster A — `music_brain/latent/__init__.py` (3-way + main)

Three branches each (re)create `latent/__init__.py` with **mutually incompatible** export lists, and each also collides with #196's 128-line version on main.

| Source | `latent/__init__.py` exports |
|--------|------------------------------|
| `origin/main` (#196) | full latent control core API (`CompanionSession`, `ConditioningProjection`, `DecodeConfig`, `EmotionTrajectory`, `MultimodalFusion`, `StemBundle`, …) — 128 lines |
| `feat/constrained-decoding` | `center, clip_norm, l2_normalize, layer_norm, min_max_scale, standardize` (from `normalization`) |
| `feat/linear-projection` | `LinearProjection` (from `projection`) |
| `feat/multimodal-fusion` | `concat_fuse, gated_fuse, normalized_average, weighted_sum` (from `fusion`) |

**Why it conflicts:** at the merge-base `latent/` did not exist, so each branch authored a fresh `__init__.py`. Against post-#196 main, all three are "both added the file with different content" conflicts — and they conflict with each other if merged sequentially.

**Resolution:** keep main's full #196 export block and **append** each branch's new imports/`__all__` entries (manual accumulation). Do not let any branch's `__init__.py` overwrite main's.

### Cluster B — `music_brain/latent/fusion.py` (semantic collision, highest-risk in the latent cluster)

This is more than a textual rewrite. **Both** main and the branch ship a `latent/fusion.py`, with **different, incompatible APIs at the same path**:

| Source | `latent/fusion.py` API | Lines |
|--------|------------------------|:---:|
| `origin/main` (#196) | classes `MultimodalFusion`, `StemBundle` | 115 |
| `feat/multimodal-fusion` | functions `concat_fuse`, `weighted_sum`, `gated_fuse`, `normalized_average` | 116 |

main's `latent/__init__.py` does `from music_brain.latent.fusion import MultimodalFusion, StemBundle`. If the branch's `fusion.py` replaces main's, **that import breaks and the entire `music_brain.latent` package fails to import** — a hard regression, not a localized conflict.

**Resolution (needs human decision):** do **not** overwrite. Either (a) move the branch's free functions into a new module (e.g. `latent/fusion_ops.py`) and re-export from `__init__`, or (b) merge the four functions into main's existing `fusion.py` alongside the classes. Option (a) is lower-risk. Note #196's `fusion.py` docstring already claims to close "Multimodal fusion engine / latent alignment" — confirm the branch isn't redundant before integrating.

### Cluster C — `music_brain/generation/__init__.py` (2-way)

`generation/` does **not** exist on main, so the **first** of these merges is clean; the **second** conflicts on `__init__.py`.

| Source | `generation/__init__.py` exports |
|--------|----------------------------------|
| `feat/context-window` | `ContextEntry, ContextWindow` (from `context_window`) |
| `feat/generation-scope` | `BudgetExceeded, LatencyBudget` (from `latency_budget`); `GenerationScope, RollbackError` (from `scope`) |

**Resolution:** merge one first; on the second, accumulate both export sets into one `__init__.py`. The underlying module files (`context_window.py`, `scope.py`, `latency_budget.py`) are disjoint — no conflict there.

---

## 4. Independent branches (zero file overlap)

These five each add **only new files** under a distinct path, touch no existing module, and do not overlap each other. They carry **no real conflicts** (only phantom #196 deletions, auto-resolved on rebase):

- `feat/dual-representation` → `music_brain/audio/dual_clip.py`
- `feat/stem-bus` → `music_brain/audio/stems.py`
- `feat/symbolic-realization` → new `music_brain/symbolic/` module
- `feat/transition-model` → new `music_brain/prediction/` module
- `feat/world-state` → `music_brain/session/world_state.py`

`dual-representation` and `stem-bus` both live under the **pre-existing** `music_brain/audio/` package but write different files (`dual_clip.py` vs `stems.py`) and neither edits `audio/__init__.py` — so they don't even conflict with each other.

> **Watch (non-blocking):** #196's `latent/fusion.py` docstring lists "Audio/MIDI dual representation" and "Stem-aware generation" among the goals it closes. There is no *file* overlap with `dual-representation` / `stem-bus`, but there may be *conceptual* redundancy. Flag for the owners; do not auto-reconcile.

---

## 5. Highest-risk branch — `feat/ttg-energy-gating`

The only branch that **modifies existing committed source** (the rest are additive). Largest, most-commits, touches the live intent pipeline.

| Metric | Value |
|--------|-------|
| Commits ahead of merge-base | **4** |
| `music_brain/api_schemas/ttg_adapter.py` | **58 → 294 lines** (+236) |
| `music_brain/pipeline/intent_pipeline.py` | **396 → 461 lines** (+65) |
| New tests | `test_ttg_energy_gating.py` (433), `test_ttg_motif_tracking.py` (212), `test_ttg_phrase_boundaries.py` (188) |

Verified that against the merge-base it touches **only** these two source files plus its three new tests (no phantom-file edits). Both modified files are on hot paths (TTG adapter + intent pipeline), so it is the most likely to interact with anything merged before it. **Merge it last** so it rebases over the final state of `music_brain/` and is validated against everything else already integrated.

---

## 6. Recommended merge order

Rebase each branch onto `origin/main` immediately before its phase (clears phantom deletions).

**Phase 1 — Independent branches (any order, lowest risk):**
`feat/dual-representation`, `feat/stem-bus`, `feat/symbolic-realization`, `feat/transition-model`, `feat/world-state`

**Phase 2 — `generation/` cluster (Cluster C):**
Merge `feat/context-window`, then `feat/generation-scope` (or vice-versa); accumulate `generation/__init__.py` exports on the second.

**Phase 3 — `latent/` cluster (Clusters A + B), with manual `__init__.py` accumulation:**
1. `feat/constrained-decoding` — append `normalization` exports to main's `latent/__init__.py`.
2. `feat/linear-projection` — append `LinearProjection`.
3. `feat/multimodal-fusion` — **resolve Cluster B first** (rehome the free functions; never overwrite `fusion.py`), then append exports.
Keep main's #196 export block intact at every step.

**Phase 4 — `feat/ttg-energy-gating` (last):**
Rebase over the fully-integrated tree and validate.

---

## 7. CI / verification gates

Per project conventions (`pytest`, flake8 `--max-line-length 100`, no `--timeout`):

- **After every merge:** `python3 -m pytest tests/ -x`
- **After Phase 3 and Phase 4:** add `python3 -m flake8 music_brain/ --max-line-length 100`
  (Phase 3 and 4 are where `__init__.py` accumulation and existing-file edits are most likely to introduce unused-import / line-length lint regressions.)
- Do **not** pass `--timeout` to pytest (pytest-timeout is not installed).

---

## Appendix — branch → diff-stat headline (vs `origin/main`, phantom included)

| Branch | files | +ins | −del (incl. phantom) |
|--------|:---:|:---:|:---:|
| `feat/constrained-decoding` | 53 | 947 | 6097 |
| `feat/linear-projection` | 48 | 301 | 6100 |
| `feat/multimodal-fusion` | 47 | 300 | 6077 |
| `feat/context-window` | 49 | 342 | 6102 |
| `feat/generation-scope` | 51 | 567 | 6102 |
| `feat/dual-representation` | 48 | 347 | 6102 |
| `feat/stem-bus` | 48 | 299 | 6102 |
| `feat/symbolic-realization` | 49 | 376 | 6102 |
| `feat/transition-model` | 49 | 345 | 6102 |
| `feat/world-state` | 48 | 299 | 6102 |
| `feat/ttg-energy-gating` | 51 | 1194 | 6105 |

The ~6,100-line deletion column is the #196 latent-control-core footprint reappearing inverted; it is **not** real change. Real change is the small `+ins` portion outside the `latent/` deletions (see §2).
