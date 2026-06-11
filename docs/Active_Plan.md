# KmiDi Active Merge Plan

**Date:** 2026-05-23
**Prepared by:** CI Merge Agent
**Main HEAD:** `abf90773` — feat(music_brain/latent): add latent control core across 5 waves (#196)
**Common merge-base:** `676581c5`
**Total branches to merge:** 11
**Stacks affected:** Python only (music_brain/, tests/unit/)
**Stacks unaffected:** C++ (engine/, include/), Rust (engine/intent_ir/), Bindings (bindings/), React (src/)

---

## Key Finding

All 11 branches share merge-base `676581c5`, which is exactly 1 commit behind main.
Main's HEAD added 20 files to `music_brain/latent/` via PR #196. Every branch's diff
shows massive deletions of those latent files — this is NOT a real conflict. It is
simply the branches not having the new main commit. **A rebase onto main resolves this
for all branches.**

---

## Pre-Merge Step (REQUIRED for ALL branches)

```bash
# For each branch, rebase onto current main to pick up latent control core (#196)
git checkout feat/<branch-name>
git rebase main
# This eliminates the shared deletion pattern across all 11 branches
```

Rebase all 11 branches onto `abf90773` before beginning any merges. This single step
eliminates the apparent "massive deletions" that every branch shows in its diff.

---

## Merge Order

### PHASE 1 — Independent Branches (no file conflicts)

These 5 branches create new modules or add standalone files. They have zero file
conflicts with each other or with any other branch. They can be merged in any order
within this phase, or even in parallel.

#### 1. feat/dual-representation
- **Adds:** `music_brain/audio/dual_clip.py` (136 lines) — DualClip aligned audio+MIDI
- **Tests:** `tests/unit/test_dual_clip.py` (154 lines)
- **Risk:** Minimal — new file only, no `__init__.py` conflicts
- **CI:** `python3 -m pytest tests/unit/test_dual_clip.py -x`

#### 2. feat/stem-bus
- **Adds:** `music_brain/audio/stems.py` (107 lines) — StemBus multi-stem container
- **Tests:** `tests/unit/test_stem_bus.py` (135 lines)
- **Risk:** Minimal — new file only
- **CI:** `python3 -m pytest tests/unit/test_stem_bus.py -x`

#### 3. feat/symbolic-realization
- **Adds:** New `music_brain/symbolic/` module
  - `music_brain/symbolic/__init__.py` (17 lines)
  - `music_brain/symbolic/realize.py` (160 lines) — chord-to-MIDI realizer
- **Tests:** `tests/unit/test_symbolic_realize.py` (142 lines)
- **Risk:** Minimal — creates entirely new module
- **CI:** `python3 -m pytest tests/unit/test_symbolic_realize.py -x`

#### 4. feat/transition-model
- **Adds:** New `music_brain/prediction/` module
  - `music_brain/prediction/__init__.py` (5 lines)
  - `music_brain/prediction/transition.py` (134 lines) — Markov baseline
- **Tests:** `tests/unit/test_transition_model.py` (149 lines)
- **Risk:** Minimal — creates entirely new module
- **CI:** `python3 -m pytest tests/unit/test_transition_model.py -x`

#### 5. feat/world-state
- **Adds:** `music_brain/session/world_state.py` (103 lines)
- **Tests:** `tests/unit/test_world_state.py` (139 lines)
- **Risk:** Minimal — new file in existing module
- **CI:** `python3 -m pytest tests/unit/test_world_state.py -x`

**Phase 1 validation after all 5 merges:**
```bash
python3 -m pytest tests/ -x
```

---

### PHASE 2 — generation/ Cluster (2-way __init__.py conflict)

These 2 branches both create `music_brain/generation/__init__.py` with different exports.
Merge them sequentially and fix the `__init__.py` after the second merge.

#### 6. feat/context-window
- **Adds:** New `music_brain/generation/` module
  - `music_brain/generation/__init__.py` — exports `ContextEntry`, `ContextWindow`
  - `music_brain/generation/context_window.py` (126 lines)
- **Tests:** `tests/unit/test_context_window.py` (154 lines)
- **Risk:** Low — creates the module first
- **CI:** `python3 -m pytest tests/unit/test_context_window.py -x`

#### 7. feat/generation-scope
- **Adds:**
  - `music_brain/generation/latency_budget.py` (91 lines)
  - `music_brain/generation/scope.py` (119 lines)
- **Tests:** `tests/unit/test_generation_scope.py` (164 lines), `tests/unit/test_latency_budget.py` (125 lines)
- **CONFLICT:** `music_brain/generation/__init__.py` — will conflict with what feat/context-window wrote
- **Manual fix required:** After merge, update `music_brain/generation/__init__.py` to export ALL symbols:
  ```python
  from .context_window import ContextEntry, ContextWindow
  from .latency_budget import LatencyBudget
  from .scope import GenerationScope, RollbackError
  ```
- **CI after fix:** `python3 -m pytest tests/unit/test_context_window.py tests/unit/test_generation_scope.py tests/unit/test_latency_budget.py -x`

**Phase 2 validation:**
```bash
python3 -m pytest tests/ -x
```

---

### PHASE 3 — latent/ Cluster (3-way __init__.py conflict)

These 3 branches all rewrite `music_brain/latent/__init__.py` with incompatible exports.
After rebase, main's `__init__.py` will have the latent control core exports. Each branch
replaces those with its own subset. Merge them one at a time and fix `__init__.py` after
each merge to accumulate all exports.

#### 8. feat/constrained-decoding
- **Adds:**
  - `music_brain/audio/chunking.py` (148 lines) — iter_chunks, pad, overlap-add
  - `music_brain/decoding/__init__.py` (19 lines) — new module
  - `music_brain/decoding/constrained.py` (136 lines) — top-k, top-p, mask, greedy, sample
  - `music_brain/latent/normalization.py` (98 lines) — l2, layer-norm, center, clip-norm, scale, standardize
- **Tests:** 3 test files (475 lines total)
- **CONFLICT:** Rewrites `music_brain/latent/__init__.py` to export normalization functions only
- **Manual fix required:** After merge, update `latent/__init__.py` to include BOTH the existing
  latent control core exports AND the new normalization exports
- **CI:** `python3 -m pytest tests/unit/test_audio_chunking.py tests/unit/test_constrained_decoding.py tests/unit/test_latent_normalization.py -x`

#### 9. feat/linear-projection
- **Adds:**
  - `music_brain/latent/projection.py` (119 lines) — LinearProjection affine bridge
- **Tests:** `tests/unit/test_linear_projection.py` (122 lines)
- **CONFLICT:** Rewrites `music_brain/latent/__init__.py` to export `LinearProjection` only
- **Manual fix required:** After merge, update `latent/__init__.py` to include existing exports
  + normalization (from step 8) + `LinearProjection`
- **CI:** `python3 -m pytest tests/unit/test_linear_projection.py -x`

#### 10. feat/multimodal-fusion
- **Adds/Modifies:**
  - `music_brain/latent/fusion.py` (116 lines) — **COMPLETE REWRITE** with concat, weighted_sum, gated, normalized_average
- **Tests:** `tests/unit/test_fusion.py` (137 lines)
- **CONFLICT:** Rewrites `music_brain/latent/__init__.py` to export fusion functions only
- **Manual fix required:** After merge, update `latent/__init__.py` to include ALL accumulated
  exports: latent control core + normalization + projection + fusion
- **CI:** `python3 -m pytest tests/unit/test_fusion.py -x`

**Phase 3 validation (critical — verify accumulated __init__.py is correct):**
```bash
python3 -m pytest tests/ -x
python3 -m flake8 music_brain/latent/ --max-line-length 100
python3 -c "import music_brain.latent; print(dir(music_brain.latent))"
```

---

### PHASE 4 — Highest-Risk Branch (merge last)

#### 11. feat/ttg-energy-gating
- **Commits:** 4 (most of any branch)
- **Net new lines:** +1137 (largest branch)
- **Modifies existing files:**
  - `music_brain/api_schemas/ttg_adapter.py` (58→294 lines) — heavy additions to TTG adapter
  - `music_brain/pipeline/intent_pipeline.py` (396→461 lines) — energy gating pipeline
- **New tests:** 3 test files (833 lines total)
  - `tests/unit/test_ttg_energy_gating.py` (433 lines)
  - `tests/unit/test_ttg_motif_tracking.py` (212 lines)
  - `tests/unit/test_ttg_phrase_boundaries.py` (188 lines)
- **Risk:** HIGHEST — this is the only branch that modifies existing production files
  rather than just adding new ones. Changes to `ttg_adapter.py` and `intent_pipeline.py`
  could have subtle interactions with main's current state.
- **Pre-merge review checklist:**
  - [ ] Verify `ttg_adapter.py` changes are additive (new schemas, not breaking existing ones)
  - [ ] Verify `intent_pipeline.py` changes don't break existing pipeline stages
  - [ ] Run full test suite including existing TTG tests
  - [ ] Check for import changes that could affect other modules

**Phase 4 validation (full suite):**
```bash
python3 -m pytest tests/ -v
python3 -m flake8 music_brain/ --max-line-length 100
```

---

## CI Strategy Summary

| Phase | CI Required | CI Skipped |
|-------|------------|------------|
| All phases | Python: pytest, flake8 | C++ build/test, Rust build/test, React build/test, Bindings |
| Phase 1 | Individual test files per branch | Full suite (run at end) |
| Phase 2 | generation/ tests + import check | — |
| Phase 3 | latent/ tests + flake8 + import check | — |
| Phase 4 | **Full test suite** + flake8 entire music_brain/ | — |

**No C++/Rust/React/Bindings CI is needed for any of the 11 merges.** All branches
are Python-only. See the individual stack analysis documents for details:
- `docs/MERGE_ANALYSIS_cpp.md`
- `docs/MERGE_ANALYSIS_rust.md`
- `docs/MERGE_ANALYSIS_react.md`
- `docs/MERGE_ANALYSIS_bindings.md`

---

## Risk Summary

| Risk Level | Branches | Key Concern |
|-----------|----------|-------------|
| Minimal | dual-representation, stem-bus, symbolic-realization, transition-model, world-state | New files only, no conflicts |
| Medium | context-window, generation-scope | 2-way `__init__.py` conflict, straightforward merge |
| High | constrained-decoding, linear-projection, multimodal-fusion | 3-way `__init__.py` conflict, requires careful export accumulation |
| Highest | ttg-energy-gating | Modifies existing production files, 4 commits, 1137 lines |

---

## Estimated Timeline

- **Phase 1:** ~30 minutes (5 independent merges, no conflicts)
- **Phase 2:** ~20 minutes (2 merges, 1 manual __init__.py fix)
- **Phase 3:** ~45 minutes (3 merges, manual __init__.py fix after each, careful validation)
- **Phase 4:** ~30 minutes (1 merge, thorough review of existing file modifications)
- **Total:** ~2 hours for all 11 branches

---

## Post-Merge Verification

After all 11 branches are merged, run the complete validation:

```bash
# Full test suite
python3 -m pytest tests/ -v --tb=short

# Lint check on all modified Python code
python3 -m flake8 music_brain/ --max-line-length 100

# Verify all new modules import cleanly
python3 -c "
import music_brain.audio.dual_clip
import music_brain.audio.stems
import music_brain.audio.chunking
import music_brain.decoding
import music_brain.generation
import music_brain.latent
import music_brain.prediction
import music_brain.symbolic
import music_brain.session.world_state
print('All modules import successfully')
"

# Verify latent __init__.py has all exports
python3 -c "
from music_brain.latent import *
print('Latent exports OK')
"

# Verify generation __init__.py has all exports
python3 -c "
from music_brain.generation import ContextEntry, ContextWindow
from music_brain.generation import LatencyBudget, GenerationScope, RollbackError
print('Generation exports OK')
"
```
