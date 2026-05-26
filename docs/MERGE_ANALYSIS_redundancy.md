# Redundancy Analysis — 11-Branch Latent Consolidation

Companion to `docs/MERGE_PLAN_2026-05-23.md`. The 11 feature branches were cut
from merge-base `676581c5`, one commit before `main` HEAD `abf90773` (PR #196,
the latent control core / 5 waves). PR #196 **already implemented** several of
the spec items these branches target, so a mechanical merge would have landed
duplicate and competing implementations. This document records the per-branch
disposition applied in branch `feat/latent-consolidation`.

## Disposition table

| Branch | Adds | vs PR #196 | Disposition |
|--------|------|------------|-------------|
| dual-representation | `audio/dual_clip.py` | `fusion.py` covers dual-rep at the *latent* level; this is a concrete audio+MIDI container | **merged as-is** |
| stem-bus | `audio/stems.py` (`StemBus`) | `fusion.StemBundle` groups *LatentFrames*; this is an audio-buffer DAW routing primitive | **merged as-is** |
| symbolic-realization | `symbolic/realize.py` | no equivalent (`ump.py` is wire-packing, not note-gen) | **merged as-is** |
| transition-model | `prediction/transition.py` | overlaps `predictors.py` cluster; kept as a Markov baseline | **merged as-is** |
| world-state | `session/world_state.py` | no concrete equivalent (`predictors.py` references a "world-model") | **merged as-is** |
| ttg-energy-gating | modifies `ttg_adapter.py`, `intent_pipeline.py` | files untouched by #196 | **merged as-is** (only branch modifying existing prod code) |
| context-window | `generation/context_window.py` | no equivalent | **merged as-is** |
| generation-scope | `generation/scope.py`, `latency_budget.py` | `latent/scope.py` *also* defines `GenerationScope` (bar-range isolation) — different meaning | **renamed** `GenerationScope`→`GenerationTransaction` |
| linear-projection | `latent/projection.py` | overlaps `conditioning_bridge.ConditioningProjection`; kept as a generic affine primitive | **merged, overlap documented** |
| constrained-decoding | `decoding/constrained.py`, `latent/normalization.py`, `audio/chunking.py` | `decoding/constrained.py` duplicates `latent/decoding.py` (same goal-list ties) | **dropped the dup decoder**, kept normalization + chunking |
| multimodal-fusion | `latent/fusion.py` (4 numpy fns) | collides with `latent/fusion.py` (`MultimodalFusion`/`StemBundle`) | **renamed** branch file → `latent/fusion_ops.py` |

## Conflict resolutions applied

- **`generation-scope`** — `generation/scope.py`'s `GenerationScope` renamed to
  `GenerationTransaction` (a commit/rollback sandbox) to avoid confusion with
  `latent/scope.py`'s `GenerationScope` (bar-range isolation). They are distinct
  concepts; both now coexist unambiguously. `generation/__init__.py` accumulates
  the context-window + scope + latency-budget exports.
- **`constrained-decoding`** — `music_brain/decoding/` package (`constrained.py`
  + `__init__.py`) and `tests/unit/test_constrained_decoding.py` dropped: they
  reimplemented temperature / top-k / top-p / mask / greedy / sample that
  `latent/decoding.py` already ships (torch/logits, integrated with the
  incremental decoder) — the branch version was numpy/probs-space and
  standalone. Net-new `latent/normalization.py` and `audio/chunking.py` kept.
- **`multimodal-fusion`** — branch's `latent/fusion.py` (raw-vector strategies
  `concat_fuse` / `weighted_sum` / `gated_fuse` / `normalized_average`) moved to
  `latent/fusion_ops.py`; `main`'s `LatentFrame`-aware `MultimodalFusion` /
  `StemBundle` kept in `fusion.py`. They are complementary layers, not
  duplicates — `gated_fuse` (latent emotional steering) has no `main` equivalent.
  Test renamed `test_fusion.py` → `test_fusion_ops.py`.
- **latent cluster `__init__.py`** — `main`'s full Wave 1–5 export surface
  preserved; normalization, projection, and fusion-ops exports appended.

## Verification

- Full unit suite: **641 passed, 53 skipped, 0 failed**.
- All new feature modules: `flake8 --max-line-length 100` clean.
- No new lint debt introduced in `music_brain/` (pre-existing findings in
  `emotion_thesaurus.py`, `api.py`, `voice/`, `intent_pipeline.py` etc. are
  unrelated to this consolidation and left untouched per scope discipline).

## Follow-ups (out of scope for this merge)

- Pre-existing `flake8` E501s in `music_brain/intent_pipeline.py` (13 lines) and
  others predate this work; `ci.yml` runs flake8 non-blocking (`|| true`), but
  `pr-review.yml` does not — worth a dedicated lint-debt pass.
- `linear-projection` vs `conditioning_bridge.ConditioningProjection` and
  `transition-model` vs `predictors.py` overlap in role; consider consolidating
  once their consumers are wired up.
