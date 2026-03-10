# Discovery: OneDrive KmiDi-remote (tree and doc diff)

**Date:** 2026-03-10  
**Source:** `~/Library/CloudStorage/OneDrive-Personal/JUCE 2/Desktop/KmiDi-remote`  
**Plan:** KmiDi folders discovery — local drive scan

## Full directory tree (summary)

- **Root:** 30+ markdown files, `Logger.py`, dirs: `ML Kelly Training`, `Production_Workflows`, `Songwriting_Guides`, `Templates`, `Theory_Reference`, `benchmarks`, `bindings`, and more (700+ items total).
- **ML Kelly Training:** backup (configs, docs/model_cards), laptop_m4_small.yaml, tests, train_mps_stub.py
- **Production_Workflows:** 30+ guides (Ambient, Bass, Compression, Country, Drum, EDM, Hip-Hop, Jazz, Lo-Fi, Mastering, Mixing, Pop, R&B, Rock, Sampling, Sound Design, Strings, Vocal, etc.), manifest.json
- **Songwriting_Guides:** Chord progressions, Co-Writing, Hook, Lyric, Melody, Rewriting, Song structure, Songwriting exercises/fundamentals, Toplining, rule_breaking_*, song_intent_schema.md
- **Templates:** Collaboration, DAiW_Cheat_Sheet/Task_Board, Daily Practice Log, Gear, Learning Topic, Mix Notes, Plugin, Project, Reference Track Analysis, Sample Pack Review, Session Notes, Song, Sound Design, Weekly Review, plus ableton_live, fl_studio, logic_pro, pro_tools
- **Theory_Reference:** Audio Recording Vocabulary, Free Plugins, Logic Pro Settings/Stock Plugins, Music Theory Vocabulary, PreSonus, music_vernacular_database.md
- **benchmarks:** ANALYSIS_SUMMARY.md, bench_groove.cpp, bench_main.cpp, bench_simd.cpp
- **bindings** and further code dirs present (full list in agent-tools output; 707 lines).

## Key docs in KmiDi-remote NOT in workspace

Workspace has no files named `KMIDI_CONSOLIDATION_SUMMARY`, `IMPLEMENTATION_PLAN`, or `DESIGN_Integration_Architecture`. The following OneDrive root-level docs are strong candidates for one-time extraction (into `docs/` or `docs/onedrive-import/`):

| OneDrive file | Purpose (infer) |
|---------------|------------------|
| ARCHITECTURE_REVIEW_2025-12-30.md | Architecture snapshot |
| CLAUDE.md, CLAUDE_AGENT_GUIDE.md | Agent/IDE guidance |
| DESIGN_Integration_Architecture.md | Integration design |
| IMPLEMENTATION_PLAN.md | Implementation plan |
| IMPLEMENTATION_ALTERNATIVES.md | Alternatives |
| KMIDI_CONSOLIDATION_SUMMARY.md | Consolidation summary |
| KMIDI_README.md, KMIDI_STRUCTURE_PLAN.md | Readme and structure |
| MERGER_INFRASTRUCTURE_COMPLETE.md | Merger status |
| INFRASTRUCTURE.md, HOW_TO_DEV_OP_101.md | Infra and ops |
| OPTIMAL_WORKFLOW_SUMMARY.md, PUSH_STRATEGY.md | Workflow and push |
| ROADMAP_Implementation.md, RELEASE_NOTES_v1.0.0.md | Roadmap and release |
| ANALYSIS_*.md, BUILD_VARIANTS.md, CHANGES_2025-12-30.md | Analysis and build |
| FINAL_VERIFICATION.md, VERIFICATION_Fix_Complete.md | Verification |
| RECOMMENDATIONS_Improvements.md, QUICKSTART_TIER123.md | Recommendations and quickstart |
| COPILOT_INSTRUCTIONS.md | Copilot |

## Comparison with workspace docs/ and experiments/

- **Workspace** has: ARCHITECTURE.md, BOOT.md, DEVELOPMENT.md, ENVIRONMENT.md, BREAK_FIX_RUN_LOG.md, CANONICAL_FOLDER_STRUCTURE.md, SSD_WORKDIR_STRUCTURE.md, and many others. It does **not** have the consolidation/merger/implementation-plan docs above.
- **Conclusion:** OneDrive KmiDi-remote holds a separate snapshot with richer consolidation/merger and implementation-plan content. Prefer **one-time extraction** of selected docs into the repo (e.g. `docs/onedrive-import/`) rather than working inside OneDrive (per CLOUD LAW).

## Suggested follow-up

1. Copy into workspace (e.g. `docs/onedrive-import/`): KMIDI_CONSOLIDATION_SUMMARY.md, IMPLEMENTATION_PLAN.md, DESIGN_Integration_Architecture.md, MERGER_INFRASTRUCTURE_COMPLETE.md, plus any of the table above as needed.
2. Optionally import Production_Workflows and Songwriting_Guides as reference (or link from docs if kept on cloud).
3. Do not sync or develop from OneDrive; treat as read-only source for a single import pass.
