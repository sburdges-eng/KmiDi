# Audit: Markdown files — obsolete, irrelevant, or for another project

This document classifies markdown files in the KmiDi repo so you can archive, delete, or update. **Canonical** docs are those referenced in [AGENTS.md](../AGENTS.md) or clearly current. **Obsolete** = one-time logs, completed checklists, or point-in-time reports. **Irrelevant** = not about KmiDi or wrong scope. **Another project/workflow** = different repo, one-time import, or external workflow.

---

## Canonical / current (keep)

| File | Notes |
|------|--------|
| [AGENTS.md](../AGENTS.md) | Canonical agent context; referenced by tools |
| [README.md](../README.md) | Project overview |
| [BUILD.md](../BUILD.md) | C++ / CMake / Tauri build reference |
| [QUICK_START.md](../QUICK_START.md) | Repo build and dev (aligned with AGENTS) |
| [docs/DEVELOPMENT.md](DEVELOPMENT.md) | Full dev guide |
| [docs/ENVIRONMENT.md](ENVIRONMENT.md) | Env vars, file layout |
| [docs/FULL_STACK_BUILD.md](FULL_STACK_BUILD.md) | React ↔ Tauri ↔ KellyFFI build |
| [docs/DATASETS_LAYOUT.md](DATASETS_LAYOUT.md) | Dataset volume layout |
| [docs/HEADLESS_ENGINE.md](HEADLESS_ENGINE.md) | Headless engine and rt_harness |
| [docs/SOURCE_INTEGRATION_PLAN.md](SOURCE_INTEGRATION_PLAN.md) | Source integration and download plan |
| [docs/AU_PLUGIN_ARCHITECTURE.md](AU_PLUGIN_ARCHITECTURE.md) | AU plugin architecture |
| [docs/SAGEMAKER_SETUP.md](SAGEMAKER_SETUP.md) | SageMaker AI training |
| [rt_harness/README.md](../rt_harness/README.md) | RT harness build and run |
| [docs/adr/](adr/) | Architecture decision records |
| [docs/research/](research/), [docs/research/sources/](research/sources/) | Research and source briefings (KmiDi-relevant) |
| [docs/INTENT_IR_SPEC.md](INTENT_IR_SPEC.md) | Intent IR spec (if still accurate) |
| [docs/CODE_REVIEW_REPORT.md](CODE_REVIEW_REPORT.md) / WHOLE_REPO_CODE_REVIEW | Substantive repo map and findings; refresh if stale |

---

## Obsolete (one-time or completed; safe to archive/remove)

One-time run logs, completion checklists, freeze reports, discovery outputs, merge logs. Keep only if you want historical record.

| File | Reason |
|------|--------|
| [BREAK_FIX_RUN_LOG.md](BREAK_FIX_RUN_LOG.md) | One-time break-and-fix verification run log |
| [DEBUGGING_PLAN_RUN.md](DEBUGGING_PLAN_RUN.md) | Point-in-time debugging run |
| [FREEZE_REPORT.md](FREEZE_REPORT.md), [FREEZE_FINAL_REPORT.md](FREEZE_FINAL_REPORT.md), [FREEZE_AUDIT_PHASE1.md](FREEZE_AUDIT_PHASE1.md), [FREEZE_AUDIT_PHASE1_CONFIRMED.md](FREEZE_AUDIT_PHASE1_CONFIRMED.md), [FREEZE_ENV_MATRIX.md](FREEZE_ENV_MATRIX.md) | Freeze/audit reports for a specific release moment |
| [GIT_GRAPH_CLEANUP.md](GIT_GRAPH_CLEANUP.md) | One-time git history cleanup options (already acted or not) |
| [PRE_WORKSPACE_COMPLETION_CHECKLIST.md](PRE_WORKSPACE_COMPLETION_CHECKLIST.md) | Migration checklist; status "ALL ITEMS COMPLETE" |
| [COMPLETED_SUMMARY.md](COMPLETED_SUMMARY.md) | V1 completion summary; historical unless refreshed |
| [START_HERE.md](START_HERE.md) | One-time "ready for development" / commit guide (2026-01-23) |
| [MERGE_LOG.md](MERGE_LOG.md), [merge-all-branches-log.md](merge-all-branches-log.md) | Merge logs |
| [STASH_INTEGRATION_LOG.md](STASH_INTEGRATION_LOG.md), [STASH_INVENTORY.md](STASH_INVENTORY.md) | Stash integration one-time logs |
| [ONEDRIVE_INTEGRATION_LOG.md](ONEDRIVE_INTEGRATION_LOG.md) | OneDrive integration log (2025-01-10); integration complete |
| [DISCOVERY_*.md](.) (all DISCOVERY_* in docs/) | Discovery run outputs (OneDrive, Xcode, SSD, workspace, datasets, etc.); one-time scans |
| [PATCH_BATCH_REPORT.md](PATCH_BATCH_REPORT.md) | One-time patch batch report |
| [PR_COMPILE_TO_MAIN.md](PR_COMPILE_TO_MAIN.md), [PR_CONFLICT_RESOLUTIONS.md](PR_CONFLICT_RESOLUTIONS.md), [PR_REVIEW_ANALYSIS.md](PR_REVIEW_ANALYSIS.md) | Point-in-time PR/conflict notes |
| [RESTRUCTURE_STATUS.md](RESTRUCTURE_STATUS.md) | Restructure status snapshot |
| [REPO_CONSOLIDATION_ANALYSIS.md](REPO_CONSOLIDATION_ANALYSIS.md) | One-time consolidation analysis |
| [PLANNING_REPO_SUMMARY.md](PLANNING_REPO_SUMMARY.md) | Planning snapshot |
| [NEXT_DEVELOPMENT_PHASE.md](NEXT_DEVELOPMENT_PHASE.md) | Phase roadmap; likely superseded by current work |

---

## Outdated or wrong (fix or remove)

| File | Issue |
|------|--------|
| [START_API.md](START_API.md) | References `./start-api.sh`; canonical API start is `python3 -m uvicorn music_brain.api:app ...` (see AGENTS.md). Update to match or remove. |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | References `scripts/verify_imports.py` (not in repo) and older script names; overlaps QUICK_START.md and AGENTS. Update to match current scripts or remove. |

---

## For another project or workflow

Content is about a different scope: multi-repo merge, one-time imports from other locations, or external reference dumps.

| File / folder | Reason |
|---------------|--------|
| [PROPOSAL_INDEX.md](PROPOSAL_INDEX.md), [PROPOSAL_QUICK_START.md](PROPOSAL_QUICK_START.md), [FORENSIC_CODE_RESTRUCTURING_PROPOSAL.md](FORENSIC_CODE_RESTRUCTURING_PROPOSAL.md), [REPOSITORY_VISUALIZATION.md](REPOSITORY_VISUALIZATION.md), [TECHNICAL_CAPABILITIES.md](TECHNICAL_CAPABILITIES.md) | **Forensic code restructuring** — merging 13 sburdges-eng GitHub repos into one; "AWAITING APPROVAL - NO CODE COMMITTED". Not core KmiDi development. |
| [docs/onedrive-import/](onedrive-import/) | One-time import runbook from OneDrive/SSD "KmiDi-remote". Useful only if you run that import again. |
| [docs/xcode-import/](xcode-import/) | One-time import from xcode clone (`/Users/.../xcode/KmiDi`). Same as above. |
| [PULSE_RECOVERY_ENTRIES.md](PULSE_RECOVERY_ENTRIES.md) | Recovered ChatGPT Pulse notifications (external dump); reference only. Optional to keep under docs/research or archive. |

---

## Possibly irrelevant or duplicate

Worth a quick check; may overlap canonical docs or describe deprecated flows.

| File | Note |
|------|------|
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Long project summary; overlaps README and AGENTS. Prefer README + AGENTS as single source; keep only if this adds unique content. |
| [PROJECT_DIRECTORY_MAP.md](PROJECT_DIRECTORY_MAP.md), [PROJECT_FEATURES_DEBATE.md](PROJECT_FEATURES_DEBATE.md), [PROJECT_SOURCE_MANIFEST.md](PROJECT_SOURCE_MANIFEST.md) | Project maps/manifests; may duplicate AGENTS layout and SOURCE_INTEGRATION_PLAN. |
| [START_HERE.md](START_HERE.md) | Already in Obsolete; also duplicates "where to start" with QUICK_START/AGENTS. |
| [CANONICAL_FOLDER_STRUCTURE.md](CANONICAL_FOLDER_STRUCTURE.md), [CORRECTED_STRUCTURE_ANALYSIS.md](CORRECTED_STRUCTURE_ANALYSIS.md) | Folder structure; AGENTS already defines layout. |
| [NAMING_CONVENTIONS.md](NAMING_CONVENTIONS.md) | Keep if enforced; else archive. |

---

## Design / polish / audit reports (point-in-time)

One-off reports; keep only for history.

| File | Note |
|------|------|
| CONTRAST_DEPTH_*.md, CREATIVE_POLISH_REPORT.md, FRONTEND_POLISH_REPORT.md, FRONTEND_DESIGN_BRIEF.md, RELEASE_POLISH_REPORT.md, ILLUSTRATION_DIRECTION.md | Design/polish reports |
| KMIDI_COPY_TONE_REPORT.md, KMIDI_INSTRUMENT_ART_DIRECTION_REPORT.md | Art direction reports |
| CROSS_CUTTING_AUDIT_REPORT.md, INPUT_VALIDATION_FINDINGS.md, DUPLICATE_CODE_ANALYSIS.md, FEATURE_GAP_ANALYSIS.md | Audit/findings; refresh or archive |

---

## KmiDi_FINAL and nested KmiDi/ trees

- **KmiDi_FINAL/** (sibling or optional integration per AGENTS): Contains many status/implementation docs (INTENT_IR_V1_*, PRROT_*, ROBUSTNESS_*, etc.). Those describe **KmiDi_FINAL**’s state, not the root monorepo. Treat as **another tree**; archive or keep only if you actively use that integration.
- **KmiDi/KmiDi/** (nested, with external/JUCE etc.): Appears to be a duplicate or legacy clone; its docs are **not** the main repo’s canonical set.
- **.tools/** (JDK legal, etc.): Third-party tooling; **not** project documentation.

---

## Skills and external references

- **skills/dataset-packaging/references/dataset-packaging-runbook.md** — Generic dataset packaging runbook (transcript → S3). **Relevant** if you use that skill for KmiDi datasets; otherwise skill-specific reference only.
- **.agents/**, **.github/** (issue/PR templates, copilot instructions): **Relevant** to this repo; keep.

---

## Summary

- **Keep as canonical:** AGENTS.md, README.md, BUILD.md, QUICK_START.md, DEVELOPMENT.md, ENVIRONMENT.md, FULL_STACK_BUILD.md, DATASETS_LAYOUT.md, HEADLESS_ENGINE.md, SOURCE_INTEGRATION_PLAN.md, AU_PLUGIN_ARCHITECTURE.md, SAGEMAKER_SETUP.md, rt_harness/README.md, adr/, research/ (and INTENT_IR_SPEC, CODE_REVIEW if still accurate).
- **Obsolete (archive/delete):** Break-fix logs, freeze/audit reports, completion checklists, discovery logs, merge logs, OneDrive/Stash integration logs, GIT_GRAPH_CLEANUP, START_HERE, PRE_WORKSPACE_COMPLETION_CHECKLIST, etc.
- **Fix or remove:** START_API.md (wrong script), QUICK_REFERENCE.md (missing script refs).
- **Another project/workflow:** Forensic restructuring proposal (multi-repo merge), onedrive-import/, xcode-import/, PULSE_RECOVERY_ENTRIES (optional research).
- **KmiDi_FINAL / KmiDi/ nested / .tools:** Separate trees or third-party; not the main repo’s doc set.

**Cleanup executed:** Unrelated docs were moved to `~/Dev/kmidi-docs-archive/unrelated/` and obsolete/one-off docs were deleted from the repo. See `scripts/docs-cleanup.sh`. START_API.md was updated to use uvicorn; QUICK_REFERENCE.md was removed. KmiDi_FINAL merge strategy: [docs/KMIDI_FINAL_MERGE_PLAN.md](KMIDI_FINAL_MERGE_PLAN.md).
