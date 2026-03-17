#!/usr/bin/env bash
# Move unrelated docs to ~/Dev (outside KmiDi); delete obsolete/one-off project markdown.
# Run from KmiDi repo root. Creates ~/Dev/kmidi-docs-archive/{unrelated,obsolete}.
set -e
REPO_ROOT="$(git rev-parse --show-toplevel)"
ARCHIVE="${ARCHIVE_ROOT:-$HOME/Dev/kmidi-docs-archive}"
cd "$REPO_ROOT"

echo "Archive root: $ARCHIVE"
mkdir -p "$ARCHIVE/unrelated"
mkdir -p "$ARCHIVE/obsolete"

# ---- Move UNRELATED (another project/workflow) to ~/Dev outside KmiDi ----
move_unrelated() {
  for f in "$@"; do
    if [ -e "docs/$f" ]; then
      mkdir -p "$ARCHIVE/unrelated/$(dirname "$f")"
      mv "docs/$f" "$ARCHIVE/unrelated/$f" && echo "Moved docs/$f -> archive/unrelated/"
    fi
  done
}

move_unrelated "PROPOSAL_INDEX.md"
move_unrelated "PROPOSAL_QUICK_START.md"
move_unrelated "FORENSIC_CODE_RESTRUCTURING_PROPOSAL.md"
move_unrelated "REPOSITORY_VISUALIZATION.md"
move_unrelated "TECHNICAL_CAPABILITIES.md"
move_unrelated "PULSE_RECOVERY_ENTRIES.md"

if [ -d "docs/onedrive-import" ]; then
  mv docs/onedrive-import "$ARCHIVE/unrelated/" && echo "Moved docs/onedrive-import -> archive/unrelated/"
fi
if [ -d "docs/xcode-import" ]; then
  mv docs/xcode-import "$ARCHIVE/unrelated/" && echo "Moved docs/xcode-import -> archive/unrelated/"
fi

# ---- Move OBSOLETE to archive (then delete from repo) or delete directly ----
# We delete from repo; optional: move to archive/obsolete first for safety.
delete_obsolete() {
  for f in "$@"; do
    if [ -e "docs/$f" ]; then
      rm -f "docs/$f" && echo "Deleted docs/$f"
    fi
  done
}

delete_obsolete "BREAK_FIX_RUN_LOG.md"
delete_obsolete "DEBUGGING_PLAN_RUN.md"
delete_obsolete "FREEZE_REPORT.md" "FREEZE_FINAL_REPORT.md" "FREEZE_AUDIT_PHASE1.md" "FREEZE_AUDIT_PHASE1_CONFIRMED.md" "FREEZE_ENV_MATRIX.md"
delete_obsolete "GIT_GRAPH_CLEANUP.md"
delete_obsolete "PRE_WORKSPACE_COMPLETION_CHECKLIST.md"
delete_obsolete "COMPLETED_SUMMARY.md"
delete_obsolete "START_HERE.md"
delete_obsolete "MERGE_LOG.md" "merge-all-branches-log.md"
delete_obsolete "STASH_INTEGRATION_LOG.md" "STASH_INVENTORY.md"
delete_obsolete "ONEDRIVE_INTEGRATION_LOG.md"
delete_obsolete "PATCH_BATCH_REPORT.md"
delete_obsolete "PR_COMPILE_TO_MAIN.md" "PR_CONFLICT_RESOLUTIONS.md" "PR_REVIEW_ANALYSIS.md"
delete_obsolete "RESTRUCTURE_STATUS.md"
delete_obsolete "REPO_CONSOLIDATION_ANALYSIS.md"
delete_obsolete "PLANNING_REPO_SUMMARY.md"
delete_obsolete "NEXT_DEVELOPMENT_PHASE.md"
delete_obsolete "QUICK_REFERENCE.md"

for f in docs/DISCOVERY_*.md; do
  [ -f "$f" ] && rm -f "$f" && echo "Deleted $f"
done

# Duplicate/irrelevant project docs
delete_obsolete "PROJECT_SUMMARY.md" "PROJECT_DIRECTORY_MAP.md" "PROJECT_FEATURES_DEBATE.md" "PROJECT_SOURCE_MANIFEST.md"
delete_obsolete "CANONICAL_FOLDER_STRUCTURE.md" "CORRECTED_STRUCTURE_ANALYSIS.md"

# Design/polish one-off reports
delete_obsolete "CONTRAST_DEPTH_REPORT.md" "CONTRAST_DEPTH_AUDIT_PHASE1.md"
delete_obsolete "CREATIVE_POLISH_REPORT.md" "FRONTEND_POLISH_REPORT.md" "FRONTEND_DESIGN_BRIEF.md" "RELEASE_POLISH_REPORT.md" "ILLUSTRATION_DIRECTION.md"
delete_obsolete "KMIDI_COPY_TONE_REPORT.md" "KMIDI_INSTRUMENT_ART_DIRECTION_REPORT.md"
delete_obsolete "CROSS_CUTTING_AUDIT_REPORT.md" "INPUT_VALIDATION_FINDINGS.md"

# Optional: keep FEATURE_GAP_ANALYSIS, DUPLICATE_CODE_ANALYSIS if still useful
delete_obsolete "FEATURE_GAP_ANALYSIS.md"
delete_obsolete "DUPLICATE_CODE_ANALYSIS.md"

# START_API.md: fixed in-repo to use uvicorn (see DOCS_CLEANUP_PLAN.md). QUICK_REFERENCE already deleted above.
echo "Done. Unrelated docs are in $ARCHIVE/unrelated; obsolete docs removed from repo."
