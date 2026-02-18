# Patchset Plan (Safe Staged Apply)

## Remote Gate
- Online source-of-truth refresh failed in this run.
- No reconciled file changes were auto-applied.

## Batch A (Exact)
- exact_match count: 3945
- action: auto-apply when remote gate is cleared.

## Batch B (Duplicate/Move)
- duplicate_or_move count: 467
- action: apply only deterministic destination mappings.

## Batch C (Heuristic Candidates)
- candidate_match count: 180
- action: manual approval required.

## Batch D (Conflicts / New Files)
- conflict count: 281
- new_file count: 1003
- action: resolve via review_queue.csv before apply.

## Rollback
- Revert staged recovery commits only (no history rewrite).
- Keep `recovery_reports/` for audit.
