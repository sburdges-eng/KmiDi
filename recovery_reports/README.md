# Recovery Reports

This directory contains the output of the recovery reconciliation process.

## Path Privacy and Security

To protect sensitive filesystem information, absolute recovery source paths are **redacted** in committed reports:

### Committed Files (Safe for Sharing)
- `run_meta.json` - Contains placeholder values like `<RECOVERY_ROOT_1>`, `<RECOVERY_ROOT_2>`, etc.
- All other reports use relative paths or repository-relative paths

### Local-Only Files (Not Committed)
- `path_mapping.local.json` - Maps placeholders to actual filesystem paths
  - **This file is in `.gitignore`** and contains the real paths
  - Only exists on the machine that ran the recovery script
  - Required for debugging or re-running the script with original paths

### How It Works

1. When `scripts/reconcile_recovery.py` runs, it:
   - Takes absolute paths as command-line arguments
   - Redacts them to `<RECOVERY_ROOT_N>` placeholders for committed reports
   - Saves the actual mapping to `path_mapping.local.json` (local-only)

2. Benefits:
   - Prevents leaking user home directories or volume mount points
   - Keeps reports shareable across teams/repositories
   - Maintains audit trail without exposing infrastructure details

### Example

**Before (exposed):**
```json
"selected_source_roots": [
  "/System/Volumes/Data/Users/seanburdges/RECOVERY_STAGED_2026-02-18/...",
  "/Volumes/KmiDi-external/COLD_STORAGEEXTERNAL"
]
```

**After (redacted):**
```json
"selected_source_roots": [
  "<RECOVERY_ROOT_1>",
  "<RECOVERY_ROOT_2>"
]
```

## Files in This Directory

### Core Reports
- `run_meta.json` - Run metadata with **redacted** source paths
- `inventory_canonical.jsonl` - Git baseline inventory
- `inventory_recovered.jsonl` - Recovered files inventory
- `matches.jsonl` - Matching analysis with confidence scores

### Decision Outputs
- `decisions.csv` - Auto-apply + manual review decisions
- `review_queue.csv` - Triage queue for manual decisions
- `patchset_plan.md` - Detailed apply strategy

### Phase Execution
- `phase2_*.csv` - Phase 2 triage batches
- `phase3_*.{md,csv,json}` - Phase 3 apply manifest and results
- `phase4_status.md` - Phase 4 status report
- `build_validation.md` - Build validation results

### Supporting Documents
- `pr_checklist.md` - PR preparation checklist
- `determinism_check.md` - Determinism validation
- `triage_analysis.md` - Triage analysis
- `final_summary.md` - Final summary

### Logs
- `logs/` - Detailed execution logs

## Privacy Policy

**Never commit `*.local.json` files** - they contain unredacted filesystem paths and are automatically excluded via `.gitignore`.

If you need to share the actual paths (e.g., for debugging), do so through a secure, private channel outside of version control.
