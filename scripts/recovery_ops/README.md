# RECOVERY_OPS Safe Deduplication Tools

**Location:** `scripts/recovery_ops/`  
**Documentation:** `docs/RECOVERY_OPS_DEDUPE/`

## Overview

Safe deduplication tools for KmiDi-related files in the RECOVERY_OPS directory. These scripts identify duplicates, determine which copies to preserve, and provide safe deletion with verification and rollback support.

## Quick Start

```bash
# 1. Analyze duplicates
python3 scripts/recovery_ops/analyze_recovery_dupes.py

# 2. Generate report
python3 scripts/recovery_ops/generate_dedupe_report.py

# 3. Verify manifest
python3 scripts/recovery_ops/verify_dedupe_manifest.py

# 4. Dry-run (simulate deletion)
python3 scripts/recovery_ops/dry_run_dedupe.py

# 5. Safe deletion (requires --live flag)
python3 scripts/recovery_ops/safe_dedupe.py --live
```

## Scripts

### 1. `analyze_recovery_dupes.py`
**Purpose:** Identify KmiDi-related duplicates in RECOVERY_OPS

**What it does:**
- Loads duplicate data from discovery analysis
- Filters to RECOVERY_OPS KmiDi-related files only
- Excludes vendor/system directories
- Determines preservation priority (CANONICAL > recent > shortest path)
- Generates preservation map and deletion manifest

**Outputs:**
- `docs/RECOVERY_OPS_DEDUPE/preservation_map.json`
- `docs/RECOVERY_OPS_DEDUPE/RECOVERY_OPS_DELETE_MANIFEST.json`
- `docs/RECOVERY_OPS_DEDUPE/dedupe_statistics.json`

### 2. `generate_dedupe_report.py`
**Purpose:** Generate human-readable deduplication report

**What it does:**
- Creates comprehensive markdown report
- Shows statistics by location
- Lists sample duplicate groups
- Provides risk assessment
- Shows space savings estimate

**Outputs:**
- `docs/RECOVERY_OPS_DEDUPE/RECOVERY_OPS_DEDUPE_REPORT.md`

### 3. `verify_dedupe_manifest.py`
**Purpose:** Verify deletion manifest before deletion

**What it does:**
- Verifies all files in manifest exist
- Verifies preservation targets exist
- Checks that preservation targets aren't marked for deletion
- Verifies file hashes match (confirms duplicates)
- Checks file sizes match

**Outputs:**
- `docs/RECOVERY_OPS_DEDUPE/verification_results.json`

**Exit code:** 0 if all checks pass, 1 if errors found

### 4. `dry_run_dedupe.py`
**Purpose:** Simulate deletion without actually deleting

**What it does:**
- Shows what would be deleted
- Calculates space that would be freed
- Breaks down by location and file extension
- Provides sample files list

**Outputs:**
- `docs/RECOVERY_OPS_DEDUPE/dry_run_results.json`
- `docs/RECOVERY_OPS_DEDUPE/DRY_RUN_REPORT.md`

### 5. `safe_dedupe.py`
**Purpose:** Safely delete duplicate files

**What it does:**
- Reads deletion manifest
- Verifies each file before deletion
- Creates detailed deletion log
- Supports rollback via log
- Progress reporting

**Usage:**
```bash
# Dry-run (default)
python3 scripts/recovery_ops/safe_dedupe.py

# Live deletion (requires confirmation)
python3 scripts/recovery_ops/safe_dedupe.py --live
```

**Outputs:**
- `docs/RECOVERY_OPS_DEDUPE/deletion_log_[dry_run|live]_[timestamp].json`
- `docs/RECOVERY_OPS_DEDUPE/DELETION_COMPLETE_[dry_run|live].md`

## Preservation Priority

Files are preserved based on this priority order:

1. **CANONICAL directories** - Highest priority (canonical copies)
2. **Most recently modified** - More current versions
3. **Shortest path** - Less nested locations
4. **Non-ARCHIVE directories** - Active files over archived

## Safety Features

1. **Exclusion Rules**
   - Never deletes files in CANONICAL directories
   - Never deletes the only copy of a file
   - Never deletes files outside RECOVERY_OPS
   - Excludes vendor/system directories

2. **Verification**
   - Hash verification before deletion
   - Preservation target verification
   - File existence checks

3. **Rollback**
   - Detailed deletion log
   - File metadata preserved
   - Preservation mapping documented

4. **Dry-Run First**
   - Always run dry-run before actual deletion
   - User approval required for live deletion
   - Detailed reporting at each step

## Current Status

**Analysis Complete:**
- 189 duplicate groups identified
- 1,507 duplicate files found
- 1,318 files marked for deletion
- 189 files to preserve
- ~0.15 GB space savings potential

**Verification:** ✅ All 1,318 files verified successfully

**Ready for:** Dry-run testing and review

## Workflow

1. **Analysis** (already complete)
   ```bash
   python3 scripts/recovery_ops/analyze_recovery_dupes.py
   ```

2. **Review Report**
   ```bash
   # View the report
   cat docs/RECOVERY_OPS_DEDUPE/RECOVERY_OPS_DEDUPE_REPORT.md
   ```

3. **Verify** (already complete)
   ```bash
   python3 scripts/recovery_ops/verify_dedupe_manifest.py
   ```

4. **Dry-Run** (already complete)
   ```bash
   python3 scripts/recovery_ops/dry_run_dedupe.py
   ```

5. **Review Dry-Run Results**
   ```bash
   cat docs/RECOVERY_OPS_DEDUPE/DRY_RUN_REPORT.md
   ```

6. **Execute Deletion** (when ready)
   ```bash
   python3 scripts/recovery_ops/safe_dedupe.py --live
   ```

## Files Generated

All outputs are in `docs/RECOVERY_OPS_DEDUPE/`:

- `RECOVERY_OPS_DEDUPE_REPORT.md` - Comprehensive report
- `RECOVERY_OPS_DELETE_MANIFEST.json` - Deletion manifest (758 KB)
- `preservation_map.json` - Preservation mapping (72 KB)
- `dedupe_statistics.json` - Statistics summary
- `verification_results.json` - Verification results (484 KB)
- `dry_run_results.json` - Dry-run results (413 KB)
- `DRY_RUN_REPORT.md` - Dry-run report
- `deletion_log_*.json` - Deletion logs (when executed)

## Notes

- All scripts focus on KmiDi-related files only
- Vendor/system files are automatically excluded
- CANONICAL directories are always preserved
- All deletions are logged for potential recovery
- Hash verification ensures files are true duplicates
