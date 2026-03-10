# Excess Duplicate Cleanup Guide

**Purpose:** Free disk space by removing files that have **MORE THAN 2 duplicates** across evidence pools.

**Strategy:** For each file group with 3+ copies, keep the 2 best versions (based on quality, recency, and size) and delete the rest.

---

## 🎯 What This Does

- Scans evidence pools: `~/_sorted` and `~/My Mac`
- Identifies files with **>2 duplicates** (same SHA-256 hash)
- Selects the **2 best versions** using quality scoring
- Marks the rest for deletion
- **Does NOT touch:** `~/KmiDi/` or `~/KmiDi_MASTER_VAULT/` (primary sources with Git)

---

## ⚡ Quick Start

### Step 1: Analyze (Safe - Read Only)

```bash
cd ~/RECOVERY_OPS
python3 cleanup_excess_duplicates.py
```

**This will:**
- Scan evidence pools
- Find duplicate groups with >2 copies
- Generate deletion list
- Create cleanup report
- **NO files are deleted yet**

**Output:**
- `REPORTS/files_to_delete.txt` - List of files to remove
- `REPORTS/CLEANUP_REPORT.md` - Detailed analysis
- `REPORTS/deletion_log.json` - Audit trail

### Step 2: Review (CRITICAL)

```bash
# Read the report
cat ~/RECOVERY_OPS/CANONICAL_REBUILD/REPORTS/CLEANUP_REPORT.md

# Check the deletion list
cat ~/RECOVERY_OPS/CANONICAL_REBUILD/REPORTS/files_to_delete.txt

# Verify space savings
grep "Space to be freed" ~/RECOVERY_OPS/CANONICAL_REBUILD/REPORTS/CLEANUP_REPORT.md
```

**⚠️ REVIEW CAREFULLY:**
- Make sure no critical files are in the deletion list
- Verify the canonical rebuild is complete
- Ensure you have Git backup (`git bundle`)

### Step 3: Execute (PERMANENT - Cannot Undo)

```bash
cd ~/RECOVERY_OPS
python3 execute_cleanup.py
```

**This will:**
- Load the deletion list
- Show files to be deleted
- **Require confirmation** (you must type `DELETE ALL`)
- Delete files with progress tracking
- Log all actions

---

## 📊 Expected Results

Based on typical fragmented recovery scenarios:

| Metric | Typical Range |
|--------|---------------|
| **Duplicate groups (>2)** | 1,000 - 10,000 |
| **Files to delete** | 5,000 - 50,000 |
| **Space freed** | 10 GB - 100 GB |

**Your actual results will vary** based on the level of fragmentation.

---

## 🔐 Safety Features

### Built-in Protections

✅ **Primary sources protected:**
- `~/KmiDi/` - NOT scanned (has .git history)
- `~/KmiDi_MASTER_VAULT/` - NOT scanned (backup vault)

✅ **Quality-based selection:**
- Best 2 versions always kept
- Uses same algorithm as recovery (AST parsing, syntax, timestamps)

✅ **Explicit confirmation required:**
- Must type `DELETE ALL` exactly
- Shows sample files before deletion
- Reports space to be freed

✅ **Complete audit trail:**
- All decisions logged
- Deletion log with timestamps
- Can trace every action

✅ **Two-phase process:**
- Phase 1: Analyze (read-only, safe)
- Phase 2: Execute (requires confirmation)

---

## 📋 Detailed Usage

### Phase 1: Analysis

```bash
cd ~/RECOVERY_OPS

# Run analysis
python3 cleanup_excess_duplicates.py

# Sample output:
#
# ============================================================
# EXCESS DUPLICATE CLEANUP ANALYZER
# ============================================================
# 
# Analyzing files with MORE THAN 2 duplicates
# Strategy: Keep 2 best versions, mark rest for deletion
#
# Scanning evidence pools for duplicates...
#   Scanning: /Users/seanburdges/_sorted
#     Indexed 1000 files...
#     Indexed 2000 files...
#   ✓ Completed: 2347 files indexed
#   Scanning: /Users/seanburdges/My Mac
#     Indexed 1000 files...
#   ✓ Completed: 1523 files indexed
#
# Found 234 file groups with >2 duplicates
#
# Analyzing 234 duplicate groups with >2 copies...
# ✓ Deletion list written to: REPORTS/files_to_delete.txt
# ✓ Cleanup report written to: REPORTS/CLEANUP_REPORT.md
# ✓ Deletion log written to: REPORTS/deletion_log.json
#
# ============================================================
# ANALYSIS COMPLETE
# ============================================================
#
# 📊 Summary:
#   - Duplicate groups (>2 copies): 234
#   - Files to keep: 468
#   - Files to delete: 702
#   - Space to free: 15.32 GB
#
# 📁 Reports:
#   - Deletion list: REPORTS/files_to_delete.txt
#   - Full report: REPORTS/CLEANUP_REPORT.md
#   - Audit log: REPORTS/deletion_log.json
#
# ⚠️  REVIEW CAREFULLY before executing deletion!
```

### Phase 2: Review

**Check the cleanup report:**

```bash
cat ~/RECOVERY_OPS/CANONICAL_REBUILD/REPORTS/CLEANUP_REPORT.md
```

**Verify deletion list:**

```bash
# View first 20 files
head -20 ~/RECOVERY_OPS/CANONICAL_REBUILD/REPORTS/files_to_delete.txt

# Count total files
grep -v "^#" ~/RECOVERY_OPS/CANONICAL_REBUILD/REPORTS/files_to_delete.txt | wc -l

# Search for specific patterns (example)
grep "important" ~/RECOVERY_OPS/CANONICAL_REBUILD/REPORTS/files_to_delete.txt
```

**Check space savings:**

```bash
grep "Space to be freed" ~/RECOVERY_OPS/CANONICAL_REBUILD/REPORTS/CLEANUP_REPORT.md
```

### Phase 3: Execution

```bash
cd ~/RECOVERY_OPS
python3 execute_cleanup.py

# Sample output:
#
# ============================================================
# EXCESS DUPLICATE CLEANUP - EXECUTION
# ============================================================
#
# Loading deletion list...
# ✓ Loaded 702 files to delete
#   Total size: 15.32 GB
#
# 📋 Sample files to be deleted (first 10):
#   - /Users/seanburdges/_sorted/backup_20250115/file1.txt
#   - /Users/seanburdges/_sorted/backup_20250115/file2.py
#   ... and 692 more
#
# ============================================================
# ⚠️  DELETION CONFIRMATION REQUIRED
# ============================================================
#
# You are about to DELETE 702 files
# Total size: 15.32 GB
#
# ⚠️  THIS ACTION IS PERMANENT AND CANNOT BE UNDONE
#
# Type 'DELETE ALL' (exactly, in caps) to proceed: DELETE ALL
#
# ✓ Confirmation received. Starting deletion...
#
# Deleting files...
#   Progress: 100/702 files (100 deleted, 0 failed)
#   Progress: 200/702 files (200 deleted, 0 failed)
#   ...
#   Progress: 700/702 files (700 deleted, 0 failed)
#
# ============================================================
# CLEANUP COMPLETE
# ============================================================
#
# ✓ Files deleted: 702
# ✗ Failed deletions: 0
# 💾 Space freed: 15.32 GB
#
# 📊 Execution log: REPORTS/execution_log.json
#
# ✅ Cleanup operation complete!
```

---

## ⚠️ Pre-Flight Checklist

**Before running cleanup, verify:**

- [ ] **Canonical rebuild is complete**
  ```bash
  ls -la ~/RECOVERY_OPS/CANONICAL_REBUILD/KmiDi
  ```

- [ ] **Git backup created**
  ```bash
  ls -lh ~/Desktop/kmidi_backup*.bundle
  ```

- [ ] **Reviewed recovery reports**
  ```bash
  cat ~/RECOVERY_OPS/CANONICAL_REBUILD/START_HERE.md
  ```

- [ ] **Build system validated** (optional but recommended)
  ```bash
  cd ~/RECOVERY_OPS/CANONICAL_REBUILD/KmiDi
  cmake build/ || echo "Validation needed"
  ```

- [ ] **Disk space is critical** (otherwise wait)
  ```bash
  df -h ~ | grep "%"
  ```

---

## 🔍 Understanding the Logic

### Duplicate Detection

Files are considered duplicates if they have:
- **Identical SHA-256 hash** (content is byte-for-byte identical)
- **Same file extension**
- Located in different directories or with different timestamps

### Best Version Selection

For each duplicate group, the 2 best versions are selected based on:

1. **Quality Score (weight: 2.0)**
   - **Python:** AST parsing, syntax validity, imports, docstrings, functions
   - **C++:** Syntax, includes, namespaces, classes, comments
   - **JSON:** Validity, structure depth, completeness
   - **All:** Length (not tiny stubs)

2. **Recency Score (weight: 1.5)**
   - Modification timestamp
   - Newer files preferred

3. **Size Score (weight: 1.0)**
   - File size
   - Larger files often more complete

**Formula:** `score = (quality × 2.0) + (recency × 1.5) + (size × 1.0)`

**Top 2 scores are kept, rest are marked for deletion.**

### Example

**File group with 4 duplicates:**
```
file1.py - Score: 25.3 (newest, best quality)    → KEEP
file2.py - Score: 23.1 (older, good quality)     → KEEP  
file3.py - Score: 18.5 (old, stub file)          → DELETE
file4.py - Score: 15.2 (oldest, incomplete)      → DELETE
```

---

## 📁 Output Files

### `files_to_delete.txt`
Plain text list of files to delete (one per line)

### `CLEANUP_REPORT.md`
Comprehensive report with:
- Summary statistics
- Example duplicate groups
- Bash commands for manual deletion
- Safety notes

### `deletion_log.json`
JSON audit trail of all decisions:
```json
[
  {
    "timestamp": "2026-01-27T10:30:00",
    "event": "duplicate_group_processed",
    "data": {
      "hash": "a3f5b8c2",
      "total": 4,
      "keeping": 2,
      "deleting": 2,
      "keepers": ["/path/to/best1.py", "/path/to/best2.py"],
      "to_delete": ["/path/to/old1.py", "/path/to/old2.py"]
    }
  }
]
```

### `execution_log.json`
JSON log of actual deletions (created during execution):
```json
[
  {
    "timestamp": "2026-01-27T10:35:00",
    "action": "deleted",
    "details": {
      "path": "/path/to/file.py",
      "size": 12345
    }
  }
]
```

---

## 🚨 Troubleshooting

### "Deletion list not found"
**Problem:** You ran `execute_cleanup.py` before `cleanup_excess_duplicates.py`

**Solution:**
```bash
cd ~/RECOVERY_OPS
python3 cleanup_excess_duplicates.py  # Run this first
```

### "Permission denied" errors during deletion
**Problem:** Some files are write-protected

**Solution:**
```bash
# Check file permissions
ls -la /path/to/problem/file

# If safe to delete, change permissions:
chmod 644 /path/to/problem/file
```

### "No files found with >2 duplicates"
**Problem:** Your evidence pools don't have excess duplicates

**Solution:** This is actually good! Your pools are already clean. No action needed.

### Script hangs during hashing
**Problem:** Large files or slow disk

**Solution:** Be patient - hashing large files takes time. The script shows progress every 1,000 files.

---

## 💡 Tips & Best Practices

### 1. Run Analysis First (Always)
Never execute cleanup without running analysis first. The analysis is read-only and safe.

### 2. Review the Report Carefully
Spend 10-15 minutes reviewing the cleanup report before executing.

### 3. Check for Critical Files
Search the deletion list for important keywords:
```bash
grep -i "config\|secret\|key\|important" ~/RECOVERY_OPS/CANONICAL_REBUILD/REPORTS/files_to_delete.txt
```

### 4. Start with Dry Run
If nervous, modify `execute_cleanup.py` to add a `--dry-run` flag that prints actions without deleting.

### 5. Incremental Cleanup
If you're not confident, manually delete just a few files first to verify the logic.

---

## 🔄 Alternative: Manual Deletion

If you prefer manual control:

```bash
# Review the deletion list
cat ~/RECOVERY_OPS/CANONICAL_REBUILD/REPORTS/files_to_delete.txt

# Delete one file at a time (safe)
rm "/path/to/specific/file.txt"

# Or use xargs for batch deletion (CAUTION)
grep -v "^#" ~/RECOVERY_OPS/CANONICAL_REBUILD/REPORTS/files_to_delete.txt | xargs -I {} rm -f "{}"
```

---

## 📊 Expected Timeline

| Phase | Duration | Effort |
|-------|----------|--------|
| **Analysis** | 5-30 minutes | Automated |
| **Review** | 10-15 minutes | Manual |
| **Execution** | 5-20 minutes | Automated |
| **Total** | 20-65 minutes | Minimal |

**Timeline depends on:**
- Number of files in evidence pools
- Disk speed
- Number of duplicate groups

---

## ✅ Post-Cleanup Verification

After cleanup completes:

```bash
# Check disk space freed
df -h ~

# Verify canonical rebuild is still intact
ls -la ~/RECOVERY_OPS/CANONICAL_REBUILD/KmiDi

# Count source files (should be unchanged)
find ~/RECOVERY_OPS/CANONICAL_REBUILD/KmiDi -name "*.cpp" | wc -l

# Check execution log
cat ~/RECOVERY_OPS/CANONICAL_REBUILD/REPORTS/execution_log.json

# Verify Git is still intact
cd ~/KmiDi && git status
```

**Everything should still work** - only excess duplicates from secondary pools were removed.

---

## 🎯 Summary

**Purpose:** Free disk space by removing excess duplicates (>2 copies)

**Safety:** Primary sources protected, 2 best versions always kept

**Process:**
1. Run `cleanup_excess_duplicates.py` (safe, read-only)
2. Review reports carefully
3. Run `execute_cleanup.py` (requires confirmation)

**Result:** Significant disk space freed while preserving best versions

---

**Questions?** Review the generated reports in `~/RECOVERY_OPS/CANONICAL_REBUILD/REPORTS/`

