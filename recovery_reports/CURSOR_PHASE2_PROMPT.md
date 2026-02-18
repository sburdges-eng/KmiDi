# Cursor Phase 2 Prompt: Manual Triage & Categorization

## Context
You are assisting with a KmiDi project recovery operation. Phase 1 (automatic matching of 4,625 files with 99-100% confidence) is complete and staged for application. 

Phase 2 requires manual review and categorization of 1,251 remaining files:
- **Phase 2C**: 248 ambiguous conflicts (75-80% confidence)
- **Phase 3**: 1,003 new files (no baseline match)

## Phase 2C: Manual Conflict Triage (248 files)

### Task: Review and Approve/Reject Ambiguous Matches

**Location**: `/private/tmp/KmiDi_recovery_20260218-043218/recovery_reports/review_queue.csv`

**Instructions**:
1. Open `review_queue.csv` and filter for `status == "conflict"`
2. For each conflict entry, examine:
   - `recovered_abs_path`: Path to recovered file
   - `top_candidate`: Best matching git file
   - `top_score`: Confidence score (0.75-0.80)
   - `second_candidate`: Alternative match
   - `second_score`: Second best score

3. Decision Logic:
   ```
   IF (top_score - second_score) > 0.05 AND top_score > 0.77:
       DECISION = "APPROVE" (use top_candidate)
   ELIF (top_score - second_score) <= 0.05:
       DECISION = "AMBIGUOUS" (needs human judgment)
   ELSE:
       DECISION = "REJECT" (treat as new file)
   ```

4. Create output file: `triage_phase2c_decisions.csv`
   Columns: `recovered_path, decision (APPROVE/REJECT/AMBIGUOUS), top_candidate, reasoning`

5. Expected Results:
   - ~180 APPROVE (apply to repo)
   - ~50 REJECT (move to Phase 3)
   - ~18 AMBIGUOUS (needs detailed review)

**Example Decision**:
```
File: src/ml/ModelTrainer.cpp
Top Match: src/ml/MLTrainer.cpp (score: 0.78)
Second Match: src/training/Trainer.cpp (score: 0.75)
Delta: 0.03 (>0.05? NO)
Decision: AMBIGUOUS - Both are similar, human review needed
Reasoning: Similar names and scores, needs content inspection
```

---

## Phase 3: New File Categorization (1,003 files)

### Task: Categorize and Decide on New Files

**Location**: `/private/tmp/KmiDi_recovery_20260218-043218/recovery_reports/inventory_recovered.jsonl`

**Filter by**: Files where `match_type == "new_file"`

**File Type Distribution** (for reference):
- Markdown (.md): 625 files → **Recommendation: KEEP** (documentation)
- Python (.py): 117 files → **Recommendation: KEEP** (source code)
- C++ Source (.cc): 73 files → **Recommendation: KEEP** (source)
- Text (.txt): 48 files → **Recommendation: REVIEW** (could be docs or logs)
- Headers (.h): 36 files → **Recommendation: KEEP** (source)
- C++ (.cpp): 34 files → **Recommendation: KEEP** (source)
- Shell (.sh): 13 files → **Recommendation: KEEP** (build/config scripts)
- Rust (.rs): 9 files → **Recommendation: KEEP** (source)
- Config (.json, .yaml): 14 files → **Recommendation: KEEP** (config)
- Other: 34 files → **Recommendation: REVIEW** (examine individually)

### Decision Categories

**KEEP** (Core Project Files):
- Source code: .cpp, .cc, .h, .hpp, .py, .js, .ts, .java, .rs, .go
- Configuration: .json, .yaml, .toml, .cmake, .gradle, .cfg
- Documentation: .md, .rst, .adoc
- Scripts: .sh, .bash, .zsh
- Criteria: Files essential to the project build/functionality

**DISCARD** (Build Artifacts & Cache):
- Object files: .o, .obj, .a, .so, .dylib
- Cache: .gradle/caches/, __pycache__/, node_modules/
- Build: dist/, build/, target/, out/
- Temp: .tmp, .swp, .log, .bak
- Criteria: Generated files, not part of source

**REVIEW** (Ambiguous):
- Large binary files: >50MB
- Pre-trained models: .pkl, .pt, .onnx, .h5
- Datasets: .csv, .parquet, .arrow
- Media: .wav, .mp3, .mp4 (if project-specific)
- Criteria: Requires human judgment to keep

### Instructions

1. **Batch Process by Extension**:
   ```bash
   # Group new files by extension
   grep '"match_type": "new_file"' inventory_recovered.jsonl | \
     jq '.ext' | sort | uniq -c | sort -rn
   ```

2. **Create Decision Rules** by extension:
   ```
   .py, .cpp, .h, .md → KEEP
   .gradle, .build, .o, __pycache__ → DISCARD
   .pkl, .pt, .csv (>50MB) → REVIEW
   ```

3. **Generate Decision File**: `new_file_decisions.csv`
   Columns: `abs_path, ext, category, decision (KEEP/DISCARD/REVIEW), size_mb, path_pattern, reasoning`

4. **Expected Distribution**:
   - KEEP: ~650 files (65%)
   - DISCARD: ~250 files (25%)
   - REVIEW: ~103 files (10%)

**Example Decision**:
```
File: src/voice/config/model.json
Ext: .json
Category: Config
Decision: KEEP
Reasoning: Configuration file essential to voice module

File: .gradle/caches/transformCache/1234567.json
Ext: .json (in .gradle)
Category: Cache
Decision: DISCARD
Reasoning: Build artifact, regenerated on build

File: datasets/training_data.parquet
Ext: .parquet
Size: 250 MB
Category: Dataset
Decision: REVIEW
Reasoning: Large dataset, verify if needed in repo
```

---

## Output Requirements

### File 1: `triage_phase2c_decisions.csv`
```csv
recovered_path,decision,top_candidate,top_score,reasoning
src/ml/ModelTrainer.cpp,APPROVE,src/ml/MLTrainer.cpp,0.78,High confidence match with significant score gap
src/ui/Panel.cpp,AMBIGUOUS,src/ui/ControlPanel.cpp,0.76,Similar names but low confidence delta
src/engine/Core.h,REJECT,src/engine/CoreLegacy.h,0.74,Too many equally scoring candidates
```

### File 2: `new_file_decisions.csv`
```csv
abs_path,ext,category,decision,size_mb,path_pattern,reasoning
/path/to/src/Main.cpp,.cpp,Source,KEEP,0.05,src/,Core source file
/path/to/build/.gradle,.gradle,Cache,DISCARD,123,build/,Build cache directory
/path/to/datasets/train.parquet,.parquet,Dataset,REVIEW,500,datasets/,Large training dataset
```

---

## Review Workflow

### Step 1: Examine Phase 2C Conflicts (60-90 minutes)
1. Open: `review_queue.csv` (filter: conflict)
2. Review: Top 10 conflicts manually
3. Create: Decision patterns based on top 10
4. Apply: Patterns to remaining 238 conflicts
5. Validate: Spot-check 20 random decisions
6. Output: `triage_phase2c_decisions.csv`

### Step 2: Categorize Phase 3 New Files (90-120 minutes)
1. Analyze: File extension distribution
2. Create: Decision rules by extension/path
3. Auto-apply: Rules to 1,003 files
4. Manual review: Edge cases (large files, ambiguous types)
5. Validate: Distribution aligns with expectations
6. Output: `new_file_decisions.csv`

### Step 3: Generate Patchsets (30 minutes)
1. Create: `patchset_phase2c_approved.patch` (apply approved conflicts)
2. Create: `new_files_keep.patch` (add approved new files)
3. Create: `decisions_summary.md` (human-readable summary)
4. Verify: Each patchset applies cleanly

### Step 4: Documentation (15 minutes)
1. Update: `recovery_reports/phase2_completion_report.md`
2. Document: Decisions made and reasoning
3. List: Ambiguous cases requiring further review
4. Summarize: Files kept, discarded, and flagged for review

---

## Success Criteria

✅ **Phase 2C Complete**:
- 248 conflicts reviewed and categorized
- ~180 approved for inclusion
- ~50 rejected (moved to Phase 3)
- ~18 flagged as ambiguous
- `triage_phase2c_decisions.csv` generated

✅ **Phase 3 Complete**:
- 1,003 new files categorized
- ~650 approved for KEEP
- ~250 marked for DISCARD
- ~103 flagged for REVIEW
- `new_file_decisions.csv` generated

✅ **Patchsets Ready**:
- Phase 2C approved conflicts ready to apply
- Phase 3 approved new files ready to add
- Clean patches generated (no conflicts)

✅ **Documentation**:
- All decisions recorded with reasoning
- Phase 2 completion report created
- Ready for code review and validation

---

## Key Files & Locations

**Input Files**:
- `/private/tmp/KmiDi_recovery_20260218-043218/recovery_reports/review_queue.csv` (conflicts & new files)
- `/private/tmp/KmiDi_recovery_20260218-043218/recovery_reports/inventory_recovered.jsonl` (file details)
- `/private/tmp/KmiDi_recovery_20260218-043218/recovery_reports/matches.jsonl` (matching scores)

**Output Files** (create in recovery_reports/):
- `triage_phase2c_decisions.csv` (conflict decisions)
- `new_file_decisions.csv` (new file categorization)
- `phase2_completion_report.md` (summary)
- `patchset_phase2c_approved.patch` (conflict patch)
- `new_files_keep.patch` (new files patch)

**Working Directory**:
`/private/tmp/KmiDi_recovery_20260218-043218/`

---

## Helpful Commands

**View conflicts**:
```bash
cd /private/tmp/KmiDi_recovery_20260218-043218
awk -F, '$1=="conflict" {print $2, $3, $4}' recovery_reports/review_queue.csv | head -20
```

**View new files by extension**:
```bash
grep '"match_type": "new_file"' recovery_reports/matches.jsonl | \
  jq -r '.recovered_abs_path' | sed 's/.*\.//' | sort | uniq -c | sort -rn
```

**Check file sizes**:
```bash
find /private/tmp/KmiDi_recovery_20260218-043218 -type f -exec ls -lh {} \; | \
  awk '$5 ~ /[MG]$/ {print $9, $5}'
```

**Verify patchset applies cleanly**:
```bash
git apply --check patchset_phase2c_approved.patch
```

---

## Questions to Guide Your Review

**For Phase 2C Conflicts**:
1. Is the top candidate a clear match (name, location, content similarity)?
2. How different is the second candidate? (score delta)
3. Could this file have been moved/renamed on the branch?
4. If ambiguous, what would be lost/gained by choosing each option?

**For Phase 3 New Files**:
1. Is this file essential to the project (source, config, docs)?
2. Is it a generated artifact (build output, cache, temp file)?
3. Is it too large or binary to include (datasets, models)?
4. Does the file path suggest its purpose/category?

---

## Notes

- **Time Estimate**: 2-4 hours total (1-1.5 hrs Phase 2C, 1.5-2.5 hrs Phase 3)
- **Effort Level**: Medium (requires judgment calls but mostly categorization)
- **Risk Level**: Medium (decisions affect final recovery integrity)
- **Reversible**: Yes (can always reject/accept later if needed)
- **Review Required**: Yes (human approval recommended before apply)

---

**Status**: Phase 2 ready to begin
**Priority**: HIGH (unblocks Phase 3 validation and merge)
**Owner**: Human triage with tool assistance

