# Cursor Phase 2 Prompts - Complete Package

## Files Included

### 1. CURSOR_PHASE2_PROMPT.md (9.7 KB)
**Comprehensive Phase 2 Prompt Document**

Complete instructions for handling Phase 2C (248 conflict triage) and Phase 3 (1,003 new file categorization).

**Contents**:
- Full context and background
- Phase 2C detailed instructions with decision logic
- Phase 3 categorization rules and examples
- Output file specifications
- Complete review workflow (4 steps)
- Success criteria
- Key files and locations
- Helpful commands
- Guidance questions

**Use Case**: Copy and paste entire content into Cursor for comprehensive Phase 2 guidance

---

### 2. PHASE2_QUICK_REFERENCE.txt (3.2 KB)
**Quick Reference Card**

Condensed one-page reference for quick lookup during triage.

**Contents**:
- Task overview
- Decision logic (compact format)
- File type rules
- Expected results
- Workflow checklist
- Output files list
- Key commands
- Decision checklist
- Success criteria

**Use Case**: Keep visible while working; quick lookup during decisions

---

## How to Use These Prompts

### Setup
```bash
# Copy prompts to recovery project
cp /tmp/CURSOR_PHASE2_PROMPT.md /private/tmp/KmiDi_recovery_20260218-043218/recovery_reports/
cp /tmp/PHASE2_QUICK_REFERENCE.txt /private/tmp/KmiDi_recovery_20260218-043218/recovery_reports/

# Navigate to project
cd /private/tmp/KmiDi_recovery_20260218-043218
```

### Option A: Comprehensive (Recommended for First-Time)
1. Open CURSOR_PHASE2_PROMPT.md
2. Read full context and instructions
3. Follow step-by-step workflow
4. Reference decision logic for complex cases

### Option B: Quick Reference (Recommended for Experienced)
1. Keep PHASE2_QUICK_REFERENCE.txt visible
2. Use decision logic from quick card
3. Reference full prompt for edge cases
4. Follow workflow checklist

### Option C: Hybrid (Recommended)
1. Start with quick reference for overview
2. Refer to full prompt for details
3. Use quick card during actual work
4. Check full prompt for validation

---

## Workflow Overview

### Timeline: 2-4 hours
```
├─ Phase 2C Triage (60-90 min)
│  ├─ Review 248 conflicts
│  ├─ Apply decision logic
│  └─ Output: triage_phase2c_decisions.csv
│
├─ Phase 3 Categorization (90-120 min)
│  ├─ Categorize 1,003 new files
│  ├─ Apply extension rules
│  └─ Output: new_file_decisions.csv
│
├─ Generate Patchsets (30 min)
│  ├─ Create phase2c_approved.patch
│  ├─ Create new_files_keep.patch
│  └─ Verify cleanly apply
│
└─ Documentation (15 min)
   ├─ Create phase2_completion_report.md
   └─ Ready for code review
```

---

## Key Decision Rules

### Phase 2C Conflict Triage
```
Confidence = top_score
Delta = top_score - second_score

IF delta > 0.05 AND top_score > 0.77:
    Decision = APPROVE → Apply to repo
ELIF delta <= 0.05:
    Decision = AMBIGUOUS → Flag for review
ELSE:
    Decision = REJECT → Move to Phase 3
```

### Phase 3 File Categorization
```
KEEP:     Source code, configs, docs (.py, .cpp, .md, .json)
DISCARD:  Build artifacts, cache (.o, .gradle, __pycache__)
REVIEW:   Large files, models, datasets (>50MB, .pkl, .pt)
```

---

## Output Files to Create

### 1. triage_phase2c_decisions.csv
```csv
recovered_path,decision,top_candidate,top_score,reasoning
src/ml/Model.cpp,APPROVE,src/ml/MLModel.cpp,0.78,Clear match
src/ui/Panel.cpp,AMBIGUOUS,src/ui/ControlPanel.cpp,0.76,Similar scores
src/engine/Core.h,REJECT,src/engine/CoreLegacy.h,0.74,Ambiguous
```

### 2. new_file_decisions.csv
```csv
abs_path,ext,category,decision,size_mb,path_pattern,reasoning
/path/src/Main.cpp,.cpp,Source,KEEP,0.05,src/,Core file
/path/build/.gradle,.gradle,Cache,DISCARD,123,build/,Build cache
/path/data/train.parquet,.parquet,Dataset,REVIEW,500,data/,Large dataset
```

### 3. phase2_completion_report.md
Summary document with:
- Decisions made
- Statistics (APPROVE/REJECT/KEEP/DISCARD/REVIEW counts)
- Reasoning for major decisions
- Ambiguous cases flagged
- Ready for merge checklist

---

## Success Criteria Checklist

- [ ] 248 conflicts reviewed and categorized
- [ ] ~180 APPROVE, ~50 REJECT, ~18 AMBIGUOUS
- [ ] 1,003 new files categorized
- [ ] ~650 KEEP, ~250 DISCARD, ~103 REVIEW
- [ ] triage_phase2c_decisions.csv generated
- [ ] new_file_decisions.csv generated
- [ ] phase2_completion_report.md created
- [ ] Patchsets generated and verified
- [ ] All decisions have reasoning
- [ ] Ready for code review

---

## Common Questions

**Q: What if a conflict is truly ambiguous?**
A: Flag it as AMBIGUOUS. Document both candidates and why they're equally scored. Let the code reviewer make the final call.

**Q: Should I keep large files (>50MB)?**
A: Unless they're source code or essential configs, REVIEW them. Likely are datasets/models that could be external storage.

**Q: What about files in .gradle or build directories?**
A: ALWAYS DISCARD. These are generated artifacts that will be recreated during next build.

**Q: How strict should I be about .md files?**
A: KEEP all .md files unless they're obviously generated (e.g., from auto-doc). Documentation is valuable.

**Q: Can I change decisions later?**
A: Yes! This is reversible. If you're unsure, flag as REVIEW and document why.

---

## Quick Start Commands

```bash
# Navigate to project
cd /private/tmp/KmiDi_recovery_20260218-043218

# View sample conflicts
awk -F, '$1=="conflict" {print $2, $3, $4}' recovery_reports/review_queue.csv | head -10

# View new files by extension
grep '"match_type": "new_file"' recovery_reports/matches.jsonl | jq -r '.ext' | sort | uniq -c | sort -rn

# Check file sizes
find . -type f -size +50M 2>/dev/null | wc -l

# Create output directory (if needed)
mkdir -p recovery_reports
```

---

## Tips for Efficiency

1. **Batch by Extension**: Group Phase 3 files by extension; apply rules in batches
2. **Decision Patterns**: For Phase 2C, establish patterns from first 10 conflicts; apply systematically
3. **Spot Checks**: After batching, randomly verify 20+ decisions for accuracy
4. **Documentation**: Keep notes on major decisions for the completion report
5. **Version Control**: Git commit decision files as you complete each phase

---

## Files Location

**Input Files**:
- `/private/tmp/KmiDi_recovery_20260218-043218/recovery_reports/review_queue.csv`
- `/private/tmp/KmiDi_recovery_20260218-043218/recovery_reports/inventory_recovered.jsonl`
- `/private/tmp/KmiDi_recovery_20260218-043218/recovery_reports/matches.jsonl`

**Output Files** (create in recovery_reports/):
- `triage_phase2c_decisions.csv`
- `new_file_decisions.csv`
- `phase2_completion_report.md`
- `patchset_phase2c_approved.patch`
- `new_files_keep.patch`

**Prompt Files**:
- `CURSOR_PHASE2_PROMPT.md` (comprehensive)
- `PHASE2_QUICK_REFERENCE.txt` (quick lookup)

---

## Status

**Phase 2 Status**: READY TO BEGIN ✓
**Priority**: HIGH (unblocks Phase 3 validation)
**Estimated Duration**: 2-4 hours
**Risk Level**: MEDIUM (decisions affect recovery integrity)
**Reversibility**: YES (can adjust decisions later)

---

**Generated**: 2026-02-18 04:44 MST
**Project**: KmiDi Recovery
**Phase**: 2 (Manual Triage & Categorization)
