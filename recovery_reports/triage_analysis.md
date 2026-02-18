# Manual Triage Analysis & Recommendations

## Overview
- **Total Items for Manual Triage**: 1,464 files
- **High-Confidence Candidates**: 180 files (92%+ score)
- **Ambiguous Conflicts**: 281 files (75-92% score)
- **New Files**: 1,003 files (no baseline match)

## Triage Strategy

### Batch 1: High-Confidence Candidates (180 files) ✅ RECOMMEND AUTO-APPROVE
**Strategy**: Accept all candidates with 92%+ confidence scores
- **Reasoning**: These have very high matching confidence
- **Examples** (score 1.0000):
  - `src/voice/CMUDictionary.cpp`
  - `src/ml/MLBridge.h`
  - `src/ml/MLBridge.cpp`
  - `src/engine/TemporalMemory.h`
  - `src/engine/QuantumEntropy.h`
  - `ML Kelly Training/train_mps_stub.py`
  - `configs/__init__.py`
- **Action**: Accept all (score >= 0.92)
- **Risk**: VERY LOW

### Batch 2: Conflicts (281 files) 🔍 REQUIRES REVIEW
**Strategy**: Categorize by conflict type
- **Possible Causes**:
  - Path has changed (file moved/renamed)
  - Content overlap with multiple files
  - Partial history divergence
- **Recommended Actions**:
  1. Examine top_score vs second_score delta
  2. If delta > 0.10 and top_score > 0.80 → Accept top candidate
  3. If delta <= 0.10 → Manual inspection needed
  4. If both candidates <0.75 → New file
- **Risk**: MEDIUM (needs human judgment)

### Batch 3: New Files (1,003 files) 📋 NEEDS CATEGORIZATION
**Strategy**: Categorize by file type and purpose
- **Categories**:
  1. **Keep**: Core source files, configs, documentation
  2. **Discard**: Build artifacts, temporary files, logs
  3. **Review**: Ambiguous cases
- **Recommended Process**:
  1. Filter by extension (e.g., .cpp, .h, .py, .md)
  2. Filter by path patterns (e.g., src/, tests/, docs/)
  3. Exclude build/cache directories
  4. Manual decision for edge cases
- **Risk**: MEDIUM (important to categorize correctly)

## Quick Decision Rules

```
IF confidence >= 0.92:
    DECISION = "Accept" → Auto-apply
ELIF confidence >= 0.80 AND (top_score - second_score) > 0.10:
    DECISION = "Accept top candidate"
ELIF confidence >= 0.75:
    DECISION = "Manual review needed"
ELSE:
    DECISION = "New file" → Categorize
```

## Recommended Triage Order

1. **Phase 1**: Accept all 180 high-confidence candidates
   - Expected Result: +180 files added
   - Risk: VERY LOW
   - Effort: Minimal (automated decision)

2. **Phase 2**: Analyze 281 conflicts
   - Separate by delta >= 0.10 and top_score > 0.80 (likely auto-accept)
   - Separate by delta < 0.10 or top_score <= 0.80 (manual review)
   - Expected: ~200 auto-accept, ~81 manual review
   - Risk: MEDIUM
   - Effort: 2-3 hours

3. **Phase 3**: Categorize 1,003 new files
   - Group by extension and path
   - Accept core project files (src/, tests/, docs/)
   - Discard build artifacts (build/, .gradle/, dist/)
   - Expected: ~600-700 keep, ~200-300 discard, ~100-200 review
   - Risk: MEDIUM
   - Effort: 3-4 hours (with automation)

## File Type Analysis Recommendations

**High Priority** (Core Project Files):
- Source: .cpp, .h, .py, .js, .ts, .java
- Config: .json, .yaml, .cmake, .gradle, .toml
- Docs: .md, .txt, .rst

**Low Priority** (Build/Cache):
- Build: .o, .obj, .a, .so, .dylib
- Cache: .gradle/, build/, dist/, __pycache__/
- Temp: .tmp, .log, .swp

**Review** (Ambiguous):
- Datasets (large .csv, .parquet)
- Models (pre-trained .pkl, .pt, .onnx)
- Media (audio, video if project-specific)

## Estimated Timeline

| Phase | Items | Effort | Status |
|-------|-------|--------|--------|
| Candidates | 180 | <30 min | Ready to auto-approve |
| Conflicts | 281 | 2-3 hrs | Requires triage logic |
| New Files | 1,003 | 3-4 hrs | Requires categorization |
| **Total** | **1,464** | **5-8 hrs** | In progress |

## Next Steps
1. Auto-approve all 180 candidates
2. Analyze conflicts with delta-based logic
3. Batch-process new files by extension/path
4. Manual review remaining ambiguous items
5. Stage and test patchset
6. Apply to main branch after validation

