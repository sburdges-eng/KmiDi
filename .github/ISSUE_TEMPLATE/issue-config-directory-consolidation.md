---
name: Config Directory Consolidation
about: Consolidate config/ and configs/ directories
title: '🔧 Config: Consolidate config/ and configs/ directories'
labels: architecture, cleanup, documentation, priority-critical
assignees: ''
---

## Problem

We currently have two separate directories for configuration files:
- `config/` (referenced in KMIDI_STRUCTURE_PLAN.md, BUILD_VARIANTS.md)
- `configs/` (actual directory containing most config files)

This creates confusion about where to place new configs and which directory is canonical.

## Current State

- `config/build-dev-mac.yaml` ← Recently added Mac dev config
- `configs/` contains 14 YAML files:
  - emotion_recognizer.yaml
  - harmony_predictor.yaml
  - melody_transformer.yaml
  - groove_predictor.yaml
  - dynamics_engine.yaml
  - instrument_recognizer.yaml
  - emotion_node_classifier.yaml
  - train-mac-smoke.yaml
  - macOS_16gb_optimized.yaml
  - nvidia_cuda_optimized.yaml
  - music_foundation_base.yaml
  - and more...

## Recommendation

Consolidate to a single `config/` directory per documented architecture:

1. Move all files from `configs/` to `config/`
2. Update all path references in code and docs
3. Remove empty `configs/` directory
4. Update .gitignore if needed

## Files Affected

- All training configs in `configs/*.yaml`
- Documentation references in:
  - KMIDI_STRUCTURE_PLAN.md
  - BUILD_VARIANTS.md
  - HOW_TO_DEV_OP_101.md
- Potential code references in:
  - scripts/train.py
  - pyproject.toml
  - Test files

## Implementation Steps

```bash
# 1. Move all files
mv configs/* config/

# 2. Update path references
grep -r "configs/" . --exclude-dir=.git

# 3. Remove old directory
rmdir configs/

# 4. Test that all configs still load
pytest tests/ -v
```

## Priority

🔴 **Critical** - Fix before production release

## Acceptance Criteria

- [ ] All config files moved to `config/` directory
- [ ] All code references updated
- [ ] All documentation updated
- [ ] No references to `configs/` remain
- [ ] All tests pass
- [ ] README/docs reflect single config directory
