---
name: Fix YAML Indentation
about: Fix mixed tabs/spaces in build-dev-mac.yaml
title: '🐛 Config: Fix YAML indentation in build-dev-mac.yaml (tabs vs spaces)'
labels: bug, config, mac-dev, priority-critical
assignees: ''
---

## Problem

`config/build-dev-mac.yaml` contains mixed tabs and spaces for indentation. YAML spec requires consistent spacing (spaces recommended).

## Location

`config/build-dev-mac.yaml` lines 11-42

## Current Code

```yaml
paths:
	data_root: ${DATA_ROOT:-data}  # ← Tab character
	models_root: ${MODELS_ROOT:-models}
	output_root: ${OUTPUT_ROOT:-output}
	checkpoints: ${MODELS_ROOT:-models}/checkpoints

performance:
	target_latency_ms: 150  # ← Tab character
	profile_enabled: true
	memory_monitor: true
	max_batch_size: 16
```

## Expected

```yaml
paths:
  data_root: ${DATA_ROOT:-data}  # ← 2 spaces
  models_root: ${MODELS_ROOT:-models}
  output_root: ${OUTPUT_ROOT:-output}
  checkpoints: ${MODELS_ROOT:-models}/checkpoints

performance:
  target_latency_ms: 150  # ← 2 spaces
  profile_enabled: true
  memory_monitor: true
  max_batch_size: 16
```

## Impact

- May cause parsing issues in some YAML parsers
- Violates YAML best practices (PEP 8 equivalent for YAML)
- Inconsistent with other config files in the repo
- CI/CD tools may fail on strict YAML validation

## Fix

Convert all indentation to 2 spaces throughout the file.

### Manual Fix
```bash
# Replace tabs with 2 spaces
sed -i 's/\t/  /g' config/build-dev-mac.yaml
```

### Or use yamllint
```bash
pip install yamllint
yamllint config/build-dev-mac.yaml
```

## Priority

🔴 **Critical** - YAML syntax issue that may cause parsing failures

## Acceptance Criteria

- [ ] All tabs replaced with spaces (2-space indentation)
- [ ] File validates with `yamllint`
- [ ] No mixed indentation warnings
- [ ] Config still loads correctly in Python (`yaml.safe_load()`)
- [ ] Consistent with other YAML files in repo
