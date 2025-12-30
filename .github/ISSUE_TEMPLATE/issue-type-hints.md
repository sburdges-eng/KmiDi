---
name: Add Missing Type Hints
about: Complete type hint coverage in scripts/train.py
title: '📝 Code Quality: Add missing type hints in scripts/train.py'
labels: code-quality, type-hints, python, priority-medium
assignees: ''
---

## Problem

The new `enforce_device_constraints()` function in `scripts/train.py` is missing a type hint on the `device` parameter, which reduces type safety and IDE autocompletion support.

## Location

`scripts/train.py` line 211

## Current Code

```python
def enforce_device_constraints(config: TrainConfig, device) -> None:
    """Clamp config for resource-limited devices (e.g., MPS/CPU smoke)."""
    # Missing type hint on 'device' parameter ↑
```

## Expected

```python
def enforce_device_constraints(config: TrainConfig, device: "torch.device") -> None:
    """Clamp config for resource-limited devices (e.g., MPS/CPU smoke)."""
```

Or with proper imports at the top:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# Later in the file:
def enforce_device_constraints(config: TrainConfig, device: torch.device) -> None:
    """Clamp config for resource-limited devices (e.g., MPS/CPU smoke)."""
```

## Additional Issues

The file uses **lazy imports** (torch imported inside `train_model()` at line 555), which makes type hints tricky. Consider:

1. Use `TYPE_CHECKING` guard for type hints only
2. Keep lazy import for runtime
3. Best of both worlds: type safety + fast startup

## Related: Magic Numbers

While fixing type hints, also extract magic numbers to named constants:

```python
# Current (lines 213-214)
max_epochs = 5 if device.type == "mps" else min(config.epochs, 10)
max_batch = 8 if device.type == "mps" else 16

# Recommended
MAX_EPOCHS_MPS = 5
MAX_EPOCHS_CPU = 10
MAX_BATCH_MPS = 8
MAX_BATCH_CPU = 16

max_epochs = MAX_EPOCHS_MPS if device.type == "mps" else min(config.epochs, MAX_EPOCHS_CPU)
max_batch = MAX_BATCH_MPS if device.type == "mps" else MAX_BATCH_CPU
```

## Impact

- Reduces type safety
- No IDE autocompletion for `device` methods
- Mypy will flag this as incomplete
- Inconsistent with rest of codebase (TrainConfig has type hints)

## Priority

🟡 **Medium** - Code quality issue, not a bug

## Acceptance Criteria

- [ ] Add type hint to `device` parameter
- [ ] Use `TYPE_CHECKING` guard to avoid circular import
- [ ] Extract magic numbers to module-level constants
- [ ] Run `mypy scripts/train.py` without errors
- [ ] Verify lazy import still works (no performance regression)
- [ ] Update other functions if they have similar issues
