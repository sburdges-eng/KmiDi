# Phase 3 Git Batching

## Commit plan
- Commit 1: exact path/hash safe imports (no-op verification or restores only)
- Commit 2: deterministic duplicate/move mappings (no-op verification or restores only)
- Commit 3: Phase 2 approved candidate/conflict resolutions

## Gate
- remote gate passed: false
- if gate is false, file content apply remains blocked and only reports are updated.
