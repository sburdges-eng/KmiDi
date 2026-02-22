## Summary

<!-- Briefly describe what this PR does and why. -->

## Changes

<!-- List the key changes made in this PR. -->

- 

## Review Checklist

### Author

- [ ] Code follows existing project conventions (see `docs/NAMING_CONVENTIONS.md`)
- [ ] Tests added or updated for changed behaviour
- [ ] No new lint or type-check warnings introduced
- [ ] Documentation updated where applicable
- [ ] No secrets, credentials, or absolute paths committed
- [ ] RT-safety rules preserved (no heap allocations on audio thread)

### Reviewer

- [ ] Logic is correct and handles edge cases
- [ ] No security or input-validation regressions
- [ ] CI checks pass (Python tests, C++ build, lint, formatting)
- [ ] Changes are minimal and focused on the stated goal

## Testing

<!-- Describe how you verified the changes (e.g., ran tests, manual steps). -->

```
pytest tests/unit/ -v
```

## Related Issues

<!-- Link any related issues, e.g., Fixes #123. -->
