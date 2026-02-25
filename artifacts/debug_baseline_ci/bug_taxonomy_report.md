# Bug Taxonomy Report

- Generated (UTC): `2026-02-22T13:35:12Z`
- Logs directory: `artifacts/debug_baseline_ci`
- Finding count: `4`

## Command Status

| Command | Exit Code |
|---|---:|
| `cargo_check` | `101` |
| `cmake_build` | `0` |
| `cmake_config` | `0` |
| `ctest` | `0` |
| `npm_build` | `127` |
| `pytest` | `0` |

## Category Counts

| Category | Count |
|---|---:|
| Logical bugs | 0 |
| Syntax bugs | 0 |
| Runtime errors | 0 |
| Semantic bugs | 0 |
| Concurrency bugs | 0 |
| Race conditions | 0 |
| Deadlocks | 0 |
| Off-by-one errors | 0 |
| Performance bugs | 0 |
| Memory leaks | 0 |
| Null pointer errors | 0 |
| Type errors | 0 |
| Overflow errors | 0 |
| Underflow errors | 0 |
| Floating point precision errors | 0 |
| Security vulnerabilities | 0 |
| Injection flaws | 0 |
| Authentication bugs | 0 |
| Authorization bugs | 0 |
| Integration bugs | 0 |
| API contract violations | 0 |
| Data corruption bugs | 0 |
| Encoding/decoding bugs | 0 |
| State machine bugs | 0 |
| Initialization bugs | 0 |
| Configuration bugs | 4 |
| Dependency bugs | 2 |
| Build/compile errors | 0 |
| Regression bugs | 0 |
| Edge case bugs | 0 |
| Boundary condition bugs | 0 |
| Resource exhaustion bugs | 0 |
| Timeout errors | 0 |
| Exception handling bugs | 0 |
| Infinite loop bugs | 0 |
| Recursion overflow bugs | 0 |
| UI rendering bugs | 0 |
| Event handling bugs | 0 |
| Caching bugs | 0 |
| Serialization bugs | 0 |
| Deserialization bugs | 0 |

## Findings

1. `cargo_check.log:1` | Categories: Configuration bugs, Dependency bugs | Evidence: `error: no matching package named `chrono` found`
2. `cargo_check.log:4` | Categories: Configuration bugs, Dependency bugs | Evidence: `As a reminder, you're using offline mode (--frozen) which can sometimes cause surprising resolution failures, if this error is too confusing you may wish to retry without `--frozen`.`
3. `ctest.log:2` | Categories: Configuration bugs | Evidence: `No tests were found!!!`
4. `npm_build.log:5` | Categories: Configuration bugs | Evidence: `sh: tsc: command not found`

## Determinism Notes

- Report ordering is stable by `(log file, line number, message)`.
- Classification is rule-based and does not depend on external services.
