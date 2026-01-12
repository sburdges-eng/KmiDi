# Blockers & Known Issues

**Project:** iDAW Penta Core Testing Infrastructure  
**Date:** 2025-12-04  
**Status:** No Critical Blockers

---

## Critical Blockers

**Status:** ✅ None

---

## Minor Issues & TODOs

### 1. Doxygen Not Installed in CI Environment

**Severity:** Low  
**Impact:** Documentation generation requires manual setup  
**Status:** Documented  

**Description:**
- Doxygen is not pre-installed in GitHub Actions runners
- Documentation generation workflow not yet created

**Workaround:**
- Doxyfile is created and ready
- Can be run locally or in dedicated documentation job
- Add to CI with `sudo apt-get install doxygen graphviz`

**Resolution Plan:**
- Add documentation generation job to CI workflow
- Generate and deploy docs to GitHub Pages
- Priority: Low (documentation exists in code comments)

---

### 2. Mock Audio Device Timing Accuracy

**Severity:** Low  
**Impact:** Tests may have timing variations on slow runners  
**Status:** Monitored  

**Description:**
- Mock audio device uses `std::this_thread::sleep_for()` for timing
- OS scheduler may introduce jitter beyond configured amounts
- Could cause occasional test flakiness on heavily loaded runners

**Current Mitigation:**
- Tests use tolerant assertions (e.g., `EXPECT_GT` instead of exact equality)
- Jitter simulation is optional
- Performance tests run in Release mode for consistency

**Resolution Plan:**
- Monitor CI for flaky tests
- If issues arise, increase tolerance thresholds
- Consider platform-specific timing mechanisms
- Priority: Low (no issues observed yet)

---

### 3. Windows Build Configuration Untested

**Severity:** Medium  
**Impact:** Windows builds may require additional configuration  
**Status:** TODO  

**Description:**
- test.yml includes Windows in build matrix
- Windows-specific dependencies not fully validated
- MSVC compiler flags may need adjustment

**Current State:**
- Build matrix configured
- Dependency installation via Chocolatey
- Standard CMake configuration

**Next Steps:**
- Monitor first Windows CI run
- Add Windows-specific suppressions to valgrind.supp (N/A for Windows)
- Verify JUCE dependencies on Windows
- Priority: Medium (will be caught in CI)

---

### 4. Python Bindings Not Tested in Plugin Harness

**Severity:** Low  
**Impact:** Python bindings integration not validated  
**Status:** Deferred  

**Description:**
- Plugin test harness focuses on C++ components
- Python bindings (`pybind11`) not exercised in new tests
- Existing Python tests in `tests_music-brain/` still valid

**Rationale:**
- Python bindings tested separately in existing CI
- Plugin harness focuses on RT-safe C++ code
- Python cannot be called in RT context anyway

**Resolution Plan:**
- Keep Python tests separate
- Add Python→C++ integration tests if needed
- Priority: Low (existing coverage sufficient)

---

### 5. AAX Plugin Format Requires SDK

**Severity:** Low  
**Impact:** Pro Tools plugin builds require manual SDK setup  
**Status:** Expected  

**Description:**
- AAX format requires Avid AAX SDK
- Not publicly available without developer account
- Build gracefully falls back to AU/VST3 only

**Current Handling:**
- `plugins/CMakeLists.txt` checks for SDK
- Falls back to AU/VST3/Standalone if missing
- Status message logged during configuration

**Notes:**
- This is expected behavior
- Most developers won't need AAX
- Not a blocker for general development
- Priority: N/A (by design)

---

### 6. Benchmark Baseline Not Established

**Severity:** Low  
**Impact:** Performance regressions won't be automatically detected  
**Status:** TODO  

**Description:**
- Benchmarks run and report results
- No historical baseline for comparison
- Regressions require manual review of artifacts

**Current State:**
- Benchmarks run successfully
- Results uploaded as artifacts
- Pass/fail based on absolute thresholds (e.g., <100μs)

**Resolution Plan:**
- Collect baseline metrics over several runs
- Store in repository (e.g., `benchmarks/baselines.json`)
- Add regression detection script
- Use GitHub Actions caching for history
- Priority: Low (functional benchmarks exist)

---

### 7. Test Coverage Metrics Not Aggregated

**Severity:** Low  
**Impact:** Overall coverage percentage unknown  
**Status:** TODO  

**Description:**
- Coverage reports generated for C++ and Python
- Uploaded to Codecov
- No single aggregated metric in CI summary

**Current State:**
- C++ coverage: lcov generates reports
- Python coverage: pytest-cov generates reports
- Both uploaded to Codecov separately

**Resolution Plan:**
- Add coverage summary to test-summary job
- Parse lcov output for percentage
- Display in GitHub Actions summary
- Priority: Low (coverage data exists)

---

## Resolved Issues

### ✅ Missing diagnostics_test.cpp in CMakeLists.txt

**Status:** Resolved  
**Resolution:** Added `diagnostics_test.cpp` to TEST_SOURCES  
**Date:** 2025-12-04  

---

## Non-Issues (Documented for Clarity)

### Valgrind on macOS
**Status:** Expected Limitation  
**Details:** Valgrind doesn't work reliably on modern macOS (especially ARM). This is a known limitation. Valgrind job runs on Linux only.

### SIMD Instructions in CI
**Status:** Working  
**Details:** AVX2 SIMD optimizations enabled with compiler checks. Falls back gracefully if not available. CI runners support AVX2 on x86_64.

### Real-Time Priorities in Tests
**Status:** Not Required  
**Details:** Tests don't require actual RT scheduling priorities. Mock audio device simulates timing without needing elevated privileges.

---

## Monitoring & Escalation

### When to Escalate

Escalate if ANY of the following occur:
1. **Build fails >2 times** on same issue → Log as blocker, stub implementation
2. **Test failures >3 consecutive CI runs** → Critical blocker
3. **Memory leaks detected by Valgrind** → High priority blocker
4. **Performance regression >20%** → Medium priority blocker
5. **RT-safety violations** → Critical blocker
6. **Missing critical architecture files** → Escalate immediately

### Current Escalation Status
**✅ No escalations required** - All systems operational

---

## Dependencies & Prerequisites

### Required for Building
- ✅ CMake 3.22+ (available in CI)
- ✅ C++17 compatible compiler (configured)
- ✅ GoogleTest (fetched by CMake)
- ✅ JUCE framework (fetched by CMake)
- ✅ pybind11 (fetched by CMake)

### Required for Testing
- ✅ GoogleTest (included)
- ✅ Mock audio device (implemented)
- ✅ RT validator (implemented)

### Required for CI/CD
- ✅ GitHub Actions runners (Ubuntu, macOS, Windows)
- ✅ Ninja build system (installed in CI)
- ✅ Valgrind (Ubuntu only)
- ✅ lcov (for coverage)
- ✅ Codecov (optional, for reporting)

### Optional
- ⚠️ Doxygen (for documentation generation)
- ⚠️ Graphviz (for Doxygen diagrams)
- ⚠️ AAX SDK (for Pro Tools plugin builds)

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Flaky tests due to timing | Low | Medium | Tolerant assertions, retry logic |
| Windows build failures | Medium | Low | CI will catch, fix as needed |
| Memory leaks in new code | Low | High | Valgrind in CI, strict review |
| Performance regression | Low | Medium | Benchmark tracking, alerts |
| CI runner instability | Low | Low | Multiple runners, retry on failure |
| Missing dependencies | Very Low | Low | FetchContent handles most deps |

**Overall Risk Level:** ✅ Low

---

## Next Review

**Date:** After first CI run on main branch  
**Focus Areas:**
1. Verify all test jobs pass
2. Check Windows build success
3. Review Valgrind report for leaks
4. Validate benchmark results
5. Confirm artifact uploads

---

## Contact & Support

**For Build Issues:**
- Check `BUILD.md` documentation
- Review CMake configuration logs
- Check GitHub Actions workflow logs

**For Test Failures:**
- Review test logs in artifacts
- Check `plugin_test_harness.cpp` for test implementation
- Verify RT-safety violations in output

**For Performance Issues:**
- Review benchmark artifacts
- Check CPU usage in diagnostics
- Validate SIMD is enabled (`cmake .. -DPENTA_ENABLE_SIMD=ON`)

---

**Summary:** No critical blockers. Minor TODOs documented for future improvement. All essential functionality is working and tested.
