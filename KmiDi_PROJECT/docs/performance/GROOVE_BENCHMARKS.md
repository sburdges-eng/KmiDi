# Groove Engine Performance Benchmarks

## Overview

This document outlines the performance benchmarks for the Groove Engine, including latency measurements for `GrooveEngine::processAudio()`, `OnsetDetector::process()`, `TempoEstimator::estimateTempo()`, and `RhythmQuantizer::quantize()`/`applySwing()`.

**Note**: The benchmark tests themselves are defined in `KMiDi_PROJECT/benchmarks/groove_latency.cpp`.

## Execution Status

As of 2026-01-11, these benchmarks *exist* in the codebase but cannot be reliably executed or reported due to challenges with the Catch2 test runner integration in the current environment. Attempts to run them directly or via `ctest` were unsuccessful.

## Expected Performance Targets (from plan)

*   **GrooveEngine latency**: `<200μs @ 48kHz/512 samples`
*   **Individual components**: Expected to be well within real-time audio constraints.

## Future Work

*   Integrate Catch2 benchmarks more reliably into the CI/CD pipeline.
*   Automate reporting of benchmark results.
*   Implement performance regressions checks.
