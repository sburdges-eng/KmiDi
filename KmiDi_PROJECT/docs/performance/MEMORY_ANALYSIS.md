# Memory Analysis and Profiling for KmiDi Project

## Overview

This document describes the memory profiling efforts for the KmiDi project, focusing on identifying and fixing memory leaks and optimizing memory usage, especially within real-time audio components.

## Existing Memory Tests

The following C++ test files are dedicated to memory-related checks:

*   `KMiDi_PROJECT/tests/cpp/test_memory.cpp`
*   `KMiDi_PROJECT/tests/cpp/rt_memory_test.cpp`
*   `KMiDi_PROJECT/tests/penta_core/rt_memory_test.cpp`

These tests are designed to:
*   Verify correct allocation and deallocation patterns.
*   Ensure lock-free data structures (like `RTMessageQueue`) behave correctly with memory.
*   Validate real-time safety rules (e.g., no memory allocation in audio callbacks).

## Profiling Approach & Limitations

Direct interactive memory profiling using tools like macOS Instruments or Linux Valgrind is not feasible within the current automated environment. Therefore, our strategy relies on:

1.  **Unit Tests**: The existing C++ memory tests act as a first line of defense against obvious leaks or incorrect memory usage patterns.
2.  **Continuous Integration (CI)**: The project's CI/CD workflows (`.github/workflows/ci.yml`) are configured to run memory testing (e.g., using Valgrind on Linux builds) to catch memory issues in a more controlled environment. This is the primary mechanism for comprehensive memory leak detection.
3.  **Code Reviews**: Manual code reviews focus on adherence to RT-safety rules and best practices for memory management (e.g., use of `std::pmr` containers where applicable).

## Memory Targets

*   **No Memory Leaks**: All dynamically allocated memory should be correctly deallocated.
*   **Minimal Allocations in RT-Threads**: Real-time audio threads should avoid dynamic memory allocations to prevent glitches.
*   **Reasonable Memory Footprint**: Overall application memory usage should be optimized for efficient resource utilization.

## Future Work

*   Integrate memory profiling tools directly into the local development workflow if automation capabilities improve.
*   Expand memory test coverage for more complex scenarios.
*   Implement automatic reporting of memory usage trends over time in CI.

## Apple Silicon low-memory tips (Metal/MPS)

*   Prefer Metal (MPS/MLX) backends over CPU; set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8` to cap VRAM use and avoid macOS swap pressure.
*   Limit BLAS/OpenMP threads when running locally: `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_MAX_THREADS=4`.
*   Keep uvicorn to a single worker and low concurrency (`--workers 1 --limit-concurrency 4`) to reduce peak RSS during API load spikes.
