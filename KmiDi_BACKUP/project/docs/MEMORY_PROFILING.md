# Memory Profiling Guide

**Date**: 2025-01-02

## Overview

Memory profiling helps identify memory leaks, excessive allocations, and memory usage patterns in KmiDi components.

## Tools

### Python

#### memory_profiler

```bash
pip install memory-profiler

# Profile a script
python -m memory_profiler script.py

# Line-by-line profiling
@profile
def my_function():
    # Code to profile
    pass
```

#### tracemalloc

Built-in Python memory profiler:

```python
import tracemalloc

tracemalloc.start()
# ... code to profile ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

### C++

#### Valgrind (Linux)

```bash
# Install Valgrind
sudo apt-get install valgrind

# Run with memory checking
valgrind --leak-check=full --show-leak-kinds=all ./penta_tests

# Generate report
valgrind --leak-check=full --log-file=valgrind.log ./penta_tests
```

#### Instruments (macOS)

```bash
# Profile with Instruments
instruments -t "Leaks" ./penta_tests
instruments -t "Allocations" ./penta_tests
```

#### AddressSanitizer

```bash
# Compile with AddressSanitizer
cmake -DCMAKE_CXX_FLAGS="-fsanitize=address -g" ..
make

# Run tests
./penta_tests
```

## Profiling Scripts

### Python Memory Profiling

```bash
# Profile harmony engine
python -m memory_profiler -m music_brain.harmony

# Profile API endpoint
python -m memory_profiler api/main.py
```

### C++ Memory Profiling

```bash
# Build with debug symbols
cmake -DCMAKE_BUILD_TYPE=Debug ..
make

# Run Valgrind
valgrind --leak-check=full --track-origins=yes ./penta_tests
```

## Memory Leak Detection

### Python

```python
import tracemalloc
import gc

# Start tracking
tracemalloc.start()

# Run code
# ...

# Check for leaks
snapshot1 = tracemalloc.take_snapshot()
# ... more code ...
snapshot2 = tracemalloc.take_snapshot()

top_stats = snapshot2.compare_to(snapshot1, 'lineno')
for stat in top_stats[:10]:
    print(stat)
```

### C++

Valgrind automatically detects:
- Memory leaks
- Use of uninitialized memory
- Invalid memory access
- Double free errors

## Common Issues

### Python

1. **Circular References**: Use weakref or break cycles
2. **Global Variables**: Clear globals when done
3. **Caching**: Limit cache size or use LRU cache

### C++

1. **Missing delete**: Ensure all `new` have matching `delete`
2. **Exception Safety**: Use RAII and smart pointers
3. **Buffer Overruns**: Use bounds checking

## Memory Usage Targets

### Python Components

- **API Service**: <500MB baseline
- **Music Generation**: <1GB per generation
- **ML Models**: <2GB total (all models loaded)

### C++ Components

- **Penta-Core**: <50MB baseline
- **Audio Processing**: <100MB per stream
- **Real-time Safety**: No allocations in audio thread

## Reporting

### Generate Report

```bash
# Python
python scripts/profile_memory.py > memory_report.txt

# C++ (Valgrind)
valgrind --leak-check=full --log-file=valgrind_report.txt ./penta_tests
```

### Analyze Report

1. **Identify Leaks**: Look for "definitely lost" in Valgrind
2. **Check Allocation Sites**: Find where memory is allocated
3. **Review Patterns**: Look for repeated allocations
4. **Fix Issues**: Address identified problems

## Best Practices

1. **Profile Regularly**: Include in CI/CD
2. **Baseline Measurements**: Track memory usage over time
3. **Fix Leaks Immediately**: Don't accumulate technical debt
4. **Use Tools**: Leverage profilers and sanitizers
5. **Document Issues**: Keep track of known issues

## See Also

- [Performance Benchmarking](PERFORMANCE_BENCHMARKING.md)
- [C++ Testing](docs/CPP_TEST_STATUS.md)
