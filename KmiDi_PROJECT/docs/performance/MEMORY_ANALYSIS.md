# KmiDi Memory Analysis Guide

> Memory profiling and optimization for real-time audio processing

## Overview

KmiDi follows a dual-heap memory architecture to ensure real-time safety:

```
┌─────────────────────────────────────────────────────────────┐
│                     Memory Architecture                      │
├──────────────────────────┬──────────────────────────────────┤
│      Side A (Audio)      │        Side B (Processing)       │
│    RT-Safe / No Alloc    │       Dynamic / May Block        │
├──────────────────────────┼──────────────────────────────────┤
│ • Pre-allocated buffers  │ • AI/ML inference               │
│ • Lock-free queues       │ • MIDI file I/O                 │
│ • Fixed-size pools       │ • UI updates                    │
│ • NO malloc() in audio   │ • Standard heap allocation      │
└──────────────────────────┴──────────────────────────────────┘
```

## Memory Targets

| Component | Max Memory | Target |
|-----------|------------|--------|
| Audio buffers | 64 MB | < 32 MB typical |
| ML models (loaded) | 512 MB | < 256 MB typical |
| Working memory | 128 MB | < 64 MB typical |
| **Total footprint** | **~700 MB** | **< 400 MB typical** |

## Profiling Tools

### Python Components

#### Using `memory_profiler`

```bash
# Install
pip install memory_profiler

# Profile function
python -m memory_profiler scripts/benchmark_performance.py
```

#### Using `tracemalloc`

```python
import tracemalloc

tracemalloc.start()

# ... run code ...

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

#### Memory benchmark script

```python
#!/usr/bin/env python3
"""Memory profiling for KmiDi components."""

import tracemalloc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "source" / "python"))

def profile_component(name, func):
    """Profile memory usage of a component."""
    tracemalloc.start()

    # Run function
    result = func()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"{name}:")
    print(f"  Current: {current / 1024 / 1024:.2f} MB")
    print(f"  Peak:    {peak / 1024 / 1024:.2f} MB")

    return result

def profile_therapy_session():
    from music_brain.structure.comprehensive_engine import TherapySession

    def run():
        session = TherapySession()
        session.process_core_input("feeling grief and loss")
        session.set_scales(7, 0.5)
        return session.generate_plan()

    return profile_component("TherapySession", run)

def profile_groove_engine():
    from music_brain.groove_engine import apply_groove

    events = [
        {"start_tick": i * 240, "velocity": 100, "pitch": 36}
        for i in range(1024)  # Large event list
    ]

    def run():
        return apply_groove(events, complexity=0.5, vulnerability=0.5)

    return profile_component("GrooveEngine (1024 events)", run)

if __name__ == "__main__":
    print("=" * 60)
    print("KmiDi Memory Profiling")
    print("=" * 60)

    profile_therapy_session()
    profile_groove_engine()
```

### C++ Components

#### Using Valgrind (Linux)

```bash
# Memory leak check
valgrind --leak-check=full ./build/benchmarks/bench_harmony

# Detailed heap analysis
valgrind --tool=massif ./build/benchmarks/bench_harmony
ms_print massif.out.*
```

#### Using Instruments (macOS)

```bash
# Open Instruments
open -a Instruments

# Or from command line
instruments -t "Allocations" ./build/benchmarks/bench_harmony
```

#### Using AddressSanitizer

Build with sanitizers enabled:

```cmake
# CMakeLists.txt
if(ENABLE_ASAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
endif()
```

```bash
cmake -DENABLE_ASAN=ON ..
cmake --build .
./benchmarks/bench_harmony
```

## Memory Optimization Guidelines

### RT-Safe Audio Processing

1. **No Dynamic Allocation**
   ```cpp
   // ❌ BAD - allocates in audio thread
   void processBlock(float* buffer, int numSamples) {
       std::vector<float> temp(numSamples);  // ALLOCATION!
   }

   // ✅ GOOD - use pre-allocated buffer
   class Processor {
       std::array<float, 4096> scratchBuffer_;

       void processBlock(float* buffer, int numSamples) {
           // Use scratchBuffer_ instead
       }
   };
   ```

2. **Lock-Free Communication**
   ```cpp
   // Use moodycamel::ReaderWriterQueue for audio<->UI communication
   #include "readerwriterqueue.h"

   moodycamel::ReaderWriterQueue<Message> audioToUI;
   ```

3. **Fixed-Size Memory Pools**
   ```cpp
   #include <memory_resource>

   // Pre-allocate pool
   std::array<std::byte, 1024 * 1024> buffer;  // 1MB
   std::pmr::monotonic_buffer_resource pool{buffer.data(), buffer.size()};

   // Allocate from pool
   std::pmr::vector<float> data{&pool};
   ```

### Python Components

1. **Reuse Objects**
   ```python
   # ❌ BAD - creates new session each time
   def process(text):
       session = TherapySession()
       return session.process_core_input(text)

   # ✅ GOOD - reuse session
   class Processor:
       def __init__(self):
           self._session = TherapySession()

       def process(self, text):
           self._session.state.affect_result = None
           return self._session.process_core_input(text)
   ```

2. **Use Generators for Large Data**
   ```python
   # ❌ BAD - loads all into memory
   def process_notes(midi_file):
       notes = list(read_all_notes(midi_file))
       return [process(n) for n in notes]

   # ✅ GOOD - stream processing
   def process_notes(midi_file):
       for note in read_notes_streaming(midi_file):
           yield process(note)
   ```

3. **NumPy Array Pre-allocation**
   ```python
   import numpy as np

   # ❌ BAD - reallocates on each call
   def process_buffer(data):
       result = np.zeros_like(data)
       # ...

   # ✅ GOOD - reuse buffer
   class Processor:
       def __init__(self, max_size=4096):
           self._buffer = np.zeros(max_size, dtype=np.float32)

       def process_buffer(self, data):
           self._buffer[:len(data)] = 0
           # ...
   ```

## Memory Leak Detection Checklist

- [ ] Run Valgrind/Instruments on all C++ components
- [ ] Check for circular references in Python (use `gc.get_referrers()`)
- [ ] Verify all file handles are closed (`with` statements)
- [ ] Check for leaked threads/processes
- [ ] Monitor memory over extended operation (24+ hour test)
- [ ] Test with large files (100MB+ MIDI/audio)
- [ ] Profile under memory pressure (limit container memory)

## Typical Memory Profiles

### Startup Sequence

```
Phase          | Memory (MB) | Notes
---------------|-------------|---------------------------
Import modules |     50      | Python + dependencies
Load ML models |    200      | CoreML/ONNX models
Init audio     |     30      | Buffers, queues
Ready state    |    280      | Total footprint at idle
```

### During Processing

```
Operation              | Additional Memory | Duration
-----------------------|-------------------|----------
Therapy session        |      5 MB         | Transient
Groove processing      |      2 MB         | Transient
MIDI rendering         |     10 MB         | Peak during export
ML inference           |     20 MB         | Peak during inference
```

## Monitoring in Production

### Docker Memory Limits

```yaml
# docker-compose.yml
services:
  kmidi-api:
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
```

### Prometheus Metrics

The API exposes memory metrics at `/metrics`:

```
kmidi_api_system_memory_percent 45.2
kmidi_api_system_memory_available_bytes 4294967296
```

### Alert Rules

```yaml
# prometheus-alerts.yml
- alert: KmiDiHighMemory
  expr: kmidi_api_system_memory_percent > 85
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "KmiDi using >85% memory"
```

## Troubleshooting

### Common Issues

1. **Memory grows continuously**
   - Check for event listener leaks
   - Look for unclosed file handles
   - Profile with tracemalloc

2. **High peak memory**
   - Check ML model loading
   - Review buffer sizes
   - Consider lazy loading

3. **OOM in container**
   - Increase memory limit
   - Enable swap
   - Profile and optimize

### Debug Commands

```bash
# Python memory snapshot
python -c "
import tracemalloc
tracemalloc.start()
# ... import your module ...
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics('lineno')[:20]:
    print(stat)
"

# Check process memory (macOS)
ps -o rss,vsz,pid,command | grep python

# Check process memory (Linux)
pmap -x $(pgrep -f kmidi)
```

---

*Last Updated: 2026-01-11*
