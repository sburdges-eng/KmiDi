# Performance Benchmarking Guide

**Date**: 2025-01-02  
**Targets**: 
- Harmony Engine: <100μs latency @ 48kHz/512 samples
- Groove Engine: <200μs latency @ 48kHz/512 samples  
- ML Models: <5ms latency per model

## Overview

Performance benchmarks ensure KmiDi components meet real-time audio processing requirements.

## Running Benchmarks

### Quick Benchmark

```bash
python scripts/benchmark_performance.py
```

### Individual Component Benchmarks

```bash
# Harmony engine only
python -c "from scripts.benchmark_performance import benchmark_harmony_engine; benchmark_harmony_engine()"

# Groove engine only
python -c "from scripts.benchmark_performance import benchmark_groove_engine; benchmark_groove_engine()"

# ML models only
python -c "from scripts.benchmark_performance import benchmark_ml_models; benchmark_ml_models()"
```

## Performance Targets

### Harmony Engine

- **Target**: <100μs mean latency
- **Sample Rate**: 48kHz
- **Buffer Size**: 512 samples
- **Measurement**: Processing time per buffer

### Groove Engine

- **Target**: <200μs mean latency
- **Sample Rate**: 48kHz
- **Buffer Size**: 512 samples
- **Measurement**: Processing time per buffer

### ML Model Inference

- **Target**: <5ms per model
- **Models**: EmotionRecognizer, HarmonyPredictor, MelodyTransformer, GroovePredictor, DynamicsEngine
- **Measurement**: End-to-end inference time

## Benchmark Results Interpretation

### Latency Percentiles

- **Mean**: Average latency across all iterations
- **Median**: 50th percentile latency
- **P95**: 95th percentile latency (95% of requests faster)
- **P99**: 99th percentile latency (99% of requests faster)
- **Max**: Maximum observed latency

### Pass/Fail Criteria

- **PASS**: Mean latency < target
- **FAIL**: Mean latency >= target
- **WARNING**: P99 latency significantly higher than mean

## Performance Optimization

### If Benchmarks Fail

1. **Profile the Code**: Identify bottlenecks
   ```bash
   python -m cProfile -o profile.stats scripts/benchmark_performance.py
   python -m pstats profile.stats
   ```

2. **Check for Allocations**: Ensure no memory allocation in hot path
3. **Optimize Algorithms**: Review algorithm complexity
4. **Use SIMD**: Consider SIMD optimizations for vector operations
5. **Cache Results**: Cache expensive computations

### Optimization Strategies

#### Harmony Engine

- Pre-compute chord progressions
- Use lookup tables for common operations
- Optimize Roman numeral parsing
- Cache scale computations

#### Groove Engine

- Pre-allocate buffers
- Use efficient timing calculations
- Optimize MIDI event processing
- Cache groove templates

#### ML Models

- Use model quantization
- Optimize tensor operations
- Batch inference when possible
- Use ONNX Runtime for faster inference

## Continuous Performance Monitoring

### CI/CD Integration

Add to CI pipeline:

```yaml
# .github/workflows/performance.yml
- name: Performance Benchmarks
  run: python scripts/benchmark_performance.py
```

### Performance Regression Detection

- Compare results against baseline
- Alert on significant regressions (>10% slowdown)
- Track performance trends over time

## See Also

- [Memory Profiling Guide](MEMORY_PROFILING.md)
- [C++ Performance](docs/CPP_TEST_STATUS.md)
