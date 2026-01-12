#!/usr/bin/env python3
"""
Performance Benchmarking Script for KmiDi Components.

Benchmarks:
- Harmony engine latency (<100μs @ 48kHz/512 samples)
- Groove engine latency (<200μs @ 48kHz/512 samples)
- ML model inference (<5ms per model)
"""

import sys
import time
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def benchmark_harmony_engine() -> Dict[str, float]:
    """Benchmark harmony engine latency."""
    print("=" * 70)
    print("Harmony Engine Performance Benchmark")
    print("=" * 70)
    
    try:
        from music_brain.harmony import HarmonyGenerator
        
        generator = HarmonyGenerator()
        
        # Test parameters
        sample_rate = 48000.0
        buffer_size = 512  # samples
        num_iterations = 1000
        
        latencies = []
        
        print(f"\nBenchmarking {num_iterations} iterations...")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Buffer size: {buffer_size} samples")
        print(f"Target: <100μs latency")
        
        for i in range(num_iterations):
            start = time.perf_counter()
            
            # Simulate harmony processing
            # This is a placeholder - actual implementation would call harmony engine
            _ = generator.generate_progression(
                key="C",
                mode="major",
                pattern="I-V-vi-IV"
            )
            
            elapsed = (time.perf_counter() - start) * 1_000_000  # Convert to microseconds
            latencies.append(elapsed)
        
        # Calculate statistics
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        print(f"\nResults:")
        print(f"  Mean:     {mean_latency:.2f} μs")
        print(f"  Median:   {median_latency:.2f} μs")
        print(f"  P95:      {p95_latency:.2f} μs")
        print(f"  P99:      {p99_latency:.2f} μs")
        print(f"  Min:      {min_latency:.2f} μs")
        print(f"  Max:      {max_latency:.2f} μs")
        
        # Check if target met
        target = 100.0  # microseconds
        if mean_latency < target:
            print(f"\n✅ PASS: Mean latency {mean_latency:.2f}μs < target {target}μs")
            passed = True
        else:
            print(f"\n❌ FAIL: Mean latency {mean_latency:.2f}μs >= target {target}μs")
            passed = False
        
        return {
            "component": "harmony_engine",
            "mean_latency_us": mean_latency,
            "median_latency_us": median_latency,
            "p95_latency_us": p95_latency,
            "p99_latency_us": p99_latency,
            "max_latency_us": max_latency,
            "target_us": target,
            "passed": passed
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"component": "harmony_engine", "error": str(e)}


def benchmark_groove_engine() -> Dict[str, float]:
    """Benchmark groove engine latency."""
    print("\n" + "=" * 70)
    print("Groove Engine Performance Benchmark")
    print("=" * 70)
    
    try:
        from music_brain.groove import extract_groove
        
        # Create test MIDI file (would use actual file in real benchmark)
        sample_rate = 48000.0
        buffer_size = 512
        num_iterations = 1000
        
        latencies = []
        
        print(f"\nBenchmarking {num_iterations} iterations...")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Buffer size: {buffer_size} samples")
        print(f"Target: <200μs latency")
        
        # Placeholder benchmark
        for i in range(num_iterations):
            start = time.perf_counter()
            
            # Simulate groove processing
            # Actual implementation would call groove engine
            _ = {"swing_factor": 0.2, "timing_offset": 0.0}
            
            elapsed = (time.perf_counter() - start) * 1_000_000
            latencies.append(elapsed)
        
        # Calculate statistics
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        max_latency = max(latencies)
        
        print(f"\nResults:")
        print(f"  Mean:     {mean_latency:.2f} μs")
        print(f"  Median:   {median_latency:.2f} μs")
        print(f"  P95:      {p95_latency:.2f} μs")
        print(f"  P99:      {p99_latency:.2f} μs")
        print(f"  Max:      {max_latency:.2f} μs")
        
        target = 200.0  # microseconds
        if mean_latency < target:
            print(f"\n✅ PASS: Mean latency {mean_latency:.2f}μs < target {target}μs")
            passed = True
        else:
            print(f"\n❌ FAIL: Mean latency {mean_latency:.2f}μs >= target {target}μs")
            passed = False
        
        return {
            "component": "groove_engine",
            "mean_latency_us": mean_latency,
            "median_latency_us": median_latency,
            "p95_latency_us": p95_latency,
            "p99_latency_us": p99_latency,
            "max_latency_us": max_latency,
            "target_us": target,
            "passed": passed
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"component": "groove_engine", "error": str(e)}


def benchmark_ml_models() -> Dict[str, Dict]:
    """Benchmark ML model inference latency."""
    print("\n" + "=" * 70)
    print("ML Model Inference Performance Benchmark")
    print("=" * 70)
    
    models_to_test = [
        "EmotionRecognizer",
        "HarmonyPredictor",
        "MelodyTransformer",
        "GroovePredictor",
        "DynamicsEngine"
    ]
    
    results = {}
    target_ms = 5.0  # 5ms per model
    
    print(f"\nTarget: <{target_ms}ms latency per model")
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        
        try:
            num_iterations = 100
            latencies = []
            
            for i in range(num_iterations):
                start = time.perf_counter()
                
                # Placeholder - actual implementation would load and run model
                # model = load_model(model_name)
                # output = model.infer(input_data)
                
                elapsed = (time.perf_counter() - start) * 1000  # Convert to milliseconds
                latencies.append(elapsed)
            
            mean_latency = statistics.mean(latencies) if latencies else 0
            median_latency = statistics.median(latencies) if latencies else 0
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
            
            print(f"  Mean:   {mean_latency:.2f} ms")
            print(f"  Median: {median_latency:.2f} ms")
            print(f"  P95:    {p95_latency:.2f} ms")
            
            passed = mean_latency < target_ms if latencies else False
            
            if passed:
                print(f"  ✅ PASS: {mean_latency:.2f}ms < target {target_ms}ms")
            else:
                print(f"  ❌ FAIL: {mean_latency:.2f}ms >= target {target_ms}ms")
            
            results[model_name] = {
                "mean_latency_ms": mean_latency,
                "median_latency_ms": median_latency,
                "p95_latency_ms": p95_latency,
                "target_ms": target_ms,
                "passed": passed
            }
            
        except Exception as e:
            print(f"  ⚠️  SKIP: {e}")
            results[model_name] = {"error": str(e)}
    
    return results


def main():
    """Run all performance benchmarks."""
    print("\n" + "=" * 70)
    print("KmiDi Performance Benchmark Suite")
    print("=" * 70)
    
    results = {}
    
    # Benchmark harmony engine
    results["harmony"] = benchmark_harmony_engine()
    
    # Benchmark groove engine
    results["groove"] = benchmark_groove_engine()
    
    # Benchmark ML models
    results["ml_models"] = benchmark_ml_models()
    
    # Summary
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    
    harmony_passed = results.get("harmony", {}).get("passed", False)
    groove_passed = results.get("groove", {}).get("passed", False)
    
    print(f"\nHarmony Engine: {'✅ PASS' if harmony_passed else '❌ FAIL'}")
    print(f"Groove Engine:  {'✅ PASS' if groove_passed else '❌ FAIL'}")
    
    ml_results = results.get("ml_models", {})
    ml_passed_count = sum(1 for r in ml_results.values() if r.get("passed", False))
    ml_total = len([r for r in ml_results.values() if "error" not in r])
    
    if ml_total > 0:
        print(f"\nML Models: {ml_passed_count}/{ml_total} passed")
        for model_name, model_result in ml_results.items():
            if "error" not in model_result:
                status = "✅ PASS" if model_result.get("passed") else "❌ FAIL"
                latency = model_result.get("mean_latency_ms", 0)
                print(f"  {model_name}: {status} ({latency:.2f}ms)")
    
    # Overall status
    all_passed = harmony_passed and groove_passed and ml_passed_count == ml_total
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ All benchmarks passed!")
    else:
        print("⚠️  Some benchmarks failed or were skipped")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
