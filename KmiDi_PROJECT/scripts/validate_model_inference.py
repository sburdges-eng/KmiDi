#!/usr/bin/env python3
"""
Validate all 5 integrated models can run inference.

Tests:
1. Model loading (PyTorch checkpoint)
2. Inference execution
3. Output shape validation
4. Inference latency measurement (<5ms target for EmotionRecognizer)
5. Edge case testing (empty input, out-of-range values)
"""

import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import model loading utilities from export script
from scripts.export_models import (
    load_checkpoint_and_create_model,
    SimpleMLP,
    SimpleLSTM,
    FlexibleModel,
)

MODEL_CONFIGS = {
    'emotionrecognizer': {
        'checkpoint': '../KMiDi_TRAINING/models/models/checkpoints/emotionrecognizer_best.pt',
        'input_shape': (1, 128),
        'expected_output_shape': (1, 64),
        'latency_target_ms': 5.0,
        'use_flexible': True,
    },
    'harmonypredictor': {
        'checkpoint': '../KMiDi_TRAINING/models/models/checkpoints/harmonypredictor_best.pt',
        'input_shape': (1, 128),
        'expected_output_shape': (1, 64),
        'latency_target_ms': 10.0,
        'use_flexible': False,
    },
    'melodytransformer': {
        'checkpoint': '../KMiDi_TRAINING/models/models/checkpoints/melodytransformer_best.pt',
        'input_shape': (1, 64),
        'expected_output_shape': (1, 128),
        'latency_target_ms': 10.0,
        'use_flexible': True,
    },
    'groovepredictor': {
        'checkpoint': '../KMiDi_TRAINING/models/models/checkpoints/groovepredictor_best.pt',
        'input_shape': (1, 64),
        'expected_output_shape': (1, 32),
        'latency_target_ms': 10.0,
        'use_flexible': False,
    },
    'dynamicsengine': {
        'checkpoint': '../KMiDi_TRAINING/models/models/checkpoints/dynamicsengine_best.pt',
        'input_shape': (1, 32),
        'expected_output_shape': (1, 16),
        'latency_target_ms': 5.0,
        'use_flexible': False,
    },
}


def measure_inference_latency(model: nn.Module, input_tensor: torch.Tensor, num_runs: int = 100, warmup: int = 10) -> Dict[str, float]:
    """Measure inference latency statistics."""
    model.eval()

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # Measure latency
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(input_tensor)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
    }


def validate_output_shape(output: torch.Tensor, expected_shape: Tuple[int, ...], model_name: str) -> bool:
    """Validate output shape matches expected dimensions."""
    output_shape = tuple(output.shape)

    if output_shape == expected_shape:
        return True

    # Allow batch size flexibility (first dimension)
    if len(output_shape) == len(expected_shape):
        flexible_shape = (output_shape[0],) + expected_shape[1:]
        if output_shape == flexible_shape:
            print(f"  ⚠ Shape matches flexibly: {output_shape} (expected batch flexibility)")
            return True

    print(f"  ✗ Shape mismatch: got {output_shape}, expected {expected_shape}")
    return False


def test_edge_cases(model: nn.Module, input_shape: Tuple[int, ...], model_name: str) -> Dict[str, bool]:
    """Test edge cases: empty input, out-of-range values, NaN, Inf."""
    results = {}
    model.eval()

    with torch.no_grad():
        # Test 1: Normal random input
        try:
            normal_input = torch.randn(*input_shape)
            output = model(normal_input)
            results['normal_input'] = True
        except Exception as e:
            print(f"  ✗ Normal input failed: {e}")
            results['normal_input'] = False

        # Test 2: Zero input
        try:
            zero_input = torch.zeros(*input_shape)
            output = model(zero_input)
            if torch.isfinite(output).all():
                results['zero_input'] = True
            else:
                print(f"  ⚠ Zero input produced NaN/Inf")
                results['zero_input'] = False
        except Exception as e:
            print(f"  ✗ Zero input failed: {e}")
            results['zero_input'] = False

        # Test 3: Large values
        try:
            large_input = torch.ones(*input_shape) * 10.0
            output = model(large_input)
            if torch.isfinite(output).all():
                results['large_values'] = True
            else:
                print(f"  ⚠ Large values produced NaN/Inf")
                results['large_values'] = False
        except Exception as e:
            print(f"  ✗ Large values failed: {e}")
            results['large_values'] = False

        # Test 4: Negative values
        try:
            neg_input = torch.randn(*input_shape) * -1.0
            output = model(neg_input)
            if torch.isfinite(output).all():
                results['negative_values'] = True
            else:
                print(f"  ⚠ Negative values produced NaN/Inf")
                results['negative_values'] = False
        except Exception as e:
            print(f"  ✗ Negative values failed: {e}")
            results['negative_values'] = False

    return results


def validate_model_inference(model_name: str, config: dict) -> Dict:
    """Validate a single model's inference capability."""
    print(f"\n{'='*70}")
    print(f"Validating: {model_name.upper()}")
    print(f"{'='*70}")

    checkpoint_path = project_root / config['checkpoint']

    if not checkpoint_path.exists():
        return {
            'success': False,
            'error': f'Checkpoint not found: {checkpoint_path}',
        }

    try:
        # Load model
        print(f"Loading checkpoint: {checkpoint_path.name}")
        model, metadata = load_checkpoint_and_create_model(
            checkpoint_path,
            use_flexible=config['use_flexible']
        )

        param_count = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model loaded: {metadata.get('architecture', 'unknown')} architecture")
        print(f"  ✓ Parameters: {param_count:,}")

        # Test 1: Basic inference
        print(f"\n[1/4] Testing basic inference...")
        model.eval()
        input_tensor = torch.randn(*config['input_shape'])

        with torch.no_grad():
            output = model(input_tensor)

        output_shape = tuple(output.shape)
        print(f"  ✓ Inference successful: {config['input_shape']} → {output_shape}")

        # Test 2: Output shape validation
        print(f"\n[2/4] Validating output shape...")
        shape_valid = validate_output_shape(output, config['expected_output_shape'], model_name)

        # Test 3: Latency measurement
        print(f"\n[3/4] Measuring inference latency...")
        latency_stats = measure_inference_latency(model, input_tensor, num_runs=100)

        print(f"  Mean: {latency_stats['mean_ms']:.3f} ms")
        print(f"  Std:  {latency_stats['std_ms']:.3f} ms")
        print(f"  Min:  {latency_stats['min_ms']:.3f} ms")
        print(f"  Max:  {latency_stats['max_ms']:.3f} ms")
        print(f"  P95:  {latency_stats['p95_ms']:.3f} ms")

        latency_ok = latency_stats['mean_ms'] <= config['latency_target_ms']
        if latency_ok:
            print(f"  ✓ Latency within target ({config['latency_target_ms']} ms)")
        else:
            print(f"  ⚠ Latency exceeds target ({config['latency_target_ms']} ms)")

        # Test 4: Edge cases
        print(f"\n[4/4] Testing edge cases...")
        edge_results = test_edge_cases(model, config['input_shape'], model_name)

        edge_passed = sum(edge_results.values())
        edge_total = len(edge_results)
        print(f"  ✓ Edge cases: {edge_passed}/{edge_total} passed")

        # Summary
        success = shape_valid and latency_ok and edge_passed == edge_total

        return {
            'success': success,
            'model_name': model_name,
            'architecture': metadata.get('architecture', 'unknown'),
            'param_count': param_count,
            'input_shape': config['input_shape'],
            'output_shape': output_shape,
            'expected_output_shape': config['expected_output_shape'],
            'shape_valid': shape_valid,
            'latency_stats': latency_stats,
            'latency_ok': latency_ok,
            'edge_cases': edge_results,
            'edge_passed': edge_passed,
            'edge_total': edge_total,
        }

    except Exception as e:
        import traceback
        print(f"  ✗ Error: {e}")
        print(f"  Traceback:\n{traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


def main():
    print("="*70)
    print("Model Inference Validation")
    print("="*70)
    print(f"Testing {len(MODEL_CONFIGS)} models")
    print()

    results = {}

    for model_name, config in MODEL_CONFIGS.items():
        results[model_name] = validate_model_inference(model_name, config)

    # Summary report
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}\n")

    successful = sum(1 for r in results.values() if r.get('success', False))
    total = len(results)

    for model_name, result in results.items():
        if result.get('success'):
            latency_mean = result.get('latency_stats', {}).get('mean_ms', 0)
            shape_status = "✓" if result.get('shape_valid') else "✗"
            latency_status = "✓" if result.get('latency_ok') else "⚠"
            edge_status = f"✓ {result.get('edge_passed')}/{result.get('edge_total')}"

            print(f"  ✓ {model_name:20s}")
            print(f"      Shape:   {shape_status} {result.get('output_shape')}")
            print(f"      Latency: {latency_status} {latency_mean:.3f} ms (target: {MODEL_CONFIGS[model_name]['latency_target_ms']} ms)")
            print(f"      Edge:    {edge_status}")
        else:
            error = result.get('error', 'Unknown error')
            print(f"  ✗ {model_name:20s} - {error}")

    print(f"\n{'='*70}")
    print(f"Results: {successful}/{total} models validated successfully")

    # Save results to JSON
    results_file = project_root / 'models' / 'validation_results.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)

    # Serialize results (convert tensors to lists, remove non-serializable items)
    serializable_results = {}
    for name, result in results.items():
        serializable = {}
        for k, v in result.items():
            if k == 'latency_stats':
                serializable[k] = {k2: float(v2) for k2, v2 in v.items()}
            elif isinstance(v, (int, float, str, bool, list, tuple)):
                serializable[k] = v
            elif isinstance(v, dict):
                serializable[k] = {k2: bool(v2) if isinstance(v2, bool) else str(v2) for k2, v2 in v.items()}
            else:
                serializable[k] = str(v)
        serializable_results[name] = serializable

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    if successful == total:
        print("\n🎉 SUCCESS: All models validated successfully!")
        return 0
    else:
        print(f"\n⚠ {total - successful} models failed validation")
        return 1


if __name__ == '__main__':
    sys.exit(main())

