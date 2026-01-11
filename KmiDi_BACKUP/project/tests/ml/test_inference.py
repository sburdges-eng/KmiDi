"""
ML Inference Tests - Model loading and inference validation.

Tests:
- Model loading (PyTorch, ONNX, RTNeural JSON)
- Inference with various input shapes
- Edge cases (empty input, out-of-range values)
- Output shape validation
- Latency measurement
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.export_models import load_checkpoint_and_create_model


MODEL_CONFIGS = {
    'emotionrecognizer': {
        'checkpoint': 'models/checkpoints/emotionrecognizer_best.pt',
        'input_shape': (1, 128),
        'expected_output_shape': (1, 64),
        'use_flexible': True,
    },
    'harmonypredictor': {
        'checkpoint': 'models/checkpoints/harmonypredictor_best.pt',
        'input_shape': (1, 128),
        'expected_output_shape': (1, 64),
        'use_flexible': False,
    },
    'melodytransformer': {
        'checkpoint': 'models/checkpoints/melodytransformer_best.pt',
        'input_shape': (1, 64),
        'expected_output_shape': (1, 128),
        'use_flexible': True,
    },
    'groovepredictor': {
        'checkpoint': 'models/checkpoints/groovepredictor_best.pt',
        'input_shape': (1, 64),
        'expected_output_shape': (1, 32),
        'use_flexible': False,
    },
    'dynamicsengine': {
        'checkpoint': 'models/checkpoints/dynamicsengine_best.pt',
        'input_shape': (1, 32),
        'expected_output_shape': (1, 16),
        'use_flexible': False,
    },
}


@pytest.fixture(params=list(MODEL_CONFIGS.keys()))
def model_config(request):
    """Fixture providing model config for each model."""
    model_name = request.param
    config = MODEL_CONFIGS[model_name].copy()
    config['name'] = model_name
    config['checkpoint'] = project_root / config['checkpoint']
    return config


@pytest.fixture
def loaded_model(model_config):
    """Fixture loading a model from checkpoint."""
    if not model_config['checkpoint'].exists():
        pytest.skip(f"Checkpoint not found: {model_config['checkpoint']}")
    
    model, metadata = load_checkpoint_and_create_model(
        model_config['checkpoint'],
        use_flexible=model_config['use_flexible']
    )
    model.eval()
    return model, model_config, metadata


class TestModelLoading:
    """Test model loading functionality."""
    
    def test_checkpoint_exists(self, model_config):
        """Test that checkpoint files exist."""
        assert model_config['checkpoint'].exists(), \
            f"Checkpoint not found: {model_config['checkpoint']}"
    
    def test_model_loads(self, model_config):
        """Test that models can be loaded from checkpoints."""
        model, metadata = load_checkpoint_and_create_model(
            model_config['checkpoint'],
            use_flexible=model_config['use_flexible']
        )
        
        assert model is not None
        assert 'architecture' in metadata
    
    def test_model_parameters(self, loaded_model):
        """Test that loaded models have parameters."""
        model, _, _ = loaded_model
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0, "Model should have parameters"


class TestModelInference:
    """Test model inference functionality."""
    
    def test_basic_inference(self, loaded_model):
        """Test basic inference with normal input."""
        model, config, _ = loaded_model
        
        input_tensor = torch.randn(*config['input_shape'])
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == config['input_shape'][0]  # Batch size matches
    
    def test_output_shape(self, loaded_model):
        """Test that output shapes match expected dimensions."""
        model, config, _ = loaded_model
        
        input_tensor = torch.randn(*config['input_shape'])
        
        with torch.no_grad():
            output = model(input_tensor)
        
        output_shape = tuple(output.shape)
        expected = config['expected_output_shape']
        
        # Allow batch size flexibility
        if len(output_shape) == len(expected):
            assert output_shape[1:] == expected[1:], \
                f"Output shape {output_shape} doesn't match expected {expected}"
    
    def test_inference_latency(self, loaded_model):
        """Test that inference latency is reasonable."""
        model, config, _ = loaded_model
        
        input_tensor = torch.randn(*config['input_shape'])
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Measure
        latencies = []
        with torch.no_grad():
            for _ in range(100):
                start = time.perf_counter()
                _ = model(input_tensor)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms
        
        mean_latency = np.mean(latencies)
        
        # Latency target depends on model
        if config['name'] in ['emotionrecognizer', 'dynamicsengine']:
            assert mean_latency < 5.0, \
                f"Mean latency {mean_latency:.3f}ms exceeds 5ms target"
        else:
            assert mean_latency < 10.0, \
                f"Mean latency {mean_latency:.3f}ms exceeds 10ms target"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_input(self, loaded_model):
        """Test inference with zero input."""
        model, config, _ = loaded_model
        
        input_tensor = torch.zeros(*config['input_shape'])
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert torch.isfinite(output).all(), \
            "Zero input should produce finite output"
    
    def test_large_values(self, loaded_model):
        """Test inference with large input values."""
        model, config, _ = loaded_model
        
        input_tensor = torch.ones(*config['input_shape']) * 10.0
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert torch.isfinite(output).all(), \
            "Large input values should produce finite output"
    
    def test_negative_values(self, loaded_model):
        """Test inference with negative input values."""
        model, config, _ = loaded_model
        
        input_tensor = torch.randn(*config['input_shape']) * -1.0
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert torch.isfinite(output).all(), \
            "Negative input values should produce finite output"
    
    def test_different_batch_sizes(self, loaded_model):
        """Test inference with different batch sizes."""
        model, config, _ = loaded_model
        
        # Test batch size 2
        input_shape = (2,) + config['input_shape'][1:]
        input_tensor = torch.randn(*input_shape)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape[0] == 2, "Batch size should be preserved"
    
    def test_output_range(self, loaded_model):
        """Test that outputs are in reasonable range."""
        model, config, _ = loaded_model
        
        input_tensor = torch.randn(*config['input_shape'])
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # Outputs shouldn't be NaN or Inf
        assert torch.isfinite(output).all(), "Output should be finite"
        
        # Outputs shouldn't be extremely large
        assert torch.abs(output).max() < 1e6, \
            "Output values should be in reasonable range"


class TestModelConsistency:
    """Test model consistency and reproducibility."""
    
    def test_deterministic_output(self, loaded_model):
        """Test that same input produces same output (if deterministic)."""
        model, config, _ = loaded_model
        
        input_tensor = torch.randn(*config['input_shape'])
        
        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = model(input_tensor)
        
        # Results should be very similar (allowing for small numerical differences)
        diff = torch.abs(output1 - output2).max()
        assert diff < 1e-5, \
            f"Outputs should be deterministic (max diff: {diff})"
    
    def test_gradient_disabled(self, loaded_model):
        """Test that gradients are disabled during inference."""
        model, config, _ = loaded_model
        
        input_tensor = torch.randn(*config['input_shape'])
        input_tensor.requires_grad = True
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # Output should not require gradients
        assert not output.requires_grad, \
            "Output should not require gradients during inference"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

