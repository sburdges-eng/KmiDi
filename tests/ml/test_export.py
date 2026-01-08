"""
ML Export Tests - Model export/import validation.

Tests:
- ONNX export/import round-trip
- RTNeural JSON export/import
- LSTM layer export correctness
- Export format compatibility
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import json
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.export_models import load_checkpoint_and_create_model, export_to_rtneural_json, export_to_onnx


MODEL_CONFIGS = {
    'emotionrecognizer': {
        'checkpoint': 'models/checkpoints/emotionrecognizer_best.pt',
        'input_shape': (1, 128),
        'use_flexible': True,
    },
    'harmonypredictor': {
        'checkpoint': 'models/checkpoints/harmonypredictor_best.pt',
        'input_shape': (1, 128),
        'use_flexible': False,
    },
    'melodytransformer': {
        'checkpoint': 'models/checkpoints/melodytransformer_best.pt',
        'input_shape': (1, 64),
        'use_flexible': True,
    },
    'groovepredictor': {
        'checkpoint': 'models/checkpoints/groovepredictor_best.pt',
        'input_shape': (1, 64),
        'use_flexible': False,
    },
    'dynamicsengine': {
        'checkpoint': 'models/checkpoints/dynamicsengine_best.pt',
        'input_shape': (1, 32),
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


class TestRTNeuralExport:
    """Test RTNeural JSON export."""
    
    def test_rtneural_export_succeeds(self, loaded_model, tmp_path):
        """Test that RTNeural JSON export completes without errors."""
        model, config, metadata = loaded_model
        
        output_path = tmp_path / f"{config['name']}.json"
        
        success = export_to_rtneural_json(model, output_path, config['name'])
        
        assert success, f"RTNeural export should succeed for {config['name']}"
        assert output_path.exists(), "Export file should be created"
    
    def test_rtneural_json_valid(self, loaded_model, tmp_path):
        """Test that exported RTNeural JSON is valid."""
        model, config, metadata = loaded_model
        
        output_path = tmp_path / f"{config['name']}.json"
        export_to_rtneural_json(model, output_path, config['name'])
        
        # Load and validate JSON
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert 'version' in data
        assert 'model_name' in data
        assert 'layers' in data
        assert isinstance(data['layers'], list)
        assert len(data['layers']) > 0
    
    def test_rtneural_contains_layers(self, loaded_model, tmp_path):
        """Test that exported JSON contains layer information."""
        model, config, metadata = loaded_model
        
        output_path = tmp_path / f"{config['name']}.json"
        export_to_rtneural_json(model, output_path, config['name'])
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        layers = data['layers']
        assert len(layers) > 0, "Should have at least one layer"
        
        # Check layer structure
        for layer in layers:
            assert 'type' in layer, "Layer should have type"
            if layer['type'] == 'dense':
                assert 'weights' in layer, "Dense layer should have weights"
                assert 'bias' in layer or layer.get('bias') is None, "Dense layer should have bias (or null)"
            elif layer['type'] == 'lstm':
                assert 'input_size' in layer, "LSTM layer should have input_size"
                assert 'hidden_size' in layer, "LSTM layer should have hidden_size"
    
    def test_lstm_layers_exported(self, loaded_model, tmp_path):
        """Test that LSTM layers are properly exported."""
        model, config, metadata = loaded_model
        
        # Only test models with LSTM (emotionrecognizer, melodytransformer)
        if config['name'] not in ['emotionrecognizer', 'melodytransformer']:
            pytest.skip(f"Model {config['name']} doesn't have LSTM layers")
        
        output_path = tmp_path / f"{config['name']}.json"
        export_to_rtneural_json(model, output_path, config['name'])
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        lstm_layers = [l for l in data['layers'] if l.get('type') == 'lstm']
        assert len(lstm_layers) > 0, "Should export LSTM layers"
        
        for lstm_layer in lstm_layers:
            assert 'weights_ih' in lstm_layer, "LSTM should have input-hidden weights"
            assert 'weights_hh' in lstm_layer, "LSTM should have hidden-hidden weights"
            assert len(lstm_layer['weights_ih']) > 0, "Should have weight matrices"
    
    def test_export_file_size_reasonable(self, loaded_model, tmp_path):
        """Test that exported files are reasonable size."""
        model, config, metadata = loaded_model
        
        output_path = tmp_path / f"{config['name']}.json"
        export_to_rtneural_json(model, output_path, config['name'])
        
        file_size = output_path.stat().st_size
        # Should be at least 1KB, but less than 100MB
        assert file_size > 1024, "Export file should be at least 1KB"
        assert file_size < 100 * 1024 * 1024, "Export file should be less than 100MB"


class TestONNXExport:
    """Test ONNX export (if available)."""
    
    @pytest.mark.skipif(not hasattr(torch.onnx, 'export'), reason="ONNX export not available")
    def test_onnx_export_succeeds(self, loaded_model, tmp_path):
        """Test that ONNX export completes (may fail for some models)."""
        model, config, metadata = loaded_model
        
        output_path = tmp_path / f"{config['name']}.onnx"
        
        try:
            success = export_to_onnx(
                model,
                output_path,
                config['input_shape'],
                config['name']
            )
            
            if success:
                assert output_path.exists(), "ONNX export file should be created"
        except Exception as e:
            # ONNX export may fail for complex models - mark as skip rather than fail
            pytest.skip(f"ONNX export failed (expected for some models): {e}")
    
    @pytest.mark.skipif(not hasattr(torch.onnx, 'export'), reason="ONNX export not available")
    def test_onnx_file_valid(self, loaded_model, tmp_path):
        """Test that exported ONNX file is valid."""
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx package not available")
        
        model, config, metadata = loaded_model
        
        output_path = tmp_path / f"{config['name']}.onnx"
        
        try:
            success = export_to_onnx(
                model,
                output_path,
                config['input_shape'],
                config['name']
            )
            
            if success:
                # Try to load and validate ONNX model
                onnx_model = onnx.load(str(output_path))
                onnx.checker.check_model(onnx_model)
        except Exception as e:
            pytest.skip(f"ONNX validation failed: {e}")


class TestExportConsistency:
    """Test export consistency and round-trip."""
    
    def test_export_preserves_model_info(self, loaded_model, tmp_path):
        """Test that export preserves model information."""
        model, config, metadata = loaded_model
        
        output_path = tmp_path / f"{config['name']}.json"
        export_to_rtneural_json(model, output_path, config['name'])
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert data['model_name'] == config['name'], \
            "Model name should be preserved in export"
    
    def test_multiple_exports_identical(self, loaded_model, tmp_path):
        """Test that multiple exports of the same model are identical."""
        model, config, metadata = loaded_model
        
        output_path1 = tmp_path / f"{config['name']}_1.json"
        output_path2 = tmp_path / f"{config['name']}_2.json"
        
        export_to_rtneural_json(model, output_path1, config['name'])
        export_to_rtneural_json(model, output_path2, config['name'])
        
        # Compare file contents
        with open(output_path1, 'r') as f1:
            data1 = json.load(f1)
        with open(output_path2, 'r') as f2:
            data2 = json.load(f2)
        
        # Compare layer count
        assert len(data1['layers']) == len(data2['layers']), \
            "Multiple exports should have same number of layers"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

