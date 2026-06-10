"""Unit tests for ONNX export and Unreal NNI integration."""

from pathlib import Path
from music_brain.video import (
    ONNXModelExporter,
    ONNXExportConfig,
    export_emotion_visual_model,
    UnrealNNIIntegration,
    NNIPluginConfig,
    UnrealNNIModelConfig,
    NNIBackend,
    create_nni_project_structure,
    setup_nni_plugin,
)


class TestONNXExportConfig:
    """Test ONNXExportConfig dataclass."""

    def test_config_defaults(self):
        """Test default ONNX export configuration."""
        config = ONNXExportConfig()
        assert config.model_name == "emotion_visual_mapper"
        assert config.input_dim == 128
        assert config.output_dim == 256
        assert config.opset_version == 17
        assert config.optimize is True

    def test_config_custom(self):
        """Test custom ONNX export configuration."""
        config = ONNXExportConfig(
            model_name="custom_model", input_dim=64, output_dim=128, opset_version=15, quantize=True
        )
        assert config.model_name == "custom_model"
        assert config.input_dim == 64
        assert config.output_dim == 128
        assert config.quantize is True


class TestONNXModelExporter:
    """Test ONNXModelExporter class."""

    def test_exporter_creation(self):
        """Test creating an ONNX exporter."""
        exporter = ONNXModelExporter()
        assert exporter is not None
        assert exporter.config is not None

    def test_exporter_with_config(self):
        """Test creating exporter with custom config."""
        config = ONNXExportConfig(model_name="test_model")
        exporter = ONNXModelExporter(config)
        assert exporter.config.model_name == "test_model"

    def test_export_pytorch_model_stub(self):
        """Test PyTorch export (stub)."""
        exporter = ONNXModelExporter()
        result = exporter.export_pytorch_model(
            model=None, output_path=Path("/tmp/test.onnx")  # Stub doesn't use this
        )
        # Stub returns False
        assert result is False

    def test_export_tensorflow_model_stub(self):
        """Test TensorFlow export (stub)."""
        exporter = ONNXModelExporter()
        result = exporter.export_tensorflow_model(model=None, output_path=Path("/tmp/test.onnx"))
        # Stub returns False
        assert result is False

    def test_optimize_onnx_model_stub(self):
        """Test ONNX optimization (stub)."""
        exporter = ONNXModelExporter()
        result = exporter.optimize_onnx_model(
            input_path=Path("/tmp/input.onnx"), output_path=Path("/tmp/output.onnx")
        )
        # Stub returns False
        assert result is False

    def test_quantize_onnx_model_stub(self):
        """Test ONNX quantization (stub)."""
        exporter = ONNXModelExporter()
        result = exporter.quantize_onnx_model(
            input_path=Path("/tmp/input.onnx"), output_path=Path("/tmp/output.onnx")
        )
        # Stub returns False
        assert result is False


class TestNNIPluginConfig:
    """Test NNIPluginConfig dataclass."""

    def test_nni_config_defaults(self):
        """Test default NNI plugin configuration."""
        config = NNIPluginConfig()
        assert config.plugin_enabled is True
        assert config.backend == NNIBackend.AUTO
        assert config.enable_gpu is True
        assert config.models_directory == "Models"

    def test_nni_config_custom(self):
        """Test custom NNI configuration."""
        config = NNIPluginConfig(backend=NNIBackend.CPU, num_threads=8, enable_gpu=False)
        assert config.backend == NNIBackend.CPU
        assert config.num_threads == 8
        assert config.enable_gpu is False


class TestUnrealNNIModelConfig:
    """Test UnrealNNIModelConfig dataclass."""

    def test_model_config_creation(self):
        """Test creating model configuration."""
        config = UnrealNNIModelConfig(
            model_name="EmotionMapper", model_asset_path="/Game/Models/EmotionMapper"
        )
        assert config.model_name == "EmotionMapper"
        assert config.model_asset_path == "/Game/Models/EmotionMapper"
        assert config.normalize_inputs is True

    def test_model_config_with_io_names(self):
        """Test model config with custom input/output names."""
        config = UnrealNNIModelConfig(
            model_name="test",
            model_asset_path="/Game/Models/test",
            input_names=["emotion_embedding"],
            output_names=["visual_params"],
        )
        assert "emotion_embedding" in config.input_names
        assert "visual_params" in config.output_names


class TestUnrealNNIIntegration:
    """Test UnrealNNIIntegration class."""

    def test_nni_integration_creation(self):
        """Test creating NNI integration."""
        nni = UnrealNNIIntegration()
        assert nni is not None
        assert nni.config is not None

    def test_nni_integration_with_config(self):
        """Test creating NNI integration with config."""
        config = NNIPluginConfig(backend=NNIBackend.DX12)
        nni = UnrealNNIIntegration(config=config)
        assert nni.config.backend == NNIBackend.DX12

    def test_deploy_model_no_project(self):
        """Test deploying model without project path."""
        nni = UnrealNNIIntegration()
        result = nni.deploy_model(onnx_path=Path("/tmp/model.onnx"), model_name="test")
        # Should fail without project path
        assert result is False

    def test_list_deployed_models(self):
        """Test listing deployed models."""
        nni = UnrealNNIIntegration()
        models = nni.list_deployed_models()
        assert isinstance(models, list)
        assert len(models) == 0  # Initially empty

    def test_get_model_info_not_found(self):
        """Test getting info for non-existent model."""
        nni = UnrealNNIIntegration()
        info = nni.get_model_info("nonexistent")
        assert info is None

    def test_remove_model_not_found(self):
        """Test removing non-existent model."""
        nni = UnrealNNIIntegration()
        result = nni.remove_model("nonexistent")
        assert result is False

    def test_validate_plugin_stub(self):
        """Test NNI plugin validation (stub)."""
        nni = UnrealNNIIntegration()
        result = nni.validate_plugin()
        # Stub returns False
        assert result is False

    def test_generate_blueprint_wrapper_no_model(self):
        """Test Blueprint generation for non-deployed model."""
        nni = UnrealNNIIntegration()
        result = nni.generate_blueprint_wrapper("nonexistent")
        assert result is False

    def test_generate_cpp_wrapper_no_model(self):
        """Test C++ wrapper generation for non-deployed model."""
        nni = UnrealNNIIntegration()
        result = nni.generate_cpp_wrapper("nonexistent")
        assert result is False


class TestNNIBackend:
    """Test NNIBackend enum."""

    def test_backend_types(self):
        """Test NNI backend types."""
        assert NNIBackend.CPU.value == "cpu"
        assert NNIBackend.DX12.value == "directx12"
        assert NNIBackend.AUTO.value == "auto"


class TestNNIHelperFunctions:
    """Test NNI helper functions."""

    def test_create_nni_project_structure_stub(self):
        """Test creating NNI project structure (stub)."""
        result = create_nni_project_structure(
            project_path=Path("/tmp/nonexistent"), project_name="TestProject"
        )
        # Stub returns False for non-existent path
        assert result is False

    def test_setup_nni_plugin_stub(self):
        """Test NNI plugin setup (stub)."""
        result = setup_nni_plugin(project_path=Path("/tmp/nonexistent"), enable_gpu=True)
        # Stub returns False
        assert result is False

    def test_export_emotion_visual_model_stub(self):
        """Test convenience export function (stub)."""
        result = export_emotion_visual_model(model=None, output_dir=Path("/tmp"), model_name="test")
        # Stub returns False
        assert result is False


class TestUnrealBridgeONNXIntegration:
    """Test UnrealBridge ONNX integration."""

    def test_unreal_bridge_onnx_config(self):
        """Test UnrealBridge with ONNX config."""
        from music_brain.video import UnrealBridge, UnrealConfig

        config = UnrealConfig(enable_nni=True, use_gpu_inference=True)
        bridge = UnrealBridge(config)
        assert bridge.config.enable_nni is True
        assert bridge.config.use_gpu_inference is True

    def test_load_onnx_model_not_connected(self):
        """Test loading ONNX model when not connected."""
        from music_brain.video import UnrealBridge

        bridge = UnrealBridge()
        result = bridge.load_onnx_model(model_name="test", model_asset_path="/Game/Models/test")
        # Should fail when not connected
        assert result is False

    def test_run_onnx_inference_model_not_loaded(self):
        """Test running inference on non-loaded model."""
        from music_brain.video import UnrealBridge

        bridge = UnrealBridge()
        result = bridge.run_onnx_inference(
            model_name="nonexistent", inputs={"input": [1.0, 2.0, 3.0]}
        )
        # Should return None for non-loaded model
        assert result is None

    def test_list_loaded_models_empty(self):
        """Test listing models when none loaded."""
        from music_brain.video import UnrealBridge

        bridge = UnrealBridge()
        models = bridge.list_loaded_models()
        assert isinstance(models, list)
        assert len(models) == 0
