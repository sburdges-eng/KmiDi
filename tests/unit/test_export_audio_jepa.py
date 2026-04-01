"""Tests for Audio JEPA export pipeline."""

import os
import platform
import tempfile

import pytest
import torch
import numpy as np

from music_brain.jepa.audio_jepa import AudioJEPAEncoder
from music_brain.jepa.config import AudioJEPAConfig

try:
    import coremltools as ct
    HAS_COREML = True
except ImportError:
    HAS_COREML = False

skip_no_coreml = pytest.mark.skipif(
    not HAS_COREML or platform.system() != "Darwin",
    reason="Requires coremltools on macOS",
)


CHECKPOINT_PATH = "checkpoints/audio_jepa/best_model.pt"


@pytest.fixture
def checkpoint():
    """Load the real checkpoint."""
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
    return torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)


@pytest.fixture
def encoder(checkpoint):
    """Reconstruct encoder from checkpoint config and weights."""
    config = AudioJEPAConfig(**checkpoint["config"])
    enc = AudioJEPAEncoder(config=config)
    enc.load_state_dict(checkpoint["encoder"])
    enc.eval()
    return enc


class TestCheckpointLoading:
    def test_checkpoint_has_required_keys(self, checkpoint):
        for key in ("encoder", "config", "epoch", "loss"):
            assert key in checkpoint, f"Missing key: {key}"

    def test_config_matches_expected_shape(self, checkpoint):
        cfg = checkpoint["config"]
        assert cfg["n_mels"] == 128
        assert cfg["max_frames"] == 512
        assert cfg["latent_dim"] == 256
        assert cfg["tier"] == "medium"

    def test_encoder_loads_and_runs(self, encoder):
        x = torch.randn(1, 1, 128, 512)
        with torch.no_grad():
            z = encoder(x)
        assert z.shape == (1, 512, 256)


class TestOnnxExport:
    def test_onnx_export_produces_valid_file(self, encoder):
        import onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test.onnx")
            dummy = torch.randn(1, 1, 128, 512)
            torch.onnx.export(
                encoder,
                dummy,
                out_path,
                opset_version=18,
                input_names=["mel"],
                output_names=["latent"],
            )
            assert os.path.exists(out_path)
            model = onnx.load(out_path)
            onnx.checker.check_model(model)

    def test_onnx_has_fixed_shapes(self, encoder):
        import onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test.onnx")
            dummy = torch.randn(1, 1, 128, 512)
            torch.onnx.export(
                encoder,
                dummy,
                out_path,
                opset_version=18,
                input_names=["mel"],
                output_names=["latent"],
            )
            model = onnx.load(out_path)
            inp = model.graph.input[0]
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            assert dims == [1, 1, 128, 512], f"Expected fixed [1,1,128,512], got {dims}"

    def test_onnx_inference_matches_pytorch(self, encoder):
        import onnxruntime as ort
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test.onnx")
            dummy = torch.randn(1, 1, 128, 512)
            torch.onnx.export(
                encoder,
                dummy,
                out_path,
                opset_version=18,
                input_names=["mel"],
                output_names=["latent"],
            )
            # PyTorch reference
            with torch.no_grad():
                pt_out = encoder(dummy).numpy()

            # ONNX Runtime
            sess = ort.InferenceSession(out_path)
            ort_out = sess.run(None, {"mel": dummy.numpy()})[0]

            np.testing.assert_allclose(pt_out, ort_out, atol=1e-5, rtol=1e-4)


class _CoreMLEncoderWrapper(torch.nn.Module):
    """Wrapper that avoids dynamic shape unpacking for coremltools tracing."""

    def __init__(self, encoder):
        super().__init__()
        self.conv = encoder.conv
        self.proj = encoder.proj
        self.layer_norm = encoder.layer_norm

    def forward(self, x):
        h = self.conv(x)
        h = h.permute(0, 3, 1, 2)
        h = h.flatten(2)
        h = self.proj(h)
        return self.layer_norm(h)


class TestCoremlExport:
    @skip_no_coreml
    def test_coreml_export_produces_mlpackage(self, encoder):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test.mlpackage")
            dummy = torch.randn(1, 1, 128, 512)
            wrapper = _CoreMLEncoderWrapper(encoder)
            wrapper.eval()
            traced = torch.jit.trace(wrapper, dummy)

            mlmodel = ct.convert(
                traced,
                inputs=[ct.TensorType(name="mel", shape=(1, 1, 128, 512))],
                minimum_deployment_target=ct.target.macOS13,
                compute_units=ct.ComputeUnit.ALL,
            )
            mlmodel.save(out_path)
            assert os.path.exists(out_path)

    @skip_no_coreml
    def test_coreml_inference_matches_pytorch(self, encoder):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test.mlpackage")
            dummy = torch.randn(1, 1, 128, 512)
            wrapper = _CoreMLEncoderWrapper(encoder)
            wrapper.eval()
            traced = torch.jit.trace(wrapper, dummy)

            mlmodel = ct.convert(
                traced,
                inputs=[ct.TensorType(name="mel", shape=(1, 1, 128, 512))],
                minimum_deployment_target=ct.target.macOS13,
                compute_units=ct.ComputeUnit.ALL,
            )
            mlmodel.save(out_path)

            # PyTorch reference
            with torch.no_grad():
                pt_out = encoder(dummy).numpy()

            # Core ML
            loaded = ct.models.MLModel(out_path)
            pred = loaded.predict({"mel": dummy.numpy()})
            cml_out = list(pred.values())[0]

            max_diff = np.abs(pt_out - cml_out).max()
            assert max_diff < 1e-2, f"Core ML max_diff={max_diff:.2e} exceeds 1e-2"
