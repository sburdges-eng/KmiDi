"""Tests for EmotionProbe model."""

import pytest

torch = pytest.importorskip("torch")

from music_brain.jepa.emotion_probe import EmotionProbe  # noqa: E402


class TestEmotionProbe:
    def test_output_shape(self):
        probe = EmotionProbe(latent_dim=256, hidden_dim=128)
        x = torch.randn(4, 256)
        out = probe(x)
        assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"

    def test_output_range(self):
        probe = EmotionProbe(latent_dim=256, hidden_dim=128)
        x = torch.randn(32, 256)
        out = probe(x)
        assert out.min() >= -1.0, "Output below -1"
        assert out.max() <= 1.0, "Output above 1"

    def test_gradient_flows(self):
        probe = EmotionProbe(latent_dim=256, hidden_dim=128)
        x = torch.randn(4, 256)
        target = torch.tensor([[0.5, 0.3], [-0.2, 0.8], [0.0, 0.0], [0.9, -0.5]])
        out = probe(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        for p in probe.parameters():
            assert p.grad is not None, "Gradient not flowing"
            assert p.grad.abs().sum() > 0, "Zero gradient"

    def test_deterministic_eval(self):
        probe = EmotionProbe(latent_dim=256, hidden_dim=128)
        probe.eval()
        x = torch.randn(2, 256)
        out1 = probe(x)
        out2 = probe(x)
        assert torch.allclose(out1, out2), "Non-deterministic in eval mode"
