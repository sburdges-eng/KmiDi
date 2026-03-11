"""Comprehensive tests for JEPA model implementations."""

import pytest
import torch

from music_brain.jepa.audio_jepa import (
    AudioJEPAEncoder,
    EMATargetEncoder,
    LatentPredictor,
)
from music_brain.jepa.chord_jepa import ChordEmbedding, ChordJEPA
from music_brain.jepa.masking import MultiBlockMasking, mask_latents
from music_brain.jepa.config import (
    AudioJEPAConfig,
    ChordJEPAConfig,
    StemJEPAConfig,
    TrainingConfig,
)
from music_brain.jepa.datasets import AudioMelDataset, ChordSequenceDataset


class TestAudioJEPAEncoder:
    """Audio-JEPA encoder tests."""

    def test_encoder_small(self):
        enc = AudioJEPAEncoder(tier="small")
        x = torch.randn(2, 1, 128, 64)
        z = enc(x)
        assert z.dim() == 3
        assert z.shape[0] == 2
        assert z.shape[2] == 128

    def test_encoder_medium(self):
        enc = AudioJEPAEncoder(tier="medium")
        x = torch.randn(2, 1, 128, 128)
        z = enc(x)
        assert z.shape[0] == 2
        assert z.shape[2] == 256

    def test_encoder_large(self):
        enc = AudioJEPAEncoder(tier="large")
        x = torch.randn(1, 1, 128, 64)
        z = enc(x)
        assert z.shape[2] == 512

    def test_encoder_from_config(self):
        cfg = AudioJEPAConfig(tier="small", latent_dim=128, n_mels=128)
        enc = AudioJEPAEncoder(config=cfg)
        x = torch.randn(1, 1, 128, 32)
        z = enc(x)
        assert z.shape[2] == 128

    def test_encoder_gradient_flow(self):
        enc = AudioJEPAEncoder(tier="small")
        x = torch.randn(1, 1, 128, 64, requires_grad=True)
        z = enc(x)
        loss = z.sum()
        loss.backward()
        assert x.grad is not None

    def test_encoder_different_lengths(self):
        enc = AudioJEPAEncoder(tier="medium")
        for T in [32, 64, 128, 256]:
            x = torch.randn(1, 1, 128, T)
            z = enc(x)
            assert z.shape[0] == 1
            assert z.shape[2] == 256


class TestLatentPredictor:
    """Latent predictor tests."""

    def test_predictor_forward(self):
        pred = LatentPredictor(latent_dim=256)
        z = torch.randn(2, 32, 256)
        out = pred(z)
        assert out.shape == z.shape

    def test_predictor_small_dim(self):
        pred = LatentPredictor(latent_dim=64, num_heads=4)
        z = torch.randn(1, 16, 64)
        out = pred(z)
        assert out.shape == (1, 16, 64)

    def test_predictor_gradient_flow(self):
        pred = LatentPredictor(latent_dim=128, num_heads=4)
        z = torch.randn(1, 8, 128, requires_grad=True)
        out = pred(z)
        loss = out.sum()
        loss.backward()
        assert z.grad is not None


class TestEMATargetEncoder:
    """EMA target encoder tests."""

    def test_ema_init(self):
        enc = AudioJEPAEncoder(tier="small")
        ema = EMATargetEncoder(enc, momentum=0.99)
        x = torch.randn(1, 1, 128, 64)
        z = ema(x)
        assert z.dim() == 3

    def test_ema_no_grad(self):
        enc = AudioJEPAEncoder(tier="small")
        ema = EMATargetEncoder(enc)
        for p in ema.encoder.parameters():
            assert not p.requires_grad

    def test_ema_update(self):
        enc = AudioJEPAEncoder(tier="small")
        ema = EMATargetEncoder(enc, momentum=0.0)
        x = torch.randn(1, 1, 128, 64)
        enc(x).sum().backward()
        with torch.no_grad():
            for p in enc.parameters():
                p.data += 0.1
        ema.update(enc)
        for p_ema, p_enc in zip(ema.encoder.parameters(), enc.parameters()):
            assert torch.allclose(p_ema, p_enc, atol=1e-6)


class TestChordJEPA:
    """Chord-JEPA tests."""

    def test_chord_jepa_forward(self):
        model = ChordJEPA(d_model=128, num_classes=170, num_heads=4,
                          num_layers=2, seq_len=64)
        z = torch.randn(2, 64, 128)
        out = model(z)
        assert out.shape == (2, 64, 170)

    def test_chord_jepa_from_config(self):
        cfg = ChordJEPAConfig(d_model=64, num_heads=4, num_layers=2,
                              seq_len=32, num_chords=100)
        model = ChordJEPA(config=cfg)
        z = torch.randn(1, 32, 64)
        out = model(z)
        assert out.shape == (1, 32, 100)

    def test_chord_embedding(self):
        emb = ChordEmbedding(num_chords=170, d_model=256)
        ids = torch.randint(0, 170, (2, 64))
        out = emb(ids)
        assert out.shape == (2, 64, 256)

    def test_chord_jepa_gradient_flow(self):
        model = ChordJEPA(d_model=64, num_classes=50, num_heads=4,
                          num_layers=1, seq_len=16)
        z = torch.randn(1, 16, 64, requires_grad=True)
        out = model(z)
        loss = out.sum()
        loss.backward()
        assert z.grad is not None


class TestMasking:
    """Multi-block masking tests."""

    def test_mask_3d(self):
        z = torch.randn(2, 32, 128)
        masked, mask = mask_latents(z, mask_ratio=0.5)
        assert masked.shape == z.shape
        assert mask.shape == (2, 32)
        assert mask.dtype == torch.bool

    def test_mask_4d(self):
        z = torch.randn(2, 1, 128, 64)
        masked, mask = mask_latents(z, mask_ratio=0.5)
        assert masked.shape == z.shape
        assert mask.shape == (2, 64)

    def test_mask_zeros_out_positions(self):
        z = torch.ones(1, 32, 64)
        masked, mask = mask_latents(z, mask_ratio=0.8, block_size=8)
        assert masked[mask.unsqueeze(-1).expand_as(masked)].sum() == 0

    def test_multi_block_masking_module(self):
        masker = MultiBlockMasking(mask_ratio=0.5, block_size=4)
        z = torch.randn(2, 32, 128)
        result = masker(z)
        assert result.masked.shape == z.shape
        assert result.mask.shape == (2, 32)
        assert result.target_indices.dim() == 2

    def test_mask_ratio_respected(self):
        z = torch.randn(4, 64, 128)
        _, mask = mask_latents(z, mask_ratio=0.5, block_size=4, num_blocks=8)
        ratio = mask.float().mean().item()
        assert 0.2 < ratio < 0.9


class TestConfigs:
    """Configuration dataclass tests."""

    def test_audio_jepa_defaults(self):
        cfg = AudioJEPAConfig()
        assert cfg.latent_dim == 256
        assert cfg.n_mels == 128
        assert cfg.tier == "medium"

    def test_chord_jepa_defaults(self):
        cfg = ChordJEPAConfig()
        assert cfg.d_model == 256
        assert cfg.num_chords == 170

    def test_stem_jepa_defaults(self):
        cfg = StemJEPAConfig()
        assert cfg.stem_embedding_dim == 128

    def test_training_config_defaults(self):
        cfg = TrainingConfig()
        assert cfg.epochs == 120
        assert cfg.instance_type == "ml.g5.xlarge"

    def test_training_config_custom(self):
        cfg = TrainingConfig(epochs=50, batch_size=8, learning_rate=1e-4)
        assert cfg.epochs == 50
        assert cfg.batch_size == 8


class TestDatasets:
    """Dataset tests (dry-run mode with random data)."""

    def test_chord_dataset_random(self):
        ds = ChordSequenceDataset(num_samples=16, seq_len=32, num_chords=50)
        assert len(ds) == 16
        sample = ds[0]
        assert sample.shape == (32,)
        assert sample.max() < 50

    def test_chord_dataset_none_files(self):
        ds = ChordSequenceDataset(files=[None, None], seq_len=16)
        assert len(ds) > 0
        sample = ds[0]
        assert sample.shape == (16,)

    def test_audio_dataset_empty(self):
        ds = AudioMelDataset(files=[], max_frames=64)
        assert len(ds) == 0


class TestEndToEndJEPA:
    """End-to-end mini training step tests."""

    def test_audio_jepa_training_step(self):
        enc = AudioJEPAEncoder(tier="small")
        pred = LatentPredictor(latent_dim=128, num_heads=4)
        x = torch.randn(2, 1, 128, 64)
        z = enc(x)
        masked, _ = mask_latents(z)
        out = pred(masked)
        loss = ((out - z) ** 2).mean()
        loss.backward()
        assert loss.item() > 0

    def test_chord_jepa_training_step(self):
        emb = ChordEmbedding(num_chords=170, d_model=128)
        model = ChordJEPA(d_model=128, num_classes=170, num_heads=4,
                          num_layers=2, seq_len=32)
        chords = torch.randint(0, 170, (2, 32))
        z = emb(chords)
        masked, _ = mask_latents(z)
        out = model(masked)
        loss = torch.nn.functional.cross_entropy(
            out.reshape(-1, 170), chords.reshape(-1)
        )
        loss.backward()
        assert loss.item() > 0

    def test_ema_training_step(self):
        enc = AudioJEPAEncoder(tier="small")
        pred = LatentPredictor(latent_dim=128, num_heads=4)
        teacher = EMATargetEncoder(enc, momentum=0.996)

        x = torch.randn(2, 1, 128, 64)
        z = enc(x)
        with torch.no_grad():
            z_target = teacher(x)
        masked, _ = mask_latents(z)
        out = pred(masked)
        loss = ((out - z_target) ** 2).mean()
        loss.backward()

        opt = torch.optim.AdamW(
            list(enc.parameters()) + list(pred.parameters()), lr=1e-4
        )
        opt.step()
        teacher.update(enc)
        assert loss.item() > 0
