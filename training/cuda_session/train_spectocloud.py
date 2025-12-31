#!/usr/bin/env python3
"""
Spectocloud Neural Model - CUDA Training Script

Train the SpectocloudViT model for real-time 3D music visualization.

Usage:
    python train_spectocloud.py --config spectocloud_training_config.yaml
    
Requirements:
    pip install torch torchvision torchaudio
    pip install transformers timm
    pip install wandb tensorboard
    pip install einops
"""

import os
import sys
import argparse
import yaml
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Check CUDA availability
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("Warning: CUDA not available, using CPU")


# =============================================================================
# Model Architecture
# =============================================================================

class SpectrogramEncoder(nn.Module):
    """Encode audio spectrogram into features."""
    
    def __init__(self, n_mels: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Linear(256 * 16, hidden_dim)
    
    def forward(self, x):
        # x: (batch, n_mels, time_frames)
        x = x.unsqueeze(1)  # Add channel dim
        x = self.conv_stack(x)
        x = x.flatten(1)
        return self.proj(x)


class EmotionEncoder(nn.Module):
    """Encode emotion embedding + MIDI features."""
    
    def __init__(self, emotion_dim: int = 64, midi_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.emotion_proj = nn.Linear(emotion_dim, hidden_dim // 2)
        self.midi_proj = nn.Linear(midi_dim, hidden_dim // 2)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
    
    def forward(self, emotion, midi_features):
        e = self.emotion_proj(emotion)
        m = self.midi_proj(midi_features)
        fused = torch.cat([e, m], dim=-1)
        return self.fusion(fused)


class PointCloudDecoder(nn.Module):
    """Decode features to 3D point cloud with properties."""
    
    def __init__(self, hidden_dim: int = 256, num_points: int = 1200, output_dim: int = 10):
        super().__init__()
        self.num_points = num_points
        self.output_dim = output_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, num_points * output_dim),
        )
    
    def forward(self, features):
        # features: (batch, hidden_dim * 2)
        out = self.decoder(features)
        out = out.view(-1, self.num_points, self.output_dim)
        
        # Split outputs
        positions = out[:, :, :3]  # xyz
        colors = torch.sigmoid(out[:, :, 3:7])  # rgba
        properties = out[:, :, 7:]  # size, glow, depth
        
        return {
            'positions': positions,
            'colors': colors,
            'properties': properties,
        }


class SpectocloudViT(nn.Module):
    """
    Spectocloud Vision Transformer
    
    Generates 3D point cloud visualizations from audio + emotion.
    """
    
    def __init__(
        self,
        n_mels: int = 128,
        emotion_dim: int = 64,
        midi_dim: int = 32,
        hidden_dim: int = 256,
        num_points: int = 1200,
        num_layers: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()
        
        self.spec_encoder = SpectrogramEncoder(n_mels, hidden_dim)
        self.emotion_encoder = EmotionEncoder(emotion_dim, midi_dim, hidden_dim)
        
        # Transformer layers for cross-modal fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.point_decoder = PointCloudDecoder(hidden_dim, num_points, output_dim=10)
    
    def forward(self, spectrogram, emotion, midi_features):
        # Encode inputs
        spec_features = self.spec_encoder(spectrogram)  # (B, hidden)
        emo_features = self.emotion_encoder(emotion, midi_features)  # (B, hidden)
        
        # Stack as sequence for transformer
        features = torch.stack([spec_features, emo_features], dim=1)  # (B, 2, hidden)
        
        # Cross-modal fusion
        features = self.transformer(features)  # (B, 2, hidden)
        
        # Pool and decode
        pooled = features.flatten(1)  # (B, hidden * 2)
        
        return self.point_decoder(pooled)


# =============================================================================
# Dataset
# =============================================================================

class SpectocloudDataset(Dataset):
    """Dataset for Spectocloud training."""
    
    def __init__(self, manifest_path: str, config: dict):
        self.config = config
        self.n_mels = config.get('n_mels', 128)
        self.time_frames = 64
        self.cache_dir = config.get('cache_dir', 'cache/spectrograms')
        self.use_cache = config.get('cache_features', False)
        
        if self.use_cache:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Load manifest or generate synthetic data
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                self.samples = [json.loads(line) for line in f]
            self.use_real_data = True
            print(f"Loaded {len(self.samples)} samples from {manifest_path}")
        else:
            print(f"WARNING: Manifest not found: {manifest_path}")
            print("WARNING: Generating synthetic training data for smoke testing only!")
            print("WARNING: This is NOT real data training!")
            self.samples = self._generate_synthetic_samples(
                config.get('synthetic_config', {}).get('num_train_samples', 10000)
            )
            self.use_real_data = False
    
    def _generate_synthetic_samples(self, num_samples: int) -> List[dict]:
        """Generate synthetic training samples."""
        samples = []
        np.random.seed(42)
        
        for i in range(num_samples):
            # Random spectrogram-like data
            n_mels = self.config.get('n_mels', 128)
            time_frames = 64
            
            # Emotion in valence-arousal-intensity space
            valence = np.random.uniform(-1, 1)
            arousal = np.random.uniform(0, 1)
            intensity = np.random.uniform(0, 1)
            
            samples.append({
                'spectrogram': np.random.randn(n_mels, time_frames).astype(np.float32),
                'emotion': np.array([valence, arousal, intensity] + [0.0] * 61, dtype=np.float32),
                'midi_features': np.random.randn(32).astype(np.float32),
                'target_positions': self._generate_target_cloud(valence, arousal, intensity),
                'target_colors': self._generate_target_colors(valence),
            })
        
        return samples
    
    def _generate_target_cloud(self, valence: float, arousal: float, intensity: float) -> np.ndarray:
        """Generate target point cloud based on emotion."""
        num_points = 1200
        
        # Spread increases with arousal
        spread = 0.16 + 0.22 * arousal
        
        # Generate points in conical pattern
        t = np.random.uniform(0, 1, num_points)  # Time dimension
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        r = np.random.exponential(spread, num_points) * (1 + t)
        
        x = t
        y = r * np.cos(theta) + 0.5
        z = r * np.sin(theta) * intensity + 0.5
        
        return np.stack([x, y, z], axis=1).astype(np.float32)
    
    def _generate_target_colors(self, valence: float) -> np.ndarray:
        """Generate target colors based on valence."""
        num_points = 1200
        
        # Valence -> color (blue to red)
        u = (valence + 1) / 2  # Map to 0-1
        
        # Coolwarm-ish colors
        r = u
        g = 0.5 - 0.5 * abs(valence)
        b = 1 - u
        a = np.random.uniform(0.3, 0.7, num_points)
        
        colors = np.zeros((num_points, 4), dtype=np.float32)
        colors[:, 0] = r + np.random.normal(0, 0.1, num_points)
        colors[:, 1] = g + np.random.normal(0, 0.1, num_points)
        colors[:, 2] = b + np.random.normal(0, 0.1, num_points)
        colors[:, 3] = a
        
        return np.clip(colors, 0, 1)
    
    def _load_audio_spectrogram(self, audio_path: str) -> np.ndarray:
        """Load audio file and compute mel spectrogram."""
        import librosa
        
        # Check cache first
        if self.use_cache:
            cache_key = hashlib.md5(audio_path.encode()).hexdigest()
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
            
            if os.path.exists(cache_path):
                return np.load(cache_path)
        
        # Load audio
        try:
            y, sr = librosa.load(audio_path, sr=44100, mono=True)
        except Exception as e:
            print(f"Warning: Failed to load {audio_path}: {e}")
            # Return random spectrogram as fallback
            return np.random.randn(self.n_mels, self.time_frames).astype(np.float32)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.n_mels,
            hop_length=512,
            n_fft=2048,
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [-1, 1]
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        # Resize to target time frames
        if mel_spec_db.shape[1] < self.time_frames:
            # Pad if too short
            pad_width = self.time_frames - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        elif mel_spec_db.shape[1] > self.time_frames:
            # Random crop if too long
            start = np.random.randint(0, mel_spec_db.shape[1] - self.time_frames + 1)
            mel_spec_db = mel_spec_db[:, start:start + self.time_frames]
        
        mel_spec_db = mel_spec_db.astype(np.float32)
        
        # Cache if enabled
        if self.use_cache:
            np.save(cache_path, mel_spec_db)
        
        return mel_spec_db
    
    def _extract_midi_features(self, midi_path: str) -> np.ndarray:
        """
        Extract simple MIDI features for conditioning.
        
        TODO: Implement richer feature extraction using pretty_midi.
        For now, returns placeholder features.
        """
        try:
            import pretty_midi
            
            midi = pretty_midi.PrettyMIDI(midi_path)
            
            # Extract simple features
            features = np.zeros(32, dtype=np.float32)
            
            # Feature 0-11: pitch class distribution
            pitch_classes = np.zeros(12)
            for instrument in midi.instruments:
                for note in instrument.notes:
                    pitch_classes[note.pitch % 12] += 1
            if pitch_classes.sum() > 0:
                pitch_classes /= pitch_classes.sum()
            features[:12] = pitch_classes
            
            # Feature 12: average velocity
            velocities = []
            for instrument in midi.instruments:
                velocities.extend([note.velocity for note in instrument.notes])
            features[12] = np.mean(velocities) / 127.0 if velocities else 0.5
            
            # Feature 13: note density (notes per second)
            if midi.get_end_time() > 0:
                total_notes = sum(len(inst.notes) for inst in midi.instruments)
                features[13] = min(1.0, total_notes / (midi.get_end_time() * 10))
            
            # Feature 14: tempo (normalized)
            tempo_changes = midi.get_tempo_changes()
            if len(tempo_changes[1]) > 0:
                avg_tempo = np.mean(tempo_changes[1])
                features[14] = min(1.0, avg_tempo / 200.0)
            
            # Remaining features: reserved for future use
            
            return features
            
        except Exception as e:
            print(f"Warning: Failed to extract MIDI features from {midi_path}: {e}")
            # Return default features
            return np.random.randn(32).astype(np.float32) * 0.1
    
    def _normalize_emotion(self, emotion: List[float]) -> np.ndarray:
        """Normalize emotion vector to 64 dimensions."""
        if len(emotion) == 64:
            return np.array(emotion, dtype=np.float32)
        elif len(emotion) == 3:
            # Expand 3D (valence, arousal, intensity) to 64D
            # First 3 are the core VAI, rest are zero-padded
            expanded = np.zeros(64, dtype=np.float32)
            expanded[:3] = emotion
            return expanded
        else:
            # Unexpected size - pad or truncate
            result = np.zeros(64, dtype=np.float32)
            result[:min(len(emotion), 64)] = emotion[:min(len(emotion), 64)]
            return result
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.use_real_data:
            # Load real data from files
            audio_path = sample['audio_path']
            midi_path = sample['midi_path']
            emotion = sample['emotion']
            
            # Load spectrogram
            spectrogram = self._load_audio_spectrogram(audio_path)
            
            # Extract MIDI features
            midi_features = self._extract_midi_features(midi_path)
            
            # Normalize emotion to 64D
            emotion = self._normalize_emotion(emotion)
            
            # Generate target point cloud (self-supervised mode)
            # Extract valence from emotion
            valence = emotion[0]
            arousal = emotion[1] if len(emotion) > 1 else 0.0
            intensity = emotion[2] if len(emotion) > 2 else 0.5
            
            target_positions = self._generate_target_cloud(valence, arousal, intensity)
            target_colors = self._generate_target_colors(valence)
            
            return {
                'spectrogram': torch.tensor(spectrogram),
                'emotion': torch.tensor(emotion),
                'midi_features': torch.tensor(midi_features),
                'target_positions': torch.tensor(target_positions),
                'target_colors': torch.tensor(target_colors),
            }
        else:
            # Synthetic data fallback
            return {
                'spectrogram': torch.tensor(sample['spectrogram']),
                'emotion': torch.tensor(sample['emotion']),
                'midi_features': torch.tensor(sample['midi_features']),
                'target_positions': torch.tensor(sample['target_positions']),
                'target_colors': torch.tensor(sample['target_colors']),
            }


# =============================================================================
# Loss Functions
# =============================================================================

def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Chamfer distance between point clouds."""
    # pred, target: (B, N, 3)
    
    # Compute pairwise distances
    diff = pred.unsqueeze(2) - target.unsqueeze(1)  # (B, N, N, 3)
    dist = torch.sum(diff ** 2, dim=-1)  # (B, N, N)
    
    # Forward and backward distances
    forward = torch.min(dist, dim=2)[0]  # (B, N)
    backward = torch.min(dist, dim=1)[0]  # (B, N)
    
    return torch.mean(forward) + torch.mean(backward)


def color_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss for colors."""
    return F.mse_loss(pred, target)


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    epoch: int,
    config: dict,
    scaler: Optional[Any] = None,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_pos_loss = 0.0
    total_color_loss = 0.0
    
    grad_accum_steps = config.get('grad_accum_steps', 1)
    use_amp = config.get('amp', False)
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        spec = batch['spectrogram'].to(device)
        emotion = batch['emotion'].to(device)
        midi = batch['midi_features'].to(device)
        target_pos = batch['target_positions'].to(device)
        target_col = batch['target_colors'].to(device)
        
        # Forward pass with AMP
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.float16 if config.get('amp_dtype', 'float16') == 'float16' else torch.bfloat16):
                outputs = model(spec, emotion, midi)
                
                # Compute losses
                pos_loss = chamfer_distance(outputs['positions'], target_pos)
                col_loss = color_loss(outputs['colors'], target_col)
                
                loss = pos_loss + 0.5 * col_loss
                loss = loss / grad_accum_steps
        else:
            outputs = model(spec, emotion, midi)
            
            # Compute losses
            pos_loss = chamfer_distance(outputs['positions'], target_pos)
            col_loss = color_loss(outputs['colors'], target_col)
            
            loss = pos_loss + 0.5 * col_loss
            loss = loss / grad_accum_steps
        
        # Backward pass
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights with gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Gradient clipping
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
                optimizer.step()
            
            optimizer.zero_grad()
            
            if scheduler:
                scheduler.step()
        
        total_loss += loss.item() * grad_accum_steps
        total_pos_loss += pos_loss.item()
        total_color_loss += col_loss.item()
        
        if batch_idx % config.get('log_every', 50) == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item() * grad_accum_steps:.4f} (pos: {pos_loss.item():.4f}, col: {col_loss.item():.4f})")
    
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'pos_loss': total_pos_loss / n_batches,
        'color_loss': total_color_loss / n_batches,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model."""
    model.eval()
    
    total_loss = 0.0
    total_pos_loss = 0.0
    total_color_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            spec = batch['spectrogram'].to(device)
            emotion = batch['emotion'].to(device)
            midi = batch['midi_features'].to(device)
            target_pos = batch['target_positions'].to(device)
            target_col = batch['target_colors'].to(device)
            
            outputs = model(spec, emotion, midi)
            
            pos_loss = chamfer_distance(outputs['positions'], target_pos)
            col_loss = color_loss(outputs['colors'], target_col)
            loss = pos_loss + 0.5 * col_loss
            
            total_loss += loss.item()
            total_pos_loss += pos_loss.item()
            total_color_loss += col_loss.item()
    
    n_batches = len(dataloader)
    return {
        'val_loss': total_loss / n_batches,
        'val_pos_loss': total_pos_loss / n_batches,
        'val_color_loss': total_color_loss / n_batches,
    }


def main(config_path: str):
    """Main training function."""
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("Spectocloud Neural Model Training")
    print("=" * 60)
    print(f"Config: {config_path}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model_cfg = config.get('model', {})
    model = SpectocloudViT(
        n_mels=config.get('data', {}).get('n_mels', 128),
        emotion_dim=64,
        midi_dim=32,
        hidden_dim=model_cfg.get('encoder', {}).get('hidden_dim', 256),
        num_points=model_cfg.get('decoder', {}).get('output_points', 1200),
        num_layers=model_cfg.get('encoder', {}).get('num_layers', 6),
        num_heads=model_cfg.get('encoder', {}).get('num_heads', 8),
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Compile model (PyTorch 2.0+)
    if config.get('hardware', {}).get('compile_mode', True):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile()")
        except (RuntimeError, AttributeError) as e:
            print(f"torch.compile() not available: {e}")
    
    # Create datasets
    data_cfg = config.get('data', {})
    train_dataset = SpectocloudDataset(
        data_cfg.get('train_manifest', 'data/manifests/spectocloud_train.jsonl'),
        data_cfg,
    )
    val_dataset = SpectocloudDataset(
        data_cfg.get('val_manifest', 'data/manifests/spectocloud_val.jsonl'),
        data_cfg,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.get('batch_size', 32),
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=data_cfg.get('pin_memory', True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.get('batch_size', 32),
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 4),
    )
    
    # Create optimizer
    optim_cfg = config.get('optim', {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg.get('lr', 1e-4),
        weight_decay=optim_cfg.get('weight_decay', 0.05),
        betas=tuple(optim_cfg.get('betas', [0.9, 0.95])),
    )
    
    # Create scheduler
    sched_cfg = config.get('scheduler', {})
    total_steps = sched_cfg.get('max_steps', len(train_loader) * config.get('training', {}).get('epochs', 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=optim_cfg.get('lr', 1e-4) * 0.01,
    )
    
    # Training loop
    train_cfg = config.get('training', {})
    epochs = train_cfg.get('epochs', 20)
    save_dir = Path('checkpoints/spectocloud')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # AMP setup
    use_amp = train_cfg.get('amp', False)
    scaler = None
    if use_amp and device.type == 'cuda':
        amp_dtype = train_cfg.get('amp_dtype', 'float16')
        if amp_dtype == 'float16':
            scaler = torch.cuda.amp.GradScaler()
            print("Using AMP with float16")
        else:
            # bfloat16 doesn't need GradScaler
            print("Using AMP with bfloat16")
    
    # Early stopping setup
    early_stop_cfg = train_cfg.get('early_stopping', {})
    early_stop_enabled = early_stop_cfg.get('enabled', False)
    patience = early_stop_cfg.get('patience', 5)
    min_delta = early_stop_cfg.get('min_delta', 0.001)
    min_epochs = train_cfg.get('min_epochs', 10)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, train_cfg, scaler
        )
        print(f"Train - Loss: {train_metrics['loss']:.4f}")
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        print(f"Val - Loss: {val_metrics['val_loss']:.4f}")
        
        # Save checkpoint
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            improvement = best_val_loss - val_metrics['val_loss']
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config,
            }, save_dir / 'best.pt')
            print(f"Saved best model (val_loss: {best_val_loss:.4f}, improvement: {improvement:.4f})")
        else:
            patience_counter += 1
        
        # Regular checkpoint (epoch-based)
        if (epoch + 1) % train_cfg.get('save_every', 5) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pt')
        
        # Early stopping check
        if early_stop_enabled and epoch >= min_epochs:
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"No improvement for {patience} epochs")
                break
            
            # Diminishing returns check
            if is_best and improvement < min_delta:
                print(f"Warning: Improvement {improvement:.4f} below threshold {min_delta:.4f}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Spectocloud model')
    parser.add_argument('--config', type=str, default='spectocloud_training_config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)
