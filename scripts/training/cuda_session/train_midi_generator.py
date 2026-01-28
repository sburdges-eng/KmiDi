#!/usr/bin/env python3
"""
MIDI Generator Transformer - CUDA Training Script

Train the EmotionMIDITransformer for emotion-conditioned MIDI generation.

Usage:
    python train_midi_generator.py --config midi_generator_training_config.yaml

Requirements:
    pip install torch transformers
    pip install wandb tensorboard
"""

import os
import sys
import argparse
import yaml
import json
import math
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Check CUDA
if torch.cuda.is_available():
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
else:
    print("Warning: CUDA not available")


# =============================================================================
# MIDI Tokenization
# =============================================================================

class MIDITokenizer:
    """Tokenize MIDI events for transformer training."""
    
    def __init__(self, vocab_size: int = 388):
        self.vocab_size = vocab_size
        
        # Token ranges
        self.note_range = (0, 128)          # MIDI notes 0-127
        self.velocity_range = (128, 160)     # 32 velocity bins
        self.timing_range = (160, 288)       # 128 timing positions
        
        # Special tokens
        self.pad_token = 384
        self.bos_token = 385
        self.eos_token = 386
        self.bar_token = 387
    
    def encode(self, midi_events: List[Dict]) -> List[int]:
        """Encode MIDI events to token sequence."""
        tokens = [self.bos_token]
        
        for event in midi_events:
            if event.get('type') == 'bar':
                tokens.append(self.bar_token)
            elif event.get('type') == 'note_on':
                # Note token
                tokens.append(event['note'] % 128)
                # Velocity token (quantized to 32 bins)
                vel_bin = min(31, event.get('velocity', 64) // 4)
                tokens.append(128 + vel_bin)
                # Timing token
                time_pos = int(event.get('time_in_bar', 0) * 127) % 128
                tokens.append(160 + time_pos)
        
        tokens.append(self.eos_token)
        return tokens
    
    def decode(self, tokens: List[int]) -> List[Dict]:
        """Decode token sequence to MIDI events."""
        events = []
        i = 0
        current_time = 0.0
        
        while i < len(tokens):
            token = tokens[i]
            
            if token == self.bar_token:
                events.append({'type': 'bar', 'time': current_time})
                current_time += 1.0  # Bar = 1 beat
            elif self.note_range[0] <= token < self.note_range[1]:
                # Note event - expect velocity and timing to follow
                note = token
                velocity = 64
                timing = 0.0
                
                if i + 1 < len(tokens):
                    vel_token = tokens[i + 1]
                    if self.velocity_range[0] <= vel_token < self.velocity_range[1]:
                        velocity = (vel_token - 128) * 4
                        i += 1
                
                if i + 1 < len(tokens):
                    time_token = tokens[i + 1]
                    if self.timing_range[0] <= time_token < self.timing_range[1]:
                        timing = (time_token - 160) / 127.0
                        i += 1
                
                events.append({
                    'type': 'note_on',
                    'note': note,
                    'velocity': velocity,
                    'time': current_time + timing,
                })
            
            i += 1
        
        return events


# =============================================================================
# Model Architecture
# =============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for music sequences."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos())
        self.register_buffer('sin', emb.sin())
    
    def forward(self, seq_len: int):
        return self.cos[:seq_len], self.sin[:seq_len]


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to queries and keys."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class EmotionCrossAttention(nn.Module):
    """Cross-attention layer for emotion conditioning."""
    
    def __init__(self, hidden_dim: int, num_heads: int, emotion_dim: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(emotion_dim, hidden_dim)
        self.v_proj = nn.Linear(emotion_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, emotion_context):
        # x: (B, T, hidden)
        # emotion_context: (B, emotion_tokens, emotion_dim)
        
        B, T, _ = x.shape
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(emotion_context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(emotion_context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, -1)
        
        return self.norm(x + self.out_proj(out))


class TransformerBlock(nn.Module):
    """Transformer block with optional emotion cross-attention."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_mult: float = 4.0,
        dropout: float = 0.1,
        use_emotion_cross_attn: bool = False,
        emotion_dim: int = 64,
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        
        self.use_emotion_cross_attn = use_emotion_cross_attn
        if use_emotion_cross_attn:
            self.emotion_cross_attn = EmotionCrossAttention(hidden_dim, num_heads, emotion_dim)
        
        ff_dim = int(hidden_dim * ff_mult)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, attn_mask=None, emotion_context=None):
        # Self attention with causal mask
        x = self.self_attn_norm(x + self._self_attn(x, attn_mask))
        
        # Emotion cross attention
        if self.use_emotion_cross_attn and emotion_context is not None:
            x = self.emotion_cross_attn(x, emotion_context)
        
        # Feed-forward
        x = self.ff_norm(x + self.ff(x))
        
        return x
    
    def _self_attn(self, x, mask):
        return self.self_attn(x, x, x, attn_mask=mask, is_causal=True)[0]


class EmotionMIDITransformer(nn.Module):
    """
    Emotion-conditioned MIDI generation transformer.
    
    Generates MIDI token sequences conditioned on emotion embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int = 388,
        max_seq_len: int = 1024,
        hidden_dim: int = 384,
        num_layers: int = 8,
        num_heads: int = 6,
        ff_mult: float = 4.0,
        dropout: float = 0.1,
        emotion_dim: int = 64,
        emotion_cross_attn_layers: List[int] = None,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        
        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        
        # Rotary position embedding
        self.rotary = RotaryEmbedding(hidden_dim // num_heads, max_seq_len)
        
        # Emotion encoder
        self.emotion_encoder = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Transformer layers
        if emotion_cross_attn_layers is None:
            emotion_cross_attn_layers = [0, 2, 4, 6]
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout,
                use_emotion_cross_attn=(i in emotion_cross_attn_layers),
                emotion_dim=hidden_dim,  # After encoding
            )
            for i in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        emotion: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        B, T = input_ids.shape
        
        # Token embeddings
        x = self.token_emb(input_ids)
        
        # Encode emotion context
        emotion_context = None
        if emotion is not None:
            # Expand emotion to context window
            emotion_context = self.emotion_encoder(emotion)
            if emotion_context.dim() == 2:
                emotion_context = emotion_context.unsqueeze(1)  # (B, 1, hidden)
        
        # Generate causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask, emotion_context=emotion_context)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        emotion: torch.Tensor,
        max_length: int = 256,
        temperature: float = 0.9,
        top_k: int = 40,
        top_p: float = 0.92,
    ) -> torch.Tensor:
        """Generate MIDI tokens autoregressively."""
        self.eval()
        
        generated = prompt.clone()
        
        for _ in range(max_length):
            if generated.shape[1] >= self.max_seq_len:
                break
            
            logits = self(generated, emotion)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][:, -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS
            if next_token.item() == 386:  # EOS token
                break
        
        return generated


# =============================================================================
# Dataset
# =============================================================================

class MIDIDataset(Dataset):
    """Dataset for MIDI generation training."""
    
    def __init__(self, manifest_path: str, config: dict, tokenizer: MIDITokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.max_seq_len = config.get('max_seq_length', 512)
        self.cache_dir = config.get('cache_dir', 'cache/midi_tokens')
        self.use_cache = config.get('cache_features', False)
        
        if self.use_cache:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                self.samples = [json.loads(line) for line in f]
            self.use_real_data = True
            print(f"Loaded {len(self.samples)} samples from {manifest_path}")
        else:
            print(f"WARNING: Manifest not found: {manifest_path}")
            print("WARNING: Generating synthetic MIDI data for smoke testing only!")
            print("WARNING: This is NOT real data training!")
            self.samples = self._generate_synthetic_samples(10000)
            self.use_real_data = False
    
    def _generate_synthetic_samples(self, num_samples: int) -> List[dict]:
        """Generate synthetic MIDI training samples."""
        samples = []
        np.random.seed(42)
        
        for _ in range(num_samples):
            # Random emotion
            valence = np.random.uniform(-1, 1)
            arousal = np.random.uniform(0, 1)
            intensity = np.random.uniform(0, 1)
            
            # Generate MIDI events based on emotion
            events = []
            n_bars = np.random.randint(2, 8)
            
            for bar in range(n_bars):
                events.append({'type': 'bar'})
                
                # More notes for higher arousal
                n_notes = int(2 + 6 * arousal)
                
                for i in range(n_notes):
                    # Higher valence = higher notes
                    base_note = 60 + int(12 * valence)
                    note = base_note + np.random.randint(-7, 8)
                    note = max(24, min(96, note))
                    
                    # Higher intensity = higher velocity
                    velocity = int(32 + 64 * intensity + np.random.randint(-16, 16))
                    velocity = max(1, min(127, velocity))
                    
                    # Timing within bar
                    time_in_bar = i / n_notes + np.random.uniform(-0.05, 0.05)
                    
                    events.append({
                        'type': 'note_on',
                        'note': note,
                        'velocity': velocity,
                        'time_in_bar': time_in_bar,
                    })
            
            tokens = self.tokenizer.encode(events)
            
            samples.append({
                'tokens': tokens,
                'emotion': np.array([valence, arousal, intensity] + [0.0] * 61, dtype=np.float32),
            })
        
        return samples
    
    def _load_and_tokenize_midi(self, midi_path: str) -> List[int]:
        """Load MIDI file and tokenize to event sequence."""
        # Check cache first
        if self.use_cache:
            cache_key = hashlib.md5(midi_path.encode()).hexdigest()
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
            
            if os.path.exists(cache_path):
                return np.load(cache_path).tolist()
        
        try:
            import pretty_midi
            
            # Load MIDI
            midi = pretty_midi.PrettyMIDI(midi_path)
            
            # Convert to events
            events = []
            
            # Get time signature changes (estimate bars)
            if len(midi.time_signature_changes) > 0:
                ts = midi.time_signature_changes[0]
                beats_per_bar = ts.numerator
                beat_duration = 60.0 / 120.0  # Assume 120 BPM if no tempo
                if len(midi.get_tempo_changes()[1]) > 0:
                    beat_duration = 60.0 / midi.get_tempo_changes()[1][0]
                bar_duration = beats_per_bar * beat_duration
            else:
                bar_duration = 2.0  # Default 2 seconds per bar
            
            # Process all notes from all instruments
            all_notes = []
            for instrument in midi.instruments:
                if not instrument.is_drum:  # Skip drums for now
                    for note in instrument.notes:
                        all_notes.append({
                            'time': note.start,
                            'pitch': note.pitch,
                            'velocity': note.velocity,
                        })
            
            # Sort by time
            all_notes.sort(key=lambda x: x['time'])
            
            # Convert to events with bar markers
            current_bar = 0
            for note in all_notes:
                note_bar = int(note['time'] / bar_duration)
                
                # Add bar markers
                while current_bar < note_bar:
                    events.append({'type': 'bar'})
                    current_bar += 1
                
                # Add note event
                time_in_bar = (note['time'] % bar_duration) / bar_duration
                events.append({
                    'type': 'note_on',
                    'note': note['pitch'],
                    'velocity': note['velocity'],
                    'time_in_bar': time_in_bar,
                })
            
            # Tokenize
            tokens = self.tokenizer.encode(events)
            
            # Cache if enabled
            if self.use_cache:
                np.save(cache_path, np.array(tokens))
            
            return tokens
            
        except Exception as e:
            print(f"Warning: Failed to load MIDI from {midi_path}: {e}")
            # Return minimal valid sequence
            return [self.tokenizer.bos_token, self.tokenizer.eos_token]
    
    def _normalize_emotion(self, emotion: List[float]) -> np.ndarray:
        """Normalize emotion vector to 64 dimensions."""
        if len(emotion) == 64:
            return np.array(emotion, dtype=np.float32)
        elif len(emotion) == 3:
            # Expand 3D (valence, arousal, intensity) to 64D
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
            # Load and tokenize MIDI file
            midi_path = sample['midi_path']
            tokens = self._load_and_tokenize_midi(midi_path)
            
            # Normalize emotion
            emotion = self._normalize_emotion(sample['emotion'])
        else:
            # Synthetic data fallback
            tokens = sample['tokens']
            emotion = sample['emotion']
        
        # Pad or truncate
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        else:
            tokens = tokens + [self.tokenizer.pad_token] * (self.max_seq_len - len(tokens))
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'emotion': torch.tensor(emotion),
        }


# =============================================================================
# Training
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
    total_correct = 0
    total_tokens = 0
    
    grad_accum_steps = config.get('grad_accum_steps', 1)
    use_amp = config.get('amp', False)
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        emotion = batch['emotion'].to(device)
        
        # Forward pass with AMP
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.float16 if config.get('amp_dtype', 'bfloat16') == 'float16' else torch.bfloat16):
                logits = model(input_ids, emotion)
                
                loss = F.cross_entropy(
                    logits.view(-1, model.vocab_size),
                    labels.view(-1),
                    ignore_index=384,  # Pad token
                    label_smoothing=0.1,
                )
                loss = loss / grad_accum_steps
        else:
            logits = model(input_ids, emotion)
            
            loss = F.cross_entropy(
                logits.view(-1, model.vocab_size),
                labels.view(-1),
                ignore_index=384,  # Pad token
                label_smoothing=0.1,
            )
            loss = loss / grad_accum_steps
        
        # Backward pass
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights with gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
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
        
        # Accuracy (excluding padding)
        mask = labels != 384
        preds = logits.argmax(dim=-1)
        total_correct += ((preds == labels) & mask).sum().item()
        total_tokens += mask.sum().item()
        
        if batch_idx % config.get('log_every', 100) == 0:
            acc = total_correct / max(1, total_tokens)
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item() * grad_accum_steps:.4f} Acc: {acc:.4f}")
    
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'accuracy': total_correct / max(1, total_tokens),
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model."""
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            emotion = batch['emotion'].to(device)
            
            logits = model(input_ids, emotion)
            
            loss = F.cross_entropy(
                logits.view(-1, model.vocab_size),
                labels.view(-1),
                ignore_index=384,
            )
            
            total_loss += loss.item()
            
            mask = labels != 384
            preds = logits.argmax(dim=-1)
            total_correct += ((preds == labels) & mask).sum().item()
            total_tokens += mask.sum().item()
    
    n_batches = len(dataloader)
    return {
        'val_loss': total_loss / n_batches,
        'val_accuracy': total_correct / max(1, total_tokens),
        'perplexity': math.exp(total_loss / n_batches),
    }


def main(config_path: str):
    """Main training function."""
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("MIDI Generator Transformer Training")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create tokenizer
    tokenizer = MIDITokenizer()
    
    # Create model
    arch = config.get('model', {}).get('architecture', {})
    model = EmotionMIDITransformer(
        vocab_size=arch.get('vocab_size', 388),
        max_seq_len=arch.get('max_seq_length', 1024),
        hidden_dim=arch.get('hidden_dim', 384),
        num_layers=arch.get('num_layers', 8),
        num_heads=arch.get('num_heads', 6),
        ff_mult=arch.get('ff_multiplier', 4.0),
        dropout=arch.get('dropout', 0.1),
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Compile
    if config.get('hardware', {}).get('compile_mode', True):
        try:
            model = torch.compile(model)
            print("Model compiled")
        except (RuntimeError, AttributeError) as e:
            print(f"torch.compile() skipped: {e}")
    
    # Create datasets
    data_cfg = config.get('data', {})
    train_dataset = MIDIDataset(
        data_cfg.get('train_manifest', 'data/manifests/midi_train.jsonl'),
        data_cfg,
        tokenizer,
    )
    val_dataset = MIDIDataset(
        data_cfg.get('val_manifest', 'data/manifests/midi_val.jsonl'),
        data_cfg,
        tokenizer,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.get('batch_size', 64),
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.get('batch_size', 64),
        shuffle=False,
    )
    
    # Optimizer
    optim_cfg = config.get('optim', {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg.get('lr', 3e-4),
        weight_decay=optim_cfg.get('weight_decay', 0.1),
    )
    
    # Scheduler
    sched_cfg = config.get('scheduler', {})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=sched_cfg.get('max_steps', 30000),
    )
    
    # Training loop
    train_cfg = config.get('training', {})
    epochs = train_cfg.get('epochs', 15)
    save_dir = Path('checkpoints/midi_generator')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # AMP setup
    use_amp = train_cfg.get('amp', False)
    scaler = None
    if use_amp and device.type == 'cuda':
        amp_dtype = train_cfg.get('amp_dtype', 'bfloat16')
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
    min_epochs = train_cfg.get('min_epochs', 5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, train_cfg, scaler
        )
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f} PPL: {val_metrics['perplexity']:.2f}")
        
        # Save checkpoint
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            improvement = best_val_loss - val_metrics['val_loss']
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'config': config,
            }, save_dir / 'best.pt')
            print(f"Saved best model (improvement: {improvement:.4f})")
        else:
            patience_counter += 1
        
        # Regular checkpoint (epoch-based)
        if (epoch + 1) % train_cfg.get('save_every', 5) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
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
    
    print("\nTraining Complete!")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='midi_generator_training_config.yaml')
    args = parser.parse_args()
    main(args.config)
