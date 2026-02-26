"""
Full Audio-JEPA training loop.
Usage: from repo root, PYTHONPATH=. python Training/train_audio_jepa.py
Or: python -m Training.train_audio_jepa (from repo root with PYTHONPATH=.)
"""

import sys
import os

# Ensure repo root and Training are on path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TRAINING = os.path.join(_REPO_ROOT, "Training")
for p in (_REPO_ROOT, _TRAINING):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from torch.utils.data import DataLoader

from Brain.audio_jepa.model import AudioJEPAEncoder, LatentPredictor
from Brain.audio_jepa.masking import mask_latents
from dataloader_audio import AudioDataset
from training.storage_paths import AUDIO_DATASETS_DIR, ensure_storage_dirs


def main():
    ensure_storage_dirs()

    # Discover all audio files under Datasets/audio (recursive: GTZAN_kaggle, etc.)
    file_list = []
    audio_root = AUDIO_DATASETS_DIR
    if os.path.isdir(audio_root):
        for root, _, files in os.walk(audio_root):
            for f in files:
                if f.endswith((".wav", ".mp3", ".flac")):
                    file_list.append(os.path.join(root, f))
    if not file_list:
        # Create dummy data for dry run
        print("No audio files in", audio_root, "; using dummy data for one epoch.")
        file_list = [None]  # will trigger dummy path in dataset below

    # Allow dummy run: single random tensor batch
    if file_list == [None]:
        dataset = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = AudioJEPAEncoder().to(device)
        predictor = LatentPredictor().to(device)
        opt = torch.optim.AdamW(
            list(encoder.parameters()) + list(predictor.parameters()),
            lr=3e-4,
        )
        dummy = torch.randn(2, 1, 128, 512, device=device)
        z = encoder(dummy)
        masked, mask = mask_latents(z)
        pred = predictor(masked)
        loss = ((pred - z) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("Epoch 0 (dummy) Loss", loss.item())
        return

    num_epochs = int(os.environ.get("AUDIO_JEPA_EPOCHS", "100"))
    print("Audio-JEPA training: {} files, {} epochs".format(len(file_list), num_epochs), flush=True)
    dataset = AudioDataset(file_list)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = AudioJEPAEncoder().to(device)
    predictor = LatentPredictor().to(device)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=3e-4,
    )
    for epoch in range(num_epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            z = encoder(batch)
            masked, mask = mask_latents(z)
            pred = predictor(masked)
            loss = ((pred - z) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        print("Epoch", epoch, "Loss", avg, flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
