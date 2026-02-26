"""
Chord-JEPA training loop: self-supervised masking + prediction.
Usage: PYTHONPATH=. python training/train_chord_jepa.py
"""

import sys
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TRAINING = os.path.dirname(os.path.abspath(__file__))
for p in (_REPO_ROOT, _TRAINING):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Brain.chord_jepa.model import ChordJEPA
from Brain.audio_jepa.masking import mask_latents
from chord_dataset import ChordDataset, CHORD_SEQ_LEN, NUM_CHORDS
from training.storage_paths import MIDI_DATASETS_DIR, ensure_storage_dirs


def main():
    ensure_storage_dirs()

    # Discover all MIDI files under Datasets/midi (recursive: MIDI_kaggle, etc.)
    midi_files = []
    midi_dir = MIDI_DATASETS_DIR
    if os.path.isdir(midi_dir):
        for root, _, files in os.walk(midi_dir):
            for f in files:
                if f.lower().endswith((".mid", ".midi")):
                    midi_files.append(os.path.join(root, f))
    if not midi_files:
        midi_files = [None] * 32  # dummy 32 samples

    num_epochs = int(os.environ.get("CHORD_JEPA_EPOCHS", "100"))
    print("Chord-JEPA training: {} files, {} epochs".format(len(midi_files), num_epochs), flush=True)
    dataset = ChordDataset(midi_files)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_model = 256
    embedding = nn.Embedding(NUM_CHORDS, d_model).to(device)
    model = ChordJEPA(d_model=d_model, num_classes=NUM_CHORDS).to(device)
    opt = torch.optim.AdamW(
        list(embedding.parameters()) + list(model.parameters()),
        lr=2e-4,
    )

    for epoch in range(num_epochs):
        total_loss = 0.0
        n_batches = 0
        for chords in loader:
            chords = chords.to(device)  # (B, seq)
            z = embedding(chords)  # (B, seq, d_model)
            masked, mask = mask_latents(z)
            pred = model(masked)  # (B, seq, NUM_CHORDS)
            target = chords
            loss = nn.functional.cross_entropy(pred.reshape(-1, NUM_CHORDS), target.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print("Epoch", epoch, "Loss", total_loss / max(n_batches, 1), flush=True)


if __name__ == "__main__":
    main()
