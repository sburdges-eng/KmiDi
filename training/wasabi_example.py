"""
WASABI Dataset Integration Example

This example shows how to use the WASABI dataset for emotion-conditioned music generation.
"""

from pathlib import Path
from torch.utils.data import DataLoader

from training.wasabi_dataset import WasabiDataset

# Initialize WASABI dataset
dataset = WasabiDataset(
    manifest_path="data/wasabi/processed/train.jsonl",
    emotion_filter=["happy", "sad", "angry", "joy"],  # Filter by emotions
    year_range=(2000, 2024),  # Filter by year range
    require_lyrics=True,  # Only include songs with lyrics
    max_samples=10000,  # Limit to 10k samples for quick testing
)

print(f"Loaded {len(dataset)} samples from WASABI")

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)

# Iterate through batches
for batch in dataloader:
    emotion_embeddings = batch["emotion_embedding"]  # Shape: (batch, emotion_dim)
    chord_embeddings = batch["chord_embedding"]  # Shape: (batch, chord_dim)
    lyrics = batch["lyrics"]  # List of lyrics strings
    titles = batch["title"]  # List of song titles
    artists = batch["artist"]  # List of artists
    
    # Use emotion_embeddings and chord_embeddings for training
    # emotion_embeddings can condition music generation models
    # chord_embeddings provide harmonic context
    
    print(f"Batch emotion embeddings shape: {emotion_embeddings.shape}")
    print(f"Batch chord embeddings shape: {chord_embeddings.shape}")
    print(f"Sample song: '{titles[0]}' by {artists[0]}")
    break

# Access full sample information
sample_info = dataset.get_sample_info(0)
print(f"\nSample info:")
print(f"  Title: {sample_info.title}")
print(f"  Artist: {sample_info.artist}")
print(f"  Year: {sample_info.year}")
print(f"  Genre: {sample_info.genre}")
print(f"  Emotions: {sample_info.lyrics_emotion}")
print(f"  Chords: {sample_info.chords[:5]}...")  # First 5 chords
