# WASABI Dataset Integration

## Overview

WASABI (Web Audio Semantic Annotation and Browsing Interface) is now integrated into the KmiDi training pipeline. WASABI provides:

- **2 million songs** with rich metadata
- **Lyrics** with emotional content analysis
- **Audio features**: chords, tempo, key, mode
- **Cultural metadata**: artists, albums, genres, release dates
- **Emotion labels** extracted from lyrics

## Files Created

1. **`training/wasabi_dataset.py`** - PyTorch Dataset class for WASABI
   - Loads from JSON/JSONL manifest files
   - Filters by emotion, genre, year, artist
   - Computes emotion and chord embeddings
   - Integrates with emotion-conditioned training

2. **`training/wasabi_example.py`** - Usage example

3. **Updated `scripts/prepare_datasets.py`** - Added WASABI download/preprocessing support

4. **Updated training configs**:
   - `training/cuda_session/midi_generator_training_config.yaml`
   - `training/integrated_training_config.yaml`

## Usage

### 1. Download and Prepare WASABI Dataset

```bash
# Download WASABI dataset
python scripts/prepare_datasets.py --dataset wasabi --download

# Preprocess into training format
python scripts/prepare_datasets.py --dataset wasabi --preprocess
```

### 2. Use in Training Scripts

```python
from training.wasabi_dataset import WasabiDataset
from torch.utils.data import DataLoader

# Initialize dataset
dataset = WasabiDataset(
    manifest_path="data/wasabi/processed/train.jsonl",
    emotion_filter=["happy", "sad", "angry", "joy"],
    year_range=(2000, 2024),
    require_lyrics=True,
)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Use in training loop
for batch in dataloader:
    emotion_emb = batch["emotion_embedding"]  # (batch, emotion_dim)
    chord_emb = batch["chord_embedding"]       # (batch, chord_dim)
    lyrics = batch["lyrics"]                   # List[str]
    # ... use for training
```

### 3. Dataset Features

Each sample provides:
- `emotion_embedding`: 16-dimensional emotion vector (happy, sad, angry, etc.)
- `chord_embedding`: 64-dimensional chord progression encoding
- `lyrics`: Full lyrics text
- `title`, `artist`, `year`, `genre`: Metadata
- `tempo`, `key`, `chords`: Audio features

## Integration with Training

WASABI is configured in training configs for:
- **Emotion-conditioned MIDI generation**: Use emotion embeddings to condition music generation
- **Lyrics-to-music**: Train models that generate music from lyrics
- **Multi-modal emotion learning**: Combine lyrics emotions with audio features

## Dataset Access

WASABI can be accessed via:
- **REST API**: https://wasabi.i3s.unice.fr/api/
- **SPARQL endpoint**: https://wasabi.i3s.unice.fr/sparql
- **GitHub**: https://github.com/micbuffa/WasabiDataset

The dataset preparation script attempts to download from GitHub, but manual download may be required.

## Next Steps

1. Download WASABI dataset using the preparation script
2. Preprocess into JSONL format
3. Use in emotion-conditioned training pipelines
4. Combine with other datasets (Lakh MIDI, MAESTRO) for comprehensive training
