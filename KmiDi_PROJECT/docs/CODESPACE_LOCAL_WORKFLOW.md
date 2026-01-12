# Codespace ↔ Local Mac Training Workflow

## Overview

```
┌─────────────────────┐         ┌─────────────────────┐
│   GitHub Codespace  │         │     Local Mac       │
│  - Edit code        │  git    │  - Run training     │
│  - Run tests        │ ◄─────► │  - Access datasets  │
│  - Quick prototypes │  push   │  - Export models    │
└─────────────────────┘  pull   └─────────────────────┘
                                         │
                                         ▼
                                ┌─────────────────────┐
                                │  /Volumes/sbdrive   │
                                │  - M4Singer (11GB)  │
                                │  - Lakh MIDI (7GB)  │
                                │  - Genius (650MB)   │
                                │  - venv             │
                                └─────────────────────┘
```

## Quick Start

### On Local Mac (Training)
```bash
cd "/Volumes/sbdrive/My Mac/Desktop/KmiDi-remote"
git pull origin main
./scripts/local_train.sh emotion_recognizer
git add models/ && git commit -m "Trained emotion_recognizer" && git push
```

### In Codespace (Development)
```bash
git pull  # Get trained models
# Edit code, run tests
git push  # Send changes to local Mac
```

## Dataset Summary

| Dataset | Size | Files | Use For |
|---------|------|-------|---------|
| M4Singer | 11GB | 20,896 WAV+MIDI | EmotionRecognizer, DynamicsEngine |
| Lakh MIDI | 7.6GB | 178,561 MIDI | HarmonyPredictor, MelodyTransformer |
| Genius Lyrics | 649MB | 480K songs | Lyric alignment (future) |
| MoodyLyrics | 640KB | Labels | Emotion ground truth |

## Training Commands

```bash
./scripts/local_train.sh --list              # Show models
./scripts/local_train.sh emotion_recognizer  # Train one
./scripts/local_train.sh                     # Train all
```

## Environment Setup (One-time)

```bash
# Create venv on external SSD
/opt/homebrew/bin/python3.12 -m venv /Volumes/sbdrive/venv
source /Volumes/sbdrive/venv/bin/activate
pip install torch torchvision torchaudio
pip install -e ".[dev]"
```

## Troubleshooting

**sbdrive not mounted**: Connect the external SSD

**Permission denied**:
```bash
sudo chown -R "$USER" /Volumes/sbdrive
```

**Import errors**: Script sets PYTHONPATH automatically
