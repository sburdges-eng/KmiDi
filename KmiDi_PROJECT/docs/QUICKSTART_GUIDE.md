# KmiDi Quick Start Guide

> Get up and running with KmiDi in under 10 minutes

---

## 🚀 5-Minute Quick Start

### Option 1: Web App (Easiest)

```bash
# Install and run
pip install streamlit
cd KmiDi_PROJECT
streamlit run streamlit_app.py
```

Open http://localhost:8501 and start creating!

### Option 2: API + CLI

```bash
# Terminal 1: Start API
cd KmiDi_PROJECT
pip install -e .
python -m uvicorn api.main:app --port 8000

# Terminal 2: Use CLI
daiw generate --key C --mode minor --pattern "I-V-vi-IV" -o song.mid
```

### Option 3: Desktop App (Tauri)

```bash
# Download pre-built app from releases, or build:
cd KmiDi_PROJECT/source/frontend
npm install
npm run tauri dev
```

---

## 📋 Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.9+ | `python --version` |
| pip | Latest | `pip --version` |
| Node.js (for Tauri) | 18+ | `node --version` |
| Rust (for Tauri) | Latest | `rustc --version` |

---

## 🔧 Installation

### Core Package

```bash
# Clone repository
git clone https://github.com/your-org/KmiDi.git
cd KmiDi/KmiDi_PROJECT

# Install Python package
pip install -e .

# Verify installation
daiw --version
python -c "import music_brain; print('OK')"
```

### Optional Dependencies

```bash
# Audio processing
pip install librosa soundfile

# Full production setup
pip install -r requirements-production.txt

# GUI dependencies
pip install PySide6  # Qt GUI
pip install streamlit  # Web app
```

---

## 🎵 Your First Generation

### Method 1: Command Line

```bash
# Generate from emotion
daiw intent suggest grief
daiw generate --key F --mode minor --tempo 70 -o grief_song.mid

# Analyze existing MIDI
daiw analyze --chords song.mid
daiw diagnose "C-G-Am-F"
```

### Method 2: Python API

```python
from music_brain.structure.comprehensive_engine import TherapySession

# Create session
session = TherapySession()

# Process emotional intent
session.process_core_input("grief hidden as love")
session.set_scales(motivation=7, chaos=0.5)

# Generate plan
plan = session.generate_plan()
print(f"Key: {plan.root_note} {plan.mode}")
print(f"Tempo: {plan.tempo_bpm} BPM")
print(f"Chords: {' - '.join(plan.chord_symbols)}")
```

### Method 3: REST API

```bash
# Start API server
python -m uvicorn api.main:app --port 8000

# Generate music (in another terminal)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"intent": {"emotional_intent": "calm and peaceful"}}'
```

### Method 4: Web Interface

```bash
streamlit run streamlit_app.py
```

1. Open http://localhost:8501
2. Type: "grief hidden as love"
3. Click **Generate**
4. Download the MIDI file

---

## 🎯 Common Workflows

### Workflow 1: Emotion to MIDI

```bash
# 1. Get suggestions for your emotion
daiw intent suggest grief

# 2. Create intent file
daiw intent new --title "Loss" -o loss.json

# 3. Edit loss.json with your emotions

# 4. Generate
daiw generate --intent-file loss.json -o output.mid
```

### Workflow 2: Humanize Existing MIDI

```bash
# Add human feel
daiw humanize drums.mid --preset lofi_depression -o humanized.mid

# Or apply genre groove
daiw apply --genre dilla drums.mid -o dilla_drums.mid
```

### Workflow 3: Analyze and Improve

```bash
# Diagnose progression
daiw diagnose "F-C-Am-Dm"

# Get jazz reharmonization
daiw reharm "F-C-Am-Dm" --style jazz

# Generate improved version
daiw generate --key F --pattern "Fmaj7-C7-Am7-Dm7" -o improved.mid
```

---

## 📊 Understanding the Output

When you generate music, KmiDi produces:

```
Detected Affect: grief (intensity: 0.67)
Secondary Affect: tenderness

Musical Plan:
  Root: C
  Mode: aeolian (natural minor)
  Tempo: 70 BPM
  Length: 32 bars
  Progression: Cm - Ab - Fm - Cm
  Complexity: 0.5
```

### What the Parameters Mean

| Parameter | Description |
|-----------|-------------|
| **Mode** | Musical scale (aeolian = sad, lydian = dreamy, etc.) |
| **Tempo** | Speed in beats per minute |
| **Length** | Song structure in bars |
| **Complexity** | How "chaotic" the timing/dynamics are (0-1) |

---

## 🎨 Emotional Modes

KmiDi maps emotions to musical modes:

| Emotion | Mode | Character |
|---------|------|-----------|
| Grief | Aeolian | Sad, heavy |
| Rage | Phrygian | Dark, aggressive |
| Awe | Lydian | Ethereal, floating |
| Nostalgia | Dorian | Wistful, bittersweet |
| Defiance | Mixolydian | Rock, rebellious |
| Fear | Phrygian | Tense, unsettling |
| Tenderness | Ionian | Gentle, warm |
| Confusion | Locrian | Unstable, searching |

---

## 🐛 Quick Troubleshooting

### "Command not found: daiw"

```bash
pip install -e .
# Or use: python -m music_brain.cli
```

### "Import error: music_brain"

```bash
cd KmiDi_PROJECT
pip install -e .
```

### "API not responding"

```bash
# Check if API is running
curl http://localhost:8000/health

# Start API
python -m uvicorn api.main:app --port 8000
```

### "Streamlit error"

```bash
pip install streamlit>=1.52.0
streamlit run streamlit_app.py --server.port 8502  # Try different port
```

---

## 📚 Next Steps

### Learn More

- [API Reference](api/API_REFERENCE.md) - Full API documentation
- [CLI Guide](cli/CLI_GUIDE.md) - Complete CLI reference
- [GUI Manual](gui/GUI_MANUAL.md) - All interfaces
- [Deployment Guide](deployment/DEPLOYMENT_GUIDE.md) - Production setup

### Try These Commands

```bash
# List available emotions
curl http://localhost:8000/emotions

# Get teaching on rule-breaking
daiw teach rulebreaking

# Analyze audio
daiw analyze-audio song.wav --all

# List humanization presets
daiw humanize --list-presets
```

### Example Projects

```bash
# 1. Lo-fi grief piece
daiw generate --key Eb --mode aeolian --tempo 65 -o lofi_grief.mid
daiw humanize lofi_grief.mid --preset lofi_depression -o final.mid

# 2. Defiant punk progression
daiw generate --key A --mode mixolydian --tempo 145 -o punk.mid
daiw humanize punk.mid --preset defiant_punk -o final_punk.mid

# 3. Dreamy ambient
daiw generate --key F --mode lydian --tempo 80 -o ambient.mid
```

---

## 🤝 Philosophy

> "The audience doesn't hear 'borrowed from Dorian.' They hear 'that part made me cry.'"

KmiDi follows the principle: **"Interrogate Before Generate"**

The tool doesn't finish art for you—it helps you be braver in expressing your emotions through music.

---

## 🆘 Getting Help

```bash
# CLI help
daiw --help
daiw generate --help

# API documentation
open http://localhost:8000/docs

# Streamlit app has built-in help
```

---

*Happy creating! 🎵*
