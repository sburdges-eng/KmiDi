# KmiDi GUI Manual

> Complete guide to all KmiDi graphical interfaces

**Version**: 1.0.0

---

## Table of Contents

1. [Overview](#overview)
2. [Streamlit Web App](#streamlit-web-app)
3. [Tauri Desktop App](#tauri-desktop-app)
4. [Qt Desktop App](#qt-desktop-app)
5. [Common Features](#common-features)
6. [Keyboard Shortcuts](#keyboard-shortcuts)
7. [Troubleshooting](#troubleshooting)

---

## Overview

KmiDi provides three GUI options:

| Interface | Platform | Best For |
|-----------|----------|----------|
| **Streamlit** | Web Browser | Quick demos, cloud deployment, sharing |
| **Tauri** | macOS, Windows, Linux | Full desktop app, production use |
| **Qt** | macOS, Windows, Linux | Power users, DAW integration |

---

## Streamlit Web App

The Streamlit app provides a beautiful, accessible web interface for emotion-driven music generation.

### Starting the App

```bash
# From project root
streamlit run streamlit_app.py

# Or with specific port
streamlit run streamlit_app.py --server.port 8501
```

**Default URL**: http://localhost:8501

### Interface Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  🎵 KmiDi - Emotion-Driven Music Generation                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌───────────────────────────────────────────┐│
│  │  Sidebar    │  │                                           ││
│  │             │  │        Main Content Area                  ││
│  │ • Mode      │  │                                           ││
│  │ • Settings  │  │   Emotion Input                           ││
│  │ • History   │  │   ┌─────────────────────────────────┐    ││
│  │             │  │   │ Describe your emotional intent │    ││
│  │             │  │   └─────────────────────────────────┘    ││
│  │             │  │                                           ││
│  │             │  │   [Generate] [Clear]                      ││
│  │             │  │                                           ││
│  │             │  │   Results Display                         ││
│  │             │  │   • Detected Affect                       ││
│  │             │  │   • Generated Chords                      ││
│  │             │  │   • Musical Parameters                    ││
│  │             │  │                                           ││
│  └─────────────┘  └───────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Features

#### Emotion Input

1. **Free-Form Text**: Enter any emotional description
   - "grief hidden as love"
   - "anxious anticipation"
   - "peaceful morning"

2. **Emotional Presets**: Quick selection buttons for common emotions

3. **Advanced Settings** (in sidebar):
   - Motivation Scale (1-10)
   - Chaos Tolerance (0-100%)
   - Output Format (MIDI/Audio)

#### Generation Results

After clicking **Generate**, you'll see:

- **Detected Affect**: Primary and secondary emotions with confidence
- **Musical Plan**:
  - Key and Mode
  - Tempo (BPM)
  - Chord Progression
  - Structure (bars)
- **Download Button**: Export generated MIDI

#### Session History

The sidebar shows previous generations, allowing you to:
- View past results
- Regenerate with modifications
- Compare different approaches

### API Mode

Enable API mode to connect to a running KmiDi API server:

```bash
# Set environment variable before running
export KMIDI_USE_API=true
export KMIDI_API_URL=http://localhost:8000

streamlit run streamlit_app.py
```

### Cloud Deployment

Deploy to Streamlit Cloud:

1. Push code to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Deploy `streamlit_app.py`

---

## Tauri Desktop App

The Tauri app provides a native desktop experience with full system integration.

### Installation

#### Pre-built Releases

Download from the releases page:
- **macOS**: `KmiDi_0.1.0_aarch64.dmg` or `KmiDi_0.1.0_x64.dmg`
- **Windows**: `KmiDi_0.1.0_x64.msi`
- **Linux**: `KmiDi_0.1.0_amd64.AppImage`

#### Build from Source

```bash
# Prerequisites
# Install Rust: https://rustup.rs
# Install Node.js: https://nodejs.org

# Build
cd source/frontend
npm install
npm run tauri build

# Output location
ls src-tauri/target/release/bundle/
```

### Interface Layout

```
┌──────────────────────────────────────────────────────────────────┐
│ KmiDi                                    ─ □ ×  │
├──────────────────────────────────────────────────────────────────┤
│ File  Edit  View  Tools  Help                                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Emotion Input                           │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ What are you feeling?                                │ │ │
│  │  │                                                      │ │ │
│  │  │                                                      │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  │                                                            │ │
│  │  ┌──────────┐  ┌───────────┐  ┌───────────┐               │ │
│  │  │ Generate │  │  Preview  │  │  Export   │               │ │
│  │  └──────────┘  └───────────┘  └───────────┘               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────┐  ┌────────────────────────────────┐│
│  │   Technical Parameters  │  │         Results                ││
│  │                         │  │                                ││
│  │   Key:    [C      ▼]   │  │   Chord Progression:           ││
│  │   Mode:   [Major  ▼]   │  │   Cm - Ab - Fm - Cm            ││
│  │   BPM:    [72     ▼]   │  │                                ││
│  │   Genre:  [Acoustic▼]  │  │   Mode: Aeolian                ││
│  │                         │  │   Tempo: 70 BPM                ││
│  └─────────────────────────┘  │   Length: 32 bars              ││
│                               └────────────────────────────────┘│
├──────────────────────────────────────────────────────────────────┤
│ Status: Ready                                    API: Connected  │
└──────────────────────────────────────────────────────────────────┘
```

### Features

#### Menu Bar

**File**
- New Project (⌘/Ctrl+N)
- Open Project (⌘/Ctrl+O)
- Save Project (⌘/Ctrl+S)
- Export MIDI (⌘/Ctrl+E)
- Export Audio
- Exit

**Edit**
- Undo (⌘/Ctrl+Z)
- Redo (⌘/Ctrl+Shift+Z)
- Preferences

**View**
- Show/Hide Sidebar
- Theme (Light/Dark)
- Full Screen

**Tools**
- Run Benchmark
- Clear Cache
- Check for Updates

**Help**
- Documentation
- Keyboard Shortcuts
- About

#### Workflow

1. **Enter Emotion**: Type your emotional intent
2. **Adjust Parameters** (optional): Key, mode, BPM, genre
3. **Generate**: Click Generate or press ⌘/Ctrl+G
4. **Preview**: Listen to generated music
5. **Export**: Save as MIDI or audio

#### System Integration

- **Notifications**: Desktop notifications for long operations
- **File Associations**: Open `.mid` files directly
- **Keyboard Shortcuts**: Full keyboard navigation
- **Auto-updates**: Check for updates automatically

### Configuration

Edit `src-tauri/tauri.conf.json` for:
- Window size and position
- API endpoint configuration
- Theme preferences

---

## Qt Desktop App

The Qt app provides a professional-grade interface for power users.

### Starting the App

```bash
# Install dependencies
pip install PySide6>=6.5.0

# Run
python kmidi_gui/main.py
```

### Interface Layout

Similar to Tauri but with:
- Native Qt look and feel
- More advanced customization
- Plugin system support (planned)

### Features

#### AI Assistant Dock (Coming Soon)

- Interactive help
- Music theory guidance
- Composition suggestions

#### Log Viewer Dock

- Real-time operation logs
- Debug information
- Performance metrics

### Customization

#### Themes

Edit `kmidi_gui/themes/` to customize:
- Colors
- Fonts
- Spacing
- Icons

#### Layouts

Save and restore custom layouts via View menu.

---

## Common Features

### Emotion-to-Music Generation

All interfaces support the core workflow:

1. **Input Emotion**: Natural language description
2. **Analysis**: AI detects affect and intensity
3. **Planning**: Musical parameters are generated
4. **Output**: MIDI/Audio file creation

### Emotional Presets

Quick-select common emotions:

| Preset | Description |
|--------|-------------|
| Calm | Peaceful, relaxed |
| Grief | Sad, mourning |
| Joy | Happy, celebratory |
| Anger | Intense, aggressive |
| Nostalgia | Wistful, remembering |
| Hope | Optimistic, forward |
| Fear | Anxious, tense |
| Love | Tender, warm |

### Technical Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| Key | C, C#, D, ... B | Musical key |
| Mode | Major, Minor, Dorian, ... | Modal quality |
| BPM | 40-200 | Tempo |
| Genre | Various | Style template |

---

## Keyboard Shortcuts

### Universal Shortcuts

| Shortcut | Action |
|----------|--------|
| ⌘/Ctrl + N | New Project |
| ⌘/Ctrl + O | Open Project |
| ⌘/Ctrl + S | Save Project |
| ⌘/Ctrl + G | Generate |
| ⌘/Ctrl + P | Preview |
| ⌘/Ctrl + E | Export |
| ⌘/Ctrl + Z | Undo |
| ⌘/Ctrl + Shift + Z | Redo |
| F1 | Help |
| Esc | Close Dialog |

### Streamlit Shortcuts

| Shortcut | Action |
|----------|--------|
| R | Rerun app |
| C | Clear cache |

---

## Troubleshooting

### Streamlit Won't Start

```bash
# Check Streamlit installed
pip show streamlit

# Install if missing
pip install streamlit>=1.52.0

# Try with verbose logging
streamlit run streamlit_app.py --logger.level debug
```

### Tauri App Won't Build

```bash
# Check Rust installation
rustc --version

# Check Node.js
node --version

# Clear build cache
cd src-tauri
cargo clean
npm run tauri build
```

### Qt App Crashes

```bash
# Check PySide6
pip show PySide6

# Reinstall
pip install --force-reinstall PySide6>=6.5.0

# Run with debugging
python kmidi_gui/main.py --debug
```

### API Connection Issues

All GUIs can connect to the API server. If connection fails:

```bash
# Verify API is running
curl http://localhost:8000/health

# Start API server
python -m uvicorn api.main:app --port 8000

# Check firewall/network
nc -zv localhost 8000
```

### Generation Takes Too Long

- Reduce complexity settings
- Use shorter bar counts
- Check CPU usage
- Consider using API mode for heavy processing

---

## Best Practices

1. **Start Simple**: Begin with basic emotions like "calm" or "sad"
2. **Experiment**: Try different parameter combinations
3. **Save Often**: Don't lose your work
4. **Use Presets**: Leverage emotional presets for quick results
5. **Preview First**: Listen before exporting
6. **Iterate**: Refine based on results

---

## See Also

- [API Reference](../api/API_REFERENCE.md)
- [CLI Guide](../cli/CLI_GUIDE.md)
- [Quick Start Guide](../QUICKSTART_GUIDE.md)
- [Deployment Guide](../deployment/DEPLOYMENT_GUIDE.md)

---

*Last Updated: 2026-01-11*
