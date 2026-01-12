# Platform Quick Reference

Quick command reference for DAiW Music-Brain on different platforms.

## Installation

### One-Line Install

**macOS:**
```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/yourusername/DAiW-Music-Brain/main/install_macos.sh)"
```

**Linux:**
```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/yourusername/DAiW-Music-Brain/main/install_linux.sh)"
```

**Windows PowerShell:**
```powershell
iex ((New-Object System.Net.WebClient).DownloadString('https://raw.githubusercontent.com/yourusername/DAiW-Music-Brain/main/install_windows.ps1'))
```

### Docker

```bash
# Pull and run
docker pull daiw/music-brain:latest
docker run --rm -it daiw/music-brain:latest daiw --help

# With docker-compose
git clone https://github.com/yourusername/DAiW-Music-Brain.git
cd DAiW-Music-Brain
docker-compose up -d
```

### Conda

```bash
conda env create -f environment.yml
conda activate daiw-music-brain
```

## Common Commands

### CLI

```bash
# Show version
daiw --version

# Get help
daiw --help

# Extract groove from MIDI
daiw extract drums.mid -o groove.json

# Apply groove to MIDI
daiw apply --genre funk track.mid -o funky_track.mid

# Analyze chord progression
daiw diagnose "F-C-Am-Dm" --key "D minor"

# Generate reharmonization
daiw reharm "F-C-Am-Dm" --style jazz

# Work with intent system
daiw intent new --title "My Song"
daiw intent process intent.json -o output.mid
daiw intent suggest grief
```

### Python API

```python
# Import
from music_brain.groove import GrooveApplicator
from music_brain.structure import Chord, diagnose_progression
from music_brain.session import CompleteSongIntent, process_intent

# Extract and apply groove
applicator = GrooveApplicator()
funk_groove = applicator.get_genre_template('funk')
applicator.apply_groove('input.mid', 'output.mid', funk_groove)

# Analyze chords
result = diagnose_progression("F-C-Am-Dm", key="D minor")
print(result.emotional_character)

# Process intent
intent = CompleteSongIntent(...)
result = process_intent(intent)
```

## Platform-Specific

### macOS

```bash
# Install dependencies
brew install python@3.11 portaudio libsndfile

# Add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc

# Allow unsigned app
xattr -d com.apple.quarantine /path/to/DAiW.app
```

### Windows

```powershell
# Install Python
winget install Python.Python.3.12

# Enable long paths
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Add to PATH
$env:PATH += ";$env:LOCALAPPDATA\Programs\Python\Python312\Scripts"
```

### Linux

```bash
# Ubuntu/Debian - Install dependencies
sudo apt-get install python3 python3-pip libasound2-dev portaudio19-dev libsndfile1-dev

# Fedora - Install dependencies
sudo dnf install python3 python3-pip alsa-lib-devel portaudio-devel libsndfile-devel

# Add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# MIDI permissions
sudo usermod -a -G audio $USER
```

## DAW Integration

### Logic Pro

```bash
# Generate and import
daiw intent process intent.json -o logic_track.mid
# File → Import → MIDI File in Logic Pro
```

### Ableton Live

```bash
# Generate MIDI
daiw intent process intent.json -o ableton_track.mid
# Drag into Ableton Live arrangement view
```

### Generic DAW

```bash
# Export standard MIDI file
daiw intent process intent.json -o track.mid --format 1 --ppq 480
```

## Troubleshooting

### Import Error

```bash
# Reinstall
pip install --force-reinstall -e .

# Check installation
pip show daiw
```

### Command Not Found

```bash
# macOS/Linux
export PATH="$HOME/.local/bin:$PATH"

# Windows
$env:PATH += ";$env:LOCALAPPDATA\Programs\Python\Python312\Scripts"
```

### Audio Library Error

```bash
# macOS
brew install portaudio libsndfile
pip install --force-reinstall soundfile librosa

# Ubuntu/Debian
sudo apt-get install portaudio19-dev libsndfile1-dev
pip install --force-reinstall soundfile librosa

# Fedora
sudo dnf install portaudio-devel libsndfile-devel
pip install --force-reinstall soundfile librosa
```

## Development

### Setup Dev Environment

```bash
# Clone repo
git clone https://github.com/yourusername/DAiW-Music-Brain.git
cd DAiW-Music-Brain

# Install in editable mode
pip install -e ".[dev,all]"

# Run tests
pytest tests_music-brain/ -v
```

### Docker Development

```bash
# Start dev container
docker-compose run --rm daiw-dev

# Inside container
pip install -e ".[all]"
pytest tests_music-brain/
```

## Resources

- **Documentation**: https://github.com/yourusername/DAiW-Music-Brain#readme
- **Installation Guide**: [INSTALL.md](INSTALL.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **DAW Integration**: [DAW_INTEGRATION.md](DAW_INTEGRATION.md)
- **Issues**: https://github.com/yourusername/DAiW-Music-Brain/issues
