# Platform-Specific Installation Instructions

## Installation Methods by Platform

### macOS

#### Via Installation Script (Recommended)
```bash
curl -sSL https://raw.githubusercontent.com/yourusername/DAiW-Music-Brain/main/install_macos.sh | bash
```

#### Via Homebrew (Coming Soon)
```bash
# Add tap
brew tap yourusername/daiw

# Install
brew install daiw-music-brain

# Verify installation
daiw --version
```

#### Manual Installation
```bash
# Install dependencies
brew install python@3.11 portaudio libsndfile

# Clone repository
git clone https://github.com/yourusername/DAiW-Music-Brain.git
cd DAiW-Music-Brain

# Install package
pip3 install -e ".[all]"

# Verify
daiw --help
```

### Windows

#### Via Installation Script (Recommended)
1. Download `install_windows.ps1`
2. Open PowerShell as Administrator
3. Run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\install_windows.ps1
```

#### Via Windows Package Manager
```powershell
# Install Python (if needed)
winget install Python.Python.3.12

# Install DAiW (coming soon)
# winget install DAiW.MusicBrain
```

#### Manual Installation
```powershell
# Install Python 3.9+ from python.org

# Clone repository
git clone https://github.com/yourusername/DAiW-Music-Brain.git
cd DAiW-Music-Brain

# Install package
python -m pip install -e ".[all]"

# Verify
daiw --help
```

### Linux

#### Via Installation Script (Recommended)
```bash
# Download and run installer
curl -sSL https://raw.githubusercontent.com/yourusername/DAiW-Music-Brain/main/install_linux.sh | bash
```

#### Ubuntu/Debian
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv \
    libasound2-dev portaudio19-dev libsndfile1-dev

# Clone repository
git clone https://github.com/yourusername/DAiW-Music-Brain.git
cd DAiW-Music-Brain

# Install package
pip3 install --user -e ".[all]"

# Verify
daiw --help
```

#### Fedora/RHEL/CentOS
```bash
# Install dependencies
sudo dnf install python3 python3-pip \
    alsa-lib-devel portaudio-devel libsndfile-devel

# Clone repository
git clone https://github.com/yourusername/DAiW-Music-Brain.git
cd DAiW-Music-Brain

# Install package
pip3 install --user -e ".[all]"

# Verify
daiw --help
```

#### Arch Linux/Manjaro
```bash
# Install dependencies
sudo pacman -S python python-pip alsa-lib portaudio libsndfile

# Install from AUR (coming soon)
# yay -S daiw-music-brain

# Or manual installation
git clone https://github.com/yourusername/DAiW-Music-Brain.git
cd DAiW-Music-Brain
pip install --user -e ".[all]"
```

#### AppImage (Universal Linux)
```bash
# Download AppImage
wget https://github.com/yourusername/DAiW-Music-Brain/releases/latest/download/DAiW-Music-Brain.AppImage

# Make executable
chmod +x DAiW-Music-Brain.AppImage

# Run
./DAiW-Music-Brain.AppImage
```

### Docker

#### Using Pre-built Image
```bash
# Pull image
docker pull daiw/music-brain:latest

# Run CLI
docker run --rm -it daiw/music-brain:latest daiw --help

# Run UI
docker run --rm -p 8501:8501 daiw/music-brain:latest streamlit run app.py
```

#### Using Docker Compose
```bash
# Clone repository
git clone https://github.com/yourusername/DAiW-Music-Brain.git
cd DAiW-Music-Brain

# Start services
docker-compose up -d

# Access UI at http://localhost:8501
```

### Conda/Mamba

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate daiw-music-brain

# Verify
daiw --help
```

## Platform-Specific Notes

### macOS

**Apple Silicon (M1/M2/M3)**
- Native ARM64 support available
- Some audio libraries require Rosetta 2 for compatibility

**Code Signing**
- Unsigned builds may require: System Preferences → Security & Privacy → Allow

**Audio Permissions**
- Grant microphone access when prompted for audio analysis features

### Windows

**Antivirus Warnings**
- PyInstaller executables may trigger false positives
- Add exception for DAiW installation directory

**MIDI Devices**
- Ensure MIDI drivers are installed for your hardware
- Check Device Manager for proper device recognition

**Path Issues**
- If `daiw` command not found, add to PATH:
  ```powershell
  $env:PATH += ";$env:LOCALAPPDATA\Programs\Python\Python312\Scripts"
  ```

### Linux

**Audio Server**
- PulseAudio or PipeWire required for audio features
- JACK support available for pro audio setups

**MIDI Permissions**
```bash
# Add user to audio group
sudo usermod -a -G audio $USER

# Reload groups (or logout/login)
newgrp audio
```

**AppImage FUSE Requirement**
```bash
# Ubuntu/Debian
sudo apt install libfuse2

# Fedora
sudo dnf install fuse

# Arch
sudo pacman -S fuse2
```

## IDE Setup

### VSCode

1. Install Python extension
2. Open DAiW repository folder
3. Select Python interpreter: `Cmd/Ctrl+Shift+P` → "Python: Select Interpreter"
4. Install recommended extensions:
   - Python
   - Pylance
   - Black Formatter
   - Python Test Explorer

**settings.json**
```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "editor.formatOnSave": true
}
```

### PyCharm

1. Open DAiW repository
2. Configure Python interpreter: Settings → Project → Python Interpreter
3. Enable pytest: Settings → Tools → Python Integrated Tools → Testing → pytest
4. Configure Black formatter: Settings → Tools → Black → Enable

### Cursor

Same as VSCode - uses VSCode configuration system.

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'music_brain'`

**Solution**:
```bash
# Reinstall in editable mode
pip install -e .

# Or verify installation
pip show daiw
```

### Audio Library Errors

**macOS**:
```bash
brew install portaudio libsndfile
pip install --force-reinstall soundfile librosa
```

**Linux**:
```bash
sudo apt-get install portaudio19-dev libsndfile1-dev
pip install --force-reinstall soundfile librosa
```

**Windows**:
- Download pre-built wheels from: https://www.lfd.uci.edu/~gohlke/pythonlibs/
- Install: `pip install downloaded_wheel.whl`

### MIDI Device Not Found

1. Check device connection
2. List available devices:
   ```python
   python -c "import mido; print(mido.get_input_names())"
   ```
3. Verify drivers installed (Windows)
4. Check permissions (Linux - see MIDI Permissions above)

### Performance Issues

**Enable SIMD optimizations** (if building from source):
```bash
pip install -e ".[all]" --config-settings="--build-option=--enable-simd"
```

**Increase buffer size** for audio processing:
```python
import os
os.environ['LIBROSA_CACHE_LEVEL'] = '50'
```

## Version-Specific Notes

### Python 3.9
- Minimum supported version
- All core features available

### Python 3.10
- Full compatibility
- Type hints improvements

### Python 3.11
- Recommended version
- ~25% faster than 3.9
- Better error messages

### Python 3.12
- Full support
- Performance optimizations
- Some optional dependencies may lag

### Python 3.13
- Experimental support
- Test thoroughly before production use

## Getting Help

- GitHub Issues: https://github.com/yourusername/DAiW-Music-Brain/issues
- Documentation: https://github.com/yourusername/DAiW-Music-Brain#readme
- Discussions: https://github.com/yourusername/DAiW-Music-Brain/discussions
