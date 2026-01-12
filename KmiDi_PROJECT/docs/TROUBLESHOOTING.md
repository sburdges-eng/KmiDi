# Platform-Specific Troubleshooting

## macOS

### Common Issues

#### 1. "command not found: daiw"

**Cause:** `~/.local/bin` not in PATH

**Solution:**
```bash
# For zsh (default on macOS Catalina+)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# For bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bash_profile
source ~/.bash_profile

# Verify
daiw --version
```

#### 2. Apple Silicon (M1/M2/M3) Compatibility

**Symptom:** Package installation fails with architecture errors

**Solution:**
```bash
# Install Rosetta 2 if needed
softwareupdate --install-rosetta

# Use native ARM Python
arch -arm64 pip install -e .

# Or force x86_64 if needed
arch -x86_64 pip install -e .
```

#### 3. Audio Library Issues

**Symptom:** `ImportError: libsndfile.dylib not found`

**Solution:**
```bash
# Install via Homebrew
brew install portaudio libsndfile

# Reinstall Python packages
pip install --force-reinstall soundfile librosa
```

#### 4. Code Signing Warnings

**Symptom:** "App cannot be opened because the developer cannot be verified"

**Solution:**
```bash
# Allow the app
xattr -d com.apple.quarantine /path/to/DAiW.app

# Or via System Preferences
# Security & Privacy → General → "Open Anyway"
```

#### 5. Microphone/Audio Permissions

**Symptom:** Audio analysis fails silently

**Solution:**
- System Preferences → Security & Privacy → Privacy → Microphone
- Enable Terminal (or your Python IDE)

---

## Windows

### Common Issues

#### 1. "Python was not found"

**Cause:** Python not installed or not in PATH

**Solution:**
```powershell
# Install via Windows Package Manager
winget install Python.Python.3.12

# Or download from python.org and check "Add to PATH"

# Verify
python --version
```

#### 2. PowerShell Execution Policy

**Symptom:** Cannot run install script

**Solution:**
```powershell
# Allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run installer
.\install_windows.ps1
```

#### 3. Visual C++ Redistributables Missing

**Symptom:** DLL load errors when importing packages

**Solution:**
```powershell
# Download and install from Microsoft:
# https://aka.ms/vs/17/release/vc_redist.x64.exe

# Or via winget
winget install Microsoft.VCRedist.2015+.x64
```

#### 4. Long Path Issues

**Symptom:** File path errors during installation

**Solution:**
```powershell
# Enable long paths (requires admin)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Restart computer
```

#### 5. MIDI Device Access

**Symptom:** MIDI devices not detected

**Solution:**
```powershell
# Check Device Manager
devmgmt.msc

# Look for "Sound, video and game controllers"
# Update drivers if needed

# List MIDI devices
python -c "import mido; print(mido.get_input_names())"
```

#### 6. Antivirus False Positives

**Symptom:** Executable blocked or quarantined

**Solution:**
- Add exception in Windows Defender
- Settings → Update & Security → Windows Security → Virus & threat protection
- Manage settings → Add exclusion → Folder
- Add DAiW installation directory

#### 7. Module Not Found After Installation

**Symptom:** `ModuleNotFoundError: No module named 'music_brain'`

**Solution:**
```powershell
# Verify installation
pip show daiw

# Reinstall in editable mode
pip install -e .

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

---

## Linux

### Ubuntu/Debian

#### 1. Missing System Libraries

**Symptom:** Package installation fails during compilation

**Solution:**
```bash
# Install development libraries
sudo apt-get update
sudo apt-get install -y \
  python3-dev \
  python3-pip \
  python3-venv \
  libasound2-dev \
  portaudio19-dev \
  libsndfile1-dev \
  build-essential

# Reinstall package
pip install -e .
```

#### 2. Permission Denied for MIDI

**Symptom:** Cannot access MIDI devices

**Solution:**
```bash
# Add user to audio group
sudo usermod -a -G audio $USER

# Reload groups (or logout/login)
newgrp audio

# Verify
groups | grep audio
```

#### 3. PulseAudio/JACK Conflicts

**Symptom:** Audio playback issues

**Solution:**
```bash
# For PulseAudio
pulseaudio --check
pulseaudio -D

# For JACK (pro audio)
sudo apt-get install jackd2
# Configure JACK with qjackctl

# Use JACK bridge if needed
sudo apt-get install pulseaudio-module-jack
```

### Fedora/RHEL/CentOS

#### 1. Missing Development Tools

**Solution:**
```bash
# Install development group
sudo dnf groupinstall "Development Tools"

# Install specific libraries
sudo dnf install -y \
  python3-devel \
  python3-pip \
  alsa-lib-devel \
  portaudio-devel \
  libsndfile-devel
```

#### 2. SELinux Permissions

**Symptom:** Permission denied errors despite correct file permissions

**Solution:**
```bash
# Check SELinux status
sestatus

# Temporarily disable (for testing)
sudo setenforce 0

# Or create policy for DAiW
# (consult SELinux documentation)
```

### Arch Linux

#### 1. Package Installation

**Solution:**
```bash
# Install dependencies
sudo pacman -S python python-pip alsa-lib portaudio libsndfile

# AUR installation (when available)
yay -S daiw-music-brain
# or
paru -S daiw-music-brain
```

#### 2. Python Environment

**Solution:**
```bash
# Use virtual environment
python -m venv ~/.venvs/daiw
source ~/.venvs/daiw/bin/activate
pip install -e .
```

### AppImage Issues

#### 1. FUSE Not Installed

**Symptom:** AppImage won't run

**Solution:**
```bash
# Ubuntu/Debian
sudo apt install libfuse2

# Fedora
sudo dnf install fuse

# Arch
sudo pacman -S fuse2

# Make executable
chmod +x DAiW-Music-Brain.AppImage
./DAiW-Music-Brain.AppImage
```

#### 2. Extract and Run

**Alternative:**
```bash
# Extract AppImage
./DAiW-Music-Brain.AppImage --appimage-extract

# Run directly
./squashfs-root/AppRun
```

---

## Docker

### Common Issues

#### 1. Permission Denied

**Solution:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Logout/login or:
newgrp docker

# Verify
docker run hello-world
```

#### 2. Container Can't Access Host Files

**Solution:**
```bash
# Use volume mounts
docker run -v $(pwd)/data:/app/data daiw-music-brain:latest

# Check permissions
ls -la data/
```

#### 3. Audio Not Working in Container

**Cause:** Docker doesn't support audio by default

**Solution (Linux):**
```bash
docker run --device /dev/snd \
  -e PULSE_SERVER=unix:/run/user/1000/pulse/native \
  -v /run/user/1000/pulse:/run/user/1000/pulse \
  daiw-music-brain:latest
```

**Solution (macOS/Windows):**
Audio passthrough not supported - use CLI tools only

---

## IDE-Specific Issues

### VSCode

#### 1. Python Interpreter Not Found

**Solution:**
- Cmd/Ctrl+Shift+P → "Python: Select Interpreter"
- Choose interpreter with DAiW installed
- Or create `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "/path/to/python"
}
```

#### 2. Import Errors Despite Installation

**Solution:**
- Reload window: Cmd/Ctrl+Shift+P → "Developer: Reload Window"
- Restart Python language server
- Clear Python cache: delete `.vscode` folder

### PyCharm

#### 1. Module Not Found

**Solution:**
- File → Settings → Project → Python Interpreter
- Ensure correct interpreter selected
- Install package in selected environment

#### 2. Tests Not Discovered

**Solution:**
- Settings → Tools → Python Integrated Tools
- Default test runner: pytest
- Regenerate test configuration

---

## General Troubleshooting

### Performance Issues

#### Slow Import Times

**Solution:**
```bash
# Use lazy imports
export PYTHONOPTIMIZE=1

# Or compile to bytecode
python -m compileall music_brain/
```

#### High Memory Usage

**Solution:**
```python
# Reduce librosa cache
import os
os.environ['LIBROSA_CACHE_LEVEL'] = '10'

# Or disable completely
os.environ['LIBROSA_CACHE_DIR'] = '/tmp'
```

### Debugging

#### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from music_brain import *
```

#### Check Installation

```bash
# Verify package
pip show daiw

# Check installed files
pip show -f daiw

# Reinstall
pip install --force-reinstall --no-cache-dir -e .
```

### Getting Help

If issues persist:

1. Check [GitHub Issues](https://github.com/yourusername/DAiW-Music-Brain/issues)
2. Search [Discussions](https://github.com/yourusername/DAiW-Music-Brain/discussions)
3. Create new issue with:
   - OS and version
   - Python version (`python --version`)
   - Complete error message
   - Steps to reproduce
   - Output of `pip show daiw`

### Diagnostic Script

Run this to collect system info:

```python
import sys
import platform

print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")
print(f"Architecture: {platform.machine()}")

try:
    import music_brain
    print(f"DAiW version: {music_brain.__version__}")
except Exception as e:
    print(f"DAiW import error: {e}")

try:
    import mido
    print(f"Mido: {mido.__version__}")
except:
    print("Mido: not installed")

try:
    import numpy
    print(f"NumPy: {numpy.__version__}")
except:
    print("NumPy: not installed")

try:
    import librosa
    print(f"Librosa: {librosa.__version__}")
except:
    print("Librosa: not installed")
```

Save as `diagnostic.py` and run with `python diagnostic.py`
