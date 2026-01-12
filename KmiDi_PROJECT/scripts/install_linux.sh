#!/bin/bash
#
# DAiW Music-Brain Installer for Linux
# =====================================
# Version: 0.2.0
#
# Supports: Ubuntu/Debian, Fedora/RHEL, Arch Linux

set -e

echo "========================================"
echo "  DAiW Music-Brain"
echo "  Version 0.2.0"
echo "  Installer for Linux"
echo "========================================"
echo ""

# Detect distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
else
    echo "❌ Cannot detect Linux distribution"
    exit 1
fi

echo "Detected distribution: $DISTRO"
echo ""

# Check for Python 3.9+
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo ""
    case $DISTRO in
        ubuntu|debian)
            echo "Install with:"
            echo "  sudo apt-get update"
            echo "  sudo apt-get install python3 python3-pip python3-venv"
            ;;
        fedora|rhel|centos)
            echo "Install with:"
            echo "  sudo dnf install python3 python3-pip"
            ;;
        arch|manjaro)
            echo "Install with:"
            echo "  sudo pacman -S python python-pip"
            ;;
    esac
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python found: $(python3 --version)"

# Check version is 3.9+
MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]); then
    echo "❌ Python 3.9 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

# Install system dependencies for audio support
echo ""
echo "Installing system dependencies..."
case $DISTRO in
    ubuntu|debian)
        sudo apt-get update
        sudo apt-get install -y python3-dev python3-pip python3-venv \
            libasound2-dev portaudio19-dev libsndfile1-dev
        ;;
    fedora|rhel|centos)
        sudo dnf install -y python3-devel python3-pip \
            alsa-lib-devel portaudio-devel libsndfile-devel
        ;;
    arch|manjaro)
        sudo pacman -S --noconfirm python-pip alsa-lib portaudio libsndfile
        ;;
    *)
        echo "⚠ Distribution-specific dependencies not configured for $DISTRO"
        echo "Continuing with Python-only installation..."
        ;;
esac

echo "✓ System dependencies installed"

# Upgrade pip
echo ""
echo "Upgrading pip..."
python3 -m pip install --user --upgrade pip
echo "✓ pip upgraded"

# Install DAiW package
echo ""
echo "Installing DAiW Music-Brain..."
python3 -m pip install --user -e .
echo "✓ DAiW installed"

# Install optional dependencies
echo ""
echo "Installing optional dependencies..."
python3 -m pip install --user -e ".[audio,ui]"
echo "✓ Optional dependencies installed"

# Add to PATH if not already there
USER_BIN="$HOME/.local/bin"
if [[ ":$PATH:" != *":$USER_BIN:"* ]]; then
    echo ""
    echo "Adding $USER_BIN to PATH..."
    case $SHELL in
        */bash)
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
            echo "Added to ~/.bashrc"
            ;;
        */zsh)
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
            echo "Added to ~/.zshrc"
            ;;
        *)
            echo "⚠ Unknown shell. Please manually add $USER_BIN to your PATH"
            ;;
    esac
    echo "Please restart your terminal or run:"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

# Create desktop entry
echo ""
read -p "Create desktop application entry? (Y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    DESKTOP_DIR="$HOME/.local/share/applications"
    mkdir -p "$DESKTOP_DIR"
    
    cat > "$DESKTOP_DIR/daiw-music-brain.desktop" << 'EOF'
[Desktop Entry]
Type=Application
Name=DAiW Music-Brain
Comment=Music production intelligence toolkit
Exec=daiw-desktop
Icon=audio-x-generic
Terminal=false
Categories=AudioVideo;Audio;Music;
Keywords=music;midi;production;daw;
EOF
    
    echo "✓ Desktop entry created"
fi

echo ""
echo "========================================"
echo "  ✓ Installation Complete!"
echo "========================================"
echo ""
echo "You can now use DAiW from the command line:"
echo "  daiw --help"
echo ""
echo "Or run the desktop UI:"
echo "  daiw-desktop"
echo ""
echo "For more information, see README.md"
echo ""
echo "Enjoy making music! 🎵"
