#!/bin/bash
# Quick build script for macOS/Linux

echo "=========================================="
echo "  Lariat Bible - Quick Build (Unix)      "
echo "=========================================="

# Navigate to desktop_app directory
cd "$(dirname "$0")"

# Install PyInstaller if not present
echo "Checking for PyInstaller..."
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    pip3 install pyinstaller
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist __pycache__ *.spec 2>/dev/null

# Build the executable
echo "Building standalone executable..."
pyinstaller \
    --name="LariatBible" \
    --windowed \
    --onefile \
    --clean \
    --noconfirm \
    main.py

# Check if build was successful
if [ -f "dist/LariatBible" ] || [ -d "dist/LariatBible.app" ]; then
    echo "✅ Build successful!"
    echo "📦 Executable location: dist/"
    
    # Make it executable on Linux
    if [ -f "dist/LariatBible" ]; then
        chmod +x dist/LariatBible
        echo "🚀 Run with: ./dist/LariatBible"
    fi
    
    # For macOS
    if [ -d "dist/LariatBible.app" ]; then
        echo "🚀 Run with: open dist/LariatBible.app"
    fi
else
    echo "❌ Build failed. Check the output above for errors."
    exit 1
fi

echo "=========================================="
echo "  Build Complete!                         "
echo "=========================================="
