#!/bin/bash
#
# iDAW Installer for macOS
# ========================
# Version: 1.0.00
#
# This script installs iDAW and creates a proper .app bundle
#

echo "========================================"
echo "  iDAW - intelligent Digital Audio Workspace"
echo "  Version 1.0.00"
echo "  Installer for macOS"
echo "========================================"
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo ""
    echo "Install Python from:"
    echo "  https://www.python.org/downloads/"
    echo ""
    echo "Or via Homebrew:"
    echo "  brew install python3"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip3 install --upgrade pip
pip3 install streamlit music21 mido numpy pydub scipy

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✓ Dependencies installed"

# Create application directory
APP_DIR="$HOME/Applications/iDAW"
mkdir -p "$APP_DIR"

# Copy files
echo ""
echo "Installing iDAW to $APP_DIR..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cp "$SCRIPT_DIR/idaw_ableton_ui.py" "$APP_DIR/"
cp "$SCRIPT_DIR/idaw_complete_pipeline.py" "$APP_DIR/"
cp "$SCRIPT_DIR/idaw_launcher.py" "$APP_DIR/"
cp "$SCRIPT_DIR/vernacular.py" "$APP_DIR/" 2>/dev/null
cp "$SCRIPT_DIR/vernacular_database.json" "$APP_DIR/" 2>/dev/null

echo "✓ Files installed"

# Create .app bundle
echo ""
echo "Creating iDAW.app..."

APP_BUNDLE="$HOME/Applications/iDAW.app"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"

# Copy Info.plist
cat > "$APP_BUNDLE/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>iDAW</string>
    <key>CFBundleIdentifier</key>
    <string>com.idaw.app</string>
    <key>CFBundleName</key>
    <string>iDAW</string>
    <key>CFBundleVersion</key>
    <string>1.0.00</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.00</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.music</string>
</dict>
</plist>
PLIST

# Create launcher
cat > "$APP_BUNDLE/Contents/MacOS/iDAW" << 'LAUNCHER'
#!/bin/bash
APP_DIR="$HOME/Applications/iDAW"
cd "$APP_DIR"

# Start Streamlit and open browser
python3 -m streamlit run idaw_ableton_ui.py \
    --server.headless=true \
    --server.port=8501 \
    --browser.gatherUsageStats=false \
    --theme.base=dark \
    --theme.primaryColor="#ff9500" \
    --theme.backgroundColor="#1e1e1e" \
    --theme.secondaryBackgroundColor="#2d2d2d" \
    --theme.textColor="#cccccc" &

sleep 2
open "http://localhost:8501"
wait
LAUNCHER

chmod +x "$APP_BUNDLE/Contents/MacOS/iDAW"

echo "✓ iDAW.app created"

# Create desktop shortcut
echo ""
echo "Creating desktop shortcut..."
ln -sf "$APP_BUNDLE" "$HOME/Desktop/iDAW.app" 2>/dev/null

echo ""
echo "========================================"
echo "  ✓ Installation Complete!"
echo "========================================"
echo ""
echo "You can now:"
echo "  1. Double-click iDAW on your Desktop"
echo "  2. Find it in ~/Applications/iDAW.app"
echo "  3. Drag it to your Dock"
echo ""
echo "First launch may take a moment to start."
echo ""
echo "Enjoy making music! 🎵"
