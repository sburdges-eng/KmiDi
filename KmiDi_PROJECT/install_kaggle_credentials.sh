#!/bin/bash
# Install Kaggle API credentials for SeanBurdges

echo "============================================================"
echo "KAGGLE API CREDENTIALS INSTALLER"
echo "============================================================"
echo ""
echo "Username: SeanBurdges"
echo ""

# Check if kaggle.json exists in Downloads
KAGGLE_JSON="$HOME/Downloads/kaggle.json"

if [ ! -f "$KAGGLE_JSON" ]; then
    echo "⚠ kaggle.json not found in Downloads folder"
    echo ""
    echo "Please download it first:"
    echo "1. Visit: https://www.kaggle.com/seanburdges/settings"
    echo "2. Scroll to 'API' section"
    echo "3. Click 'Create New Token'"
    echo "4. The file will download to ~/Downloads/kaggle.json"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✓ Found kaggle.json in Downloads"
echo ""

# Create .kaggle directory
KAGGLE_DIR="$HOME/.kaggle"
mkdir -p "$KAGGLE_DIR"

# Move the file
echo "Installing credentials..."
mv "$KAGGLE_JSON" "$KAGGLE_DIR/kaggle.json"

# Set secure permissions (required by Kaggle API)
chmod 600 "$KAGGLE_DIR/kaggle.json"

echo "✓ Credentials installed to: $KAGGLE_DIR/kaggle.json"
echo "✓ Permissions set correctly (600)"
echo ""

# Verify installation
if [ -f "$KAGGLE_DIR/kaggle.json" ]; then
    echo "============================================================"
    echo "INSTALLATION COMPLETE!"
    echo "============================================================"
    echo ""
    echo "You can now download datasets:"
    echo "  python3 setup_kaggle_datasets.py"
    echo ""
    echo "Or test the connection:"
    echo "  kaggle datasets list --max-size 10"
    echo ""
else
    echo "✗ Installation failed"
    exit 1
fi
