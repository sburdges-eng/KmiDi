#!/bin/bash
#
# Kelly Song Project - Main Setup Script
# Run this to generate all files and prepare for Logic Pro
#

echo "========================================"
echo "Kelly Song Project - Setup"
echo "========================================"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not installed."
    exit 1
fi

# Install mido if needed
echo "Checking dependencies..."
python3 -c "import mido" 2>/dev/null || pip3 install mido

echo ""
echo "Step 1: Generating MIDI files..."
echo "--------------------------------"
python3 generate_midi.py

echo ""
echo "Step 2: Setting up Logic Pro structure..."
echo "-----------------------------------------"
python3 setup_logic_project.py

echo ""
echo "========================================"
echo "SETUP COMPLETE"
echo "========================================"
echo ""
echo "Files created in: $SCRIPT_DIR"
echo ""
echo "MIDI files:"
ls -la *.mid 2>/dev/null || echo "  (MIDI files in MIDI folder)"
echo ""
echo "Next steps:"
echo "  1. Open Logic Pro"
echo "  2. Create new project at 72 BPM, key of A minor"
echo "  3. Read LOGIC_PRO_SETUP.txt for detailed instructions"
echo "  4. Import kelly_song_reference.mid as a guide track"
echo "  5. Create your audio tracks (Guitar L, Guitar R, Vocal)"
echo "  6. Record. Remember Kelly."
echo ""
echo "========================================"
