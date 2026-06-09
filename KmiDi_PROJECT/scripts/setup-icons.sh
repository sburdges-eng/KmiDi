#!/bin/bash
# Historical helper in legacy/alternate tree: setup Tauri icon files
# Creates placeholder PNG icons from existing ICNS file or generates simple placeholders
# Not current runnable-command authority for the canonical repo root.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICONS_DIR="${SCRIPT_DIR}/../src-tauri/icons"

echo "Setting up Tauri icon files..."

# Create icons directory if it doesn't exist
mkdir -p "${ICONS_DIR}"

# Check if we have icon.icns (macOS icon)
if [ -f "${ICONS_DIR}/icon.icns" ]; then
    echo "Found icon.icns, extracting PNGs..."
    
    # Try to extract PNGs from ICNS (requires iconutil on macOS)
    if command -v iconutil &> /dev/null; then
        # Create temporary directory for extraction
        TEMP_DIR=$(mktemp -d)
        iconutil --convert iconset "${ICONS_DIR}/icon.icns" --output "${TEMP_DIR}/icon.iconset" 2>/dev/null || true
        
        # Copy extracted PNGs if available
        if [ -d "${TEMP_DIR}/icon.iconset" ]; then
            # Look for standard sizes
            for size in 32 128 256 512; do
                if [ -f "${TEMP_DIR}/icon.iconset/icon_${size}x${size}.png" ]; then
                    cp "${TEMP_DIR}/icon.iconset/icon_${size}x${size}.png" "${ICONS_DIR}/${size}x${size}.png"
                    echo "  ✓ Created ${size}x${size}.png"
                fi
                if [ -f "${TEMP_DIR}/icon.iconset/icon_${size}x${size}@2x.png" ]; then
                    cp "${TEMP_DIR}/icon.iconset/icon_${size}x${size}@2x.png" "${ICONS_DIR}/${size}x${size}@2x.png"
                    echo "  ✓ Created ${size}x${size}@2x.png"
                fi
            done
            rm -rf "${TEMP_DIR}"
        fi
    fi
fi

# Create placeholder PNGs for missing sizes using ImageMagick or sips (macOS)
create_placeholder() {
    local size=$1
    local output="${ICONS_DIR}/${size}x${size}.png"
    
    if [ ! -f "${output}" ]; then
        # Try ImageMagick first
        if command -v convert &> /dev/null; then
            convert -size ${size}x${size} xc:gray -pointsize $((size/4)) -fill white -gravity center -annotate +0+0 "KmiDi" "${output}" 2>/dev/null && echo "  ✓ Created ${output} (ImageMagick)" && return 0
        fi
        
        # Try sips on macOS
        if command -v sips &> /dev/null && [ -f "${ICONS_DIR}/icon.icns" ]; then
            sips -z ${size} ${size} "${ICONS_DIR}/icon.icns" --out "${output}" 2>/dev/null && echo "  ✓ Created ${output} (sips)" && return 0
        fi
        
        # Fallback: create a simple colored square using Python
        if command -v python3 &> /dev/null; then
            python3 << EOF
from PIL import Image, ImageDraw, ImageFont
import os

size = ${size}
img = Image.new('RGB', (size, size), color='#4A90E2')
draw = ImageDraw.Draw(img)

# Try to add text
try:
    font_size = max(12, size // 8)
    # Use default font
    draw.text((size//2, size//2), 'K', fill='white', anchor='mm')
except:
    pass

img.save('${output}')
EOF
            if [ -f "${output}" ]; then
                echo "  ✓ Created ${output} (Python PIL)"
                return 0
            fi
        fi
        
        echo "  ⚠ Could not create ${output} - manual creation required"
        return 1
    else
        echo "  ✓ ${output} already exists"
        return 0
    fi
}

# Create required icon sizes
echo "Creating placeholder icons for missing sizes..."
create_placeholder 32
create_placeholder 128
create_placeholder 256
create_placeholder 512

# Create 2x retina version
if [ ! -f "${ICONS_DIR}/128x128@2x.png" ]; then
    if [ -f "${ICONS_DIR}/256x256.png" ]; then
        cp "${ICONS_DIR}/256x256.png" "${ICONS_DIR}/128x128@2x.png"
        echo "  ✓ Created 128x128@2x.png from 256x256.png"
    else
        create_placeholder 256
        if [ -f "${ICONS_DIR}/256x256.png" ]; then
            cp "${ICONS_DIR}/256x256.png" "${ICONS_DIR}/128x128@2x.png"
        fi
    fi
fi

echo ""
echo "Icon setup complete!"
echo "Required icons:"
for icon in "32x32.png" "128x128.png" "128x128@2x.png"; do
    if [ -f "${ICONS_DIR}/${icon}" ]; then
        echo "  ✓ ${icon}"
    else
        echo "  ✗ ${icon} (MISSING - Tauri build will fail)"
    fi
done
