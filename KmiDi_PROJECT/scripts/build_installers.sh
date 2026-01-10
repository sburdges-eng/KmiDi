#!/bin/bash
# KmiDi Installer Build Script
# =============================
# Creates platform-specific installers

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build/installers"
VERSION="1.0.0"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

build_macos_installer() {
    echo "Creating macOS .pkg installer..."
    
    # Requirements: pkgbuild, productbuild (macOS only)
    if ! command -v pkgbuild &> /dev/null; then
        echo "Error: pkgbuild not found. macOS only."
        return 1
    fi
    
    # Create package structure
    PKG_ROOT="$BUILD_DIR/macos/pkg_root"
    mkdir -p "$PKG_ROOT/Applications/KmiDi.app/Contents"
    
    # Copy application files (placeholder - actual implementation would copy real files)
    # cp -r "$PROJECT_ROOT/kmidi_gui" "$PKG_ROOT/Applications/KmiDi.app/Contents/"
    
    # Create package info
    pkgbuild \
        --root "$PKG_ROOT" \
        --identifier com.kmidi.app \
        --version "$VERSION" \
        --install-location "/" \
        "$BUILD_DIR/KmiDi-$VERSION.pkg"
    
    echo "✅ macOS installer created: $BUILD_DIR/KmiDi-$VERSION.pkg"
}

build_linux_installer() {
    echo "Creating Linux .deb installer..."
    
    # Create Debian package structure
    DEB_ROOT="$BUILD_DIR/linux/deb"
    mkdir -p "$DEB_ROOT/DEBIAN"
    mkdir -p "$DEB_ROOT/usr/local/bin"
    mkdir -p "$DEB_ROOT/usr/local/share/kmidi"
    
    # Create control file
    cat > "$DEB_ROOT/DEBIAN/control" <<EOF
Package: kmidi
Version: $VERSION
Section: sound
Priority: optional
Architecture: amd64
Depends: python3 (>= 3.9), python3-pip
Maintainer: KmiDi Team <team@kmidi.com>
Description: KmiDi - Emotion-Driven Music Generation
 KmiDi generates music from emotional intent for music therapy,
 songwriting, and creative expression.
EOF
    
    # Copy application files (placeholder)
    # cp -r "$PROJECT_ROOT/kmidi_gui" "$DEB_ROOT/usr/local/share/kmidi/"
    
    # Build .deb package
    dpkg-deb --build "$DEB_ROOT" "$BUILD_DIR/kmidi_${VERSION}_amd64.deb"
    
    echo "✅ Debian package created: $BUILD_DIR/kmidi_${VERSION}_amd64.deb"
    
    # Note: .rpm creation would require rpmbuild and similar structure
    echo "For .rpm package, use: rpmbuild -bb kmidi.spec"
}

# Main script execution
echo -e "${GREEN}KmiDi Installer Build Script${NC}"
echo "=================================="
echo ""

# Create build directory
mkdir -p "$BUILD_DIR"

# Detect platform
OS=$(uname -s)

if [ "$OS" = "Darwin" ]; then
    echo "Building macOS installer..."
    build_macos_installer
elif [ "$OS" = "Linux" ]; then
    echo "Building Linux installer..."
    build_linux_installer
else
    echo "Unsupported platform: $OS"
    exit 1
fi

echo ""
echo -e "${GREEN}Build complete!${NC}"
echo "Installers available in: $BUILD_DIR"
