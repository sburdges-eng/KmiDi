#!/bin/bash
# Build script for Dart Strike Mac App

echo "🎯 Building Dart Strike for Mac..."

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install Node.js first."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Build for Mac
echo "🔨 Building Mac app..."
npm run dist-mac

echo "✅ Build complete! Check the 'dist' folder for:"
echo "  - Dart Strike.dmg (installer)"
echo "  - Dart Strike-mac.zip (portable)"
echo ""
echo "To install on Mac:"
echo "  1. Open Dart Strike.dmg"
echo "  2. Drag Dart Strike to Applications folder"
echo "  3. Open from Applications (may need to right-click > Open first time)"
