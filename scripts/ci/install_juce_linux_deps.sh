#!/usr/bin/env bash
# install_juce_linux_deps.sh — apt packages required to build JUCE 8 on Linux.
#
# JUCE 8's juceaide (the build-time helper that generates resource files) and
# the GUI modules (juce_graphics, juce_gui_basics) link against X11, freetype,
# fontconfig, xkbcommon, and OpenGL. CI workflows that build any target which
# pulls in external/JUCE need these packages on Linux.
#
# Audio / engine deps (libasound2-dev, etc.) are NOT installed here — those
# are project-specific and stay in each workflow's apt block.
#
# Usage:
#   - name: Install JUCE Linux deps
#     run: scripts/ci/install_juce_linux_deps.sh

set -euo pipefail

if [[ "${RUNNER_OS:-Linux}" != "Linux" ]] && [[ "$(uname -s)" != "Linux" ]]; then
    echo "install_juce_linux_deps: not Linux, skipping"
    exit 0
fi

sudo apt-get update -qq

# JUCE 8 build-time + runtime Linux deps.
sudo apt-get install -y --no-install-recommends \
    libfreetype-dev \
    libfontconfig1-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxcomposite-dev \
    libxkbcommon-dev \
    libgl1-mesa-dev
