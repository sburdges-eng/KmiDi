#!/usr/bin/env bash
# install_juce_linux_deps.sh — apt packages required to build JUCE 8 on Linux.
#
# JUCE 8's juceaide (the build-time helper that generates resource files) and
# the GUI modules (juce_graphics, juce_gui_basics) link against X11, freetype,
# fontconfig, xkbcommon, and OpenGL. juce_core needs libcurl headers and
# juce_audio_devices needs ALSA headers, so every KellyCore/penta build pulls
# those too. CI workflows that build any target which pulls in external/JUCE
# need these packages on Linux.
#
# Usage:
#   - name: Install JUCE Linux deps
#     run: scripts/ci/install_juce_linux_deps.sh

set -euo pipefail

if [[ -n "${RUNNER_OS:-}" && "${RUNNER_OS}" != "Linux" ]] || [[ "$(uname -s)" != "Linux" ]]; then
    echo "install_juce_linux_deps: not Linux (RUNNER_OS=${RUNNER_OS:-unset}, uname=$(uname -s)), skipping"
    exit 0
fi

sudo apt-get update -qq

# JUCE 8 build-time + runtime Linux deps.
sudo apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    libasound2-dev \
    libfreetype-dev \
    libfontconfig1-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxcomposite-dev \
    libxkbcommon-dev \
    libgl1-mesa-dev
