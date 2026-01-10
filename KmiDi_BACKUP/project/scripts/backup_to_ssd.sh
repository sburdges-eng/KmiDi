#!/bin/bash

# KmiDi Project Backup Script
# Backs up all essential project files to SSD labeled "KmiDi-DONE"
# Excludes build artifacts, cache, and temporary files

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="/Users/seanburdges/KmiDi-1"
SSD_LABEL="KmiDi-DONE"
SSD_MOUNT="/Volumes/${SSD_LABEL}"

# Function to print status
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if SSD is mounted
check_ssd() {
    if [ ! -d "$SSD_MOUNT" ]; then
        print_error "SSD '${SSD_LABEL}' not found at ${SSD_MOUNT}"
        print_status "Available volumes:"
        ls -1 /Volumes/ | grep -v "Macintosh HD"
        echo ""
        print_status "Please connect the SSD and ensure it's labeled '${SSD_LABEL}'"
        print_status "Or specify the mount path manually:"
        echo "  $0 /path/to/ssd"
        exit 1
    fi
    
    # Check if SSD is writable
    if [ ! -w "$SSD_MOUNT" ]; then
        print_error "SSD is not writable. Please check permissions."
        exit 1
    fi
    
    print_success "SSD found at ${SSD_MOUNT}"
}

# Allow manual SSD path override
if [ -n "$1" ]; then
    SSD_MOUNT="$1"
    print_status "Using custom SSD path: ${SSD_MOUNT}"
fi

# Check SSD
check_ssd

# Create backup directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="${SSD_MOUNT}/KmiDi-Backup-${TIMESTAMP}"
BACKUP_ROOT="${BACKUP_DIR}/KmiDi-1"

print_status "Creating backup directory: ${BACKUP_DIR}"
mkdir -p "${BACKUP_ROOT}"

# Function to copy directory with exclusions
copy_with_exclusions() {
    local src="$1"
    local dest="$2"
    local name="$3"
    
    if [ ! -d "$src" ]; then
        print_warning "Directory not found: ${src}"
        return
    fi
    
    print_status "Copying ${name}..."
    
    # Use rsync with exclusions
    rsync -av --progress \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='*.pyo' \
        --exclude='.pytest_cache' \
        --exclude='.coverage' \
        --exclude='build/' \
        --exclude='build-*/' \
        --exclude='cmake-build-*/' \
        --exclude='CMakeCache.txt' \
        --exclude='CMakeFiles/' \
        --exclude='node_modules/' \
        --exclude='dist/' \
        --exclude='.parcel-cache/' \
        --exclude='target/' \
        --exclude='*.log' \
        --exclude='*.tmp' \
        --exclude='*.bak' \
        --exclude='.vscode/' \
        --exclude='.idea/' \
        --exclude='.DS_Store' \
        --exclude='*.swp' \
        --exclude='*.swo' \
        --exclude='checkpoints/' \
        --exclude='logs/' \
        --exclude='output/' \
        --exclude='cache/' \
        --exclude='*.wav' \
        --exclude='*.mp3' \
        --exclude='*.flac' \
        --exclude='*.aiff' \
        --exclude='*.mid' \
        --exclude='*.midi' \
        --exclude='*.vst3' \
        --exclude='*.clap' \
        --exclude='.git/' \
        --exclude='*.pt' \
        --exclude='*.pth' \
        --exclude='*.onnx' \
        --exclude='*.ckpt' \
        --exclude='*.h5' \
        --exclude='*.npy' \
        --exclude='*.npz' \
        --exclude='*.pkl' \
        --exclude='*.pickle' \
        "${src}/" "${dest}/"
    
    print_success "Copied ${name}"
}

# Function to copy file
copy_file() {
    local src="$1"
    local dest="$2"
    local name="$3"
    
    if [ ! -f "$src" ]; then
        print_warning "File not found: ${src}"
        return
    fi
    
    print_status "Copying ${name}..."
    mkdir -p "$(dirname "$dest")"
    cp "$src" "$dest"
    print_success "Copied ${name}"
}

# Start backup
print_status "Starting backup of KmiDi project..."
echo ""

# Core source code directories
print_status "=== Copying Core Source Code ==="
copy_with_exclusions "${PROJECT_ROOT}/music_brain" "${BACKUP_ROOT}/music_brain" "Music Brain"
copy_with_exclusions "${PROJECT_ROOT}/penta_core" "${BACKUP_ROOT}/penta_core" "Penta Core"
copy_with_exclusions "${PROJECT_ROOT}/iDAW_Core" "${BACKUP_ROOT}/iDAW_Core" "iDAW Core"
copy_with_exclusions "${PROJECT_ROOT}/src" "${BACKUP_ROOT}/src" "Source Code"
copy_with_exclusions "${PROJECT_ROOT}/src_penta-core" "${BACKUP_ROOT}/src_penta-core" "Penta Core C++"
copy_with_exclusions "${PROJECT_ROOT}/include" "${BACKUP_ROOT}/include" "C++ Headers"
copy_with_exclusions "${PROJECT_ROOT}/cpp_music_brain" "${BACKUP_ROOT}/cpp_music_brain" "C++ Music Brain"
copy_with_exclusions "${PROJECT_ROOT}/python" "${BACKUP_ROOT}/python" "Python Bindings"
copy_with_exclusions "${PROJECT_ROOT}/bindings" "${BACKUP_ROOT}/bindings" "Language Bindings"
copy_with_exclusions "${PROJECT_ROOT}/kmidi_gui" "${BACKUP_ROOT}/kmidi_gui" "KmiDi GUI"

# MCP servers
print_status "=== Copying MCP Servers ==="
copy_with_exclusions "${PROJECT_ROOT}/mcp_workstation" "${BACKUP_ROOT}/mcp_workstation" "MCP Workstation"
copy_with_exclusions "${PROJECT_ROOT}/mcp_todo" "${BACKUP_ROOT}/mcp_todo" "MCP TODO"
copy_with_exclusions "${PROJECT_ROOT}/mcp_penta_swarm" "${BACKUP_ROOT}/mcp_penta_swarm" "MCP Penta Swarm"
copy_with_exclusions "${PROJECT_ROOT}/daiw_mcp" "${BACKUP_ROOT}/daiw_mcp" "DAiW MCP"

# Data files
print_status "=== Copying Data Files ==="
copy_with_exclusions "${PROJECT_ROOT}/data" "${BACKUP_ROOT}/data" "Data Files"
if [ -d "${PROJECT_ROOT}/Data_Files" ]; then
    copy_with_exclusions "${PROJECT_ROOT}/Data_Files" "${BACKUP_ROOT}/Data_Files" "Data Files (alt)"
fi

# Documentation
print_status "=== Copying Documentation ==="
copy_with_exclusions "${PROJECT_ROOT}/docs" "${BACKUP_ROOT}/docs" "Documentation"
copy_with_exclusions "${PROJECT_ROOT}/Production_Workflows" "${BACKUP_ROOT}/Production_Workflows" "Production Workflows"
copy_with_exclusions "${PROJECT_ROOT}/Songwriting_Guides" "${BACKUP_ROOT}/Songwriting_Guides" "Songwriting Guides"
copy_with_exclusions "${PROJECT_ROOT}/Theory_Reference" "${BACKUP_ROOT}/Theory_Reference" "Theory Reference"
copy_with_exclusions "${PROJECT_ROOT}/vault" "${BACKUP_ROOT}/vault" "Vault (Obsidian)"
copy_with_exclusions "${PROJECT_ROOT}/Templates" "${BACKUP_ROOT}/Templates" "Templates"

# Tests
print_status "=== Copying Tests ==="
copy_with_exclusions "${PROJECT_ROOT}/tests" "${BACKUP_ROOT}/tests" "Tests"
if [ -d "${PROJECT_ROOT}/tests_music-brain" ]; then
    copy_with_exclusions "${PROJECT_ROOT}/tests_music-brain" "${BACKUP_ROOT}/tests_music-brain" "Music Brain Tests"
fi
if [ -d "${PROJECT_ROOT}/tests_penta-core" ]; then
    copy_with_exclusions "${PROJECT_ROOT}/tests_penta-core" "${BACKUP_ROOT}/tests_penta-core" "Penta Core Tests"
fi

# Scripts and tools
print_status "=== Copying Scripts and Tools ==="
copy_with_exclusions "${PROJECT_ROOT}/scripts" "${BACKUP_ROOT}/scripts" "Scripts"
copy_with_exclusions "${PROJECT_ROOT}/tools" "${BACKUP_ROOT}/tools" "Tools"

# Examples
print_status "=== Copying Examples ==="
copy_with_exclusions "${PROJECT_ROOT}/examples" "${BACKUP_ROOT}/examples" "Examples"

# External libraries (JUCE, etc.)
print_status "=== Copying External Libraries ==="
copy_with_exclusions "${PROJECT_ROOT}/external" "${BACKUP_ROOT}/external" "External Libraries"

# Configuration files
print_status "=== Copying Configuration Files ==="
copy_file "${PROJECT_ROOT}/pyproject.toml" "${BACKUP_ROOT}/pyproject.toml" "pyproject.toml"
copy_file "${PROJECT_ROOT}/requirements.txt" "${BACKUP_ROOT}/requirements.txt" "requirements.txt"
copy_file "${PROJECT_ROOT}/requirements-production.txt" "${BACKUP_ROOT}/requirements-production.txt" "requirements-production.txt"
copy_file "${PROJECT_ROOT}/CMakeLists.txt" "${BACKUP_ROOT}/CMakeLists.txt" "CMakeLists.txt"
copy_file "${PROJECT_ROOT}/package.json" "${BACKUP_ROOT}/package.json" "package.json"
copy_file "${PROJECT_ROOT}/tsconfig.json" "${BACKUP_ROOT}/tsconfig.json" "tsconfig.json"
copy_file "${PROJECT_ROOT}/vite.config.ts" "${BACKUP_ROOT}/vite.config.ts" "vite.config.ts"
copy_file "${PROJECT_ROOT}/tailwind.config.js" "${BACKUP_ROOT}/tailwind.config.js" "tailwind.config.js"
copy_file "${PROJECT_ROOT}/postcss.config.js" "${BACKUP_ROOT}/postcss.config.js" "postcss.config.js"
copy_file "${PROJECT_ROOT}/.gitignore" "${BACKUP_ROOT}/.gitignore" ".gitignore"
copy_file "${PROJECT_ROOT}/LICENSE" "${BACKUP_ROOT}/LICENSE" "LICENSE"
copy_file "${PROJECT_ROOT}/VERSION" "${BACKUP_ROOT}/VERSION" "VERSION"
copy_file "${PROJECT_ROOT}/environment.yml" "${BACKUP_ROOT}/environment.yml" "environment.yml"
copy_file "${PROJECT_ROOT}/env.example" "${BACKUP_ROOT}/env.example" "env.example"
copy_file "${PROJECT_ROOT}/Doxyfile" "${BACKUP_ROOT}/Doxyfile" "Doxyfile"
copy_file "${PROJECT_ROOT}/Dockerfile.streamlit" "${BACKUP_ROOT}/Dockerfile.streamlit" "Dockerfile.streamlit"
copy_file "${PROJECT_ROOT}/Procfile" "${BACKUP_ROOT}/Procfile" "Procfile"

# Important root documentation files
print_status "=== Copying Root Documentation ==="
for doc in README.md KMIDI_README.md CLAUDE.md CLAUDE_AGENT_GUIDE.md COPILOT_INSTRUCTIONS.md; do
    if [ -f "${PROJECT_ROOT}/${doc}" ]; then
        copy_file "${PROJECT_ROOT}/${doc}" "${BACKUP_ROOT}/${doc}" "${doc}"
    fi
done

# Copy all markdown files in root (important documentation)
find "${PROJECT_ROOT}" -maxdepth 1 -name "*.md" -type f -exec cp {} "${BACKUP_ROOT}/" \;

# CI/CD workflows
print_status "=== Copying CI/CD Configuration ==="
if [ -d "${PROJECT_ROOT}/.github" ]; then
    copy_with_exclusions "${PROJECT_ROOT}/.github" "${BACKUP_ROOT}/.github" "GitHub Workflows"
fi

# Deployment configs
print_status "=== Copying Deployment Configuration ==="
if [ -d "${PROJECT_ROOT}/deployment" ]; then
    copy_with_exclusions "${PROJECT_ROOT}/deployment" "${BACKUP_ROOT}/deployment" "Deployment"
fi

# Web frontend
print_status "=== Copying Web Frontend ==="
if [ -d "${PROJECT_ROOT}/web" ]; then
    copy_with_exclusions "${PROJECT_ROOT}/web" "${BACKUP_ROOT}/web" "Web Frontend"
fi
if [ -f "${PROJECT_ROOT}/index.html" ]; then
    copy_file "${PROJECT_ROOT}/index.html" "${BACKUP_ROOT}/index.html" "index.html"
fi

# Assets
print_status "=== Copying Assets ==="
if [ -d "${PROJECT_ROOT}/assets" ]; then
    copy_with_exclusions "${PROJECT_ROOT}/assets" "${BACKUP_ROOT}/assets" "Assets"
fi

# Create backup manifest
print_status "=== Creating Backup Manifest ==="
MANIFEST_FILE="${BACKUP_DIR}/BACKUP_MANIFEST.txt"
{
    echo "KmiDi Project Backup Manifest"
    echo "=============================="
    echo ""
    echo "Backup Date: $(date)"
    echo "Source: ${PROJECT_ROOT}"
    echo "Destination: ${BACKUP_DIR}"
    echo ""
    echo "Contents:"
    echo "---------"
    find "${BACKUP_ROOT}" -type f | wc -l | xargs echo "Total files:"
    find "${BACKUP_ROOT}" -type d | wc -l | xargs echo "Total directories:"
    echo ""
    echo "Directory Structure:"
    du -sh "${BACKUP_ROOT}"/* 2>/dev/null | sort -h
    echo ""
    echo "Excluded Items:"
    echo "- Build directories (build/, cmake-build-*/)"
    echo "- Cache directories (__pycache__/, .pytest_cache/, node_modules/)"
    echo "- Log files (*.log)"
    echo "- Temporary files (*.tmp, *.bak)"
    echo "- Large binary files (*.wav, *.mp3, *.pt, *.pth, etc.)"
    echo "- Generated output (output/, checkpoints/, logs/)"
    echo "- Git repository (.git/)"
    echo ""
    echo "To restore:"
    echo "1. Copy ${BACKUP_ROOT} to your desired location"
    echo "2. Run: pip install -e ."
    echo "3. Run: npm install (if using web frontend)"
    echo "4. Follow setup instructions in README.md"
} > "${MANIFEST_FILE}"

print_success "Backup manifest created: ${MANIFEST_FILE}"

# Create README on SSD
print_status "=== Creating SSD README ==="
README_FILE="${BACKUP_DIR}/README.txt"
{
    echo "KmiDi Project - Final Backup"
    echo "============================"
    echo ""
    echo "This SSD contains a complete backup of the KmiDi project."
    echo ""
    echo "Backup Date: $(date)"
    echo "Project: KmiDi (Kelly - Therapeutic iDAW)"
    echo ""
    echo "Contents:"
    echo "- KmiDi-1/: Complete project source code"
    echo "- BACKUP_MANIFEST.txt: Detailed backup information"
    echo ""
    echo "Project Structure:"
    echo "- music_brain/: Python Music Intelligence Toolkit"
    echo "- penta_core/: C++ Real-time Audio Engines"
    echo "- iDAW_Core/: JUCE Plugin Suite"
    echo "- mcp_workstation/: MCP Multi-AI Orchestration"
    echo "- docs/: Complete documentation"
    echo "- data/: Music theory data files"
    echo ""
    echo "To restore this project:"
    echo "1. Copy KmiDi-1/ to your development machine"
    echo "2. Install dependencies: pip install -e ."
    echo "3. Build C++ components: mkdir build && cd build && cmake .. && make"
    echo "4. See README.md in KmiDi-1/ for detailed setup instructions"
    echo ""
    echo "For questions or issues, refer to the documentation in docs/"
} > "${README_FILE}"

print_success "SSD README created: ${README_FILE}"

# Calculate backup size
BACKUP_SIZE=$(du -sh "${BACKUP_DIR}" | cut -f1)
print_success "Backup complete!"
echo ""
echo "Backup Summary:"
echo "  Location: ${BACKUP_DIR}"
echo "  Size: ${BACKUP_SIZE}"
echo "  Files: $(find "${BACKUP_ROOT}" -type f | wc -l | xargs)"
echo "  Directories: $(find "${BACKUP_ROOT}" -type d | wc -l | xargs)"
echo ""
print_success "All essential project files have been backed up to the SSD!"
