# 09. Documentation & Repo Hygiene Specs

## Overview

The boring stuff that saves months. Proper documentation and repo organization prevents confusion and technical debt.

## README & Preview Asset Spec

### What Gets Documented

**✓ REQUIRED DOCUMENTATION:**
- Clear project description (what it does)
- Installation instructions (step-by-step)
- Basic usage examples
- System requirements
- Troubleshooting section
- License information

**❌ FORBIDDEN IN README:**
- Internal implementation details
- Developer-only instructions
- Raw technical specifications
- TODO lists or roadmap details

**Implementation:**
```markdown
<!-- README.md - User-focused documentation -->
# DAiW - Digital Audio intelligent Workstation

Emotion-driven music production that stays out of your way.

## What is DAiW?

DAiW is a music production environment that uses emotion as the starting point for musical creation. Instead of technical parameters, you begin with how you want to feel, and DAiW helps translate that into musical elements.

## Quick Start

### Standalone App (macOS)
```bash
# Download from releases
# Double-click DAiW.app
# Select emotion → Generate music → Export
```

### Plugin (DAW Integration)
1. Install VST3/AU/CLAP plugin
2. Load in Logic Pro, Reaper, or Ableton Live
3. Adjust emotion parameters in your DAW

## Features

- **Emotion-First Workflow**: Start with feeling, not technical parameters
- **AI-Assisted Composition**: Harmony, melody, and rhythm suggestions
- **Professional Integration**: Works in all major DAWs
- **Standalone Production**: Complete music production environment

## System Requirements

- **macOS**: 10.12 or later
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for application
- **Audio Interface**: Any Core Audio compatible device

## Troubleshooting

### Plugin Not Loading
- Ensure you're using the correct plugin format for your DAW
- Check that DAiW has audio input permissions
- Try rescanning plugins in your DAW

### Audio Artifacts
- Lower ML Intensity parameter
- Increase buffer size in audio settings
- Check for conflicting plugins

### Emotion Not Responding
- Verify AI features are enabled in preferences
- Check internet connection for cloud features
- Restart the application

## Support

For issues and questions:
- Check the troubleshooting section above
- Search existing GitHub issues
- Create a new issue with system details

## License

Commercial license. See LICENSE file for details.
```

### Where Screenshots Live

**✓ ORGANIZED ASSETS:**
```
docs/
├── screenshots/
│   ├── standalone/
│   │   ├── emotion_wheel.png
│   │   ├── timeline_view.png
│   │   └── export_dialog.png
│   └── plugins/
│       ├── master_eq.png
│       ├── emotion_panel.png
│       └── parameter_controls.png
└── previews/
    ├── standalone_app.mp4
    └── plugin_demo.mp4
```

**❌ FORBIDDEN:**
- Screenshots in root directory
- Unnamed or unclear screenshots
- Outdated screenshots
- Screenshots with debug information

**Implementation:**
```bash
# Screenshot organization script
#!/bin/bash
# organize_screenshots.sh

SCREENSHOT_DIR="docs/screenshots"
PREVIEW_DIR="docs/previews"

# Create organized structure
mkdir -p "$SCREENSHOT_DIR/standalone"
mkdir -p "$SCREENSHOT_DIR/plugins"
mkdir -p "$PREVIEW_DIR"

# Move and rename screenshots
mv screenshot_001.png "$SCREENSHOT_DIR/standalone/emotion_wheel.png"
mv screenshot_002.png "$SCREENSHOT_DIR/standalone/timeline_view.png"
mv screenshot_003.png "$SCREENSHOT_DIR/plugins/master_eq.png"

# Generate README references
echo "![Emotion Wheel](docs/screenshots/standalone/emotion_wheel.png)" >> screenshots.md
echo "![Timeline View](docs/screenshots/standalone/timeline_view.png)" >> screenshots.md
echo "![Master EQ](docs/screenshots/plugins/master_eq.png)" >> screenshots.md
```

### Naming Conventions

**✓ FILE NAMING:**
- `feature_description.png` (snake_case, descriptive)
- `v1.0_screenshot_name.png` (versioned when needed)
- `platform_feature_name.png` (platform-specific)

**❌ FORBIDDEN:**
- `IMG_1234.png` (generic names)
- `screenshot.png` (too vague)
- `new_feature.png` (temporary names)
- Spaces or special characters

**Implementation:**
```bash
# Naming convention validation
validate_screenshot_name() {
    local filename="$1"

    # Check for valid pattern
    if [[ ! $filename =~ ^[a-z0-9_]+\.png$ ]]; then
        echo "ERROR: Screenshot name '$filename' doesn't match convention"
        echo "Use: feature_description.png (lowercase, underscores, .png only)"
        return 1
    fi

    # Check for descriptive names
    if [[ $filename == "screenshot.png" ]] || [[ $filename == "image.png" ]]; then
        echo "ERROR: '$filename' is too generic"
        return 1
    fi

    return 0
}

# Usage in CI/CD
for screenshot in docs/screenshots/**/*.png; do
    if ! validate_screenshot_name "$(basename "$screenshot")"; then
        exit 1
    fi
done
```

## Large-Repo Safety Spec

### Scoped Edits

**✓ REQUIRED:**
- Edit only files you understand completely
- Test changes in isolation
- Use feature branches for significant changes
- Document rationale for architectural changes

**❌ FORBIDDEN:**
- Editing files "just to see what happens"
- Making changes without understanding dependencies
- Bulk find-and-replace without testing
- Changing core architecture without design review

**Implementation:**
```bash
# Pre-commit hook for scoped edits
#!/bin/bash
# pre-commit-scoped-edits.sh

# Get list of changed files
CHANGED_FILES=$(git diff --cached --name-only)

# Check for dangerous patterns
if echo "$CHANGED_FILES" | grep -q "CMakeLists.txt"; then
    echo "WARNING: CMakeLists.txt changed. Ensure you understand build implications."
    echo "Consider testing build on multiple platforms."
fi

if echo "$CHANGED_FILES" | grep -q "include/.*\.h"; then
    echo "WARNING: Header file changed. Check for breaking API changes."
    echo "Ensure all dependent code is updated."
fi

# Check edit scope
FILE_COUNT=$(echo "$CHANGED_FILES" | wc -l)
if [ "$FILE_COUNT" -gt 10 ]; then
    echo "WARNING: Large number of files changed ($FILE_COUNT)."
    echo "Consider breaking into smaller commits."
fi

# Require rationale for certain files
for file in $CHANGED_FILES; do
    case $file in
        "docs/specs/"*)
            echo "Spec file changed. Please ensure spec compliance audit is updated."
            ;;
        "source/plugins/"*)
            echo "Plugin code changed. Test in multiple DAW hosts."
            ;;
        "source/cpp/src/ui/"*)
            echo "UI code changed. Test visual regression."
            ;;
    esac
done
```

### Authoritative Roots

**✓ SINGLE SOURCES OF TRUTH:**
- `docs/specs/` - Authoritative specifications
- `source/cpp/src/` - Authoritative implementation
- `KmiDi_PROJECT/` - Authoritative build configuration

**❌ FORBIDDEN:**
- Duplicate specifications
- Multiple build systems
- Competing implementations
- Unofficial documentation

**Implementation:**
```bash
# Authoritative root validation
validate_authoritative_roots() {
    # Check for duplicate specs
    SPEC_COUNT=$(find . -name "*.md" -exec grep -l "spec" {} \; | wc -l)
    if [ "$SPEC_COUNT" -gt 9 ]; then
        echo "ERROR: Too many spec files. Consolidate into docs/specs/"
        return 1
    fi

    # Check for multiple CMakeLists.txt in root directories
    CMAKE_COUNT=$(find . -maxdepth 2 -name "CMakeLists.txt" | wc -l)
    if [ "$CMAKE_COUNT" -gt 1 ]; then
        echo "WARNING: Multiple CMakeLists.txt found. Ensure KmiDi_PROJECT/CMakeLists.txt is authoritative."
    fi

    # Check for competing UI implementations
    UI_DIRS=$(find source -name "*ui*" -type d | wc -l)
    if [ "$UI_DIRS" -gt 1 ]; then
        echo "WARNING: Multiple UI directories found. Ensure source/cpp/src/ui/ is authoritative."
    fi
}

# CI validation
validate_authoritative_roots || exit 1
```

### What Is Read-Only Forever

**❌ NEVER EDIT:**
- `docs/specs/*.md` (once approved, read-only)
- `LICENSE` (legal document)
- `version` files (generated by CI)
- Historical changelogs
- Archived specifications

**✓ CAN EDIT:**
- Implementation code (with tests)
- Documentation (user-facing)
- Build scripts (with validation)
- Configuration (with review)

**Implementation:**
```bash
# Read-only file protection
readonly_files=(
    "docs/specs/*.md"
    "LICENSE"
    "CHANGELOG.md"
    "version"
)

check_readonly_edit() {
    local file="$1"

    for pattern in "${readonly_files[@]}"; do
        if [[ $file == $pattern ]] || [[ $file == ${pattern//\*/*} ]]; then
            echo "ERROR: $file is read-only. Do not edit approved specifications."
            echo "Create new spec version or implementation change only."
            exit 1
        fi
    done
}

# Pre-commit hook
for file in $(git diff --cached --name-only); do
    check_readonly_edit "$file"
done
```

## Audit Checklist

### README & Documentation Compliance
- [ ] Clear project description (what it does)
- [ ] Step-by-step installation instructions
- [ ] Basic usage examples
- [ ] System requirements listed
- [ ] Troubleshooting section with common issues
- [ ] License information included
- [ ] No internal implementation details

### Screenshot & Asset Organization Compliance
- [ ] Screenshots in organized directory structure
- [ ] Descriptive naming conventions (snake_case, feature_description.png)
- [ ] No generic names (IMG_1234.png, screenshot.png)
- [ ] Preview videos in appropriate format
- [ ] Assets referenced correctly in documentation

### Large-Repo Safety Compliance
- [ ] Scoped edits (understand before changing)
- [ ] Authoritative roots identified and respected
- [ ] Read-only files protected from editing
- [ ] Pre-commit hooks validate changes
- [ ] Large changes broken into smaller commits

## Code Examples

### ✅ CORRECT: README Structure
```markdown
# DAiW - Digital Audio intelligent Workstation

> Emotion-driven music production that stays out of your way.

## Quick Start

### Standalone App (macOS)
1. Download from [releases](https://github.com/user/daiw/releases)
2. Open DAiW.app
3. Select emotion → Generate → Export

### Plugin Installation
1. Download plugin for your DAW
2. Install VST3/AU/CLAP
3. Load in your DAW
4. Adjust emotion parameters

## Features

- **Emotion-First**: Start with feeling, not parameters
- **AI-Assisted**: Smart suggestions without taking control
- **Professional**: Works in Logic Pro, Reaper, Ableton Live
- **Standalone**: Complete production environment

## Requirements

- macOS 10.12+
- 4GB RAM minimum
- Audio interface (recommended)

## Troubleshooting

### Plugin not appearing in DAW
- Ensure correct plugin format (VST3 for Reaper, AU for Logic)
- Rescan plugins in DAW
- Check plugin permissions

### Audio glitches
- Increase buffer size in DAW
- Lower ML Intensity parameter
- Disable other heavy plugins

## License

Commercial license. See LICENSE for details.
```

### ✅ CORRECT: Screenshot Organization
```bash
# docs/screenshots/ directory structure
docs/
└── screenshots/
    ├── standalone/
    │   ├── emotion_wheel_selection.png
    │   ├── timeline_with_overlays.png
    │   ├── export_audio_dialog.png
    │   └── inspector_emotion_bars.png
    └── plugins/
        ├── master_eq_user_curve.png
        ├── emotion_panel_readonly.png
        ├── parameter_automation.png
        └── ml_hint_panel.png
```

### ✅ CORRECT: Pre-commit Validation
```bash
#!/bin/bash
# .git/hooks/pre-commit

# Check for scoped edits
CHANGED_FILES=$(git diff --cached --name-only)
FILE_COUNT=$(echo "$CHANGED_FILES" | wc -l)

if [ "$FILE_COUNT" -gt 20 ]; then
    echo "WARNING: Large commit ($FILE_COUNT files)"
    echo "Consider splitting into smaller, focused commits"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for read-only files
for file in $CHANGED_FILES; do
    case $file in
        docs/specs/*.md)
            echo "ERROR: Specification files are read-only"
            echo "Create implementation changes only"
            exit 1
            ;;
        LICENSE)
            echo "ERROR: LICENSE file is read-only"
            exit 1
            ;;
    esac
done

# Validate screenshot naming
for file in $CHANGED_FILES; do
    if [[ $file == docs/screenshots/*.png ]]; then
        filename=$(basename "$file")
        if [[ ! $filename =~ ^[a-z0-9_]+\.png$ ]]; then
            echo "ERROR: Screenshot '$filename' doesn't follow naming convention"
            echo "Use: feature_description.png"
            exit 1
        fi
    fi
done
```

### ❌ WRONG: Poor Documentation
```markdown
<!-- WRONG: Too technical, not user-focused -->
# DAiW

A JUCE-based audio plugin that uses machine learning for emotion-to-music conversion via OSC communication with a Python backend server.

## Installation

Clone the repo, run `cmake .. && make`, copy the .vst3 to your VST3 directory. Make sure Python server is running on port 8080.

## API

The plugin exposes these parameters:
- ml_intensity (0-100)
- harmony_influence (0-100)
- groove_influence (0-100)

See source code for implementation details.
```

## Non-Compliance Fixes

### If README Is Developer-Focused:
1. Rewrite from user perspective
2. Remove technical implementation details
3. Add step-by-step usage instructions
4. Include troubleshooting section
5. Add clear feature descriptions

### If Screenshots Are Disorganized:
1. Create organized directory structure
2. Rename screenshots with descriptive names
3. Update documentation references
4. Remove outdated screenshots
5. Add naming convention validation

### If No Repo Safety Measures:
1. Implement pre-commit hooks for validation
2. Define read-only files policy
3. Add scoped edit guidelines
4. Create authoritative root documentation
5. Implement CI checks for repo hygiene

### If Multiple Competing Documentation:
1. Consolidate into single README
2. Remove duplicate information
3. Create clear information hierarchy
4. Update all references to point to authoritative docs
5. Archive old documentation appropriately