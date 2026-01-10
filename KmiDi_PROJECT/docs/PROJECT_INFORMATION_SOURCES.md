# Project Information Sources

**Date**: 2025-01-07  
**Status**: Active Documentation

---

## Overview

The KmiDi project relies on three primary sources of information for development, recovery, and data management:

1. **Active Project Repository** (`/Users/seanburdges/KmiDi-1`)
2. **Recovery Operations Archive** (`/Users/seanburdges/RECOVERY_OPS`)
3. **External Drive Storage** (`/Volumes/sbdrive`)

---

## 1. Active Project Repository

**Location**: `/Users/seanburdges/KmiDi-1`  
**Size**: ~1.5GB  
**Purpose**: Current working codebase and active development

### Contents
- **Core Codebase**: Unified monorepo combining music_brain, penta_core, iDAW_Core
- **Documentation**: Complete project documentation, guides, and references
- **Source Code**: Python, C++, TypeScript implementations
- **Configuration**: Build configs, environment files, project settings
- **Tests**: Comprehensive test suites for all components

### Key Directories
```
KmiDi-1/
├── music_brain/          # Python music intelligence toolkit
├── penta_core/           # C++ real-time audio engines
├── iDAW_Core/            # JUCE plugin suite
├── mcp_workstation/      # Multi-AI orchestration
├── mcp_todo/             # Cross-AI task management
├── docs/                 # Project documentation
├── data/                 # Music theory data files
├── tests/                # Test suites
└── scripts/              # Utility scripts
```

### Status
- ✅ Active development
- ✅ Version controlled (Git)
- ✅ CI/CD configured
- ✅ Production-ready components

---

## 2. Recovery Operations Archive

**Location**: `/Users/seanburdges/RECOVERY_OPS`  
**Size**: ~110GB  
**Purpose**: Comprehensive backup, archive, and recovery data

### Contents
- **Archived Projects**: Historical versions and backups
- **Audio/MIDI Data**: Large datasets and audio libraries
- **Trained Models**: ML model checkpoints and weights
- **Documentation**: Historical documentation and guides
- **Music Projects**: Song projects, Logic Pro sessions, MIDI files

### Key Directories
```
RECOVERY_OPS/
├── ARCHIVE/
│   ├── kelly-music-brain-clean/    # Archived codebase
│   └── KmiDi-remote/               # Git remote backup
├── AUDIO_MIDI_DATA/
│   └── kelly-audio-data/           # Audio datasets
├── CANONICAL/
│   └── KmiDi/                      # Canonical reference
├── ML_TRAINED_MODELS/              # Model checkpoints
├── Music/
│   ├── AudioVault/                 # Audio sample library
│   ├── iDAW_Output/                # Generated projects
│   └── Logic/                      # Logic Pro sessions
└── kelly-project/                  # Legacy project structure
```

### Use Cases
- **Recovery**: Restore lost files or previous versions
- **Reference**: Historical code and documentation
- **Data Access**: Large datasets not in active repo
- **Model Access**: Trained ML models and checkpoints

---

## 3. External Drive Storage

**Location**: `/Volumes/sbdrive`  
**Size**: ~39GB  
**Purpose**: External storage for large media files and libraries

### Contents
- **Emotion Libraries**: Emotion_Instrument_Library, Emotion_Scale_Library
- **Audio Files**: MP3 collections, audio samples
- **Code Archives**: Miscellaneous code and projects
- **Media Assets**: Images, audio, and other media files

### Key Directories
```
sbdrive/
├── Emotion_Instrument_Library/     # Instrument emotion mappings
├── Emotion_Scale_Library/          # Scale emotion characteristics
├── MP3/                             # Audio file collections
├── MISC CODE/                       # Additional code archives
└── [various project files]          # Other archived projects
```

### Use Cases
- **Media Storage**: Large audio and media files
- **Library Access**: Emotion and scale libraries
- **Backup**: Additional backup location
- **Archive**: Long-term storage for less frequently accessed data

---

## Data Flow and Relationships

```
┌─────────────────────────────────────────────────────────┐
│  Active Project (KmiDi-1)                               │
│  - Current development                                   │
│  - Version controlled                                    │
│  - Production code                                       │
└──────────────┬──────────────────────────────────────────┘
               │
               ├─── References ────┐
               │                   │
               ▼                   ▼
┌──────────────────────┐  ┌──────────────────────┐
│  Recovery Archive    │  │  External Drive     │
│  (RECOVERY_OPS)      │  │  (sbdrive)           │
│  - Historical data   │  │  - Media libraries  │
│  - Large datasets    │  │  - Emotion data      │
│  - Model checkpoints │  │  - Audio files      │
└──────────────────────┘  └──────────────────────┘
```

---

## Usage Guidelines

### When to Use Each Source

#### Active Project (`KmiDi-1`)
- ✅ All active development work
- ✅ Current code changes
- ✅ Running tests and builds
- ✅ Creating new features

#### Recovery Archive (`RECOVERY_OPS`)
- 🔍 Finding historical code or documentation
- 🔍 Accessing large datasets not in active repo
- 🔍 Restoring previous versions of files
- 🔍 Accessing trained model checkpoints
- 🔍 Reference for music projects and sessions

#### External Drive (`sbdrive`)
- 📦 Accessing emotion libraries
- 📦 Large media file storage
- 📦 Long-term archive storage
- 📦 Backup of critical data

---

## Synchronization Strategy

### Active Project → Recovery Archive
- Manual backup of important milestones
- Archive completed features before major refactoring
- Store trained models and large outputs

### Active Project → External Drive
- Copy emotion libraries when needed
- Archive large generated outputs
- Store media assets

### Recovery Archive ↔ External Drive
- Cross-reference for data recovery
- Verify data integrity
- Maintain redundant backups

---

## Recovery Procedures

### If Active Project is Corrupted
1. Check `RECOVERY_OPS/ARCHIVE/kelly-music-brain-clean/` for recent backup
2. Check `RECOVERY_OPS/CANONICAL/KmiDi/` for canonical reference
3. Restore from Git history if version controlled
4. Rebuild from archived components if necessary

### If Data is Missing
1. Search `RECOVERY_OPS/AUDIO_MIDI_DATA/` for audio datasets
2. Check `RECOVERY_OPS/ML_TRAINED_MODELS/` for model checkpoints
3. Look in `sbdrive/` for emotion libraries and media files
4. Check `RECOVERY_OPS/Music/` for music projects

### If Models are Needed
1. Check `RECOVERY_OPS/ML_TRAINED_MODELS/` for checkpoints
2. Verify model compatibility with current codebase
3. Copy to active project if needed
4. Update model paths in configuration

---

## Maintenance

### Regular Tasks
- **Weekly**: Verify all three sources are accessible
- **Monthly**: Archive important milestones to RECOVERY_OPS
- **Quarterly**: Review and clean up outdated archives
- **As Needed**: Sync critical data between sources

### Backup Strategy
- Active project: Git version control + periodic RECOVERY_OPS backups
- Recovery archive: Maintain integrity, verify accessibility
- External drive: Keep mounted and accessible for media libraries

---

## Document Information

**Last Updated**: 2025-01-07  
**Maintained By**: Project Team

---

## Notes

- All three sources are required for complete project functionality
- Active development should primarily occur in `KmiDi-1`
- Large datasets and models are stored in `RECOVERY_OPS` to keep active repo manageable
- External drive provides additional storage and backup redundancy
- Regular synchronization ensures data availability and recovery capability

