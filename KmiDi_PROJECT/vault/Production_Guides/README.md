# Production Guides

This directory contains comprehensive production guides for music creation.

## Available Guides

- **Drum Programming Guide.md** - Making programmed drums sound human
- **Dynamics and Arrangement Guide.md** - Creating emotional impact through dynamics
- **Electronic EDM Production Guide.md** - EDM-specific production techniques

## Note on Duplicate Files

These guides also exist in `Production_Workflows/` directory. Both locations contain **identical content**. 

**Source of Truth**: This directory (`vault/Production_Guides/`)

**Why Duplicates Exist**: 
- `vault/Production_Guides/` - Part of the Obsidian knowledge base
- `Production_Workflows/` - Quick access for workflow integration

**Recommendation**: 
- For editing: Use files in `vault/Production_Guides/`
- Both copies are kept in sync automatically (they are identical)
- If you edit one, copy changes to the other to maintain consistency

**Last Sync Check**: 2025-01-08 - Files verified as identical

---

For integration with code, see `music_brain/production/` modules:
- `dynamics_engine.py` - Implements Dynamics and Arrangement Guide principles
- `emotion_production.py` - Maps emotions to production techniques
- `groove/drum_humanizer.py` - Applies Drum Programming Guide rules