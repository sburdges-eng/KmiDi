# Folder Color Application Guide

## Automated Method

Run the script:
```bash
./scripts/apply_folder_colors.sh
```

## Manual Method (If Script Doesn't Work)

### Step 1: Open Finder
Navigate to: `/Users/seanburdges/KmiDi-1/`

### Step 2: Apply Yellow/Gold Labels

**For Active Development Folders:**
1. Select these folders (Cmd+Click to multi-select):
   - `src/`
   - `src/plugin/`
   - `src/gui/`
   - `src/bridge/`
   - `src/core/`
   - `include/` (if exists)
   - `tests/`

2. Right-click → **Tags** → Select **Yellow**
   - OR use keyboard: Select → **Cmd+Option+3**

**For Active Development Documents:**
1. Select these files:
   - `PROJECT_SOURCE_MANIFEST.md`
   - `src/INDEX.md`
   - `PROJECT_DIRECTORY_MAP.md`
   - `MIGRATION_COMPLETE_SUMMARY.md`
   - `src/ACTIVE_DEVELOPMENT.md`
   - `src/plugin/ACTIVE_DEVELOPMENT.md`
   - `src/gui/ACTIVE_DEVELOPMENT.md`
   - `src/bridge/ACTIVE_DEVELOPMENT.md`

2. Right-click → **Tags** → Select **Yellow**

### Step 3: Verify

Folders and files should now display with yellow/gold color in Finder.

## Troubleshooting

If labels don't appear:
1. Check System Preferences → Security & Privacy → Privacy → Full Disk Access
2. Ensure Terminal/iTerm has Full Disk Access enabled
3. Try manual method above
4. Restart Finder: Cmd+Option+Esc → Finder → Relaunch
