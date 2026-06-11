# UI Testing Checklist

Historical note
- This checklist lives under a legacy/alternate tree and is not current repo architecture or runnable-command authority.
- Any references below to Tauri-only startup paths should be treated as historical unless revalidated against the active repo `package.json` and `docs/DEVELOPMENT.md`.

This document provides a comprehensive checklist for testing all UI buttons, functions, and interactions.

## Prerequisites

1. Start the dev server: `npm run dev:react`
2. Open browser to: http://localhost:1420
3. Open browser DevTools (F12) for console logs

## Automated Testing

### Option 1: Browser Console Script
1. Open browser console (F12)
2. Copy and paste the contents of `scripts/test-ui-all.js`
3. Run `runUITests()` in the console

### Option 2: Python Playwright Script
```bash
# Install dependencies
pip install playwright
playwright install chromium

# Run tests
python scripts/test-ui-automated.py
```

## Manual Testing Checklist

### ✅ App Header
- [ ] **Side Toggle Button** - Click to switch between Side A and Side B
- [ ] **API Status Indicator** - Check if it shows Online/Offline/Checking
- [ ] **Error Dismiss Button** - If error appears, click to dismiss

### ✅ Side A: Professional DAW
- [ ] **Test Generate Music Button** - Click and verify API call
- [ ] Verify loading state shows "Generating..."
- [ ] Check console for API response or errors

### ✅ Side B: Therapeutic Interface

#### Lyric Panel
- [ ] **Load .txt/.lrc Button** - Click to open file picker (requires Tauri)
- [ ] **Save Lyrics Button** - Type lyrics, click save, verify status message
- [ ] **Refresh Button** - Click to reload lyrics from backend
- [ ] **Clear Button** - Click to clear lyrics
- [ ] **Textarea** - Type/paste lyrics, verify text updates

#### Emotion Wheel
- [ ] **Load Emotions Button** - Click to load emotion data
- [ ] **Base Emotion Buttons** - Click each base emotion (angry, happy, sad, etc.)
- [ ] **Intensity Level Buttons** - After selecting base, click intensity levels
- [ ] **Sub-Emotion Buttons** - After selecting intensity, click sub-emotions
- [ ] **Clear Button** - After full selection, click to reset
- [ ] Verify selected emotion displays correctly

#### GhostWriter Section
- [ ] **Generate Music Button** - Click after selecting emotion
- [ ] Verify button is disabled when no emotion selected
- [ ] Check console for API response

#### Interrogator Section
- [ ] **Start Interrogation Button** - Click to test interrogation
- [ ] Check console for response
- [ ] Verify loading state

### ✅ Production Workflow Guides

#### Guide Navigation
- [ ] **Search Input** - Type to filter guides
- [ ] **Topic Filter Pills** - Click "All" and individual topics
- [ ] **Preview Button** - Click to preview guide in viewer
- [ ] **Copy Path Button** - Click to copy guide path to clipboard
- [ ] **Open Link** - Click to open guide in new tab
- [ ] Verify guide count updates with filters

#### Guide Viewer
- [ ] Select a guide to preview
- [ ] Verify markdown renders correctly
- [ ] Check topic chips display
- [ ] Verify title and slug display

### ✅ SpectoCloud Panel

#### Controls
- [ ] **Load Humanizer Config Button** - Click to load config
- [ ] **Preset Buttons** - Click preview, standard, high
- [ ] **Mode Dropdown** - Switch between static and animation
- [ ] **FPS Input** - Change FPS value
- [ ] **Rotate Checkbox** - Toggle rotate option
- [ ] **Anchor Density Dropdown** - Change between sparse/normal/dense
- [ ] **Particles Input** - Change particle count
- [ ] **Duration Input** - Change duration value
- [ ] **Frame Index Input** - Change frame index (static mode only)

#### MIDI Input
- [ ] **MIDI Events Textarea** - Paste/edit JSON array
- [ ] **Upload JSON File** - Upload a JSON file with MIDI events
- [ ] **MIDI File Path Input** - Enter path to MIDI file

#### Render
- [ ] **Render Static/Animation Button** - Click to render
- [ ] Verify loading state
- [ ] Check output displays path and frame count
- [ ] Verify error handling for invalid input

## Expected Behaviors

### API Status
- Should show "Checking..." initially
- Should show "Online" if Music Brain API is running
- Should show "Offline" if API is not reachable

### Button States
- Buttons should be disabled during loading
- Buttons should show loading text ("Generating...", "Loading...", etc.)
- Error messages should appear for failed API calls

### Emotion Wheel
- Should show 3-step selection process
- Selected emotion should display at top
- Clear button should reset all selections

### Guide Navigation
- Search should filter in real-time
- Topic filters should update guide count
- Preview should show guide content in viewer

## Common Issues

### API Not Running
- Error: "Music Brain API is not running"
- Solution: Start API with `python -m music_brain.api`

### File Picker Not Working
- Error: "File picker requires Tauri"
- Historical workaround from the alternate tree: run in Tauri mode only if that tree still exposes a validated Tauri dev command.

### Buttons Disabled
- Check if required data is loaded (emotions, etc.)
- Check if previous action is still loading
- Verify API is online

## Test Results Template

```
Date: ___________
Tester: ___________

Total Tests: ___
Passed: ___
Failed: ___
Skipped: ___

Failed Tests:
- [List any failed tests]

Notes:
[Any observations or issues]
```
