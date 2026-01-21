# Intent IR Debugging Guide

## UI Debugging Workflow

### Step 1: Classify the Bug

Every UI bug falls into exactly one bucket:

- **Bucket A - Rendering bug**: Wrong colors, layout glitches, jank/flicker
- **Bucket B - State reflection bug**: UI shows wrong emotion, parameters don't match sound
- **Bucket C - Control bug**: User edits don't change sound, changes apply late
- **Bucket D - Backend masquerading as UI**: UI looks wrong but logs show correct values

### Step 2: Make Intent IR Visible

Use the `IntentIRInspector` component to display the full IntentFrame. This shows:
- Raw values, no formatting
- Provenance coloring
- Real-time updates

### Step 3: Debugging Order

1. Log IR at UI boundary
2. Log IR after Rust validation
3. Log IR received by C++
4. Log parameters derived from IR

If step 1 is wrong → UI bug
If step 1 right, step 2 wrong → Rust bug
If step 2 right, step 3 wrong → FFI bug
If step 3 right, output wrong → engine bug

### Step 4: Static Intent Injection

Use `IntentInjector` component to inject a known, static IntentFrame:
- No ML
- No user input
- No audio thread mutation

If UI still breaks → UI bug
If UI becomes perfect → backend timing bug

## Plugin Debugging Workflow

### Step 1: Classify the Plugin Bug

- **Bucket P1 - Host lifecycle bug**: UI resets randomly, parameters snap back
- **Bucket P2 - Parameter sync bug**: UI shows value A, audio uses value B
- **Bucket P3 - Thread violation**: Random crashes, glitches when opening UI
- **Bucket P4 - Host-specific insanity**: Works in Reaper, broken in Logic

### Step 2: Mandatory Logging Points

Log every time these fire:
- `prepareToPlay`
- `releaseResources`
- `processBlock` (first call only)
- `setStateInformation`
- `getStateInformation`
- Editor constructor/destructor
- Parameter change callbacks

### Step 3: Golden Plugin Test

1. Load plugin
2. Inject static IntentFrame
3. Disable UI edits, ML, host automation
4. Verify stability

### Step 4: Host-Specific Debugging

Use `HostDebugger` to:
- Detect host type
- Apply host-specific workarounds
- Log host quirks
- Verify parameter sync

## Common Bug Patterns

### Emotion wheel snaps back
**Cause**: UI mutating instead of emitting IR
**Fix**: Use `useIntentIR` hook, emit IR updates

### Sliders lag behind sound
**Cause**: UI rendering live state instead of snapshot
**Fix**: Freeze UI state on audio ticks

### Visuals disagree with audio
**Cause**: Engine interpretation bug, not UI
**Fix**: Check engine contract, verify IR consumption

### Plugin UI resets
**Cause**: Host lifecycle event ignored
**Fix**: Log lifecycle events, handle editor destruction

## IR Inspection Guide

### Using IntentIRInspector

1. Add component to your UI
2. Pass current IntentFrame as prop
3. Expand sections to see raw values
4. Check provenance coloring

### Using PluginIRInspector

1. Add component to plugin editor
2. Call `updateFrame()` with IntentFrame
3. Component updates thread-safely
4. Check provenance display

## Migration Guide

### From Old System

1. Replace direct emotion/music parameter passing with IntentFrame
2. Use `IntentFrameAdapter` for backward compatibility
3. Update engines to consume IntentFrame snapshots
4. Remove side channels

### Testing Migration

1. Run static IntentFrame injection test
2. Verify all engines consume IR correctly
3. Check logs for IR flow
4. Verify no side channels exist
