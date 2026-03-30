# Plugin Compliance Audit

## Overview

Audit of all plugins (VST3, AU, CLAP) against specifications in `docs/specs/01-09/`.

**Status: PARTIAL COMPLIANCE - Core Issues**

## Plugin Inventory

### Existing Plugins
1. **Pencil** - Sketching/drafting audio ideas
2. **Chalk** - Lo-fi/bitcrusher effect
3. **Eraser** - Audio removal/cleanup
4. **Palette** - Tonal coloring/mixing
5. **Parrot** - Sample playback/mimicry
6. **Press** - Dynamics/compression
7. **Smudge** - Audio blending/smoothing
8. **Stamp** - Stutter/repeater effect
9. **Stencil** - Sidechain/ducking effect
10. **Trace** - Pattern following/automation

**Total: 10 plugins**

## Critical Issues

### ❌ Parameter Exposure Violation

**Spec Requires:** All plugins must expose these parameters:
- `ml_intensity` (0-100%) - Overall AI influence strength
- `melody_influence` (0-100%) - How much AI affects melody generation
- `harmony_influence` (0-100%) - How much AI affects harmony choices
- `groove_influence` (0-100%) - How much AI affects rhythmic feel
- `dynamics_influence` (0-100%) - How much AI affects volume/envelope shaping

**Current Reality:** Each plugin exposes its own custom parameters:
- Pencil: `sketchiness`, `detail`, `pressure`
- Chalk: `bitdepth`, `downsample`, `noise`
- Press: `ratio`, `threshold`, `attack`, `release`
- Etc.

**Impact:** Plugins cannot be controlled by the unified ML influence system.

### ❌ Plugin Editor Layout Violation

**Spec Requires:** All plugin editors must follow fixed region layout:
```
Header: Plugin name, bypass, preset selector
Emotion Panel: Read-only valence/arousal/dominance/complexity bars
Parameter Panel: ML influence controls (automatable)
Master EQ Panel: User curve (solid) + AI curve (ghost) + explicit apply
ML Hint Panel: Confidence bars + short text hints
```

**Current Reality:** Each plugin has custom editor with its own parameters and layout.

**Impact:** Inconsistent user experience, no unified ML control system.

### ❌ Master EQ Implementation Violation

**Spec Requires:**
- User curve = truth (solid line, what gets applied)
- AI curve = suggestion (dashed/ghost line)
- Explicit "Apply AI" button
- AI adjustments limited to ±1.5 dB
- Shelf-first philosophy

**Current Reality:**
- `MasterEQComponent` exists but `applySuggestedCurve()` is placeholder
- No clear user curve vs AI curve distinction
- AI can make unlimited adjustments

**Impact:** Users cannot trust or control AI EQ suggestions.

## Specification Compliance Matrix

### 01. Foundation / System UI Specs

| Requirement | Status | Notes |
|-------------|--------|-------|
| Windowing & Host Compliance | ✅ PASS | JUCE AudioProcessorEditor handles properly |
| Input & Interaction | ⚠️ PARTIAL | Basic JUCE interactions, missing accessibility |
| Performance & Frame Budget | ⚠️ PARTIAL | JUCE handles RT safety, but need verification |

### 02. Layout & Navigation Specs

| Requirement | Status | Notes |
|-------------|--------|-------|
| Standalone App Layout | N/A | Not applicable |
| Plug-in Layout | ❌ FAIL | Each plugin has custom layout, not unified regions |
| Persistence Rules | ❌ FAIL | No parameter persistence rules implemented |
| Hierarchy & Visual Priority | ❌ FAIL | No consistent visual hierarchy across plugins |

### 03. Visual System Specs

| Requirement | Status | Notes |
|-------------|--------|-------|
| Color Token & Theme | ❌ FAIL | No semantic color system in plugins |
| Typography & Spacing | ❌ FAIL | Inconsistent fonts and spacing |
| Control Styling | ❌ FAIL | Basic JUCE controls, no hybrid flat design |

### 04. Core Musical UI Specs

| Requirement | Status | Notes |
|-------------|--------|-------|
| Timeline UI | N/A | Not applicable |
| JUCE Embedding | ✅ PASS | Plugins are JUCE-only as required |
| Plugin Editor Layout | ❌ FAIL | No unified layout - each plugin is different |

### 05. AI / ML Visibility Specs

| Requirement | Status | Notes |
|-------------|--------|-------|
| AI/ML Visibility Rules | ❌ FAIL | No unified AI control system |
| Emotion Inspector | ❌ FAIL | No emotion bars in plugin UIs |
| ML Overlay Rendering | N/A | Not applicable to plugins |
| AI Explanation | ❌ FAIL | No ML hints or explanations |

### 06. Control & Trust Specs

| Requirement | Status | Notes |
|-------------|--------|-------|
| AI Trust & Consent | ❌ FAIL | No global or per-domain AI controls |
| Undo / History Semantics | ❌ FAIL | No undo system in plugins |

### 07. Plugin-Specific Specs

| Requirement | Status | Notes |
|-------------|--------|-------|
| Parameter Exposure & Automation | ❌ FAIL | Wrong parameters - no ML influence controls |
| Master EQ UI | ❌ FAIL | Exists but doesn't follow user/AI curve spec |
| EQ + Chain Coexistence | ❌ FAIL | No small-move philosophy, no back-off logic |

### 08. Output & Verification Specs

| Requirement | Status | Notes |
|-------------|--------|-------|
| Export UI | N/A | Not applicable to plugins |
| Preview & Visualization | ❌ FAIL | Debug views enabled in production builds |

### 09. Documentation & Repo Hygiene Specs

| Requirement | Status | Notes |
|-------------|--------|-------|
| README & Preview Asset | ❌ FAIL | No user documentation for plugins |
| Large-Repo Safety | ⚠️ PARTIAL | Some safety measures exist, incomplete |

## Required Fixes

### Immediate (Parameter Architecture)

1. **Unify Plugin Parameters**
   - Replace all custom parameters with ML influence parameters
   - Add `ml_intensity`, `melody_influence`, `harmony_influence`, `groove_influence`, `dynamics_influence`
   - Keep existing effect-specific parameters as secondary

2. **Implement Unified Plugin Editor Layout**
   - Create base `DAiWPluginEditor` class with fixed regions
   - Header: Plugin name, bypass, preset selector
   - Emotion Panel: Read-only bars (valence, arousal, dominance, complexity)
   - Parameter Panel: ML influence controls (automatable)
   - Master EQ Panel: User/AI curve display with explicit apply
   - ML Hint Panel: Confidence bars and text hints

3. **Fix Master EQ Implementation**
   - Implement proper user curve (solid) vs AI curve (ghost) display
   - Add explicit "Apply AI" button functionality
   - Limit AI adjustments to ±1.5 dB
   - Implement shelf-first philosophy
   - Add back-off logic when user touches controls

### Medium Priority

4. **Add AI Control System**
   - Global AI enable/disable
   - Per-domain AI toggles (melody, harmony, groove, dynamics, EQ)
   - Immediate effect on all plugins
   - Persistent preferences

5. **Implement Emotion Display**
   - Add read-only emotion bars to all plugin editors
   - Connect to unified emotion state
   - Update cadence throttled (max 10Hz)

6. **Add ML Hints System**
   - Confidence bars for AI processing
   - Short text hints (1-2 words)
   - No prescriptive language
   - Non-intrusive display

### Long Term

7. **Visual System Implementation**
   - Semantic color tokens
   - Consistent typography (system fonts)
   - Hybrid flat control styling
   - Proper spacing system

8. **Documentation**
   - User-facing documentation for each plugin
   - Installation and usage guides
   - Troubleshooting sections

## Architecture Decision

### Option 1: Full Rewrite (Recommended)
- Create new `DAiWPluginBase` class with unified parameters and layout
- Migrate all existing plugins to inherit from this base
- Implement proper Master EQ and emotion display
- **Effort:** 6-8 weeks

### Option 2: Gradual Migration
- Add ML influence parameters alongside existing ones
- Create optional unified layout components
- Fix Master EQ implementation
- **Effort:** 4-6 weeks

### Option 3: Plugin-Specific Approach
- Keep existing plugin architectures
- Add "DAiW wrapper" plugin that chains to existing plugins
- Wrapper provides unified interface
- **Effort:** 3-4 weeks

**Recommendation:** Option 1 (Full Rewrite) - provides clean, maintainable architecture that matches specifications.

## Files Requiring Changes

### Core Architecture
- `source/plugins/iDAW_Core/include/PluginBase.h` - Add unified parameters
- `source/plugins/iDAW_Core/include/PluginEditor.h` - Add unified layout
- `source/plugins/iDAW_Core/src/PluginBase.cpp` - Implement ML parameters
- `source/plugins/iDAW_Core/src/PluginEditor.cpp` - Implement unified layout

### Master EQ System
- `source/cpp/src/ui/MasterEQComponent.h` - Update to follow user/AI curve spec
- `source/cpp/src/ui/MasterEQComponent.cpp` - Implement explicit apply functionality
- `source/cpp/src/ui/AIEQSuggestionEngine.h` - Limit to ±1.5 dB adjustments

### All Plugin Editors
- `source/plugins/iDAW_Core/plugins/*/Editor.cpp` - Update to use unified layout
- `source/plugins/iDAW_Core/plugins/*/Processor.cpp` - Add ML influence parameters

### New Components Required
- `source/plugins/iDAW_Core/include/EmotionPanel.h` - Read-only emotion bars
- `source/plugins/iDAW_Core/include/MLHintPanel.h` - Confidence bars and hints
- `source/plugins/iDAW_Core/include/ParameterPanel.h` - Unified ML controls

## Estimated Effort

- **Unified Parameter System:** 1 week
- **Unified Editor Layout:** 2 weeks
- **Master EQ Fix:** 1 week
- **Emotion Display System:** 1 week
- **ML Hints System:** 1 week
- **Migrate All Plugins:** 2 weeks
- **Testing & Polish:** 1 week

**Total: 9 weeks** for full compliance.

## Success Criteria

1. ✅ All plugins expose ML influence parameters
2. ✅ All plugins use unified editor layout
3. ✅ Master EQ shows user curve (solid) and AI curve (ghost)
4. ✅ Explicit "Apply AI" button works in Master EQ
5. ✅ AI adjustments limited to ±1.5 dB
6. ✅ Emotion bars display in all plugin editors
7. ✅ ML hints provide non-prescriptive feedback
8. ✅ Global and per-domain AI controls work
9. ✅ All plugins build and load in DAWs
10. ✅ User documentation exists for all plugins