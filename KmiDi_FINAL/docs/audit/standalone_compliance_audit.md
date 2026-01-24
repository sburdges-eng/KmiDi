# Standalone App Compliance Audit

## Overview

Audit of standalone application against specifications in `docs/specs/01-09/`.

**Status: MAJOR ARCHITECTURAL MISMATCH**

## Critical Issues

### ❌ Architecture Violation

**Spec:** Standalone app must use **AppKit + JUCE + SwiftUI**
- AppKit: Windowing/layout
- JUCE: Timeline + audio rendering
- SwiftUI: Inspectors only

**Current Implementation:**
- Tauri + React (frontend)
- Qt + custom widgets (backend - `source/cpp/src/gui/main.cpp`)

**Impact:** Complete rewrite required to comply with specs.

### ❌ Layout Violation

**Spec Required Layout:**
```
Window (AppKit)
├── Timeline (Center, JUCE)
├── Inspector (Left, SwiftUI island)
├── Browser (Right, AppKit)
└── Bottom Panel (Optional, docked)
```

**Current Layout:**
- React-based Side A/Side B toggle
- No timeline component
- No inspector/browser separation
- No JUCE timeline integration

### ❌ Emotion UI Violation

**Spec:** Emotion inspector shows read-only bars:
- Valence (-1 → 1) horizontal bar
- Arousal (0 → 1) horizontal bar
- Dominance (0 → 1) horizontal bar
- Complexity (0 → 1) horizontal bar

**Current:** Interactive emotion wheel with selection interface.

## Specification Compliance Matrix

### 01. Foundation / System UI Specs

| Requirement | Status | Notes |
|-------------|--------|-------|
| Windowing & Host Compliance | ❌ FAIL | Uses Tauri/React instead of AppKit+JUCE+SwiftUI |
| Input & Interaction | ⚠️ PARTIAL | Basic React interactions, missing macOS specifics |
| Performance & Frame Budget | ❌ FAIL | React overhead violates performance requirements |

### 02. Layout & Navigation Specs

| Requirement | Status | Notes |
|-------------|--------|-------|
| Standalone App Layout | ❌ FAIL | Missing timeline, inspector, browser separation |
| Plug-in Layout | N/A | Not applicable to standalone |
| Persistence Rules | ❌ FAIL | No layout persistence implemented |
| Hierarchy & Visual Priority | ❌ FAIL | No proper visual hierarchy |

### 03. Visual System Specs

| Requirement | Status | Notes |
|-------------|--------|-------|
| Color Token & Theme | ❌ FAIL | No semantic color system |
| Typography & Spacing | ❌ FAIL | No consistent spacing system |
| Control Styling | ❌ FAIL | Basic HTML controls, no hybrid flat design |

### 04. Core Musical UI Specs

| Requirement | Status | Notes |
|-------------|--------|-------|
| Timeline UI | ❌ FAIL | No timeline component exists |
| JUCE Embedding | ❌ FAIL | No JUCE integration |
| Plugin Editor Layout | N/A | Not applicable |

### 05. AI / ML Visibility Specs

| Requirement | Status | Notes |
|-------------|--------|-------|
| AI/ML Visibility Rules | ⚠️ PARTIAL | Basic API integration exists |
| Emotion Inspector | ❌ FAIL | Wrong UI (wheel vs bars) |
| ML Overlay Rendering | ❌ FAIL | No overlays implemented |
| AI Explanation | ❌ FAIL | No explanation system |

### 06. Control & Trust Specs

| Requirement | Status | Notes |
|-------------|--------|-------|
| AI Trust & Consent | ❌ FAIL | No AI enable/disable controls |
| Undo / History Semantics | ❌ FAIL | No undo system |

### 07. Plugin-Specific Specs

| Requirement | Status | Notes |
|-------------|--------|-------|
| Parameter Exposure | N/A | Not applicable |
| Master EQ UI | N/A | Not applicable |
| EQ + Chain Coexistence | N/A | Not applicable |

### 08. Output & Verification Specs

| Requirement | Status | Notes |
|-------------|--------|-------|
| Export UI | ❌ FAIL | No export functionality |
| Preview & Visualization | ❌ FAIL | No preview system |

### 09. Documentation & Repo Hygiene Specs

| Requirement | Status | Notes |
|-------------|--------|-------|
| README & Preview Asset | ❌ FAIL | No user-focused documentation |
| Large-Repo Safety | ❌ FAIL | No safety measures |

## Required Fixes

### Immediate (Breaking Changes)

1. **Replace Tauri/React with AppKit+JUCE+SwiftUI Architecture**
   - Remove Tauri configuration
   - Implement JUCE application with AppKit window
   - Add SwiftUI inspectors

2. **Implement Required Layout Structure**
   - Create JUCE timeline component (center)
   - Implement SwiftUI emotion inspector (left)
   - Add AppKit browser panel (right)
   - Add optional bottom panel

3. **Fix Emotion UI**
   - Replace emotion wheel with read-only bars
   - Remove interactive selection
   - Add proper valence/arousal/dominance/complexity display

### Medium Priority

4. **Add AI Controls**
   - Global AI enable/disable
   - Per-domain AI toggles
   - Immediate effect implementation

5. **Implement Export System**
   - MIDI export
   - Audio export
   - Progress reporting

6. **Add Undo System**
   - User action undo
   - AI suggestion handling
   - Transaction boundaries

### Long Term

7. **Visual System Implementation**
   - Semantic color tokens
   - Consistent spacing system
   - Hybrid flat control styling

8. **Performance Optimization**
   - Remove React overhead
   - Implement JUCE-native UI
   - Meet frame budget requirements

## Estimated Effort

- **Architecture Rewrite:** 4-6 weeks (complete replacement)
- **Layout Implementation:** 2-3 weeks
- **Emotion UI Fix:** 1 week
- **AI Controls:** 1 week
- **Export System:** 1 week
- **Visual System:** 2 weeks
- **Performance:** 1 week

**Total: 12-16 weeks** for full compliance.

## Recommendation

**MAJOR DECISION REQUIRED:**

The current implementation is fundamentally incompatible with the specifications. Two options:

1. **Rewrite to comply** with AppKit+JUCE+SwiftUI architecture
2. **Update specifications** to match current Tauri/React implementation

The specifications appear to be designed for a native macOS application, while the current implementation uses cross-platform web technologies. Choose the architecture that best serves the product vision.

## Files Requiring Changes

### Complete Replacement
- `source/frontend/src-tauri/` (remove)
- `source/cpp/src/App.tsx` (replace)
- `source/cpp/src/components/` (replace)

### New Implementation Required
- JUCE timeline component
- AppKit window management
- SwiftUI emotion inspector
- AI control system
- Export functionality
- Undo system

### Configuration Updates
- `CMakeLists.txt` (remove Tauri, add JUCE app)
- Build system reconfiguration
- Dependencies update