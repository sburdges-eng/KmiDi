# KmiDi UI Redesign - Qt Desktop Application

**Version**: 1.0.0  
**Date**: 2025-01-07  
**Status**: Design Phase

---

## Design Philosophy

Following the "Interrogate Before Generate" principle, the UI must:
- **Separate GUI from Logic** (non-negotiable)
- **Start Simple** (one window, one button, one output)
- **Handle Long Tasks** (threading, progress, cancellation)
- **Be Boring and Predictable** (clear labels, sensible defaults, human errors)
- **Look Professional** (audio tool aesthetic, not flashy)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Qt GUI Layer                          │
│  (main_window.py, views/, widgets.py)                    │
│  - Buttons, sliders, displays                            │
│  - Event handlers (NO logic)                             │
└──────────────────────┬────────────────────────────────────┘
                       │ signals/slots
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 Controller Layer                         │
│  (controllers/actions.py)                                │
│  - Routes GUI events to core                             │
│  - Manages worker threads                                │
│  - Updates UI from results                               │
└──────────────────────┬────────────────────────────────────┘
                       │ function calls
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    Core Logic                            │
│  (core/engine.py, models.py, storage.py, network.py)   │
│  - Pure business logic                                   │
│  - No GUI dependencies                                   │
│  - Can run headless (CLI, tests)                         │
└─────────────────────────────────────────────────────────┘
```

**Rule**: If the GUI vanished tomorrow, the project should still run.

---

## UI Layout Sketch

### Main Window Structure

```
┌─────────────────────────────────────────────────────────────┐
│ File  Edit  View  Tools  Help                                │ ← Menu Bar
├─────────────────────────────────────────────────────────────┤
│ [Toolbar: Generate | Preview | Export | Settings]          │ ← Toolbar
├──────────┬───────────────────────────────────────────────────┤
│          │                                                   │
│          │                                                   │
│  NAV     │              MAIN WORKSPACE                      │
│  PANEL   │              (Emotion Input,                     │
│          │               Parameter Controls,                 │
│  [List]  │               Results Display)                   │
│          │                                                   │
│          │                                                   │
├──────────┴───────────────────────────────────────────────────┤
│ [AI Assistant Dock]                                          │ ← Right Dock
│ Task: File Similarity                                         │
│ Status: Complete                                             │
│ Confidence: 0.92                                              │
│ [Preview] [Apply] [Ignore]                                   │
├─────────────────────────────────────────────────────────────┤
│ [Logs Dock]                                                  │ ← Bottom Dock
│ • analyzed 124 files                                          │
│ • embeddings cached                                           │
│ • 3 suggestions pending                                        │
└─────────────────────────────────────────────────────────────┘
│ Status: Ready | API: Online | Files: 124                    │ ← Status Bar
└─────────────────────────────────────────────────────────────┘
```

### Layout Breakdown

**Left Panel (Navigation)**
- Project list
- Recent files
- Workspace switcher

**Center (Main Workspace)**
- Emotion input (text area)
- Parameter sliders (Valence, Arousal, Intensity, etc.)
- Results display (chords, MIDI preview, theory panel)

**Right Dock (AI Assistant)**
- Analysis results
- Confidence indicators
- Action buttons (Preview, Apply, Ignore)

**Bottom Dock (Logs)**
- Operation logs
- Error messages
- Progress indicators

**Status Bar**
- Connection status
- File count
- Current operation

---

## Component Hierarchy

```
MainWindow
├── MenuBar
│   ├── File (New, Open, Save, Export)
│   ├── Edit (Undo, Redo, Preferences)
│   ├── View (Docks, Toolbars)
│   ├── Tools (AI Analysis, Batch Process)
│   └── Help (About, Documentation)
├── ToolBar
│   ├── Generate
│   ├── Preview
│   ├── Export
│   └── Settings
├── CentralWidget
│   ├── NavigationPanel (left)
│   │   └── ProjectList
│   └── WorkspacePanel (center)
│       ├── EmotionInput
│       ├── ParameterPanel
│       └── ResultsDisplay
├── DockWidgets
│   ├── AIAssistantDock (right)
│   │   ├── AnalysisSummary
│   │   ├── ConfidenceIndicator
│   │   └── ActionButtons
│   └── LogsDock (bottom)
│       └── LogView
└── StatusBar
    ├── ConnectionStatus
    ├── FileCount
    └── OperationStatus
```

---

## Event Flow (Non-Negotiable)

```
User Action (Button Click)
    ↓
GUI Event Handler (main_window.py)
    ↓
Controller Method (controllers/actions.py)
    ↓
Core Logic Function (core/engine.py)
    ↓
Result Object
    ↓
Controller Updates UI (signals/slots)
    ↓
GUI Refresh
```

**Rules**:
- No logic in handlers
- No UI updates in core
- No exceptions

---

## File Structure

```
kmidi_gui/
├── core/                      # Pure logic, no GUI
│   ├── __init__.py
│   ├── engine.py              # Main processing engine
│   ├── models.py               # Data structures
│   ├── storage.py              # File/DB operations
│   ├── network.py              # API/worker communication
│   └── ai/                     # AI analysis modules
│       ├── analyzer.py
│       ├── job_queue.py
│       └── cache.py
├── gui/                        # Qt widgets only
│   ├── __init__.py
│   ├── main_window.py          # Main window
│   ├── views/
│   │   ├── emotion_input.py
│   │   ├── parameter_panel.py
│   │   └── results_display.py
│   ├── widgets/
│   │   ├── confidence_indicator.py
│   │   └── log_view.py
│   └── docks/
│       ├── ai_assistant_dock.py
│       └── logs_dock.py
├── controllers/                # Bridge GUI ↔ Core
│   ├── __init__.py
│   ├── actions.py              # Action handlers
│   └── workers.py              # Background threads
├── themes/
│   ├── audio_dark.qss          # Audio tool theme
│   └── palette.py              # Color palette
├── cli/                        # CLI using same core
│   └── main.py
├── main.py                     # Application entry
└── requirements.txt
```

---

## AI Integration Patterns

### Job Queue Architecture

```
User Action
    ↓
Job Enqueued (with input hash)
    ↓
Cache Check (if hash exists, return cached)
    ↓
Background Worker (QThread)
    ↓
AI Execution (local or remote)
    ↓
Result Stored (with decision record)
    ↓
UI Notified (signal)
    ↓
User Approval Required
    ↓
Execute (only if approved)
```

### Safe Decision Schema

**AI Can Propose**:
- `propose_base` (which file is canonical)
- `propose_merge` (merge candidates)
- `propose_archive` (archive candidates)
- `propose_ignore` (ignore suggestions)

**AI Cannot**:
- Delete files
- Overwrite files
- Move files automatically
- Act without explicit user approval

**Decision Record**:
```python
{
    "decision_id": "dec-8421",
    "ai_job": "job-1029",
    "proposal": {
        "base": "notes_v3.txt",
        "merge": ["notes_v2.txt"],
        "archive": ["notes_v1.txt"]
    },
    "confidence": 0.91,
    "approved": false  # Must be True before execution
}
```

---

## Audio Tool Theme

### Color Palette

- **Background**: `#1e1e1e` (dark gray, not black)
- **Panel**: `#202020`
- **Text**: `#d0d0d0` (soft gray, not white)
- **Accent**: `#3a6ea5` (muted blue)
- **Button**: `#2a2a2a`
- **Border**: `#2a2a2a`

### Visual Rules

- No GroupBox frames (use labels + spacing)
- Horizontal sliders > knobs
- Flat panels with faint separators
- Dense but readable
- No gradients, no gloss
- Rounded corners: 4-6px max

---

## Implementation Phases

### Phase 1: Minimal Window (Week 1)
- [x] Qt project structure
- [ ] Main window skeleton
- [ ] Audio theme applied
- [ ] One button, one output
- [ ] Core/GUI separation verified

### Phase 2: Basic Workspace (Week 2)
- [ ] Emotion input widget
- [ ] Parameter sliders
- [ ] Results display
- [ ] Menu bar + toolbar
- [ ] Status bar

### Phase 3: Docks & AI (Week 3)
- [ ] AI Assistant dock
- [ ] Logs dock
- [ ] Job queue implementation
- [ ] Worker threads
- [ ] Safe decision schema

### Phase 4: Polish (Week 4)
- [ ] Confidence indicators
- [ ] Progress bars
- [ ] Error handling
- [ ] Keyboard shortcuts
- [ ] Settings persistence

---

## Testing Strategy

### Core Logic Tests
- Run without GUI
- CLI interface
- Unit tests

### GUI Tests
- Widget rendering
- Event handling
- Thread safety
- UI responsiveness

### Integration Tests
- End-to-end workflows
- AI job processing
- File operations
- Network communication

---

## Migration Path

### From Current React/Tauri UI

1. **Preserve Core Logic**: Keep `music_brain/` and `penta_core/` unchanged
2. **Replace GUI Layer**: Qt replaces React/Tauri UI
3. **Maintain API**: Keep Music Brain API for compatibility
4. **Gradual Migration**: Run both UIs in parallel during transition

### From JUCE Plugin UI

1. **Extract Logic**: Move business logic to `core/`
2. **Qt Wrappers**: Create Qt widgets that call C++ core
3. **Shared State**: Use same data models
4. **Plugin Bridge**: Qt app can still export to JUCE plugins

---

## Success Criteria

✅ **GUI never "does" the work** - All logic in core/  
✅ **Core can run headless** - CLI works without GUI  
✅ **UI never freezes** - All long tasks in threads  
✅ **Looks professional** - Audio tool aesthetic  
✅ **User never confused** - Clear labels, sensible defaults  
✅ **AI never acts alone** - All actions require approval  
✅ **Undo always exists** - Reversible operations  

---

## Next Steps

1. Create Qt project structure
2. Implement audio theme
3. Build minimal main window
4. Connect to existing core logic
5. Add AI integration patterns
6. Test with real workflows

---

## References

- Qt Documentation: https://doc.qt.io/
- PySide6 Guide: https://doc.qt.io/qtforpython/
- Audio Tool Design Patterns: See `docs/UI_REDESIGN_Qt.md`
- AI Integration Guide: See `docs/AI_INTEGRATION.md`

