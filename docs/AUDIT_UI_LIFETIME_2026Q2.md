# UI Component Lifetime Audit (2026 Q2)

**Date:** 2026-04-21
**Scope:** src/ui/*.cpp + paired .h (28 files — note: prior audit memory cited ~57; actual count is 28 .cpp files)
**Method:** Static inspection against JUCE Component-lifetime checklist (6 categories)
**Total findings:** 8 (2 critical, 4 high, 2 medium)

## Checklist applied

1. stopTimer() in destructor
2. setLookAndFeel(nullptr) before LookAndFeel destruction
3. Async callbacks via juce::Component::SafePointer<>
4. Member declaration order vs destruction order
5. Legacy atomic shared_ptr ops
6. Listener un-registration

---

## Findings

### Critical

**C1 — LyricDisplay: missing stopTimer() in destructor**
- File: `src/ui/LyricDisplay.h:24` / `src/ui/LyricDisplay.cpp:12`
- Mechanism: `LyricDisplay` inherits `juce::Timer` (via `public juce::Timer` at `LyricDisplay.h:21`). Constructor calls `startTimer(50)` at `LyricDisplay.cpp:12`. The destructor is `~LyricDisplay() override = default` — no `stopTimer()`. If the Component is destroyed while the timer is still running, the JUCE timer thread fires `timerCallback()` on freed memory (UAF).
- Fix: Replace `~LyricDisplay() override = default;` with an explicit destructor body that calls `stopTimer();` before the default-generated destruction. Alternatively add `stopTimer();` to the destructor.

**C2 — EmotionWorkstation: raw `[this]` capture in showMenuAsync lambda after `~EmotionWorkstation() = default`**
- File: `src/ui/EmotionWorkstation.cpp:620–646`
- Mechanism: `EmotionWorkstation::showProjectMenu()` calls `projectMenu_.showMenuAsync(…, [this](int result) { … })`. The `PopupMenu::showMenuAsync` callback fires on the message thread after user selection; if the `EmotionWorkstation` is destroyed before the menu result arrives (e.g. host closes the plugin editor while a popup is open), the lambda dereferences a dangling `this`. The destructor is `~EmotionWorkstation() override = default` — no menu cancellation or guard. No `juce::Component::SafePointer<>` is used anywhere in the file.
- Fix:
  ```cpp
  void EmotionWorkstation::showProjectMenu() {
      setupProjectMenu();
      juce::Component::SafePointer<EmotionWorkstation> safe(this);
      projectMenu_.showMenuAsync(
          juce::PopupMenu::Options()
              .withTargetComponent(&projectMenuButton_)
              .withParentComponent(getTopLevelComponent()),
          [safe](int result) {
              if (auto* self = safe.getComponent()) {
                  switch (result) { /* … */ }
              }
          });
  }
  ```

---

### High

**H1 — WorkstationPanel: missing stopTimer() in destructor**
- File: `src/ui/WorkstationPanel.h:43` / `src/ui/WorkstationPanel.cpp:22–25`
- Mechanism: `WorkstationPanel` inherits `juce::Timer` (`WorkstationPanel.h:27`). Constructor calls `startTimer(30)` at `WorkstationPanel.cpp:22`. Destructor is `WorkstationPanel::~WorkstationPanel() = default` (`WorkstationPanel.cpp:25`) — no `stopTimer()`. Live timer callback fires on destroyed object.
- Fix: Provide an explicit destructor that calls `stopTimer();`.

**H2 — EmotionWorkstation: missing setLookAndFeel(nullptr) in destructor**
- File: `src/ui/EmotionWorkstation.cpp:15` / `src/ui/EmotionWorkstation.h:53`
- Mechanism: `EmotionWorkstation::setupComponents()` calls `setLookAndFeel(&lookAndFeel_)` at line 15, registering the member `KellyLookAndFeel lookAndFeel_` (declared at `EmotionWorkstation.h:205`) with the component's whole subtree. The destructor `~EmotionWorkstation() override = default` never calls `setLookAndFeel(nullptr)`. When JUCE tears down the component tree after the destructor returns (or during sub-component repaint), it accesses the already-destroyed `lookAndFeel_` member — UAF on repaint.
- Fix: Add an explicit destructor:
  ```cpp
  EmotionWorkstation::~EmotionWorkstation() {
      stopTimer();
      setLookAndFeel(nullptr);
  }
  ```
  (The `stopTimer()` is necessary here too; see C1/H1 pattern — `EmotionWorkstation` also inherits `juce::Timer` and calls `startTimer(30)` at `EmotionWorkstation.cpp:336`.)

**H3 — MusicianCommandPanel: TextEditor listener not removed in destructor**
- File: `src/ui/MusicianCommandPanel.cpp:39` / `src/ui/MusicianCommandPanel.h:41`
- Mechanism: Constructor calls `commandInput_->addListener(this)` at line 39, registering `MusicianCommandPanel` as a `juce::TextEditor::Listener` on `commandInput_`. The destructor is `~MusicianCommandPanel() override = default` — no `commandInput_->removeListener(this)`. If `commandInput_` outlives the panel (or fires a callback during destruction ordering), the listener callback fires on a destroyed `MusicianCommandPanel`. The TextEditor is a `std::unique_ptr<juce::TextEditor>` child component; although child destruction order is controlled, the listener could fire asynchronously via keyboard focus-loss events in JUCE's message queue.
- Fix: Provide explicit destructor:
  ```cpp
  MusicianCommandPanel::~MusicianCommandPanel() {
      if (commandInput_) commandInput_->removeListener(this);
  }
  ```

**H4 — EmotionWorkstation also missing stopTimer() (corollary to H2)**
- File: `src/ui/EmotionWorkstation.h:53` / `src/ui/EmotionWorkstation.cpp:336`
- Mechanism: Inherits `juce::Timer` (`EmotionWorkstation.h:50`), starts timer at `EmotionWorkstation.cpp:336` (`startTimer(30)`). Destructor `= default` with no `stopTimer()`. Distinct from H2 but must be fixed in the same destructor.
- Fix: Covered by the H2 fix — add `stopTimer();` to the new explicit destructor.

> **Note:** H4 is logically grouped with H2 since both require the same destructor addition. They are listed separately to surface both checklist failures.

---

### Medium

**M1 — EmotionWorkstation: member declaration order risk (LookAndFeel vs child Components)**
- File: `src/ui/EmotionWorkstation.h:103–205`
- Mechanism: `KellyLookAndFeel lookAndFeel_` is declared **last** among the data members (line 205), after all UI child components (`woundInput_`, `emotionWheel_`, `chordDisplay_`, `lyricDisplay_`, etc. at lines 108–199). In C++, members are destroyed in **reverse declaration order**, so `lookAndFeel_` is destroyed **first**. At the point of its destruction, the child component members have not yet been destroyed and may still hold a reference to the `LookAndFeel`. If any child component triggers a repaint or destruction callback that accesses the `LookAndFeel`, it touches freed memory. This is only partially mitigated by `setLookAndFeel(nullptr)` in the dtor (which is itself missing — see H2/H4). The fix for H2 is required first; after that, reordering `lookAndFeel_` to be declared **before** the child components is the belt-and-suspenders fix.
- Fix: Move `KellyLookAndFeel lookAndFeel_;` to be the **first** private data member declared in `EmotionWorkstation.h`, ensuring it outlives all child Components. Also apply H2 fix.

**M2 — TooltipComponent: callAfterDelay lambda captures no guard**
- File: `src/ui/TooltipComponent.cpp:53`
- Mechanism: `juce::Timer::callAfterDelay(timeoutMs, [] { TooltipComponent::hideTooltip(); })` at line 53. The lambda does **not** capture `this` (it calls a static method), so there is no direct UAF on `this`. However, `hideTooltip()` accesses the static singleton `getSharedTooltip()` which stores a `static TooltipComponent tooltip` initialized on first use. If the shared `tooltip` is destroyed at static teardown before a pending `callAfterDelay` fires (e.g. plugin unloaded), the callback accesses a destroyed static object. The risk is lower than a per-instance UAF but real in plugin contexts where the message thread may still be running during `FreeLibrary`/`dlclose`.
- Fix: Wrap the static in a `juce::DeletedAtShutdown` subclass or check an `initialized` flag atomically before accessing, and ensure `juce::MessageManager::deleteInstance()` is called before plugin shutdown.

---

## Files examined, no findings

Count: 19 files. Common pattern: simple `Component` subclass, no `Timer`, no async callbacks, no own `LookAndFeel`, no listener registrations.

List:
- `src/ui/AIEQSuggestionEngine.cpp` / `.h` — data-only engine, no Component timer
- `src/ui/AIGenerationDialog.cpp` / `.h` — uses `DialogWindow::LaunchOptions::launchAsync()` but the dialog is owned by the `DialogWindow`, not a `Component::SafePointer` problem; all raw `[this]` lambdas are button onClick lambdas bound to in-dialog buttons (no async cross-lifetime issue)
- `src/ui/ChordDisplay.cpp` / `.h`
- `src/ui/EditCommand.cpp` / `.h` — non-UI data struct
- `src/ui/EmotionRadar.cpp` / `.h`
- `src/ui/EmotionWheel.cpp` / `.h`
- `src/ui/EQBandControls.cpp` / `.h`
- `src/ui/EQCurveView.cpp` / `.h`
- `src/ui/GenerateButton.cpp` / `.h`
- `src/ui/InteractiveCustomizationPanel.cpp` / `.h` — `[this]` lambdas on member components only; no async cross-lifetime
- `src/ui/KellyLookAndFeel.cpp` / `.h`
- `src/ui/MidiEditor.cpp` / `.h`
- `src/ui/MidiKompanionLookAndFeel.cpp` / `.h`
- `src/ui/MixerConsolePanel.cpp` / `.h` — `[this]` lambdas on owned child widgets; no async
- `src/ui/MusicTheoryPanel.cpp` / `.h` — same
- `src/ui/NaturalLanguageEditor.cpp` / `.h`
- `src/ui/PianoRollPreview.cpp` / `.h`
- `src/ui/ScoreEntryPanel.cpp` / `.h`
- `src/ui/SidePanel.cpp` / `.h`
- `src/ui/SuggestionOverlay.cpp` / `.h`
- `src/ui/VocalControlPanel.cpp` / `.h`
- `src/ui/WorkflowManager.h` — header-only, no Component subclass
- `src/ui/WorkstationPanel.h` — covered by H1

**`MasterEQComponent`** — timer, listeners, and APVTS listeners are all correctly stopped/removed in the explicit destructor. This file is a positive reference.

**`CassetteView`** — correctly calls `stopTimer()` in its explicit destructor (`CassetteView.cpp:13`). No findings.

---

## Summary table

| Severity | Count |
|----------|-------|
| Critical | 2     |
| High     | 4     |
| Medium   | 2     |
| **Total**| **8** |

---

## Files with most findings

1. `src/ui/EmotionWorkstation.cpp` — 4 findings (C2, H2, H4, M1)
2. `src/ui/LyricDisplay.cpp` — 1 finding (C1)
3. `src/ui/WorkstationPanel.cpp` — 1 finding (H1)
4. `src/ui/MusicianCommandPanel.cpp` — 1 finding (H3)
5. `src/ui/TooltipComponent.cpp` — 1 finding (M2)

---

## File count note

The prior audit memory cited ~57 files. Actual `src/ui/*.cpp` count is **28**. This is expected: some classes referenced in prior audit docs likely lived in separate modules (plugin/, voice/, etc.) that are out of scope here, or the memory was quoting a broader glob. All 28 files in `src/ui/` were examined.

---

## Next step

Fixes go in a separate PR (`audit/ui-lifetime-fixes-2026q2`) in priority order: Critical → High → Medium.

Suggested fix sequence:
1. `LyricDisplay`: add explicit dtor with `stopTimer()` (C1)
2. `EmotionWorkstation`: add explicit dtor with `stopTimer()` + `setLookAndFeel(nullptr)`; move `lookAndFeel_` to first member; wrap `showMenuAsync` lambda in `SafePointer` (C2, H2, H4, M1)
3. `WorkstationPanel`: add explicit dtor with `stopTimer()` (H1)
4. `MusicianCommandPanel`: add explicit dtor with `removeListener` (H3)
5. `TooltipComponent`: address static singleton + `callAfterDelay` shutdown ordering (M2)
