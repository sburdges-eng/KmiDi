# Memory & Safety Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 12 memory leak / safety findings from recursive audit — 5 HIGH, 5 MEDIUM, 2 LOW.

**Architecture:** Group by subsystem to minimize file-reading overhead. 6 tasks, each self-contained.

**Tech Stack:** C++20, Python 3.11+, ONNX Runtime, JUCE 8

---

## Task 1: Remove hardcoded model paths (HIGH × 2)

**Files:**
- Modify: `src/plugin/PluginProcessor.cpp:345-365`

- [ ] **Step 1: Read current prepareToPlay fallback logic**

Read `src/plugin/PluginProcessor.cpp` lines 340-370.

- [ ] **Step 2: Replace hardcoded paths with env/bundle-relative lookup**

Replace the two hardcoded `/Users/seanburdges/...` fallbacks with:

```cpp
// Model paths: prefer bundle-relative, fall back to KELLY_MODEL_ROOT env var.
// Never use hardcoded absolute paths (data governance violation).
auto modelRoot = []() -> std::filesystem::path {
    if (auto* env = std::getenv("KELLY_MODEL_ROOT"))
        return std::filesystem::path(env);
    return std::filesystem::path{}; // no fallback — fail explicitly
}();

auto jepaPath = modelRoot / "audio_jepa_v01.onnx";
auto probePath = modelRoot / "emotion_probe_v01.onnx";
```

- [ ] **Step 3: Build and verify**

Run: `cmake --build build --target KellyFFI -j8`
Expected: clean build, no hardcoded path warnings.

- [ ] **Step 4: Commit**

```bash
git add src/plugin/PluginProcessor.cpp
git commit -m "fix(plugin): remove hardcoded model paths, use KELLY_MODEL_ROOT

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Replace pickle.load with safe alternatives (HIGH × 2)

**Files:**
- Modify: `music_brain/learning/openweight_learning.py:230-250,400-420`

- [ ] **Step 1: Read the two pickle.load call sites**

Read lines 225-250 and 395-420.

- [ ] **Step 2: Replace with torch.load(weights_only=True)**

At line ~237:
```python
# SECURITY: Use weights_only=True to prevent arbitrary code execution
# from crafted .pkl files (pickle deserialization attack).
weights = torch.load(path, map_location="cpu", weights_only=True)
```

At line ~409 (load_all_weights):
```python
weights = torch.load(pkl_path, map_location="cpu", weights_only=True)
```

If any call site actually uses `pickle.load` directly (not `torch.load`), replace with:
```python
import safetensors.torch as st
weights = st.load_file(path)
```
or if safetensors isn't available, at minimum add:
```python
# WARNING: pickle is unsafe. Migrate to safetensors when possible.
weights = torch.load(path, map_location="cpu", weights_only=True)
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/ -q -k "learning or weight"`

- [ ] **Step 4: Commit**

```bash
git add music_brain/learning/openweight_learning.py
git commit -m "fix(security): replace pickle.load with weights_only=True

Prevents arbitrary code execution from crafted .pkl weight files.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Fix PowerShell command injection (HIGH)

**Files:**
- Modify: `music_brain/agents/unified_hub.py:255-280`

- [ ] **Step 1: Read the TTS subprocess call**

Read lines 250-280.

- [ ] **Step 2: Replace shell interpolation with temp file approach**

```python
import tempfile

def _speak_windows(self, text: str) -> None:
    """TTS via PowerShell — write text to temp file to avoid injection."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        tmp_path = f.name
    try:
        subprocess.run(
            ["powershell", "-Command",
             f"Get-Content '{tmp_path}' | ForEach-Object {{ "
             f"Add-Type -AssemblyName System.Speech; "
             f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
             f"$s.Speak($_) }}"],
            timeout=30,
        )
    finally:
        os.unlink(tmp_path)
```

Or simpler — use `pyttsx3` if available, shell-out only as last resort.

- [ ] **Step 3: Commit**

```bash
git add music_brain/agents/unified_hub.py
git commit -m "fix(security): prevent command injection in Windows TTS path

Write text to temp file instead of interpolating into PowerShell command.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Fix RT safety — mutex in processBlock + LockFreeQueue alloc (MEDIUM × 2)

**Files:**
- Modify: `src/plugin/PluginProcessor.cpp:570-600` (mutex → double-buffer)
- Modify: `src/plugin/PluginProcessor.h` (add shadow buffer member)
- Modify: `src/core/memory.cpp:85-100` (document or fix RT-unsafe alloc)

- [ ] **Step 1: Read processBlock mutex usage**

Read `src/plugin/PluginProcessor.cpp` lines 565-610 and the header.

- [ ] **Step 2: Replace try_lock pattern with atomic double-buffer**

In the header, add:
```cpp
// Double-buffer for generated MIDI — RT thread reads current, non-RT writes shadow then swaps.
struct GeneratedMidiData {
    std::vector<juce::MidiMessage> chords;
    std::vector<juce::MidiMessage> melody;
    std::vector<juce::MidiMessage> bass;
};
std::array<GeneratedMidiData, 2> midiBuffers_;
std::atomic<int> activeMidiBuffer_{0};
```

In processBlock, replace:
```cpp
// RT-safe: read from active buffer (no lock)
const auto& midi = midiBuffers_[activeMidiBuffer_.load(std::memory_order_acquire)];
```

In the non-RT setter:
```cpp
// Non-RT: write to shadow buffer, then swap
int shadow = 1 - activeMidiBuffer_.load(std::memory_order_relaxed);
midiBuffers_[shadow] = newData;
activeMidiBuffer_.store(shadow, std::memory_order_release);
```

- [ ] **Step 3: Add RT-unsafe documentation to LockFreeQueue**

In `src/core/memory.cpp`, add comment at class level:
```cpp
// WARNING: Despite the name, this queue heap-allocates nodes in push().
// NOT safe for real-time audio threads. Use moodycamel::ReaderWriterQueue
// or pre-allocated ring buffer for RT paths.
```

- [ ] **Step 4: Build and test**

Run: `cmake --build build --target KellyFFI -j8 && cmake --build build --target KellyCore -j8`

- [ ] **Step 5: Commit**

```bash
git add src/plugin/PluginProcessor.cpp src/plugin/PluginProcessor.h src/core/memory.cpp
git commit -m "fix(rt): replace mutex in processBlock with lock-free double-buffer

Audio thread was using try_lock which risks MIDI dropouts and is
not truly RT-safe. Switch to atomic buffer swap pattern.
Also document that LockFreeQueue::push() heap-allocates (not RT-safe).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Fix ONNX void* leak + FFI mutex deadlock risks (MEDIUM × 3)

**Files:**
- Modify: `src/ml/ONNXInference.cpp:85-115,165-180` (void* → unique_ptr)
- Modify: `src/ml/ONNXInference.h` (member types)
- Modify: `src/bridge/kelly_ffi.cpp:735-755` (callback outside lock)
- Modify: `src/bridge/intent_ir_ffi.cpp:215-230` (validate outside lock)

- [ ] **Step 1: Read ONNXInference raw new pattern**

Read `src/ml/ONNXInference.cpp` lines 80-120 and the header.

- [ ] **Step 2: Replace void* with unique_ptr**

In the header:
```cpp
std::unique_ptr<Ort::Env> env_;
std::unique_ptr<Ort::Session> session_;
std::unique_ptr<Ort::MemoryInfo> memoryInfo_;
```

In loadModel():
```cpp
env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "KMiDi");
session_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), sessionOptions);
memoryInfo_ = std::make_unique<Ort::MemoryInfo>(
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
```

Remove manual delete in destructor — unique_ptr handles it.

- [ ] **Step 3: Fix kelly_ffi.cpp callback-under-mutex**

At line ~742, change pattern from:
```cpp
std::lock_guard<std::mutex> lock(wrapper->mutex);
// ... modify state ...
wrapper->callback(event);  // BAD: callback under lock
```
To:
```cpp
decltype(wrapper->callback) cb;
EventData event_copy;
{
    std::lock_guard<std::mutex> lock(wrapper->mutex);
    // ... modify state ...
    cb = wrapper->callback;
    event_copy = event;
}
if (cb) cb(event_copy);  // callback OUTSIDE lock
```

- [ ] **Step 4: Fix intent_ir_ffi.cpp validate-under-mutex**

At line ~222, same pattern — copy frame, release lock, validate, re-acquire:
```cpp
IntentFrame frame_copy;
{
    std::lock_guard<std::mutex> lock(g_state.mutex);
    frame_copy = g_state.current_frame;
}
bool valid = validate_intent_frame_ffi(&frame_copy);
if (valid) {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    g_state.current_frame = frame_copy;
}
```

- [ ] **Step 5: Build**

Run: `cmake --build build -j8`

- [ ] **Step 6: Commit**

```bash
git add src/ml/ONNXInference.cpp src/ml/ONNXInference.h src/bridge/kelly_ffi.cpp src/bridge/intent_ir_ffi.cpp
git commit -m "fix(memory): unique_ptr for ONNX, callbacks outside mutex

- Replace void* raw new with unique_ptr for Ort::Env/Session/MemoryInfo
  (prevents leak if Session ctor throws).
- Move FFI callback invocation outside mutex to prevent deadlock on
  re-entry from Rust/Tauri side.
- Move intent_ir validation outside mutex for same reason.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Low-severity hardening (LOW × 2)

**Files:**
- Modify: `src/bridge/kelly_ffi.cpp:50-60` (fixed-size error buffer)
- Modify: `include/penta/common/RTMemoryPool.h:45-55` (debug assertion)

- [ ] **Step 1: Replace std::string error storage with fixed buffer**

```cpp
// Thread-local fixed-size error buffer — no heap alloc, RT-safe.
static thread_local char tl_error_buf[512] = {};

void set_last_error(const char* msg) {
    std::strncpy(tl_error_buf, msg ? msg : "", sizeof(tl_error_buf) - 1);
    tl_error_buf[sizeof(tl_error_buf) - 1] = '\0';
}

const char* kelly_get_error_message() {
    return tl_error_buf;
}
```

- [ ] **Step 2: Add debug SPSC assertion to RTMemoryPool**

```cpp
#ifndef NDEBUG
    std::atomic<std::thread::id> owner_thread_{};

    void assertSingleProducer() {
        auto expected = std::thread::id{};
        auto current = std::this_thread::get_id();
        if (!owner_thread_.compare_exchange_strong(expected, current))
            assert(expected == current && "RTMemoryPool: SPSC violation — multiple producers");
    }
#else
    void assertSingleProducer() {}
#endif
```

Call `assertSingleProducer()` at the top of `push()`.

- [ ] **Step 3: Build and test**

Run: `cmake --build build -j8 && ctest --test-dir build --output-on-failure` (if BUILD_TESTS=ON)

- [ ] **Step 4: Commit**

```bash
git add src/bridge/kelly_ffi.cpp include/penta/common/RTMemoryPool.h
git commit -m "fix(rt): fixed-size error buffer + SPSC debug assertion

- set_last_error now uses thread_local char[512] instead of std::string
  (safe if ever called from RT context).
- RTMemoryPool gets debug-mode thread ID assertion to catch SPSC
  contract violations early.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Execution order

Tasks 1-3 (HIGH) are independent — can run in parallel.
Task 4 depends on reading processBlock first.
Task 5 touches 4 files across 2 subsystems but no overlap with others.
Task 6 is independent.

**Recommended:** Parallel dispatch Tasks 1+2+3, then 4+5+6.
