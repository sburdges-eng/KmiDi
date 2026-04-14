# noexcept-lie RT Audit — 2026Q2

Scan produced by `scripts/audit/noexcept_rt_audit.py`. 21 findings across 2 source roots.

Each entry is a function declared `noexcept` whose body allocates, locks, or throws. In an RT callback context each is one failure away from `std::terminate`. AU/VST3 validation rejects plugins whose `processBlock` reaches any of these.

**Flag legend**:
- `new` / `lock` / `throw` → CRIT (program will terminate on failure)
- `string-alloc` / `container-grow` / `to_string` → HIGH (implicit heap alloc via libc++; bad_alloc escapes as terminate)

## CRIT (2)

- `src/core/memory.cpp:33` — `MutexMemoryPool::allocate` [lock]
- `src/core/memory.cpp:44` — `MutexMemoryPool::deallocate` [lock]

## HIGH (15)

- `src/groove/GrooveEngine.cpp:165` — `GrooveEngine::detectTimeSignature` [container-grow]
- `src/groove/GrooveEngine.cpp:242` — `GrooveEngine::analyzeSwing` [container-grow]
- `src/harmony/VoiceLeading.cpp:12` — `VoiceLeading::findOptimalVoicing` [container-grow]
- `src/harmony/VoiceLeading.cpp:125` — `VoiceLeading::generateVoicingCandidates` [container-grow]
- `src/harmony/VoiceLeading.cpp:253` — `VoiceLeading::voiceProgression` [container-grow]
- `src/prrot/AudioValidator.cpp:14` — `AudioValidator::validate` [to_string, container-grow]
- `src/prrot/AudioValidator.cpp:166` — `AudioValidator::estimateNoiseFloor` [container-grow]
- `src/prrot/MidiShaper.cpp:11` — `MidiShaper::shapeMidiNotes` [to_string, container-grow]
- `src/prrot/MidiShaper.cpp:154` — `MidiShaper::computeNoteProbabilities` [container-grow]
- `src/prrot/PRROTEngine.cpp:204` — `PRROTEngine::analyzePhonemes` [to_string, container-grow]
- `src/prrot/PRROTEngine.cpp:270` — `PRROTEngine::detectBreathMarkers` [container-grow]
- `src/prrot/PhonemeSegmenter.cpp:49` — `PhonemeSegmenter::segment` [to_string, container-grow]
- `src/prrot/PitchTracker.cpp:91` — `PitchTracker::trackPitchSequence` [container-grow]
- `src_penta-core/harmony/VoiceLeading.cpp:12` — `VoiceLeading::findOptimalVoicing` [container-grow]
- `src_penta-core/harmony/VoiceLeading.cpp:125` — `VoiceLeading::generateVoicingCandidates` [container-grow]

## MED (4)

- `src/prrot/BreathDetector.cpp:14` — `BreathDetector::detectBreath` [to_string]
- `src/prrot/EnvelopeGenerator.cpp:12` — `EnvelopeGenerator::generateArticulationEnvelope` [to_string]
- `src/prrot/PRROTEngine.cpp:47` — `PRROTEngine::processAudioSegment` [to_string]
- `src/prrot/PitchTracker.cpp:17` — `PitchTracker::trackPitch` [to_string]

## Fix patterns

- **`container-grow`**: pre-allocate `std::vector::reserve(max_n)` in `prepareToPlay`; write via index assignment, never `push_back`. Or switch to `boost::container::static_vector<T, N>` / `std::array<T, N>` with a size counter.
- **`to_string` / `string-alloc`**: replace log lines with `juce::Logger::writeToLog` gated by `#ifdef JUCE_DEBUG`, or move the log to the UI thread via a lock-free SPSC queue.
- **`new`**: allocate in `prepareToPlay`; the RT path only writes into preallocated buffers.
- **`lock`**: replace with `std::atomic<T>` or a lock-free SPSC queue (`readerwriterqueue` is already linked).
- **`throw`**: return an error code or `std::optional`. RT code cannot throw across audio callbacks.

## CI gate suggestion

Once CRIT count is zero, add to the CI workflow:
```yaml
- name: noexcept RT audit (regression gate)
  run: python3 scripts/audit/noexcept_rt_audit.py > /dev/null
```

The script exits non-zero if any CRIT finding exists.
