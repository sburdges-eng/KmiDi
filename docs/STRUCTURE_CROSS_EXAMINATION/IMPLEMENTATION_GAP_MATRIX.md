# Implementation Gap Matrix

This matrix captures the executable follow-up set from the structure cross-examination.

| Area | Required | Current | Action |
|---|---|---|---|
| API intent flow | Full intent first | Partially full + fallback | Defaulted to full pipeline in `music_brain/api.py` |
| Harmony deps path | `music_brain/harmony/deps` | Missing | Added bridge package + compatibility module |
| Voice runtime classification | Export + runtime wrapper | Missing | Added `voice_classifier.py` and API endpoint |
| Audio render integration | MIDI-to-audio in generation path | Missing | Added `render_midi_to_audio` + API integration |
| Build instructions | Kelly naming and targets | Missing root `BUILD.md` | Added root `BUILD.md` |

## Next Deep-Dive Items

- Expand plugin/C++ boundary verification.
- Tighten type consistency across Python/TS/C++ bridge.
- Formalize CI checks for spec compliance.
