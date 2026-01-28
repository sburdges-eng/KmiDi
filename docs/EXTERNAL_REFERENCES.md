# External References & Data Sources

References to external projects, datasets, and APIs used by KmiDi.

---

## Magenta (Music & Art with ML)

- **Repo:** [github.com/magenta/magenta](https://github.com/magenta/magenta)  
  *Note: This repository was [archived](https://github.com/magenta/magenta) by the owner (Jan 2026). It is read-only; the project has moved to individual repos under the [Magenta GitHub Organization](https://github.com/magenta).*
- **What we use:**  
  - **Groove MIDI Dataset** – expressive drum performances (e.g. `magentadata/datasets/groove/groove-v1.0.0-midionly.zip`) for groove training.  
  - **MAESTRO**, **NSynth**, and other datasets served from `storage.googleapis.com/magentadata/` (see `scripts/prepare_datasets.py` / `scripts/utilities/prepare_datasets.py`).
- **Docs / current work:** [magenta.tensorflow.org](https://magenta.tensorflow.org), [Magenta.js](https://github.com/magenta/magenta-js) for browser models.

---

## Other references

- **JUCE:** [github.com/juce-framework/JUCE](https://github.com/juce-framework/JUCE) – audio/plugin framework (see `external/JUCE`).
- **Path and dependency audit:** `docs/EXTERNAL_PATH_REFERENCES.md` (paths outside KmiDi-compile).
