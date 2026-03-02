# MT3 transcription baseline

**MT3** (Multi-Task Multitrack Music Transcription) is an open-source, Transformer-based model (T5X) that transcribes multi-instrument audio into MIDI-like tokens. It provides a runnable sequence-to-sequence baseline for automatic music transcription — analogous to speech-to-text but for music — with pretrained checkpoints (piano-only and multi-instrument) and an official **Google Colab notebook** for inference without local setup.

- **Paper / task:** Formalizes audio → symbolic transcription; predicts note events (pitch, timing, instrument) from raw audio with a learned token vocabulary.
- **Repo / Colab:** [Google MT3](https://github.com/google-research/melody-transcription) (inference examples, checkpoints). Official Colab lets you upload audio and get transcribed MIDI.
- **Licensing:** Official MT3 checkpoints are typically **Apache-2.0**. Community forks (e.g. **YourMT3**) may use **GPL**; check compatibility before combining with Apache-2.0 assets.

**Relation to KmiDi:** MT3 is a candidate **token decoder** for latent audio representations (e.g. WavJEPA / JTFS latents → MT3-style transformer head or adapted vocabulary). It offers a real transformer token head to adapt or probe when bridging latent audio to symbolic output. Ongoing work on MT3 (e.g. mitigating instrument leakage in multi-track output) is relevant if we extend toward latent→symbol pipelines.
