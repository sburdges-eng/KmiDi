# WavJEPA pretrained waveform encoder released

**Item:** WavJEPA pretrained waveform encoder released  
**Task:** Integrate and plan acquisition of public WavJEPA checkpoints as frozen feature extractors.  
**Secondary tasks:** Use as frozen front-end per WAVJEPA_LATENT_PIPELINE; capture checkpoint id, preprocessing contract, license, and hash before any download or freeze-lane use.  
**Context:** WavJEPA; exp_002 references labhamlet/wavjepa-base (HF).  
**Verification status:** VERIFIED_WITH_PRIMARY_SOURCES  
**Verification basis:** official Hugging Face model cards plus official GitHub repository README and license.  
**Known facts:**
- Public checkpoints exist on Hugging Face, including `labhamlet/wavjepa-base` and `labhamlet/wavjepa-nat-base`.
- The official usage path is Hugging Face `AutoModel.from_pretrained(...)` plus `AutoFeatureExtractor.from_pretrained(...)`, currently with `trust_remote_code=True`.
- The public inference examples use 16 kHz waveform input and 2 s windows (`160000` samples for mono; Nat examples use multi-channel input).
- Upstream preprocessing guidance calls for RMS normalization followed by instance normalization.
- The official GitHub codebase `labhamlet/wavjepa` is BSD-3-Clause licensed.
- The Hugging Face model card for `labhamlet/wavjepa-base` lists the checkpoint license as MIT.
- KmiDi's current design intent remains unchanged: use WavJEPA only as a frozen feature extractor.
**Unknowns:** Exact checkpoint SHA256 to pin; whether KmiDi should vendor a local copy or keep weights in `KELLY_MODELS_PATH`; whether `trust_remote_code=True` should be replaced with a vendored local loader before freeze-readiness.  
**Assumptions:** The chosen checkpoint remains public and can be mirrored into an offline local artifact store.  
**Constraints:** No runtime cloud dependency; no fine-tuning or in-repo training of WavJEPA predictor/target encoder; keep acquisition reproducible with manifest + hash.  
**Ambiguities:** Base vs Nat checkpoint for KmiDi use cases; pooling strategy for downstream probes; PyTorch vs ONNX vs Core ML export target.  
**Source text / data:**
- Hugging Face model card: `labhamlet/wavjepa-base`
- Hugging Face model card: `labhamlet/wavjepa-nat-base`
- GitHub repository: `labhamlet/wavjepa`
**Captured guidance:**
- Treat WavJEPA as a ready-to-use frozen waveform foundation model for feature extraction and downstream probes.
- Verify the license of the specific checkpoint artifact, not just the repository license, before commercial or release use.
- Prefer exporting a pinned local artifact (PyTorch, ONNX, or Core ML) for production paths instead of relying on live remote model fetches.
**Output format:** Manifest entry with checkpoint id, upstream URLs, license, SHA256, preprocessing contract, approved storage path, and export status.
