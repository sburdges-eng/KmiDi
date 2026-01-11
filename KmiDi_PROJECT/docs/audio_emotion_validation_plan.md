Audio emotion extraction validation plan (music-therapy aligned)
===============================================================

Dataset selection (clinician-grounded, small)
---------------------------------------------
- Start with publicly peer-reviewed music emotion corpora that include continuous or categorical affect labels by trained raters: DEAM (valence/arousal; 1.8k excerpts) and EMO-Music (categorical emotions; ~744 clips).
- Prefer clips with available clinician or therapist annotations; when missing, sample 200–300 clips across both sets and have 2+ therapists label primary emotion + intensity on a 0–1 scale.
- If you have internal therapy notes, add them as `therapist` rows in the manifest for a gold holdout (see `val_gold` split below).
- Directory layout (non-negotiable for loaders):
  - `datasets/DEAM/{audio/,annotations/,metadata.csv}`
  - `datasets/EMO_Music/{audio/,annotations/,metadata.csv}`
  - Audio format: `.wav`, target sample rate: 22050 Hz.

Metrics
-------
- Per-emotion F1 (macro + per-class) on categorical labels.
- Concordance with clinician labels: weighted Cohen’s κ (or Fleiss’ for >2 raters) using categorical outputs.
- Calibration: Expected Calibration Error (ECE) over predicted emotion probabilities.
- Inter-rater agreement: report κ between therapists and between therapist vs public labels as the baseline ceiling.

Validation protocol
-------------------
- k-fold cross-validation (k=5) stratified by `emotion_label` and `genre`; ensure tempo and energy ranges are represented per fold.
- Hold out a therapist-annotated “gold” set (`split=val_gold`) untouched by CV for a final spot check.
- Preprocess consistently: loudness normalization to -23 LUFS target (or consistent repo default), peak-limit to -1 dBFS, resample to model sample rate.
- Input features: whatever the model uses (e.g., log-mel, wav2vec). Keep training-time and eval-time feature config identical.

Error analysis to report
------------------------
- Confusion pairs (e.g., sad↔calm, happy↔excited, calm↔neutral) via confusion matrix per fold and aggregated.
- Genre/tempo bias: per-genre and per-tempo-bin accuracy and F1; highlight drops >10% from macro mean.
- Loudness normalization effects: compare metrics with/without normalization on a subset to detect sensitivity.
- Intensity calibration: plot reliability curves per emotion; flag under- or over-confident bins.

Manifest (CSV) expectations
---------------------------
- File: `datasets/validation/audio_emotion_manifest.csv`.
- Columns: `file_path, emotion_label, intensity, genre, tempo_bpm, split, annotator_type, notes`.
- `split` values: `train` for CV folds, `val_gold` for therapist holdout. Add a `fold` column if you pre-assign folds; otherwise stratify programmatically.
- `annotator_type`: `clinician` (public dataset labels) vs `therapist` (internal gold).
- Keep paths workspace-relative. Ensure files exist before running evaluation.
- Unified (valence/arousal) manifest:
  - File: `datasets/validation/emotion_manifest.json` created by `scripts/normalize_emotion_datasets.py`.
  - Schema: `id, dataset (DEAM|EMO_Music), audio_path, valence, arousal, split (train|val|test)`.
  - Reproducible splits via `--train-ratio/--val-ratio/--seed`.

Baseline evaluation sketch
--------------------------
- Model inference: run your current audio emotion classifier to emit per-class probabilities for each manifest row.
- Metrics:
  - F1: macro + per-class on `emotion_label`.
  - Calibration: ECE over predicted emotion probabilities.
  - Concordance: Cohen’s κ between model argmax labels and clinician/therapist labels.
  - Inter-rater: κ among human raters as a reference.
- Confusion matrix: aggregate over folds, normalize per true class, and save a figure (e.g., `output/plots/confusion_matrix.png`).

Suggested implementation steps
------------------------------
1) Download DEAM and EMO-Music into `datasets/DEAM` and `datasets/EMO_Music`; run loudness normalization and verify sample rates.
2) Fill `datasets/validation/audio_emotion_manifest.csv`; add `fold` if you want deterministic splits.
3) Add a small eval script (e.g., `scripts/eval_audio_emotion.py`) to:
   - Load manifest, optionally stratify/split.
   - Run model inference (batch or streaming).
   - Compute F1, κ, ECE; log per-genre/tempo metrics.
   - Save confusion matrix and reliability curves.
4) Hold back `val_gold` rows from any hyperparameter tuning; report them separately.

Outputs to share
----------------
- Metrics table: macro F1, per-class F1, κ (model vs clinician/therapist), ECE.
- Confusion matrix plot; per-genre/tempo breakdown.
- Short error notes: top confusion pairs, any loudness normalization sensitivity, calibration notes.
