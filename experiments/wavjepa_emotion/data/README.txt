Emotion experiment data (in place). The loader finds data here by default.

Layout:
  data/
    emotions/
      ravdess/   <- RAVDESS WAVs (or symlink); empty until you add/symlink
      cremad/    <- symlink to CREMA-D (7439 WAVs) on Sean's SSD

CREMA-D source (symlinked): Datasets_Project/emotion_cremad/CREMA-D-master/AudioWAV on Sean's SSD (imported_dataset_outputs).

Override: set KMIDI_DATASETS_PATH or AUDIO_DATA_ROOT to use another root.
Prepare RAVDESS: scripts/utilities/prepare_datasets.py --dataset emotion_ravdess --download.
