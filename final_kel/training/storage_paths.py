"""
Canonical storage locations for training assets.

All defaults point to an external drive. Override with:
  KMIDI_DATA_ROOT=/Volumes/KMIDI_DATA
  KMIDI_RUN_ID=<run_id>
  KMIDI_DATASET_VERSION=<dataset_version>
"""

import os

ENV_DATA_ROOT = "KMIDI_DATA_ROOT"
ENV_EXTERNAL_ROOT = "KMIDI_EXTERNAL_ROOT"
ENV_RUN_ID = "KMIDI_RUN_ID"
ENV_DATASET_VERSION = "KMIDI_DATASET_VERSION"

DEFAULT_EXTERNAL_ROOT = "/Volumes/KMIDI_DATA"
FALLBACK_EXTERNAL_ROOTS = (
    DEFAULT_EXTERNAL_ROOT,
    "/Volumes/KmiDi-external/KMIDI_DATA",
    "/Volumes/KmiDi-external/KmiDiEXTERNAL/KMIDI_DATA",
    "/Volumes/KmiDi-external/KmiDiEXTERNAL/KmiDi_TRAINING_DATA",
    "/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi_TRAINING_DATA",
    "/Volumes/KmiDi-external/COLD_STORAGEEXTERNAL/KmiDi_TRAINING_DATA",
)


def _normalize(path):
    return os.path.abspath(os.path.expanduser(path))


def _can_create_under(path):
    parent = os.path.dirname(path.rstrip("/")) or "/"
    return os.path.isdir(parent) and os.access(parent, os.W_OK)


def resolve_external_root():
    env_root = os.environ.get(ENV_DATA_ROOT)
    if env_root:
        return _normalize(env_root)
    env_root = os.environ.get(ENV_EXTERNAL_ROOT)
    if env_root:
        return _normalize(env_root)
    for candidate in FALLBACK_EXTERNAL_ROOTS:
        if os.path.isdir(candidate) or _can_create_under(candidate):
            return _normalize(candidate)
    return _normalize(DEFAULT_EXTERNAL_ROOT)


EXTERNAL_ROOT = resolve_external_root()

RUN_ID = os.environ.get(ENV_RUN_ID, "local")
DATASET_VERSION = os.environ.get(ENV_DATASET_VERSION, "current")

DATASETS_DIR = os.path.join(EXTERNAL_ROOT, "datasets")
AUDIO_RAW_DIR = os.path.join(DATASETS_DIR, "audio", "raw")
AUDIO_NORMALIZED_DIR = os.path.join(DATASETS_DIR, "audio", "normalized", "sr_44100_lufs_-14")
AUDIO_FEATURES_MELS_DIR = os.path.join(DATASETS_DIR, "audio", "features", "mels")
MIDI_RAW_DIR = os.path.join(DATASETS_DIR, "midi", "raw")
MIDI_NORMALIZED_DIR = os.path.join(DATASETS_DIR, "midi", "normalized")
MIDI_TOKENIZED_DIR = os.path.join(DATASETS_DIR, "midi", "tokenized", "v1")
PAIRED_SPECTOCLOUD_ROOT = os.path.join(DATASETS_DIR, "paired", "spectocloud")
PAIRED_DATASET_DIR = os.path.join(PAIRED_SPECTOCLOUD_ROOT, DATASET_VERSION)
PAIRED_MANIFESTS_DIR = os.path.join(PAIRED_DATASET_DIR, "manifests")
PAIRED_INDEXES_DIR = os.path.join(PAIRED_DATASET_DIR, "indexes")

AUDIO_DATASETS_DIR = AUDIO_RAW_DIR
MIDI_DATASETS_DIR = MIDI_RAW_DIR
PROCESSED_DATASET_FILE = os.path.join(PAIRED_INDEXES_DIR, "kmidi_multimodal.npy")

PAIRED_TRAIN_MANIFEST = os.path.join(PAIRED_MANIFESTS_DIR, "train.jsonl")
PAIRED_VAL_MANIFEST = os.path.join(PAIRED_MANIFESTS_DIR, "val.jsonl")
PAIRED_TEST_MANIFEST = os.path.join(PAIRED_MANIFESTS_DIR, "test.jsonl")
PAIRED_FILE_INDEX_PARQUET = os.path.join(PAIRED_INDEXES_DIR, "file_index.parquet")

RELEASES_DIR = os.path.join(EXTERNAL_ROOT, "manifests", "releases", DATASET_VERSION)
LINEAGE_DIR = os.path.join(EXTERNAL_ROOT, "manifests", "lineage", DATASET_VERSION)
MANIFEST_RUN_FILE = os.path.join(RELEASES_DIR, "manifest_run.json")
SOURCE_HASHES_FILE = os.path.join(LINEAGE_DIR, "source_hashes.json")

STAGING_AWS_SYNC_DIR = os.path.join(EXTERNAL_ROOT, "staging", "aws_sync", RUN_ID)
CHECKPOINTS_DIR = os.path.join(EXTERNAL_ROOT, "checkpoints_mirror", "cloud", RUN_ID)
EXPORT_DIR = os.path.join(CHECKPOINTS_DIR, "export")
EXPORT_AUDIO_ONNX = os.path.join(EXPORT_DIR, "audio_jepa.onnx")
EXPERIMENTS_DIR = os.path.join(EXTERNAL_ROOT, "logs", "dataset_builds")
GENERATED_DIR = os.path.join(STAGING_AWS_SYNC_DIR, "generated")
RELEASE_DIR = os.path.join(STAGING_AWS_SYNC_DIR, "release")
QUARANTINE_DIR = os.path.join(EXTERNAL_ROOT, "quarantine", "rejected_or_corrupt")


def ensure_storage_dirs():
    os.makedirs(AUDIO_RAW_DIR, exist_ok=True)
    os.makedirs(AUDIO_NORMALIZED_DIR, exist_ok=True)
    os.makedirs(AUDIO_FEATURES_MELS_DIR, exist_ok=True)
    os.makedirs(MIDI_RAW_DIR, exist_ok=True)
    os.makedirs(MIDI_NORMALIZED_DIR, exist_ok=True)
    os.makedirs(MIDI_TOKENIZED_DIR, exist_ok=True)
    os.makedirs(PAIRED_MANIFESTS_DIR, exist_ok=True)
    os.makedirs(PAIRED_INDEXES_DIR, exist_ok=True)
    os.makedirs(RELEASES_DIR, exist_ok=True)
    os.makedirs(LINEAGE_DIR, exist_ok=True)
    os.makedirs(STAGING_AWS_SYNC_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)
    os.makedirs(RELEASE_DIR, exist_ok=True)
    os.makedirs(QUARANTINE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_DATASET_FILE), exist_ok=True)
