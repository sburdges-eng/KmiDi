# Transfer Pass 02 (MIDI)
Date: 2026-02-13
Mode: manifest-driven selective ingest, no-overwrite (cp -n), checksum dedupe

## Sources
- /Volumes/KmiDi-external/DatasetsEXTERNAL/midi
- /Volumes/KmiDi-external/DevEXTERNAL/_FORENSIC_READONLY_KMIDI (filtered to examples/midi and CODE/mid paths)

## Manifests
- /Volumes/KmiDi-external/musicgen-local/ml/data/manifests/sources/midi/2026-02-13-midi-sources.txt
- /Volumes/KmiDi-external/musicgen-local/ml/data/manifests/sources/midi/2026-02-13-midi-sha256.txt
- /Volumes/KmiDi-external/musicgen-local/ml/data/manifests/sources/midi/2026-02-13-midi-dedupe-map.tsv

## Results
- Manifest entries: 326
- Newly copied this pass: 15
- Skipped (already present): 311
- Duplicate files removed (checksum dedupe): 15
- Final MIDI files in staging: 311

## Destination
- /Volumes/KmiDi-external/musicgen-local/ml/data/raw/midi_staging
