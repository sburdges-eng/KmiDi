# 6x6x6 Ingestion Plan

## Sources

- `emotion_thesaurus/metadata.json`
- Emotion JSON collections and blend mappings
- Legacy markdown notes and mappings

## Ingestion Pipeline

1. Discover source files and validate JSON schema.
2. Normalize records into:
   - emotion id
   - descriptors
   - blend relations
   - musical mapping hints
3. Chunk and index for retrieval.
4. Persist source hash and ingest timestamp for reproducibility.

## Index Fields

- `source_path`
- `emotion`
- `blend`
- `keywords`
- `musical_parameters`
- `content`

## Validation

- Reject malformed JSON rows.
- Flag duplicate records with hash collision report.
- Require at least one citation-bearing source per indexed chunk.
