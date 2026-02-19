# Research AI Specification

## Goal

Provide a background research assistant for Kelly that can answer music/emotion questions with citations and strict guardrails.

## Scope

- Corpus-backed RAG over curated markdown/spec docs.
- Citation-required responses (source path + section).
- Non-clinical behavioral policy: no medical diagnosis/treatment guidance.
- Local persona/state by default; optional encrypted cloud sync.

## Inputs

- Emotion intent text.
- Internal docs (`docs/`, specs, architecture notes).
- 6x6x6 emotion thesaurus data (see ingestion plan).

## Outputs

- Ranked answer with citations.
- Optional structured mapping:
  - `valence`
  - `arousal`
  - `suggested_mode`
  - `tempo_range`

## Guardrails

- Block medical/legal advice.
- Mark uncertainty and missing evidence.
- Refuse unsupported claims without citations.

## Service Contract

- Endpoint concept: `/research/query`
- Request:
  - `question: str`
  - `context: optional`
- Response:
  - `answer: str`
  - `citations: list`
  - `confidence: float`
  - `safety_flags: list`
