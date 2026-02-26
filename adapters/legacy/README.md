# Legacy Adapters

This directory is reserved for adapter wrappers around imported KmiDi/iDAW modules.

Current scaffolding:

- `NoOpAdapters.h`: default no-op implementations for all interface domains.
- `EmotionThesaurusEnricher.*`: first real read-only intent enricher (flag-gated by `KELLY_ENABLE_EMOTION_ENRICHMENT`).

Rules:

- adapters must never bypass `IntentResult` safety semantics.
- failures must degrade to `ABSTAIN` or `INVALID`, never synthetic fallback output.
- escalation behavior must remain explicit and token-gated.
