# Core Bridge

Reserved for wrappers that delegate validation/enrichment to `minimal_listening_core`.

Primary guardrail:

- bridge failures must not produce output; fail closed to `ABSTAIN` or `INVALID`.
