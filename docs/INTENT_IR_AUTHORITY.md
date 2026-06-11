# Intent IR Authority

Status: authoritative for approved workbook Pass C
Last updated: 2026-06-08

Purpose
- Define where Intent IR is mandatory.
- Define who owns intent meaning.
- Define how schema/version changes happen.
- Prevent contract drift between UI, Python, Rust, C++, and persistence.

## Core rule

Intent IR is canonical for:
- engine-facing intent
- persisted intent

Nothing in those paths may bypass validated Intent IR.

## Systems that must speak Intent IR natively

- native engine
- plugin/runtime
- Python backend
- persistence layer

## Systems that may use looser intermediate forms

These may exist only before canonical validation:
- UI drafts
- Python orchestration representations
- API request adapters and DTOs
- importers and converters
- migration-normalized forms

## Forbidden bypasses

These paths must not skip Intent IR validation:
- plugin -> engine
- persisted project intent -> load/apply pipeline
- backend -> engine generation requests

## Semantic authority order

When meanings conflict, the winner order is:
1. `shared_schemas/`
2. Rust validator
3. product docs
4. Python request schema

Meaning ownership is human-owned and expressed through:
- `shared_schemas/`
- validator logic
- architecture/product docs

## Canonical validator of last resort

- Rust validator

This means:
- UI preflight is useful but not final
- Python/API validation is useful but not final
- C++ boundary checks are useful but not final
- persistence-load validation is required but not final without Rust truth

## Required validation points

Intent must be validated at:
- UI preflight
- API boundary
- Rust validator
- C++ boundary
- persistence load

## Intended flow

Preferred flow:
1. UI draft
2. normalization
3. validate Intent IR
4. orchestrate
5. engine adaptation
6. playback/render

## Loss and defaults rules

Lossy transformations are allowed only:
- before canonical validation
- in derived playback hints

Defaults may apply only at:
- normalization stage
- migration layer

Defaults do not belong in:
- engine runtime truth

## Immutability rule

After validation, persisted and engine-facing intent is immutable.

That does not prohibit:
- creating a new validated intent snapshot
- migrating an older intent into a newer validated intent
- deriving non-canonical playback hints from canonical intent

It does prohibit:
- ad hoc mutation of canonical meaning after validation
- silent coercion of invalid input into engine truth
- hidden runtime-only semantic rewrites

## Versioning policy

Explicit schema versioning is required now for:
- persisted projects
- external API surfaces

Breaking schema changes are allowed only with:
- explicit version boundary
- human review

Compatibility window
- persisted projects: forever
- API compatibility: transitional only

Old saved projects
- must remain loadable through migration

Version negotiation posture
- hybrid

## Change process when semantics change

If intent meaning changes, the required process is:
1. update schema source of truth
2. update Rust validator
3. regenerate generated types and mirrors
4. update migration docs
5. add compatibility tests
6. human review before merge

Required generated artifacts that are never hand-edited:
- `src/types/Intent.ts`
- `engine/intent_ir/src/generated/intent.rs`
- Python validation mirrors

Preferred sync pattern
- codegen

## Drift checks that must fail

The following checks are mandatory gates:
- schema sync check
- Rust tests
- Python schema tests
- generated diff checks

## Repo anchors

Primary authority files and directories:
- `shared_schemas/CompleteSongIntentRequest.json`
- `shared_schemas/intent_frame_schema.json`
- `shared_schemas/emotion_schema.json`
- `scripts/sync_entities.py`
- `engine/intent_ir/src/ffi.rs`
- `engine/intent_ir/src/generated/`
- `src/types/Intent.ts`
- `music_brain/engine_api/schema.py`

Related but narrower surface
- `/generate` request handling in `music_brain/api.py` is an API boundary contract, not the top semantic authority for engine truth

## Failure split

User-facing failures
- bad inputs
- unsupported combinations
- user-correctable contract violations

Internal failures
- contract drift
- invariant violations
- schema/validator mismatch
- generated artifact drift

## Human review requirements

Always require human review for:
- exported intent semantic changes
- breaking schema changes
- migration policy changes
- any change that alters persisted meaning

Agent edits are allowed for:
- schema plus generated artifact updates
- validator alignment
- drift fixes

But only with strict checks, and semantics changes still require human review.
