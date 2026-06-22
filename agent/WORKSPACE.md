# KmiDi — structured workspace

## Auto-install on open

When you open this workspace (or the repo root), the **Resolve workspace** task runs `uv sync`, which installs workspace deps. This app is currently a stub (no extra deps beyond root). When you add modules below, add their dependencies to `apps/kmidi/pyproject.toml`; they will be installed automatically on next open. See the **top-level README** for enabling automatic tasks.

## Modules / files to create

| Status   | Module / file              | Description |
|----------|----------------------------|-------------|
| Stub     | `tests/test_placeholder.py`| Placeholder test. |
| To add   | `kmidi/__init__.py`        | Package init. |
| To add   | `kmidi/audio.py`           | Audio I/O and buffers (e.g. for JEPA or DAW). |
| To add   | `kmidi/midi.py`            | MIDI in/out or sequencing stubs. |
| To add   | `kmidi/daw.py`             | Optional: DAW-style transport / project (iDAW scope). |
| To add   | `main.py`                  | Optional: CLI or server entry. |

## Dependencies to add (in pyproject.toml when needed)

- `jepa` (workspace) if using JEPA for audio representations.
- Optional: `sounddevice`, `mido`, `numpy`, `torch` for audio/MIDI.

## Todos

- [ ] Add `kmidi` package and `audio` / `midi` modules.
- [ ] Add `pyproject.toml` dependencies when implementing (e.g. jepa, sounddevice).
- [ ] Add `main.py` or CLI entry and document in README.
- [ ] Optional: integrate with libs/jepa for audio embeddings.
