# Spectocloud — Spine Visualization

Spectocloud: MIDI/emotion → spectral cloud visualization. Spine-included per [CONTRACTS.md](CONTRACTS.md) §5b and [INTEGRATION_MAP.md](INTEGRATION_MAP.md).

## Path (canonical)

| Layer | Path |
|-------|------|
| **Body** | `KmiDi_CANON/body/hooks/useMusicBrain.ts` — `renderSpectocloud(payload)` → `POST /spectocloud/render` |
| **API** | `POST /spectocloud/render` (FastAPI in `KmiDi_CANON/brain/api_server.py`) |
| **Brain** | `KmiDi_CANON/brain/music_brain/visualization/spectocloud.py` |

## Request (SpectocloudRenderRequest)

| Field | Type | Required | Description |
|-------|------|----------|--------------|
| `midi_events` | `Array<Record<string, any>>` | One of midi_events or midi_file_path | MIDI event list from body |
| `midi_file_path` | `string` | One of midi_events or midi_file_path | Path to MIDI file |
| `duration` | `number` | No | Duration in seconds (animation) |
| `emotion_trajectory` | `Array<...>` | No | Emotion waypoints over time |
| `mode` | `"static" \| "animation"` | No | Default `"static"` |
| `frame_idx` | `number` | No | Frame index for static mode |
| `output_path` | `string` | No | Output path (image or frame dir) |
| `fps` | `number` | No | Frames per second (default 24) |
| `rotate` | `boolean` | No | Rotate cloud |
| `anchor_density` | `string` | No | Density preset |
| `n_particles` | `number` | No | Particle count (if supported) |

## Response (SpectocloudRenderResponse)

| Field | Type | Description |
|-------|------|--------------|
| `status` | `string` | `"completed"` or `"error"` |
| `mode` | `"static" \| "animation"` | Mode used |
| `output_path` | `string` | Path to output (image or frame directory) |
| `frames` | `number` | Number of frames (1 for static) |

On error, response may include `details`.

## Backend (Python)

- **Module:** `music_brain.visualization.spectocloud`
- **Entry:** `render_spectocloud_from_request(payload: dict) -> dict`
- **API surface:** `DAiWAPI.render_spectocloud(payload)` in `music_brain/api.py`

**Implementation:** Full particle/spectral render with musical anchors, electrostatic storm, and 3D matplotlib rendering. Supports static frame and animation modes. Optional dependencies: matplotlib (for rendering), mido (for MIDI file loading). Graceful degradation: writes manifest if matplotlib unavailable.

## Running the API

From repo root (see [BOOT.md](BOOT.md) — Brain HTTP API):

```bash
PYTHONPATH=KmiDi_CANON/brain uvicorn KmiDi_CANON.brain.api_server:app --host 127.0.0.1 --port 8000
```

Body uses `http://127.0.0.1:8000`; `POST /spectocloud/render` with JSON body.

## Recovery

**Status: COMPLETE (2026-01-31)** — Full particle/spectral render restored from commit 6d4d67c5, adapted for KmiDi_CANON layout. Optional dependencies: matplotlib, mido. Graceful degradation when deps unavailable.
