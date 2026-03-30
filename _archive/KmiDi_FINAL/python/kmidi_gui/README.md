# KmiDi Qt GUI

Professional desktop GUI for KmiDi using Qt (PySide6).

## Architecture

```
GUI (Qt Widgets)
    ↓ signals
Controller Layer
    ↓ function calls
Core Logic (Pure Python)
```

**Rule**: Core logic can run without GUI (CLI, tests).

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

## Structure

- `core/` - Pure business logic, no GUI
- `gui/` - Qt widgets only, no logic
- `controllers/` - Bridge GUI ↔ Core
- `themes/` - Audio tool styling

## Design Principles

1. **Separate GUI from Logic** - Core can run headless
2. **Start Simple** - One window, one button, one output
3. **Handle Long Tasks** - Background threads for AI/network
4. **Be Predictable** - Clear labels, sensible defaults
5. **Look Professional** - Audio tool aesthetic

## Features

- ✅ Minimal main window
- ✅ Audio tool theme
- ✅ Background generation worker
- ✅ Status bar and logging
- ⏳ AI Assistant dock (coming soon)
- ⏳ Logs dock (coming soon)
- ⏳ Parameter panels (coming soon)

## Development

### Running

```bash
python main.py
```

### Testing Core Logic

```bash
# Core can run without GUI
python -c "from core.engine import get_engine; print(get_engine())"
```

### Adding Features

1. Add GUI widget in `gui/`
2. Add controller method in `controllers/actions.py`
3. Add core logic in `core/engine.py`
4. Connect signals in `main.py`

## Theme

The audio tool theme provides:
- Dark, neutral backgrounds
- Soft gray text
- Muted blue accent
- Eye-fatigue-resistant colors

See `themes/audio_dark.qss` and `themes/palette.py`.

## Next Steps

See `docs/UI_REDESIGN_Qt.md` for full design document and roadmap.

