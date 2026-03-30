# KmiDi Qt GUI - Quick Start

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
# Run GUI
python main.py
```

## Running CLI (Headless)

```bash
# Generate music from command line
python cli/main.py "grief hidden as love"
```

This demonstrates that **core logic can run without GUI**.

## Project Structure

```
kmidi_gui/
├── core/              # Pure logic (no GUI)
│   ├── engine.py      # Music generation
│   ├── models.py      # Data structures
│   └── ai/            # AI analysis
├── gui/               # Qt widgets (no logic)
│   └── main_window.py
├── controllers/       # Bridge GUI ↔ Core
│   └── actions.py
├── themes/            # Audio tool styling
│   ├── audio_dark.qss
│   └── palette.py
└── main.py            # Application entry
```

## Architecture

```
GUI (Qt) → Controller → Core Logic
```

**Key Rule**: Core logic has no GUI dependencies and can run headless.

## Features

- ✅ Minimal main window
- ✅ Audio tool theme
- ✅ Background workers (non-blocking)
- ✅ Status bar and logging
- ✅ CLI interface (headless)

## Next Steps

1. **Add Parameter Panels**: Sliders for Valence, Arousal, etc.
2. **Add AI Assistant Dock**: Analysis results, confidence indicators
3. **Add Logs Dock**: Operation logs, error messages
4. **Connect to Music Brain API**: Replace placeholder with real API calls

See `docs/UI_REDESIGN_Qt.md` for full design document.

## Testing Core Logic

```python
# Test core without GUI
from kmidi_gui.core.engine import get_engine
from kmidi_gui.core.models import EmotionIntent

engine = get_engine()
intent = EmotionIntent(core_event="grief hidden as love")
result = engine.generate_music(intent)
print(result)
```

## Theme Customization

Edit `themes/audio_dark.qss` to customize appearance.

Color constants are in `themes/palette.py`.

## Troubleshooting

### Import Errors

Make sure you're running from the `kmidi_gui/` directory or have it in your Python path.

### Qt Not Found

Install PySide6:
```bash
pip install PySide6
```

### Theme Not Loading

Check that `themes/audio_dark.qss` exists relative to `main.py`.

