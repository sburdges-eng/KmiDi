import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

def load_json_config(path: Path, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Load JSON from a file, returning the fallback if it fails or does not exist."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logging.exception("Failed to load %s", path)
    return fallback

def save_json_config(path: Path, data: Dict[str, Any]) -> bool:
    """Save a dictionary to a JSON file."""
    try:
        # Create temp file first, then atomically rename
        fd = path.with_suffix('.tmp')
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(fd, path)
        return True
    except Exception:
        logging.exception("Failed to save full configuration to %s", path)
        return False
