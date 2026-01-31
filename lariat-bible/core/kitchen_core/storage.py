from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover - external dependency
    yaml = None

from .models import Recipe, InventoryItem


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load YAML files. Install with `pip install pyyaml`."
        )
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_recipe_from_path(path: Path) -> Recipe:
    d = _load_yaml(path)
    return Recipe.from_dict(d)


def load_inventory_from_path(path: Path) -> list[InventoryItem]:
    d = _load_yaml(path)
    items = d.get("items", [])
    return [InventoryItem.from_dict(it) for it in items]
