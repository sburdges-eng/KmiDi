from typing import Dict, Tuple
from .models import Recipe, InventoryItem


def compute_shopping_list(
    recipe: Recipe,
    inventory: list[InventoryItem],
) -> Dict[Tuple[str, str], float]:
    """Compute a minimal shopping list (name, unit) -> needed quantity.
    Quantities are naive numeric subtraction.
    Units are assumed consistent for a given item.
    """
    inv_map: Dict[Tuple[str, str], float] = {}
    for it in inventory:
        key = (it.name.lower(), it.unit)
        inv_map[key] = inv_map.get(key, 0.0) + it.quantity

    needed: Dict[Tuple[str, str], float] = {}
    for ing in recipe.ingredients:
        key = (ing.name.lower(), ing.unit)
        have = inv_map.get(key, 0.0)
        req = ing.quantity
        if req > have:
            needed[key] = needed.get(key, 0.0) + (req - have)

    return needed
