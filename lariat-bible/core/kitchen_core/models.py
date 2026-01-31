from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Ingredient:
    name: str
    quantity: float
    unit: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Ingredient":
        return Ingredient(
            name=str(d.get("name")),
            quantity=float(d.get("quantity", 0)),
            unit=str(d.get("unit", "")),
        )


@dataclass
class Recipe:
    name: str
    servings: int
    ingredients: List[Ingredient]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Recipe":
        ings = [Ingredient.from_dict(i) for i in d.get("ingredients", [])]
        return Recipe(
            name=d.get("name", ""), servings=int(d.get("servings", 1)), ingredients=ings
        )


@dataclass
class InventoryItem:
    name: str
    quantity: float
    unit: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "InventoryItem":
        return InventoryItem(
            name=str(d.get("name")),
            quantity=float(d.get("quantity", 0)),
            unit=str(d.get("unit", "")),
        )
