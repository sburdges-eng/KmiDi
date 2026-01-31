import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from kitchen_core.models import Ingredient, Recipe, InventoryItem
from kitchen_core.analysis import compute_shopping_list


class AnalysisTests(unittest.TestCase):
    def test_compute_shopping_list_basic(self):
        recipe = Recipe(
            name="Test",
            servings=1,
            ingredients=[Ingredient(name="eggs", quantity=3, unit="pcs")],
        )
        inventory = [InventoryItem(name="eggs", quantity=1, unit="pcs")]
        shopping = compute_shopping_list(recipe, inventory)
        self.assertEqual(shopping.get(("eggs", "pcs")), 2)


if __name__ == "__main__":
    unittest.main()
