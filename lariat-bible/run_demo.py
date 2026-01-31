"""Demo runner for the `kitchen_core` package.

Run this script from the repo root (or directly):
    python3 run_demo.py

It loads `data/sample_recipes/omelette.yaml` and `data/sample_inventory/inventory.yaml`,
computes a shopping list, and prints it.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
# Ensure the package under core/ is importable
sys.path.insert(0, str(REPO_ROOT / "core"))

try:
    from kitchen_core import storage, analysis
except Exception as exc:  # pragma: no cover - demo
    print("Failed to import kitchen_core package:", exc)
    print("Make sure `core/` is on the PYTHONPATH or run from the repo root.")
    raise

try:
    from kitchen_core.claude_terminal import send_prompt
except Exception:
    send_prompt = None


def _format_shopping_for_prompt(recipe, shopping):
    parts = [f"Recipe: {recipe.name} (serves {recipe.servings})", "Shopping list:"]
    for (name, unit), qty in shopping.items():
        parts.append(f"- {name}: {qty} {unit}")
    return "\n".join(parts)


def main() -> None:
    recipe_path = REPO_ROOT / "data" / "sample_recipes" / "omelette.yaml"
    inventory_path = REPO_ROOT / "data" / "sample_inventory" / "inventory.yaml"

    try:
        recipe = storage.load_recipe_from_path(recipe_path)
    except ImportError as e:
        print(e)
        print("Install PyYAML: pip install pyyaml")
        return

    inventory = storage.load_inventory_from_path(inventory_path)
    shopping = analysis.compute_shopping_list(recipe, inventory)

    print(f"Recipe: {recipe.name} (serves {recipe.servings})")
    if not shopping:
        print("No items needed — inventory covers the recipe.")
    else:
        print("Shopping list:")
        for (name, unit), qty in shopping.items():
            print(f"- {name}: {qty} {unit}")

    # If Claude is available and an API key is set, ask Claude to verify or summarize.
    api_key = None
    try:
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
            "CLAUDE_API_KEY"
        )
    except Exception:
        api_key = None

    if send_prompt and api_key:
        prompt = (
            "Please review the following recipe and shopping list. "
            "Confirm whether any items are missing, suggest optimizations, "
            "and provide a short summary.\n\n"
            + _format_shopping_for_prompt(recipe, shopping)
        )
        try:
            print("\nAsking Claude to review the shopping list...\n")
            response = send_prompt(prompt, api_key=api_key)
            print("Claude response:\n")
            print(response)
        except ImportError as ie:
            print(ie)
            print("Install requests: pip install requests")
        except Exception as e:
            print("Error calling Claude:", e)


if __name__ == "__main__":
    main()
