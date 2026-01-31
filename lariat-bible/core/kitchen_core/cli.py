"""Small CLI helpers for kitchen_core demo purposes."""

from pathlib import Path


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def print_shopping_list(shopping_list: dict):
    if not shopping_list:
        print("Nothing to buy — inventory covers the recipe.")
        return
    print("Shopping list:")
    for (name, unit), qty in shopping_list.items():
        print(f"- {name}: {qty} {unit}")
