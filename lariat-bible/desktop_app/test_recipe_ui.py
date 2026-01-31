#!/usr/bin/env python3
"""
Test Recipe Manager UI
Standalone test of the integrated recipe manager
"""

import tkinter as tk
from tkinter import ttk
from features.recipe_manager import RecipeManager


def main():
    # Create main window
    root = tk.Tk()
    root.title("Lariat Bible - Recipe Manager Demo")
    root.geometry("1200x800")

    # Create recipe manager
    recipe_manager = RecipeManager()

    # Create main frame
    main_frame = ttk.Frame(root)
    main_frame.pack(fill='both', expand=True)

    # Create UI
    recipe_manager.create_ui(main_frame)

    # Print summary
    print("\n" + "=" * 80)
    print("RECIPE MANAGER - INTEGRATION DEMO")
    print("=" * 80)
    print(f"\n✓ Loaded {len(recipe_manager.recipes)} recipes")
    print(f"✓ Loaded {len(recipe_manager.ingredients)} ingredients")

    # Show categories
    by_category = recipe_manager.get_recipes_by_category()
    print(f"\nRecipes by Category:")
    for category, recipes in sorted(by_category.items()):
        print(f"  • {category}: {len(recipes)} recipes")

    print("\n" + "=" * 80)
    print("Click on a recipe to see details and ingredient pricing!")
    print("=" * 80 + "\n")

    # Run the app
    root.mainloop()


if __name__ == "__main__":
    main()
