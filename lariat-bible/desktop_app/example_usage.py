#!/usr/bin/env python3
"""
Practical Usage Examples for Enhanced Data Importers
Demonstrates real-world use cases for the Lariat Bible data importers
"""

from data_importers import UnifiedImporter, ExcelImporter
import json


def example_1_recipe_lookup():
    """Example: Look up a specific recipe with all its ingredients and costs"""
    print("=" * 80)
    print("EXAMPLE 1: Recipe Lookup with Ingredient Costs")
    print("=" * 80)

    excel_importer = ExcelImporter()

    # Get ingredient database for pricing
    ingredients_db = excel_importer.import_ingredient_database()
    if not ingredients_db:
        print("Could not load ingredient database")
        return

    # Create a lookup dictionary for quick ingredient price lookup
    ingredient_prices = {ing['name']: ing for ing in ingredients_db}

    # Import a specific recipe
    recipe = excel_importer.import_single_recipe(
        "LARIAT MENU BIBLE PT.18 V3.xlsx",
        "The Trio"
    )

    if recipe:
        print(f"\nRecipe: {recipe['name']}")
        print(f"Category: {recipe['category']}")
        print(f"Yield: {recipe['base_yield']}")
        print(f"\nIngredients:")

        total_cost = 0
        for ing in recipe['ingredients']:
            # Try to find cost in database (simplified matching)
            cost_info = ingredient_prices.get(ing['name'])
            if cost_info and cost_info.get('cost_per_oz'):
                estimated_cost = ing['scaled_qty'] * cost_info['cost_per_oz']
                total_cost += estimated_cost
                print(f"  • {ing['name']}: {ing['scaled_qty']} {ing['unit']} (≈${estimated_cost:.2f})")
            else:
                print(f"  • {ing['name']}: {ing['scaled_qty']} {ing['unit']}")

        print(f"\nEstimated Total Cost: ${total_cost:.2f}")


def example_2_all_recipes_summary():
    """Example: Get summary of all recipes by category"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: All Recipes Summary by Category")
    print("=" * 80)

    excel_importer = ExcelImporter()

    # Import all recipes
    recipes_data = excel_importer.import_menu_bible_recipes()
    if not recipes_data:
        print("Could not load recipes")
        return

    # Organize by category
    by_category = {}
    for recipe in recipes_data['recipes']:
        category = recipe.get('category', 'Unknown')
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(recipe)

    # Display summary
    print(f"\nTotal Recipes: {len(recipes_data['recipes'])}")
    print(f"Categories: {len(by_category)}\n")

    for category, recipes in sorted(by_category.items()):
        print(f"\n{category.upper()}: ({len(recipes)} recipes)")
        for recipe in recipes:
            ingredient_count = len(recipe['ingredients'])
            print(f"  • {recipe['name']} - {ingredient_count} ingredients")


def example_3_ingredient_search():
    """Example: Search ingredients by category and price range"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Ingredient Search - Dairy Products Under $5/package")
    print("=" * 80)

    excel_importer = ExcelImporter()

    # Import ingredient database
    ingredients = excel_importer.import_ingredient_database()
    if not ingredients:
        print("Could not load ingredients")
        return

    # Filter: Dairy category, cost under $5 per package
    dairy_under_5 = [
        ing for ing in ingredients
        if ing.get('category') == 'Dairy' and
           ing.get('cost_per_package', 999) < 5.0
    ]

    print(f"\nFound {len(dairy_under_5)} dairy products under $5/package:\n")

    for ing in dairy_under_5:
        print(f"• {ing['name']}")
        print(f"  ${ing['cost_per_package']:.2f}/{ing['purchase_unit']}")
        print(f"  Package size: {ing['package_size']} {ing['purchase_unit']}")
        print(f"  Cost per lb: ${ing['cost_per_lb']:.2f}")
        print()


def example_4_export_recipe_json():
    """Example: Export a recipe to JSON format"""
    print("=" * 80)
    print("EXAMPLE 4: Export Recipe to JSON")
    print("=" * 80)

    excel_importer = ExcelImporter()

    # Import specific recipe
    recipe = excel_importer.import_single_recipe(
        "LARIAT MENU BIBLE PT.18 V3.xlsx",
        "El Jefe Burger"
    )

    if recipe:
        # Export to JSON
        output_file = "el_jefe_burger_recipe.json"
        with open(output_file, 'w') as f:
            json.dump(recipe, f, indent=2)

        print(f"\n✓ Recipe exported to: {output_file}")
        print(f"\nRecipe preview:")
        print(f"  Name: {recipe['name']}")
        print(f"  Category: {recipe['category']}")
        print(f"  Ingredients: {len(recipe['ingredients'])}")
        print(f"\nFirst 3 ingredients:")
        for ing in recipe['ingredients'][:3]:
            print(f"  • {ing['name']}: {ing['scaled_qty']} {ing['unit']}")


def example_5_most_expensive_ingredients():
    """Example: Find most expensive ingredients by cost per lb"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Top 10 Most Expensive Ingredients (per lb)")
    print("=" * 80)

    excel_importer = ExcelImporter()

    # Import ingredient database
    ingredients = excel_importer.import_ingredient_database()
    if not ingredients:
        print("Could not load ingredients")
        return

    # Filter out ingredients without cost_per_lb and sort
    with_costs = [ing for ing in ingredients if ing.get('cost_per_lb', 0) > 0]
    sorted_by_cost = sorted(with_costs, key=lambda x: x['cost_per_lb'], reverse=True)

    print(f"\nTop 10 most expensive ingredients:\n")

    for i, ing in enumerate(sorted_by_cost[:10], 1):
        print(f"{i}. {ing['name']}")
        print(f"   ${ing['cost_per_lb']:.2f}/lb (Category: {ing.get('category', 'N/A')})")
        print(f"   Purchase: ${ing.get('cost_per_package', 0):.2f} per {ing.get('package_size', 0)} {ing.get('purchase_unit', '')}")
        print()


def example_6_unified_import():
    """Example: Use unified importer to get all data at once"""
    print("=" * 80)
    print("EXAMPLE 6: Unified Import - All Restaurant Data")
    print("=" * 80)

    # Use unified importer
    importer = UnifiedImporter()

    print("\nImporting all restaurant data...\n")

    # Get everything at once
    all_data = importer.import_all_restaurant_data()

    # Display summary
    print("Import Summary:")
    print(f"  • Menu items: {len(all_data.get('menu', {}).get('items', []))}")
    print(f"  • Smart costing recipes: {len(all_data.get('smart_costing', {}).get('recipes', []))}")
    print(f"  • CSV recipes: {len(all_data.get('recipes', []))}")
    print(f"  • Inventory items: {len(all_data.get('inventory', []))}")
    print(f"  • Vendor price products: {len(all_data.get('vendor_prices', {}).get('products', []))}")

    # Get complete statistics
    summary = importer.get_complete_summary()
    print(f"\nDetailed Statistics:")
    print(f"  CSV files imported: {summary['csv']['files_imported']}")
    print(f"  Excel sheets imported: {summary['excel']['sheets_imported']}")
    print(f"  Total rows imported: {summary['excel']['total_rows'] + summary['csv']['total_rows']}")


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "LARIAT BIBLE DATA IMPORTER EXAMPLES" + " " * 23 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Run examples
    example_1_recipe_lookup()
    example_2_all_recipes_summary()
    example_3_ingredient_search()
    example_4_export_recipe_json()
    example_5_most_expensive_ingredients()
    example_6_unified_import()

    print("\n" + "=" * 80)
    print("✓ ALL EXAMPLES COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
