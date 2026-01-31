#!/usr/bin/env python3
"""
Enhanced Test Data Importers
Demonstrates improved import capabilities with correct sheet names
"""

from data_importers import UnifiedImporter, ExcelImporter
from pathlib import Path


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print("─" * 80)


def main():
    print_section("LARIAT BIBLE - ENHANCED DATA IMPORTERS TEST")

    # Initialize importers
    importer = UnifiedImporter()
    excel_importer = importer.excel_importer

    # Test 1: Import Menu Bible INDEX
    print_subsection("1. MENU BIBLE - INDEX SHEET")

    menu_file = "LARIAT MENU BIBLE PT.18 V3.xlsx"
    if Path(importer.data_directory / menu_file).exists():
        print(f"\n📊 Importing: {menu_file} (INDEX sheet)")

        menu_data = excel_importer.import_menu_bible()
        if menu_data:
            print(f"   ✓ Imported {len(menu_data['items'])} menu items")
            print(f"   ✓ Categories: {len(menu_data['categories'])}")

            if menu_data['categories']:
                print(f"   Categories: {', '.join(menu_data['categories'][:5])}")
                if len(menu_data['categories']) > 5:
                    print(f"   ... and {len(menu_data['categories']) - 5} more")

            # Show sample items
            if menu_data['items']:
                print(f"\n   Sample menu items:")
                for item in menu_data['items'][:3]:
                    print(f"     • {item['name']}")
                    if item['category']:
                        print(f"       Category: {item['category']}")
                    if item['cost']:
                        print(f"       Cost: ${item['cost']:.2f}")
                    if item['price']:
                        print(f"       Price: ${item['price']:.2f}")
        else:
            print("   ⚠ Could not import menu data")

    # Test 2: Import all recipe sheets from Menu Bible
    print_subsection("2. MENU BIBLE - ALL RECIPE SHEETS")

    if Path(importer.data_directory / menu_file).exists():
        print(f"\n📖 Importing all recipe sheets from: {menu_file}")

        recipes_data = excel_importer.import_menu_bible_recipes()
        if recipes_data:
            print(f"   ✓ Imported {len(recipes_data['recipes'])} recipe sheets")

            # Show sample recipes
            if recipes_data['recipes']:
                print(f"\n   Sample recipes:")
                for recipe in recipes_data['recipes'][:5]:
                    ingredient_count = len(recipe.get('ingredients', []))
                    print(f"     • {recipe['name']} ({ingredient_count} ingredients)")
                    if recipe.get('category'):
                        print(f"       Category: {recipe['category']}")
                    if ingredient_count > 0:
                        # Show first 2 ingredients
                        print(f"       Sample ingredients:")
                        for ing in recipe['ingredients'][:2]:
                            print(f"         - {ing['name']}: {ing['scaled_qty']} {ing['unit']}")

                if len(recipes_data['recipes']) > 5:
                    print(f"     ... and {len(recipes_data['recipes']) - 5} more recipes")
        else:
            print("   ⚠ Could not import recipe sheets")

    # Test 3: Import Smart Costing with correct sheets
    print_subsection("3. SMART COSTING - RECIPE DATA")

    smart_costing_file = "SMART COSTING 1/LARIAT_SMART_COSTING_COMPLETE_1.xlsx"
    if Path(importer.data_directory / smart_costing_file).exists():
        print(f"\n💰 Importing: {smart_costing_file}")

        # Show available sheets
        sheets = excel_importer.get_sheet_names(smart_costing_file)
        if sheets:
            print(f"   Available sheets ({len(sheets)}): {', '.join(sheets[:8])}")
            if len(sheets) > 8:
                print(f"   ... and {len(sheets) - 8} more")

        # Import recipe data
        costing_data = excel_importer.import_smart_costing()
        if costing_data:
            print(f"\n   ✓ Imported from sheet: {costing_data['sheet_name']}")
            print(f"   ✓ Recipes: {len(costing_data['recipes'])}")
            print(f"   ✓ Total Cost: ${costing_data['total_cost']:.2f}")

            # Show sample recipes
            if costing_data['recipes']:
                print(f"\n   Sample costed recipes:")
                for recipe in costing_data['recipes'][:3]:
                    print(f"     • {recipe['name']}")
                    if recipe['servings']:
                        print(f"       Servings: {recipe['servings']}")
                    if recipe['total_cost']:
                        print(f"       Total Cost: ${recipe['total_cost']:.2f}")
                    if recipe['cost_per_serving']:
                        print(f"       Cost/Serving: ${recipe['cost_per_serving']:.2f}")
        else:
            print("   ⚠ Could not import smart costing data")

    # Test 4: Import Ingredient Database
    print_subsection("4. INGREDIENT DATABASE")

    if Path(importer.data_directory / smart_costing_file).exists():
        print(f"\n🥘 Importing Ingredient Database from: {smart_costing_file}")

        ingredients = excel_importer.import_ingredient_database()
        if ingredients:
            print(f"   ✓ Imported {len(ingredients)} ingredients")

            # Show sample ingredients
            print(f"\n   Sample ingredients:")
            for ingredient in ingredients[:5]:
                print(f"     • {ingredient['name']}")
                if ingredient.get('category'):
                    print(f"       Category: {ingredient['category']}")
                if ingredient.get('purchase_unit'):
                    print(f"       Purchase Unit: {ingredient['purchase_unit']}")
                if ingredient.get('package_size'):
                    print(f"       Package Size: {ingredient['package_size']}")
                if ingredient.get('cost_per_package'):
                    print(f"       Cost/Package: ${ingredient['cost_per_package']:.2f}")
                if ingredient.get('cost_per_lb'):
                    print(f"       Cost/lb: ${ingredient['cost_per_lb']:.2f}")

            if len(ingredients) > 5:
                print(f"     ... and {len(ingredients) - 5} more ingredients")
        else:
            print("   ⚠ Could not import ingredient database")

    # Test 5: Import complete restaurant data
    print_subsection("5. COMPLETE RESTAURANT DATA IMPORT")

    print("\n🔄 Importing all restaurant data...")
    all_data = importer.import_all_restaurant_data()

    if all_data['menu']:
        print(f"\n✓ MENU DATA:")
        print(f"   Items: {len(all_data['menu']['items'])}")
        print(f"   Categories: {len(all_data['menu']['categories'])}")

    if all_data['smart_costing']:
        print(f"\n✓ SMART COSTING DATA:")
        print(f"   Recipes: {len(all_data['smart_costing']['recipes'])}")
        print(f"   Total Cost: ${all_data['smart_costing']['total_cost']:.2f}")

    if all_data['vendor_prices']:
        print(f"\n✓ VENDOR PRICING DATA:")
        vendors = all_data['vendor_prices'].get('vendors', [])
        products = all_data['vendor_prices'].get('products', [])
        print(f"   Products: {len(products)}")
        if vendors:
            print(f"   Vendors: {', '.join(vendors)}")

    if all_data['inventory']:
        print(f"\n✓ INVENTORY DATA:")
        print(f"   Items: {len(all_data['inventory'])}")

    if all_data['recipes']:
        print(f"\n✓ RECIPE DATA (CSV):")
        print(f"   Recipes: {len(all_data['recipes'])}")

    # Test 6: Summary
    print_subsection("6. IMPORT SUMMARY")

    summary = importer.get_complete_summary()

    print(f"\n📊 CSV Imports:")
    print(f"   Files: {summary['csv']['files_imported']}")
    print(f"   Total Rows: {summary['csv']['total_rows']}")

    print(f"\n📊 Excel Imports:")
    print(f"   Files: {summary['excel']['files_imported']}")
    print(f"   Sheets: {summary['excel']['sheets_imported']}")
    print(f"   Total Rows: {summary['excel']['total_rows']}")

    print(f"\n📊 Google Sheets Imports:")
    print(f"   Sheets: {summary['google_sheets']['sheets_imported']}")
    print(f"   Total Rows: {summary['google_sheets']['total_rows']}")

    # Final message
    print_section("✓ ENHANCED IMPORT TEST COMPLETE!")

    print("\n💡 NEW FEATURES:")
    print("   • Menu Bible INDEX sheet import")
    print("   • Individual recipe sheet import (all recipes from Menu Bible)")
    print("   • Smart Costing with correct sheet names (SMART TEMPLATE, Ingredient Database)")
    print("   • Dedicated Ingredient Database import")
    print("   • Enhanced error handling and fallbacks")


if __name__ == "__main__":
    main()
