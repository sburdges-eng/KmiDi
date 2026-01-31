#!/usr/bin/env python3
"""
Test Data Importers
Demo script to test CSV, Excel, and Google Sheets importers
"""

from data_importers import UnifiedImporter, ExcelImporter, CSVImporter
from pathlib import Path


def main():
    print("=" * 80)
    print("LARIAT BIBLE - DATA IMPORTERS TEST")
    print("=" * 80)

    # Initialize unified importer
    importer = UnifiedImporter()

    # Test 1: List all available files
    print("\n1. LISTING ALL AVAILABLE FILES")
    print("-" * 80)

    files = importer.list_all_files()

    print(f"\nFound {len(files['csv_files'])} CSV files:")
    for f in files['csv_files'][:5]:  # Show first 5
        print(f"  • {f}")
    if len(files['csv_files']) > 5:
        print(f"  ... and {len(files['csv_files']) - 5} more")

    print(f"\nFound {len(files['excel_files'])} Excel files:")
    for f in files['excel_files'][:5]:  # Show first 5
        print(f"  • {f}")
    if len(files['excel_files']) > 5:
        print(f"  ... and {len(files['excel_files']) - 5} more")

    # Test 2: Import Excel files
    print("\n2. IMPORTING EXCEL FILES")
    print("-" * 80)

    excel_importer = importer.excel_importer

    # Check sheets in menu bible
    menu_file = "LARIAT MENU BIBLE PT.18 V3.xlsx"
    if Path(importer.data_directory / menu_file).exists():
        print(f"\n📊 Analyzing: {menu_file}")
        sheets = excel_importer.get_sheet_names(menu_file)
        if sheets:
            print(f"   Sheets available: {', '.join(sheets)}")

            # Import first sheet
            data = excel_importer.import_excel(menu_file, sheet_name=sheets[0])
            if data:
                print(f"   ✓ Imported {len(data)} rows from '{sheets[0]}'")
                if data:
                    print(f"   Columns: {', '.join(list(data[0].keys())[:5])}...")

    # Check smart costing file
    smart_costing_file = "SMART COSTING 1/LARIAT_SMART_COSTING_COMPLETE_1.xlsx"
    if Path(importer.data_directory / smart_costing_file).exists():
        print(f"\n💰 Analyzing: {smart_costing_file}")
        sheets = excel_importer.get_sheet_names(smart_costing_file)
        if sheets:
            print(f"   Sheets available: {', '.join(sheets)}")

    # Test 3: Import all restaurant data
    print("\n3. IMPORTING ALL RESTAURANT DATA")
    print("-" * 80)

    all_data = importer.import_all_restaurant_data()

    if all_data['menu']:
        print(f"\n✓ MENU DATA:")
        print(f"   Items: {len(all_data['menu']['items'])}")
        print(f"   Categories: {', '.join(all_data['menu']['categories'][:5])}")
        if len(all_data['menu']['categories']) > 5:
            print(f"   ... and {len(all_data['menu']['categories']) - 5} more")

    if all_data['smart_costing']:
        print(f"\n✓ SMART COSTING DATA:")
        print(f"   Recipes: {len(all_data['smart_costing']['recipes'])}")
        print(f"   Total Cost: ${all_data['smart_costing']['total_cost']:.2f}")

    if all_data['vendor_prices']:
        print(f"\n✓ VENDOR PRICING DATA:")
        vendors = all_data['vendor_prices'].get('vendors', [])
        products = all_data['vendor_prices'].get('products', [])
        print(f"   Products: {len(products)}")
        print(f"   Vendors: {', '.join(vendors)}")

    if all_data['inventory']:
        print(f"\n✓ INVENTORY DATA:")
        print(f"   Items: {len(all_data['inventory'])}")

    if all_data['recipes']:
        print(f"\n✓ RECIPE DATA:")
        print(f"   Recipes: {len(all_data['recipes'])}")

    # Test 4: Summary
    print("\n4. IMPORT SUMMARY")
    print("-" * 80)

    summary = importer.get_complete_summary()

    print(f"\nCSV Imports:")
    print(f"   Files: {summary['csv']['files_imported']}")
    print(f"   Total Rows: {summary['csv']['total_rows']}")

    print(f"\nExcel Imports:")
    print(f"   Files: {summary['excel']['files_imported']}")
    print(f"   Sheets: {summary['excel']['sheets_imported']}")
    print(f"   Total Rows: {summary['excel']['total_rows']}")

    print(f"\nGoogle Sheets Imports:")
    print(f"   Sheets: {summary['google_sheets']['sheets_imported']}")
    print(f"   Total Rows: {summary['google_sheets']['total_rows']}")

    # Test 5: Export sample data
    print("\n5. EXPORT TEST")
    print("-" * 80)

    if all_data['menu'] and all_data['menu']['items']:
        # Export first 10 menu items to CSV
        sample_items = all_data['menu']['items'][:10]
        output_file = "menu_sample_export.csv"

        if importer.export_data(sample_items, output_file):
            print(f"✓ Exported sample menu data to: {output_file}")

    print("\n" + "=" * 80)
    print("✓ IMPORT TEST COMPLETE!")
    print("=" * 80)

    # Print Google Sheets instructions
    print("\n📝 GOOGLE SHEETS SETUP:")
    print("-" * 80)
    print("To import from Google Sheets:")
    print("1. Set environment variable: export GOOGLE_SHEETS_API_KEY='your-key'")
    print("2. Or make sheet public and use import_public_sheet() method")
    print("3. Get spreadsheet ID from URL or use full URL with import_sheet_by_url()")


if __name__ == "__main__":
    main()
