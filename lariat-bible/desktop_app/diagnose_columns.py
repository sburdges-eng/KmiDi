#!/usr/bin/env python3
"""
Diagnose Column Names
Show actual column names in the data files to fix parsing logic
"""

from data_importers import ExcelImporter
from pathlib import Path
from data_importers.csv_importer import CSVImporter
import os


def main():
    print("=" * 80)
    print("COLUMN NAME DIAGNOSTIC")
    print("=" * 80)

    excel_importer = ExcelImporter()

    # Diagnose Menu Bible INDEX
    print("\n1. MENU BIBLE - INDEX SHEET")
    print("-" * 80)
    menu_file = "LARIAT MENU BIBLE PT.18 V3.xlsx"
    if Path(excel_importer.data_directory / menu_file).exists():
        data = excel_importer.import_excel(menu_file, sheet_name='INDEX')
        if data:
            print(f"Rows imported: {len(data)}")
            print(f"\nColumn names: {list(data[0].keys())}")
            print(f"\nFirst row sample:")
            for key, value in list(data[0].items())[:10]:
                print(f"  {key}: {value}")

    # Diagnose Smart Costing - SMART TEMPLATE
    print("\n\n2. SMART COSTING - SMART TEMPLATE SHEET")
    print("-" * 80)
    smart_file = "SMART COSTING 1/LARIAT_SMART_COSTING_COMPLETE_1.xlsx"
    if Path(excel_importer.data_directory / smart_file).exists():
        data = excel_importer.import_excel(smart_file, sheet_name='SMART TEMPLATE')
        if data:
            print(f"Rows imported: {len(data)}")
            print(f"\nColumn names: {list(data[0].keys())}")
            print(f"\nFirst row sample:")
            for key, value in list(data[0].items())[:10]:
                print(f"  {key}: {value}")

    # Diagnose Ingredient Database
    print("\n\n3. INGREDIENT DATABASE SHEET")
    print("-" * 80)
    if Path(excel_importer.data_directory / smart_file).exists():
        data = excel_importer.import_excel(smart_file, sheet_name='Ingredient Database')
        if data:
            print(f"Rows imported: {len(data)}")
            print(f"\nColumn names: {list(data[0].keys())}")
            print(f"\nFirst row sample:")
            for key, value in list(data[0].items())[:10]:
                print(f"  {key}: {value}")

            # Show a few more samples
            if len(data) > 5:
                print(f"\nSample rows with data:")
                for i, row in enumerate(data[:10]):
                    # Find first non-None value
                    non_none_values = [(k, v) for k, v in row.items() if v is not None and str(v).strip()]
                    if non_none_values:
                        print(f"  Row {i+1}: {dict(list(non_none_values)[:5])}")

    # Diagnose Menu Bible recipe sheet
    print("\n\n4. MENU BIBLE - SAMPLE RECIPE SHEET")
    print("-" * 80)
    if Path(excel_importer.data_directory / menu_file).exists():
        data = excel_importer.import_excel(menu_file, sheet_name='The Trio')
        if data:
            print(f"Rows imported: {len(data)}")
            print(f"\nColumn names: {list(data[0].keys())}")
            print(f"\nFirst few rows:")
            for i, row in enumerate(data[:5]):
                non_none_values = [(k, v) for k, v in row.items() if v is not None and str(v).strip()]
                if non_none_values:
                    print(f"  Row {i+1}: {dict(list(non_none_values)[:5])}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

    # Extra: Diagnose CSV templates in repo data/csv_templates
    print("\n" + "=" * 80)
    print("CSV TEMPLATE DIAGNOSTIC")
    print("=" * 80)
    csv_imp = CSVImporter(data_directory='.')
    csv_dir = Path('data/csv_templates')
    if csv_dir.exists():
        for f in sorted(os.listdir(csv_dir)):
            if not f.lower().endswith('.csv'):
                continue
            path = csv_dir / f
            print(f"\nFile: {path}")
            try:
                # show raw header line bytes to expose BOM or invisible chars
                with open(path, 'rb') as fh:
                    first_bytes = fh.readline()
                print(f"  Raw first-line bytes repr: {repr(first_bytes[:200])}")
            except Exception:
                pass

            data = csv_imp.import_csv(str(path))
            if not data:
                print("  Could not parse or no data rows (may have header only)")
                continue
            # print header keys with repr to expose invisible whitespace
            headers = list(data[0].keys())
            print("  Parsed headers:")
            for h in headers:
                print(f"    {repr(h)} -> canonical: {h.strip()}")
            print("  First row sample:")
            print("   ", data[0])


if __name__ == "__main__":
    main()
