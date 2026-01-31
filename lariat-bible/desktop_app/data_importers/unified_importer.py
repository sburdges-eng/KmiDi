"""
Unified Data Importer
Automatically detect and import data from CSV, Excel, or Google Sheets
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

from .csv_importer import CSVImporter
from .excel_importer import ExcelImporter
from .sheets_importer import GoogleSheetsImporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedImporter:
    """Unified importer that handles CSV, Excel, and Google Sheets"""

    def __init__(self, data_directory: str = None):
        """
        Initialize Unified Importer

        Args:
            data_directory: Path to data directory
        """
        self.csv_importer = CSVImporter(data_directory)
        self.excel_importer = ExcelImporter(data_directory)
        self.sheets_importer = GoogleSheetsImporter()

        self.data_directory = self.csv_importer.data_directory

    def import_file(self, filename: str, **kwargs) -> Optional[List[Dict]]:
        """
        Auto-detect file type and import

        Args:
            filename: File to import (CSV, Excel, or Google Sheets URL)
            **kwargs: Additional arguments for specific importer

        Returns:
            List of dictionaries representing data
        """
        # Check if it's a URL (Google Sheets)
        if filename.startswith('http'):
            logger.info("Detected Google Sheets URL")
            return self.sheets_importer.import_sheet_by_url(filename, **kwargs)

        # Check file extension
        file_path = Path(filename)
        ext = file_path.suffix.lower()

        if ext == '.csv':
            logger.info("Detected CSV file")
            return self.csv_importer.import_csv(filename, **kwargs)

        elif ext in ['.xlsx', '.xls']:
            logger.info("Detected Excel file")
            return self.excel_importer.import_excel(filename, **kwargs)

        else:
            logger.error(f"Unsupported file type: {ext}")
            return None

    def list_all_files(self) -> Dict[str, List[str]]:
        """
        List all importable files

        Returns:
            Dictionary with CSV and Excel file lists
        """
        return {
            'csv_files': self.csv_importer.list_csv_files(),
            'excel_files': self.excel_importer.list_excel_files()
        }

    def import_all_restaurant_data(self) -> Dict:
        """
        Import all restaurant data from available files

        Returns:
            Dictionary with all imported data
        """
        all_data = {
            'menu': None,
            'recipes': [],
            'inventory': [],
            'vendor_prices': None,
            'smart_costing': None
        }

        # Try to import menu bible
        try:
            all_data['menu'] = self.excel_importer.import_menu_bible()
        except Exception as e:
            logger.warning(f"Could not import menu bible: {e}")

        # Try to import smart costing
        try:
            all_data['smart_costing'] = self.excel_importer.import_smart_costing()
        except Exception as e:
            logger.warning(f"Could not import smart costing: {e}")

        # Try to import vendor matching
        try:
            all_data['vendor_prices'] = self.excel_importer.import_vendor_matching()
        except Exception as e:
            logger.warning(f"Could not import vendor matching: {e}")

        # Import all CSV files
        csv_files = self.csv_importer.list_csv_files()
        for csv_file in csv_files:
            try:
                if 'inventory' in csv_file.lower():
                    inv_data = self.csv_importer.import_inventory(csv_file)
                    if inv_data:
                        all_data['inventory'].extend(inv_data)

                elif 'recipe' in csv_file.lower():
                    recipe_data = self.csv_importer.import_recipes(csv_file)
                    if recipe_data:
                        all_data['recipes'].extend(recipe_data)

                elif 'vendor' in csv_file.lower() or 'price' in csv_file.lower():
                    if not all_data['vendor_prices']:
                        all_data['vendor_prices'] = self.csv_importer.import_vendor_prices(csv_file)

            except Exception as e:
                logger.warning(f"Could not import {csv_file}: {e}")

        # Cross-file referential checks: ensure recipe ingredients reference known inventory or vendor products
        inv_names = set()
        for item in all_data['inventory']:
            name = item.get('name')
            if name:
                inv_names.add(name.strip().lower())

        vendor_products = set()
        vp = all_data.get('vendor_prices')
        if vp and isinstance(vp, dict):
            for p in vp.get('products', []):
                if p:
                    vendor_products.add(str(p).strip().lower())

        # check recipes
        for recipe in all_data.get('recipes', []):
            for ing in recipe.get('ingredients', []):
                iname = (ing.get('name') or '').strip().lower()
                if not iname:
                    logger.warning(f"Recipe '{recipe.get('name')}' has ingredient with empty name")
                    continue
                if iname not in inv_names and iname not in vendor_products:
                    logger.error(f"Referential mismatch: ingredient '{ing.get('name')}' in recipe '{recipe.get('name')}' not found in inventory or vendor lists")

        return all_data

    def export_data(self, data: List[Dict], output_filename: str) -> bool:
        """
        Export data (auto-detect format)

        Args:
            data: Data to export
            output_filename: Output filename

        Returns:
            True if successful
        """
        ext = Path(output_filename).suffix.lower()

        if ext == '.csv':
            return self.csv_importer.export_to_csv(data, output_filename)
        elif ext in ['.xlsx', '.xls']:
            return self.excel_importer.export_to_excel(data, output_filename)
        else:
            logger.error(f"Unsupported export format: {ext}")
            return False

    def get_complete_summary(self) -> Dict:
        """
        Get complete summary of all imported data

        Returns:
            Summary dictionary
        """
        return {
            'csv': self.csv_importer.get_summary(),
            'excel': self.excel_importer.get_summary(),
            'google_sheets': self.sheets_importer.get_summary()
        }


if __name__ == "__main__":
    # Test the unified importer
    importer = UnifiedImporter()

    print("=" * 80)
    print("UNIFIED DATA IMPORTER")
    print("=" * 80)

    # List all files
    files = importer.list_all_files()
    print("\nAvailable CSV files:")
    for f in files['csv_files']:
        print(f"  - {f}")

    print("\nAvailable Excel files:")
    for f in files['excel_files']:
        print(f"  - {f}")

    # Import all restaurant data
    print("\n" + "=" * 80)
    print("Importing all restaurant data...")
    print("=" * 80)

    data = importer.import_all_restaurant_data()

    if data['menu']:
        print(f"\n✓ Menu: {len(data['menu']['items'])} items in {len(data['menu']['categories'])} categories")

    if data['smart_costing']:
        print(f"✓ Smart Costing: {len(data['smart_costing']['recipes'])} recipes")

    if data['vendor_prices']:
        print(f"✓ Vendor Prices: {len(data['vendor_prices'].get('products', []))} products")

    if data['inventory']:
        print(f"✓ Inventory: {len(data['inventory'])} items")

    if data['recipes']:
        print(f"✓ Recipes: {len(data['recipes'])} recipes")

    print("\n" + "=" * 80)
    print("Import complete!")
    print("=" * 80)
