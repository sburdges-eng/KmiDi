"""
CSV Data Importer
Import data from CSV files into the application
"""

import csv
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Normalizers for better numeric/unit parsing
from .normalizers import (
    strip_currency_and_separators,
    parse_fractional_quantity,
    normalize_unit,
    safe_float_from_string,
)


class CSVImporter:
    """Import and process CSV files"""

    def __init__(self, data_directory: str = None):
        """
        Initialize CSV Importer

        Args:
            data_directory: Path to data directory
        """
        if data_directory is None:
            # Default to data folder in desktop_app
            data_directory = Path(__file__).parent.parent / 'data'

        self.data_directory = Path(data_directory)
        self.imported_data = {}

    def list_csv_files(self, subdirectory: str = None) -> List[str]:
        """
        List all CSV files in directory

        Args:
            subdirectory: Optional subdirectory to search

        Returns:
            List of CSV file paths
        """
        search_path = self.data_directory
        if subdirectory:
            search_path = search_path / subdirectory

        if not search_path.exists():
            logger.warning(f"Directory does not exist: {search_path}")
            return []

        csv_files = list(search_path.glob('**/*.csv'))
        return [str(f.relative_to(self.data_directory)) for f in csv_files]

    def import_csv(self, filename: str, has_header: bool = True,
                   delimiter: str = ',', encoding: str = 'utf-8') -> Optional[List[Dict]]:
        """
        Import a CSV file

        Args:
            filename: CSV filename (can be relative path)
            has_header: Whether CSV has header row
            delimiter: CSV delimiter (default: comma)
            encoding: File encoding

        Returns:
            List of dictionaries representing rows
        """
        file_path = self.data_directory / filename

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        try:
            # detect encoding and delimiter heuristically; allow caller to override
            try:
                detected_enc, detected_delim = self._detect_encoding_and_delimiter(file_path)
            except Exception:
                detected_enc, detected_delim = encoding, delimiter

            # Prefer detected values if caller passed defaults
            use_encoding = encoding or detected_enc
            use_delimiter = delimiter or detected_delim

            # If detection disagrees with provided, log an informational warning
            if detected_enc and detected_enc != encoding:
                logger.info(f"Detected encoding '{detected_enc}' differs from requested '{encoding}'; using detected.")
                use_encoding = detected_enc

            if detected_delim and detected_delim != delimiter:
                logger.info(f"Detected delimiter '{detected_delim}' differs from requested '{delimiter}'; using detected.")
                use_delimiter = detected_delim

            with open(file_path, 'r', encoding=use_encoding, newline='') as f:
                if has_header:
                    reader = csv.DictReader(f, delimiter=use_delimiter)
                    data = list(reader)
                else:
                    reader = csv.reader(f, delimiter=use_delimiter)
                    data = [{'col_' + str(i): val for i, val in enumerate(row)}
                           for row in reader]

                self.imported_data[filename] = data
                logger.info(f"Imported {len(data)} rows from {filename} (encoding={use_encoding}, delimiter={use_delimiter})")
                return data

        except Exception as e:
            logger.error(f"Error importing {filename}: {e}")
            return None

    def import_vendor_prices(self, filename: str) -> Optional[Dict]:
        """
        Import vendor price comparison CSV

        Expected columns: product, vendor1_price, vendor2_price, etc.

        Args:
            filename: CSV filename

        Returns:
            Dictionary with product pricing data
        """
        data = self.import_csv(filename)
        if not data:
            return None

        vendor_data = {
            'products': [],
            'vendors': set(),
            'comparisons': []
        }

        for row in data:
            product_name = row.get('product') or row.get('item') or row.get('Product')
            if product_name:
                vendor_data['products'].append(product_name)

                # Extract vendor prices
                for key, value in row.items():
                    if 'price' in key.lower() and key.lower() != 'product':
                        vendor_name = key.replace('_price', '').replace('_per_lb', '')
                        vendor_data['vendors'].add(vendor_name)

                        try:
                            price = float(value) if value else 0.0
                            vendor_data['comparisons'].append({
                                'product': product_name,
                                'vendor': vendor_name,
                                'price': price
                            })
                        except (ValueError, TypeError):
                            continue

        vendor_data['vendors'] = list(vendor_data['vendors'])
        logger.info(f"Imported {len(vendor_data['products'])} products from {len(vendor_data['vendors'])} vendors")
        return vendor_data

    def import_inventory(self, filename: str) -> Optional[List[Dict]]:
        """
        Import inventory data CSV

        Expected columns: item, quantity, unit, par_level, etc.

        Args:
            filename: CSV filename

        Returns:
            List of inventory items
        """
        data = self.import_csv(filename)
        if not data:
            return None

        inventory_items = []
        seen_names = set()
        for i, row in enumerate(data, start=1):
            item = {
                'name': row.get('item') or row.get('item_name') or row.get('name'),
                'quantity': safe_float_from_string(row.get('quantity') or row.get('current_stock')),
                'unit': normalize_unit(row.get('unit') or 'ea'),
                'par_level': safe_float_from_string(row.get('par_level')),
                'cost': safe_float_from_string(strip_currency_and_separators(row.get('cost') or row.get('unit_cost'))),
                'category': row.get('category') or 'General'
            }
            # basic validation and warnings
            if not item['name']:
                logger.error(f"{filename}: row {i}: missing item name; skipping row")
                continue
            if item['name'] in seen_names:
                logger.warning(f"{filename}: row {i}: duplicate item name '{item['name']}'")
            seen_names.add(item['name'])

            if item['quantity'] == 0:
                logger.warning(f"{filename}: row {i}: quantity is zero or missing for '{item['name']}'")
            if item['cost'] == 0:
                logger.warning(f"{filename}: row {i}: cost is zero or missing for '{item['name']}'")

            inventory_items.append(item)

        logger.info(f"Imported {len(inventory_items)} inventory items")
        return inventory_items

    def import_recipes(self, filename: str) -> Optional[List[Dict]]:
        """
        Import recipe data CSV

        Expected columns: recipe_name, ingredient, quantity, unit, cost

        Args:
            filename: CSV filename

        Returns:
            List of recipe data grouped by recipe name
        """
        data = self.import_csv(filename)
        if not data:
            return None

        recipes = {}
        for i, row in enumerate(data, start=1):
            recipe_name = row.get('recipe_name') or row.get('recipe')
            if not recipe_name:
                logger.warning(f"{filename}: row {i}: missing recipe_name; skipping row")
                continue

            if recipe_name not in recipes:
                recipes[recipe_name] = {
                    'name': recipe_name,
                    'ingredients': [],
                    'total_cost': 0.0,
                    'servings': self._safe_int(row.get('yield_servings') or row.get('servings'))
                }

            ingredient = {
                'name': row.get('ingredient'),
                'quantity': safe_float_from_string(row.get('quantity')),
                'unit': normalize_unit(row.get('unit')),
                'cost': safe_float_from_string(strip_currency_and_separators(row.get('total_cost') or row.get('cost')))
            }

            # validate ingredient
            if not ingredient['name']:
                logger.warning(f"{filename}: row {i}: missing ingredient name for recipe '{recipe_name}'; skipping ingredient")
                continue
            if ingredient['quantity'] is None or ingredient['quantity'] == 0:
                logger.warning(f"{filename}: row {i}: missing/zero quantity for ingredient '{ingredient['name']}' in recipe '{recipe_name}'")
            if ingredient['cost'] == 0:
                logger.warning(f"{filename}: row {i}: cost is zero or missing for ingredient '{ingredient['name']}' in recipe '{recipe_name}'")

            recipes[recipe_name]['ingredients'].append(ingredient)
            recipes[recipe_name]['total_cost'] += ingredient['cost']

        recipe_list = list(recipes.values())
        logger.info(f"Imported {len(recipe_list)} recipes")
        return recipe_list

    def export_to_csv(self, data: List[Dict], output_filename: str,
                     fieldnames: List[str] = None) -> bool:
        """
        Export data to CSV file

        Args:
            data: List of dictionaries to export
            output_filename: Output CSV filename
            fieldnames: Optional list of field names (uses first dict keys if not provided)

        Returns:
            True if successful
        """
        if not data:
            logger.error("No data to export")
            return False

        output_path = self.data_directory / output_filename

        try:
            if fieldnames is None:
                fieldnames = list(data[0].keys())

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)

            logger.info(f"Exported {len(data)} rows to {output_filename}")
            return True

        except Exception as e:
            logger.error(f"Error exporting to {output_filename}: {e}")
            return False

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float"""
        try:
            return float(value) if value else default
        except (ValueError, TypeError):
            return default

    def _safe_int(self, value: Any, default: int = 0) -> int:
        """Safely convert value to int"""
        try:
            return int(value) if value else default
        except (ValueError, TypeError):
            return default

    def _detect_encoding_and_delimiter(self, file_path: Path, sample_size: int = 8192):
        """Detect likely file encoding and delimiter from a small sample.

        Returns (encoding, delimiter). Tries utf-8, utf-8-sig, latin-1 for decoding
        then uses csv.Sniffer to guess delimiter among common candidates.
        """
        # read raw bytes
        raw = file_path.open('rb').read(sample_size)

        # try common decodings
        decodings = ('utf-8', 'utf-8-sig', 'latin-1')
        decoded = None
        used_enc = None
        for enc in decodings:
            try:
                decoded = raw.decode(enc)
                used_enc = enc
                break
            except Exception:
                continue

        if decoded is None:
            # fallback to latin-1 forcibly
            decoded = raw.decode('latin-1', errors='replace')
            used_enc = 'latin-1'

        # sniff delimiter
        sniff = csv.Sniffer()
        dialect = None
        for delim in [',', ';', '\t', '|']:
            try:
                dialect = sniff.sniff(decoded, delimiters=[delim])
                detected_delim = dialect.delimiter
                break
            except Exception:
                detected_delim = None

        return used_enc, detected_delim

    def get_summary(self) -> Dict:
        """
        Get summary of imported data

        Returns:
            Summary dictionary
        """
        return {
            'files_imported': len(self.imported_data),
            'total_rows': sum(len(data) for data in self.imported_data.values()),
            'files': list(self.imported_data.keys())
        }


if __name__ == "__main__":
    # Test the importer
    importer = CSVImporter()

    print("Available CSV files:")
    csv_files = importer.list_csv_files()
    for f in csv_files:
        print(f"  - {f}")

    print("\nImporter ready to use!")
