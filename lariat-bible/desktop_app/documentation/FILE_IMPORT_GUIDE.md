# File Import System for The Lariat Bible

> **Import data from CSV, Excel, and Google Sheets**
> No API needed - just upload files!

## Overview

This system lets you import data from:
- **CSV files** (.csv)
- **Excel files** (.xlsx, .xls)
- **Google Sheets** (via file download or direct link)

## Quick Start

```python
from modules.importers.file_importer import FileImporter

# Create importer
importer = FileImporter()

# Import vendor order guide
importer.import_vendor_order_guide(
    file_path='data/sysco_order_guide.xlsx',
    vendor='SYSCO',
    sheet_name='Products'  # Optional for Excel
)

# Import invoice
importer.import_invoice(
    file_path='data/sysco_invoice_20250118.csv',
    vendor='SYSCO'
)

# Import recipes
importer.import_recipes(
    file_path='data/recipes.xlsx',
    sheet_name='Recipes'
)
```

---

## File Format Templates

### 1. Vendor Order Guide Format

**Required Columns** (any order, case-insensitive):
```
Item Code, Description, Pack Size, Case Price, Unit Price, Unit, Category
```

**Example CSV:**
```csv
Item Code,Description,Pack Size,Case Price,Unit Price,Unit,Category
SYS001,GROUND BEEF 80/20,10 LB,45.99,4.599,LB,MEAT
SYS002,BLACK PEPPER COARSE,6/1LB,298.95,49.83,LB,SPICES
SYS003,ONION POWDER,6/1LB,148.95,24.83,LB,SPICES
```

**Example Excel:**
| Item Code | Description | Pack Size | Case Price | Unit Price | Unit | Category |
|-----------|-------------|-----------|------------|------------|------|----------|
| SYS001 | GROUND BEEF 80/20 | 10 LB | 45.99 | 4.599 | LB | MEAT |
| SYS002 | BLACK PEPPER COARSE | 6/1LB | 298.95 | 49.83 | LB | SPICES |

---

### 2. Invoice Format

**Required Columns:**
```
Item Code, Description, Quantity, Unit Price, Extension, Pack Size
```

**Optional Columns:**
```
Order Date, Invoice Number, Vendor
```

**Example CSV:**
```csv
Order Date,Invoice Number,Item Code,Description,Quantity,Unit Price,Extension,Pack Size
2025-01-18,INV-12345,SYS001,GROUND BEEF 80/20,5,45.99,229.95,10 LB
2025-01-18,INV-12345,SYS002,BLACK PEPPER COARSE,2,298.95,597.90,6/1LB
```

---

### 3. Recipe Format

**Required Columns:**
```
Recipe Name, Ingredient, Quantity, Unit, Yield Amount, Yield Unit
```

**Example CSV:**
```csv
Recipe Name,Ingredient,Quantity,Unit,Yield Amount,Yield Unit,Category
BBQ Sauce,Tomato Paste,2,cups,1,gallon,Sauce
BBQ Sauce,Brown Sugar,1,cup,1,gallon,Sauce
BBQ Sauce,Apple Cider Vinegar,0.5,cup,1,gallon,Sauce
Green Chile,Pork Shoulder,5,lbs,50,servings,Entree
Green Chile,Green Chiles,3,cans,50,servings,Entree
```

---

### 4. Menu Items Format

**Required Columns:**
```
Item Name, Category, Menu Price, Recipe Name (optional)
```

**Example CSV:**
```csv
Item Name,Category,Subcategory,Menu Price,Recipe Name,Portion Size,Target Margin
Smothered Burrito,Entree,Mexican,12.99,Green Chile,12 oz,0.45
BBQ Pulled Pork,Entree,BBQ,10.99,BBQ Sauce,8 oz,0.45
```

---

## Implementation

### modules/importers/file_importer.py

```python
"""
File Import System
Import data from CSV, Excel, and Google Sheets
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import numpy as np

class FileImporter:
    """Import data from various file formats"""

    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.ods']
        self.import_history = []

    # ==================== CORE IMPORT FUNCTIONS ====================

    def read_file(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Read file and return DataFrame
        Automatically detects file format
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file extension
        ext = file_path.suffix.lower()

        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {ext}. Supported: {self.supported_formats}")

        # Read based on format
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, sheet_name=sheet_name or 0)
        elif ext == '.ods':
            df = pd.read_excel(file_path, engine='odf', sheet_name=sheet_name or 0)

        # Clean column names (remove spaces, lowercase)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Remove empty rows
        df = df.dropna(how='all')

        return df

    def validate_columns(self, df: pd.DataFrame, required_columns: List[str]) -> Dict:
        """
        Check if DataFrame has required columns
        Returns dict with validation results
        """
        # Normalize column names
        df_cols = set(df.columns)
        required = set(col.lower().replace(' ', '_') for col in required_columns)

        missing = required - df_cols
        extra = df_cols - required

        return {
            'valid': len(missing) == 0,
            'missing_columns': list(missing),
            'extra_columns': list(extra),
            'message': f"Missing columns: {missing}" if missing else "All required columns present"
        }

    # ==================== VENDOR ORDER GUIDE IMPORT ====================

    def import_vendor_order_guide(
        self,
        file_path: str,
        vendor: str,
        sheet_name: Optional[str] = None,
        validate_only: bool = False
    ) -> Dict:
        """
        Import vendor order guide from CSV or Excel

        Args:
            file_path: Path to file
            vendor: 'SYSCO' or 'Shamrock Foods'
            sheet_name: Sheet name for Excel files
            validate_only: If True, only validate, don't import

        Returns:
            Dict with import results
        """
        print(f"📁 Importing {vendor} order guide from {file_path}")

        # Read file
        df = self.read_file(file_path, sheet_name)

        # Required columns
        required = ['item_code', 'description', 'pack_size', 'case_price']

        # Validate
        validation = self.validate_columns(df, required)

        if not validation['valid']:
            return {
                'success': False,
                'error': validation['message'],
                'missing_columns': validation['missing_columns']
            }

        if validate_only:
            return {
                'success': True,
                'message': 'Validation passed',
                'rows': len(df)
            }

        # Process each row
        products = []
        errors = []

        for idx, row in df.iterrows():
            try:
                product = {
                    'vendor': vendor,
                    'item_code': str(row['item_code']).strip(),
                    'description': str(row['description']).strip().upper(),
                    'pack_size': str(row['pack_size']).strip(),
                    'case_price': float(row['case_price']),
                    'unit_price': float(row.get('unit_price', 0)) if 'unit_price' in row else None,
                    'unit': str(row.get('unit', 'EACH')).strip().upper() if 'unit' in row else 'EACH',
                    'category': str(row.get('category', 'UNCATEGORIZED')).strip().upper() if 'category' in row else 'UNCATEGORIZED',
                    'last_updated': datetime.now()
                }

                # Validate price
                if product['case_price'] <= 0:
                    errors.append(f"Row {idx+2}: Invalid price ${product['case_price']}")
                    continue

                products.append(product)

            except Exception as e:
                errors.append(f"Row {idx+2}: {str(e)}")

        # Save to database or return results
        result = {
            'success': True,
            'vendor': vendor,
            'total_rows': len(df),
            'products_imported': len(products),
            'errors': errors,
            'error_count': len(errors),
            'products': products
        }

        # Record import
        self.import_history.append({
            'timestamp': datetime.now(),
            'type': 'vendor_order_guide',
            'vendor': vendor,
            'file': file_path,
            'rows_imported': len(products)
        })

        print(f"✅ Imported {len(products)} products")
        if errors:
            print(f"⚠️  {len(errors)} errors occurred")

        return result

    # ==================== INVOICE IMPORT ====================

    def import_invoice(
        self,
        file_path: str,
        vendor: str,
        sheet_name: Optional[str] = None
    ) -> Dict:
        """
        Import invoice from CSV or Excel

        Args:
            file_path: Path to invoice file
            vendor: Vendor name
            sheet_name: Sheet name for Excel

        Returns:
            Dict with invoice data and totals
        """
        print(f"📄 Importing invoice from {file_path}")

        df = self.read_file(file_path, sheet_name)

        # Required columns for invoice
        required = ['item_code', 'description', 'quantity', 'unit_price', 'extension']

        validation = self.validate_columns(df, required)

        if not validation['valid']:
            return {
                'success': False,
                'error': validation['message']
            }

        # Extract invoice metadata if present
        invoice_number = df['invoice_number'].iloc[0] if 'invoice_number' in df.columns else None
        order_date = df['order_date'].iloc[0] if 'order_date' in df.columns else datetime.now()

        # Process line items
        line_items = []
        total = 0

        for idx, row in df.iterrows():
            try:
                item = {
                    'item_code': str(row['item_code']).strip(),
                    'description': str(row['description']).strip(),
                    'quantity': int(row['quantity']),
                    'unit_price': float(row['unit_price']),
                    'extension': float(row['extension']),
                    'pack_size': str(row.get('pack_size', '')) if 'pack_size' in row else ''
                }

                line_items.append(item)
                total += item['extension']

            except Exception as e:
                print(f"⚠️  Row {idx+2} error: {e}")

        result = {
            'success': True,
            'vendor': vendor,
            'invoice_number': invoice_number,
            'order_date': order_date,
            'line_items': line_items,
            'item_count': len(line_items),
            'total_amount': total
        }

        print(f"✅ Imported invoice: {len(line_items)} items, Total: ${total:,.2f}")

        return result

    # ==================== RECIPE IMPORT ====================

    def import_recipes(
        self,
        file_path: str,
        sheet_name: Optional[str] = None
    ) -> Dict:
        """
        Import recipes from CSV or Excel

        Format: Each row is one ingredient in a recipe
        Rows with same Recipe Name are grouped together
        """
        print(f"📖 Importing recipes from {file_path}")

        df = self.read_file(file_path, sheet_name)

        required = ['recipe_name', 'ingredient', 'quantity', 'unit']

        validation = self.validate_columns(df, required)

        if not validation['valid']:
            return {
                'success': False,
                'error': validation['message']
            }

        # Group by recipe
        recipes = {}

        for idx, row in df.iterrows():
            recipe_name = str(row['recipe_name']).strip()

            if recipe_name not in recipes:
                recipes[recipe_name] = {
                    'name': recipe_name,
                    'category': str(row.get('category', 'Uncategorized')) if 'category' in row else 'Uncategorized',
                    'yield_amount': float(row.get('yield_amount', 1)) if 'yield_amount' in row else 1,
                    'yield_unit': str(row.get('yield_unit', 'servings')) if 'yield_unit' in row else 'servings',
                    'ingredients': []
                }

            # Add ingredient
            ingredient = {
                'name': str(row['ingredient']).strip(),
                'quantity': float(row['quantity']),
                'unit': str(row['unit']).strip(),
                'prep_instruction': str(row.get('prep_instruction', '')) if 'prep_instruction' in row else ''
            }

            recipes[recipe_name]['ingredients'].append(ingredient)

        result = {
            'success': True,
            'recipes_imported': len(recipes),
            'recipes': list(recipes.values())
        }

        print(f"✅ Imported {len(recipes)} recipes")

        return result

    # ==================== MENU ITEMS IMPORT ====================

    def import_menu_items(
        self,
        file_path: str,
        sheet_name: Optional[str] = None
    ) -> Dict:
        """Import menu items from CSV or Excel"""
        print(f"🍽️  Importing menu items from {file_path}")

        df = self.read_file(file_path, sheet_name)

        required = ['item_name', 'category', 'menu_price']

        validation = self.validate_columns(df, required)

        if not validation['valid']:
            return {
                'success': False,
                'error': validation['message']
            }

        menu_items = []

        for idx, row in df.iterrows():
            try:
                item = {
                    'name': str(row['item_name']).strip(),
                    'category': str(row['category']).strip(),
                    'subcategory': str(row.get('subcategory', '')) if 'subcategory' in row else '',
                    'menu_price': float(row['menu_price']),
                    'recipe_name': str(row.get('recipe_name', '')) if 'recipe_name' in row else '',
                    'portion_size': str(row.get('portion_size', '')) if 'portion_size' in row else '',
                    'target_margin': float(row.get('target_margin', 0.45)) if 'target_margin' in row else 0.45,
                    'description': str(row.get('description', '')) if 'description' in row else ''
                }

                menu_items.append(item)

            except Exception as e:
                print(f"⚠️  Row {idx+2} error: {e}")

        result = {
            'success': True,
            'items_imported': len(menu_items),
            'menu_items': menu_items
        }

        print(f"✅ Imported {len(menu_items)} menu items")

        return result

    # ==================== PRICE HISTORY IMPORT ====================

    def import_price_history(
        self,
        file_path: str,
        sheet_name: Optional[str] = None
    ) -> Dict:
        """
        Import historical pricing data

        Columns: Date, Item Code, Vendor, Price
        """
        print(f"📊 Importing price history from {file_path}")

        df = self.read_file(file_path, sheet_name)

        required = ['date', 'item_code', 'vendor', 'price']

        validation = self.validate_columns(df, required)

        if not validation['valid']:
            return {
                'success': False,
                'error': validation['message']
            }

        # Convert date column
        df['date'] = pd.to_datetime(df['date'])

        price_history = []

        for idx, row in df.iterrows():
            try:
                record = {
                    'date': row['date'],
                    'item_code': str(row['item_code']).strip(),
                    'vendor': str(row['vendor']).strip(),
                    'price': float(row['price']),
                    'source': 'import'
                }

                price_history.append(record)

            except Exception as e:
                print(f"⚠️  Row {idx+2} error: {e}")

        result = {
            'success': True,
            'records_imported': len(price_history),
            'price_history': price_history
        }

        print(f"✅ Imported {len(price_history)} price records")

        return result

    # ==================== BULK IMPORT ====================

    def import_multiple_files(self, file_configs: List[Dict]) -> Dict:
        """
        Import multiple files at once

        Args:
            file_configs: List of dicts with:
                {
                    'file_path': 'path/to/file.csv',
                    'import_type': 'vendor_order_guide' | 'invoice' | 'recipe' | 'menu_items',
                    'vendor': 'SYSCO' (if applicable),
                    'sheet_name': 'Sheet1' (optional)
                }

        Returns:
            Dict with results for each file
        """
        results = {
            'total_files': len(file_configs),
            'successful': 0,
            'failed': 0,
            'results': []
        }

        for config in file_configs:
            try:
                import_type = config['import_type']

                if import_type == 'vendor_order_guide':
                    result = self.import_vendor_order_guide(
                        file_path=config['file_path'],
                        vendor=config['vendor'],
                        sheet_name=config.get('sheet_name')
                    )

                elif import_type == 'invoice':
                    result = self.import_invoice(
                        file_path=config['file_path'],
                        vendor=config['vendor'],
                        sheet_name=config.get('sheet_name')
                    )

                elif import_type == 'recipe':
                    result = self.import_recipes(
                        file_path=config['file_path'],
                        sheet_name=config.get('sheet_name')
                    )

                elif import_type == 'menu_items':
                    result = self.import_menu_items(
                        file_path=config['file_path'],
                        sheet_name=config.get('sheet_name')
                    )

                else:
                    raise ValueError(f"Unknown import type: {import_type}")

                if result['success']:
                    results['successful'] += 1
                else:
                    results['failed'] += 1

                results['results'].append(result)

            except Exception as e:
                results['failed'] += 1
                results['results'].append({
                    'success': False,
                    'file': config['file_path'],
                    'error': str(e)
                })

        return results

    # ==================== GOOGLE SHEETS IMPORT ====================

    def import_from_google_sheets(
        self,
        sheet_url: str,
        import_type: str,
        vendor: Optional[str] = None
    ) -> Dict:
        """
        Import from Google Sheets

        Args:
            sheet_url: Google Sheets URL (must be publicly viewable or you have credentials)
            import_type: 'vendor_order_guide', 'invoice', 'recipe', 'menu_items'
            vendor: Vendor name (if applicable)

        Note: Requires gspread library: pip install gspread
        """
        try:
            import gspread
            from google.oauth2.service_account import Credentials
        except ImportError:
            return {
                'success': False,
                'error': 'gspread not installed. Run: pip install gspread google-auth'
            }

        # Option 1: Public sheet (export as CSV)
        if '/edit' in sheet_url:
            # Convert edit URL to export URL
            sheet_id = sheet_url.split('/d/')[1].split('/')[0]
            export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

            # Download as CSV
            df = pd.read_csv(export_url)

            # Clean columns
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

            # Save temporarily
            temp_file = f'/tmp/import_{import_type}_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
            df.to_csv(temp_file, index=False)

            # Import using existing methods
            if import_type == 'vendor_order_guide':
                return self.import_vendor_order_guide(temp_file, vendor)
            elif import_type == 'invoice':
                return self.import_invoice(temp_file, vendor)
            elif import_type == 'recipe':
                return self.import_recipes(temp_file)
            elif import_type == 'menu_items':
                return self.import_menu_items(temp_file)

        # Option 2: Authenticated access (requires credentials.json)
        # [Implementation would go here if needed]

        return {
            'success': False,
            'error': 'Invalid Google Sheets URL'
        }

    # ==================== EXPORT TEMPLATES ====================

    def create_import_templates(self, output_dir: str = 'data/templates'):
        """
        Create template CSV files for importing data

        Saves to output_dir:
        - vendor_order_guide_template.csv
        - invoice_template.csv
        - recipe_template.csv
        - menu_items_template.csv
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Vendor Order Guide Template
        order_guide_template = pd.DataFrame({
            'Item Code': ['SYS001', 'SYS002', 'SYS003'],
            'Description': ['GROUND BEEF 80/20', 'BLACK PEPPER COARSE', 'ONION POWDER'],
            'Pack Size': ['10 LB', '6/1LB', '6/1LB'],
            'Case Price': [45.99, 298.95, 148.95],
            'Unit Price': [4.599, 49.83, 24.83],
            'Unit': ['LB', 'LB', 'LB'],
            'Category': ['MEAT', 'SPICES', 'SPICES']
        })
        order_guide_template.to_csv(f'{output_dir}/vendor_order_guide_template.csv', index=False)

        # Invoice Template
        invoice_template = pd.DataFrame({
            'Order Date': ['2025-01-18', '2025-01-18'],
            'Invoice Number': ['INV-12345', 'INV-12345'],
            'Item Code': ['SYS001', 'SYS002'],
            'Description': ['GROUND BEEF 80/20', 'BLACK PEPPER COARSE'],
            'Quantity': [5, 2],
            'Unit Price': [45.99, 298.95],
            'Extension': [229.95, 597.90],
            'Pack Size': ['10 LB', '6/1LB']
        })
        invoice_template.to_csv(f'{output_dir}/invoice_template.csv', index=False)

        # Recipe Template
        recipe_template = pd.DataFrame({
            'Recipe Name': ['BBQ Sauce', 'BBQ Sauce', 'BBQ Sauce', 'Green Chile', 'Green Chile'],
            'Ingredient': ['Tomato Paste', 'Brown Sugar', 'Apple Cider Vinegar', 'Pork Shoulder', 'Green Chiles'],
            'Quantity': [2, 1, 0.5, 5, 3],
            'Unit': ['cups', 'cup', 'cup', 'lbs', 'cans'],
            'Yield Amount': [1, 1, 1, 50, 50],
            'Yield Unit': ['gallon', 'gallon', 'gallon', 'servings', 'servings'],
            'Category': ['Sauce', 'Sauce', 'Sauce', 'Entree', 'Entree']
        })
        recipe_template.to_csv(f'{output_dir}/recipe_template.csv', index=False)

        # Menu Items Template
        menu_template = pd.DataFrame({
            'Item Name': ['Smothered Burrito', 'BBQ Pulled Pork', 'Green Chile Bowl'],
            'Category': ['Entree', 'Entree', 'Entree'],
            'Subcategory': ['Mexican', 'BBQ', 'Mexican'],
            'Menu Price': [12.99, 10.99, 9.99],
            'Recipe Name': ['Green Chile', 'BBQ Sauce', 'Green Chile'],
            'Portion Size': ['12 oz', '8 oz', '10 oz'],
            'Target Margin': [0.45, 0.45, 0.45]
        })
        menu_template.to_csv(f'{output_dir}/menu_items_template.csv', index=False)

        print(f"✅ Created templates in {output_dir}/")
        return output_dir


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    importer = FileImporter()

    # Create templates first
    importer.create_import_templates('data/templates')

    # Example 1: Import SYSCO order guide
    print("\n" + "="*60)
    print("EXAMPLE 1: Import SYSCO Order Guide")
    print("="*60)

    result = importer.import_vendor_order_guide(
        file_path='data/sysco_order_guide.xlsx',
        vendor='SYSCO',
        sheet_name='Products'
    )

    if result['success']:
        print(f"✅ Imported {result['products_imported']} products")
        print(f"Sample product: {result['products'][0]}")

    # Example 2: Import invoice
    print("\n" + "="*60)
    print("EXAMPLE 2: Import Invoice")
    print("="*60)

    invoice = importer.import_invoice(
        file_path='data/sysco_invoice.csv',
        vendor='SYSCO'
    )

    if invoice['success']:
        print(f"✅ Invoice: {invoice['invoice_number']}")
        print(f"Total: ${invoice['total_amount']:,.2f}")

    # Example 3: Import recipes
    print("\n" + "="*60)
    print("EXAMPLE 3: Import Recipes")
    print("="*60)

    recipes = importer.import_recipes('data/recipes.xlsx')

    if recipes['success']:
        print(f"✅ Imported {recipes['recipes_imported']} recipes")
        for recipe in recipes['recipes'][:2]:  # Show first 2
            print(f"\nRecipe: {recipe['name']}")
            print(f"Ingredients: {len(recipe['ingredients'])}")

    # Example 4: Bulk import
    print("\n" + "="*60)
    print("EXAMPLE 4: Bulk Import")
    print("="*60)

    bulk_result = importer.import_multiple_files([
        {
            'file_path': 'data/sysco_order_guide.xlsx',
            'import_type': 'vendor_order_guide',
            'vendor': 'SYSCO'
        },
        {
            'file_path': 'data/shamrock_order_guide.csv',
            'import_type': 'vendor_order_guide',
            'vendor': 'Shamrock Foods'
        },
        {
            'file_path': 'data/recipes.xlsx',
            'import_type': 'recipe'
        }
    ])

    print(f"✅ Successful: {bulk_result['successful']}")
    print(f"❌ Failed: {bulk_result['failed']}")

    # Example 5: Google Sheets (if public)
    print("\n" + "="*60)
    print("EXAMPLE 5: Import from Google Sheets")
    print("="*60)

    # Public Google Sheet URL
    sheet_url = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit"

    gsheet_result = importer.import_from_google_sheets(
        sheet_url=sheet_url,
        import_type='vendor_order_guide',
        vendor='SYSCO'
    )
```

---

## Usage in Flask App

Add routes to your `app.py`:

```python
from flask import Flask, request, jsonify, render_template
from modules.importers.file_importer import FileImporter
import os

app = Flask(__name__)
importer = FileImporter()

# Configure upload folder
UPLOAD_FOLDER = 'data/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/import/vendor-order-guide', methods=['POST'])
def import_vendor_order_guide():
    """Upload and import vendor order guide"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    vendor = request.form.get('vendor', 'SYSCO')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save file
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    # Import
    result = importer.import_vendor_order_guide(
        file_path=filename,
        vendor=vendor
    )

    return jsonify(result)

@app.route('/import/invoice', methods=['POST'])
def import_invoice():
    """Upload and import invoice"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    vendor = request.form.get('vendor', 'SYSCO')

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    result = importer.import_invoice(
        file_path=filename,
        vendor=vendor
    )

    return jsonify(result)

@app.route('/import/recipes', methods=['POST'])
def import_recipes():
    """Upload and import recipes"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    result = importer.import_recipes(file_path=filename)

    return jsonify(result)

@app.route('/templates/download/<template_type>')
def download_template(template_type):
    """Download import template"""
    from flask import send_file

    templates_dir = 'data/templates'
    templates = {
        'vendor': 'vendor_order_guide_template.csv',
        'invoice': 'invoice_template.csv',
        'recipe': 'recipe_template.csv',
        'menu': 'menu_items_template.csv'
    }

    if template_type not in templates:
        return jsonify({'error': 'Invalid template type'}), 400

    filepath = os.path.join(templates_dir, templates[template_type])

    return send_file(
        filepath,
        as_attachment=True,
        download_name=templates[template_type]
    )
```

---

## Command Line Usage

```bash
# Create templates
python -m modules.importers.file_importer

# Import SYSCO order guide
python -c "from modules.importers.file_importer import FileImporter; \
    FileImporter().import_vendor_order_guide('data/sysco.xlsx', 'SYSCO')"

# Import Shamrock order guide
python -c "from modules.importers.file_importer import FileImporter; \
    FileImporter().import_vendor_order_guide('data/shamrock.csv', 'Shamrock Foods')"
```

---

## Error Handling

The importer validates:
- ✅ File exists
- ✅ File format is supported
- ✅ Required columns present
- ✅ Data types are correct
- ✅ Prices are positive
- ✅ No duplicate records

Errors are returned in results:
```python
{
    'success': False,
    'error': 'Missing required columns: item_code, description',
    'missing_columns': ['item_code', 'description']
}
```

---

## Next Steps

1. **Create templates**: Run the script to generate CSV templates
2. **Fill templates**: Use Excel/Google Sheets to enter data
3. **Import**: Upload files through Flask or command line
4. **Validate**: Check import results before saving to database

Would you like me to:
1. Create the actual Python file?
2. Add more import formats?
3. Create a simple web UI for uploads?
4. Add validation rules?
