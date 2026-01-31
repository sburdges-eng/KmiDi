# Data Importer Enhancements

## Summary

Successfully enhanced the data importers to properly extract data from the Lariat Bible restaurant files with correct sheet names and column mappings.

## What Was Improved

### 1. Excel Importer - Menu Bible
**File**: `data_importers/excel_importer.py`

#### New Methods:
- `import_menu_bible()` - Now explicitly imports from INDEX sheet (with fallback)
- `import_single_recipe(filename, recipe_name)` - Parses individual recipe sheets with proper structure
- `import_menu_bible_recipes()` - Imports all 22 recipe sheets with ingredients

#### Recipe Structure Parsed:
```python
{
    'name': 'The Trio',
    'category': 'Appetizer',
    'base_yield': '1 serving',
    'scaled_yield': '1 serving',
    'scale_factor': 1,
    'ingredients': [
        {
            'name': 'House-made guacamole',
            'base_qty': 4.0,
            'unit': 'ounces',
            'scaled_qty': 4.0
        },
        # ... more ingredients
    ]
}
```

### 2. Excel Importer - Smart Costing
**Updated Sheet Names**: Changed from generic names to actual sheet names discovered in the files:
- `'SMART TEMPLATE'` - Recipe costing template
- `'Ingredient Database'` - Master ingredient pricing
- `'🚀 START HERE'` - Getting started guide

#### New Method:
- `import_ingredient_database()` - Properly parses the Ingredient Database with correct column mappings

#### Ingredient Database Structure:
```python
{
    'name': 'Heavy Cream',
    'category': 'Dairy',
    'purchase_unit': 'qt',
    'package_size': 1.0,
    'cost_per_package': 4.89,
    'cost_per_lb': 2.29,
    'cost_per_oz': 0.14
}
```

### 3. Column Mapping Fixes

**Ingredient Database**:
- Column: `'🗄️ INGREDIENT DATABASE - MASTER PRICING & CONVERSIONS'` → Ingredient Name
- Column_1 → CATEGORY
- Column_2 → PURCHASE UNIT
- Column_3 → PACKAGE SIZE
- Column_4 → COST PER PKG
- Column_5 → COST/LB
- Column_6 → COST/OZ

**Recipe Sheets**:
- First Column → Recipe name (header)
- Row 0, Column_1 → Category
- Row 1, Column_1 → Base Yield
- Row 1, Column_4 → Scaled Yield
- Row 4+ → Ingredients data

## Test Results

### Successfully Imported:

#### Menu Bible Recipes (22 total)
- Jalapeño Cheddar Cornbread (11 ingredients)
- The Trio (6 ingredients)
- The Pig Wings (12 ingredients)
- Summer Toast (10 ingredients)
- Mountain Mac & Cheese (10 ingredients)
- Chicken Wings
- The Rope Salad
- Beet Salad
- Green Chile
- Classic BLT
- The Rope Burger
- El Jefe Burger
- Nashville Hot Chicken Sandwich
- Fish and Chips
- Baja Fish Tacos
- Maria Birria Tacos
- Pork Chop
- Green Chile Breakfast Burrito
- Sourdough French Toast
- Huevos Rancheros
- Breakfast Sandwich
- Nashville Hot Chicken and Waffles

#### Ingredient Database (139 ingredients)
Sample ingredients with pricing:
- Heavy Cream: $4.89/qt, $2.29/lb
- Whole Milk: $4.50/gal, $0.53/lb
- Buttermilk: $5.20/gal, $0.61/lb
- Cream Cheese: $8.50/3lb, $2.83/lb
- Cheddar Cheese: $22.50/5lb
- Pepper Jack Cheese: $24.00/5lb
- ... and 133 more

### Import Statistics:
- **Excel Files**: 3
- **Sheets Imported**: 26
- **Total Rows**: 960
- **Recipes with Ingredients**: 22
- **Ingredients in Database**: 139

## Usage Examples

### Import All Recipes
```python
from data_importers import ExcelImporter

excel_importer = ExcelImporter()

# Import all recipe sheets
recipes_data = excel_importer.import_menu_bible_recipes()

for recipe in recipes_data['recipes']:
    print(f"{recipe['name']} - {recipe['category']}")
    print(f"Ingredients: {len(recipe['ingredients'])}")
    for ing in recipe['ingredients']:
        print(f"  - {ing['name']}: {ing['scaled_qty']} {ing['unit']}")
```

### Import Ingredient Database
```python
from data_importers import ExcelImporter

excel_importer = ExcelImporter()

# Import ingredient pricing database
ingredients = excel_importer.import_ingredient_database()

for ing in ingredients:
    print(f"{ing['name']} ({ing['category']})")
    print(f"  ${ing['cost_per_package']}/{ing['purchase_unit']}")
    print(f"  ${ing['cost_per_lb']}/lb")
```

### Import Single Recipe
```python
from data_importers import ExcelImporter

excel_importer = ExcelImporter()

# Import specific recipe
recipe = excel_importer.import_single_recipe(
    "LARIAT MENU BIBLE PT.18 V3.xlsx",
    "The Trio"
)

print(f"Recipe: {recipe['name']}")
print(f"Category: {recipe['category']}")
print(f"Yield: {recipe['base_yield']}")
print(f"\nIngredients:")
for ing in recipe['ingredients']:
    print(f"  - {ing['name']}: {ing['scaled_qty']} {ing['unit']}")
```

### Use Unified Importer
```python
from data_importers import UnifiedImporter

# Initialize unified importer
importer = UnifiedImporter()

# Import all restaurant data at once
all_data = importer.import_all_restaurant_data()

# Access different data types
print(f"Menu items: {len(all_data['menu']['items'])}")
print(f"Recipes: {len(all_data['recipes'])}")
print(f"Inventory: {len(all_data['inventory'])}")
```

## Files Created/Modified

1. **data_importers/excel_importer.py**
   - Updated `import_smart_costing()` with correct sheet names
   - Updated `import_menu_bible()` to use INDEX sheet
   - Added `import_single_recipe()` method
   - Updated `import_menu_bible_recipes()` to use new parsing
   - Added `import_ingredient_database()` method with correct columns

2. **test_enhanced_importers.py**
   - Comprehensive test demonstrating all new features
   - Shows sample data from each import type

3. **diagnose_columns.py**
   - Diagnostic tool to inspect actual column names
   - Useful for debugging future import issues

4. **IMPORTER_ENHANCEMENTS.md** (this file)
   - Documentation of improvements and usage

## Testing

Run the enhanced test:
```bash
python3 test_enhanced_importers.py
```

Run column diagnostic (for troubleshooting):
```bash
python3 diagnose_columns.py
```

## Next Steps

Potential enhancements:
1. Add recipe costing calculations (combine recipes with ingredient database)
2. Create API endpoints to expose recipe and ingredient data
3. Build UI components to display recipes and ingredients
4. Add search/filter functionality for recipes and ingredients
5. Export recipes to PDF or printable format
6. Calculate total menu costs based on recipes and ingredients
