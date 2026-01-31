# Order Guide Import Templates

**Created**: November 18, 2025 (10:00 PM)
**Purpose**: Parse vendor order guides (Sysco & Shamrock) into unified catalog

---

## Overview

The Order Guide Parser system supports importing product catalogs from multiple vendors:
- **Sysco** - Using SYSCO PH template
- **Shamrock** - Using SHAM PH template
- **Combined** - Using COMBO PH template (includes purchase history)

---

## Vendor Templates

### 1. SYSCO PH Template (Sysco Import)

**Format**: Sysco Purchase History Export
**File**: `COMBO PH NO TOUCH (1).xlsx` → Sheet: `SYSCO PH`

**Structure**:
```
Row 1: Header (H | L0601 | 59 | 75356 | Purchase History | 2)
Row 2: Field Names (F | SUPC | Case Qty | Split Qty | Code | Item Status | Replaced Item | Pack | Size | Unit | Brand | Mfr # | Desc | Cat | Case $)
Row 3+: Products (P | product_code | ... | pack | size | unit | brand | mfr# | description | category | price)
```

**Key Fields**:
- **SUPC**: Sysco Universal Product Code (column 2)
- **Pack**: Package quantity (column 8)
- **Size**: Package size (column 9)
- **Unit**: Unit of measure (column 10)
- **Brand**: Brand name (column 11)
- **Mfr #**: Manufacturer number (column 12)
- **Desc**: Product description (column 13)
- **Cat**: Category (column 14)
- **Case $**: Case price (column 15)

**Example Product**:
```json
{
  "vendor": "Sysco",
  "product_code": "6438545",
  "pack": 1,
  "size": "15LB",
  "unit": "LB",
  "brand": "RANCH & GRILL",
  "mfr_number": "10139862895",
  "description": "BACON, APPLEWOOD SMOKED",
  "price": "$65.50"
}
```

**Total Products**: 432

---

### 2. SHAM PH Template (Shamrock Import)

**Format**: Shamrock Order Guide
**File**: `COMBO PH NO TOUCH (1).xlsx` → Sheet: `SHAM PH`

**Structure**:
```
Rows 1-4: Header/Metadata
Row 5: Column Headers (# | Product # | Description | | | Pack Size | Brand | LWP | Avg | Price | Unit)
Row 6+: Products (number | product_code | description | | | pack_size | brand | lwp | avg | price | unit)
```

**Key Fields**:
- **Product #**: Shamrock product code (column 3)
- **Description**: Product description (column 4)
- **Pack Size**: Package size (column 7)
- **Brand**: Brand name (column 8)
- **LWP**: Last Week's Purchase quantity (column 9)
- **Avg**: Average usage (column 10)
- **Price**: Unit price (column 11)
- **Unit**: Unit of measure (column 12)

**Example Product**:
```json
{
  "vendor": "Shamrock",
  "product_code": "2175761",
  "description": "VINEGAR, WHT 5% DISTILLED BTL",
  "pack_size": "4/128/OZ",
  "brand": "KATYO",
  "lwp": 4,
  "avg": "2 CS",
  "price": 27.00,
  "unit": "CS"
}
```

**Total Products**: 227

---

### 3. COMBO PH Template (Combined with Purchase History)

**Format**: Combined Sysco + Purchase History
**File**: `COMBO PH NO TOUCH (1).xlsx` → Sheet: `COMBO PH`

**Structure**:
```
Row 1: Header (same as SYSCO PH)
Row 2: Field Names (F | SUPC | Case Qty | Split Qty | Code | ... | PLUS weekly/monthly purchase columns)
Row 3+: Products (P | product_code | ... | PLUS purchase history data)
```

**Key Fields**: Same as SYSCO PH, PLUS:
- **Purchase History**: Columns 21-444 contain historical purchase data (weekly/monthly quantities)
- **Net Wt**: Net weight
- **Stock**: Stock status (STOCKED/OUT OF STOCK)
- **Agr**: Agreement status

**Example Product**:
```json
{
  "vendor": "Combined",
  "product_code": "6438545",
  "pack": 1,
  "size": "15LB",
  "unit": "LB",
  "brand": "RANCH & GRILL",
  "mfr_number": "10139862895",
  "purchase_history": {
    "Net Wt": 15,
    "Stock": "STOCKED",
    "Agr": "N",
    "Week1": 5,
    "Week2": 3,
    "Week3": 4
  }
}
```

**Total Products**: 432 (with extensive purchase history)

---

## Usage

### Parse All Vendors from File

```python
from data_importers.order_guide_parser import OrderGuideParser

parser = OrderGuideParser()
results = parser.parse_all_vendors('path/to/order_guide.xlsx')

print(f"Sysco products: {results['sysco']['total_products']}")
print(f"Shamrock products: {results['shamrock']['total_products']}")
print(f"Combo products: {results['combo']['total_products']}")
```

### Parse Single Vendor

```python
# Auto-detect vendor
data = parser.parse_file('path/to/order_guide.xlsx')

# Or specify vendor explicitly
sysco_data = parser.parse_file('path/to/file.xlsx', vendor='sysco')
shamrock_data = parser.parse_file('path/to/file.xlsx', vendor='shamrock')
```

### Create Unified Catalog

```python
from data_importers.order_guide_parser import create_unified_catalog

catalog = create_unified_catalog(
    results['sysco']['products'],
    results['shamrock']['products']
)

print(f"Total unique products: {catalog['total_unique_products']}")
```

---

## Unified Catalog Format

The parser creates a unified catalog that combines products from all vendors:

```json
{
  "total_unique_products": 659,
  "products": [
    {
      "id": "CAT-000001",
      "vendors": ["Sysco"],
      "sysco_code": "6438545",
      "description": "BACON, APPLEWOOD SMOKED",
      "pack": 1,
      "size": "15LB",
      "unit": "LB",
      "brand": "RANCH & GRILL",
      "sysco_price": 65.50
    },
    {
      "id": "CAT-000433",
      "vendors": ["Shamrock"],
      "shamrock_code": "2175761",
      "description": "VINEGAR, WHT 5% DISTILLED BTL",
      "pack_size": "4/128/OZ",
      "brand": "KATYO",
      "shamrock_price": 27.00,
      "unit": "CS"
    }
  ],
  "vendor_map": {
    "SYSCO_6438545": 1,
    "SHAM_2175761": 433
  }
}
```

---

## Import Statistics

**From Current File**: `COMBO PH NO TOUCH (1).xlsx`

| Vendor | Sheet Name | Products | Columns | Notes |
|--------|------------|----------|---------|-------|
| Sysco | SYSCO PH | 432 | 26 | Standard purchase history |
| Shamrock | SHAM PH | 227 | 12 | Includes LWP and Avg usage |
| Combined | COMBO PH | 432 | 444 | Extensive purchase history |

**Total Unique Products**: 659 (after combining Sysco + Shamrock)

---

## Field Mappings

### Sysco → Unified Catalog
- `SUPC` → `sysco_code`
- `Pack` → `pack`
- `Size` → `size`
- `Unit` → `unit`
- `Brand` → `brand`
- `Desc` → `description`
- `Case $` → `sysco_price`

### Shamrock → Unified Catalog
- `Product #` → `shamrock_code`
- `Description` → `description`
- `Pack Size` → `pack_size`
- `Brand` → `brand`
- `Price` → `shamrock_price`
- `Unit` → `unit`
- `LWP` → `last_week_purchase`
- `Avg` → `avg_usage`

---

## Product Categories Found

### Food Products:
- Vinegar, Trout, Tomatoes, Tortillas, Tomatillos
- Bacon, Sugar, Syrups (Coke, Sprite, Ginger Ale)
- Oils, Seasonings, Sauces

### Non-Food Products:
- Paper towels, Tissues, Straws
- Cleaning supplies, Disposables

---

## Sample Products by Category

### Proteins:
```
[JIT] TROUT, RAINBOW WHL 9-11Z (4884541) - $9.98/LB
BACON, APPLEWOOD SMOKED (6438545) - $65.50/CS
```

### Vegetables:
```
TOMATO, HEIRLOOM BABY MEDLEY (4677031) - $29.14/CS
TOMATO, GRP RED (2043241) - $20.23/CS
TOMATILLO, IMP CAN WHOLE (0017925) - $37.31/CS
```

### Beverages:
```
SYRUP, COKE CLASSIC BIB (1959761) - $118.31/CS
SYRUP, SPRITE BIB (1959891) - $117.30/CS
SYRUP, GINGER ALE BIB (3328181) - $63.86/CS
```

### Condiments:
```
VINEGAR, WHT 5% DISTILLED BTL (2175761) - $27.00/CS
VINEGAR, MALT (1918171) - $35.13/CS
```

### Supplies:
```
TOWEL, PAPR MULTIFOLD WHT (4212861) - $44.20/CS
TISSUE, BATH 2PLY 500 SHEET RL WHT (4212701) - $75.67/CS
STRAW, BEV PP 7.75" BLK UNWRPD JUMBO (0032622) - $99.24/CS
```

---

## Files Created

1. **Parser**: `/Users/seanburdges/lariat-bible/desktop_app/data_importers/order_guide_parser.py`
2. **Catalog**: `/Users/seanburdges/lariat-bible/desktop_app/unified_catalog.json`
3. **Documentation**: This file

---

## Next Steps

1. ✅ **Import Templates Created**
   - Sysco PH parser (432 products)
   - Shamrock PH parser (227 products)
   - Combo PH parser (432 products + history)

2. **Potential Enhancements**:
   - [ ] Product matching across vendors (find same product from different suppliers)
   - [ ] Price comparison tool
   - [ ] Purchase history analysis
   - [ ] Automated ordering based on avg usage
   - [ ] Integration with recipe manager (link ingredients to vendor products)
   - [ ] Weekly order generation based on recipes scheduled

3. **Integration Points**:
   - Connect to Recipe Manager ingredient database
   - Link vendor products to recipe ingredients
   - Calculate recipe costs using vendor prices
   - Generate shopping lists from menu plans

---

**Status**: ✅ COMPLETE
**Templates**: 3 vendor formats supported
**Products Imported**: 659 unique products
**Ready for**: Catalog integration, price comparison, order management
