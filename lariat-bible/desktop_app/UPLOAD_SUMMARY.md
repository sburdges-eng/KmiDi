# Desktop Folders Upload Summary ✅

## Upload Complete!

Successfully uploaded **BANQUET BEO** and **LARIAT** folders from desktop to the application data directory.

## Upload Location

```
/Users/seanburdges/lariat-bible/desktop_app/data/uploaded/
```

## Upload Statistics

| Folder | Size | Files | Description |
|--------|------|-------|-------------|
| BANQUET BEO | 40 KB | 3 files | Banquet event orders and invoices |
| LARIAT | 1.4 GB | 785 files | Complete restaurant management data |
| **TOTAL** | **1.4 GB** | **788 files** | All uploaded data |

## BANQUET BEO Contents

**3 Files:**
1. Kitchen Order guide 7_12_25.xlsx (23 KB)
2. Lariat Invoice McMahon 7_18.xlsx (6.9 KB)
3. Lariat Invoice Bob Clauss 8_2 5-7pm.xlsx (7.0 KB)

**Purpose:** Banquet event management, catering orders, and client invoices

## LARIAT Folder Contents

**Major Components:**

### Excel Files & Documents (Root Level):
- LARIAT MENU BIBLE PT.18 V3.xlsx (59 KB) - Complete menu with 22 recipes
- EMERGENCY ABSENCE.xlsx (13 KB)
- QUARTER REPORT.xlsx (16 KB)
- Lariat Recipe Book.docx (24 KB)

### SMART COSTING 1 Folder:
- LARIAT_SMART_COSTING_COMPLETE_1.xlsx - Ingredient database with 139 items
- FINAL_SUMMARY.txt
- HOW_TO_USE_GUIDE.txt
- RECIPE_LIST.txt
- SCALING_FEATURE_GUIDE.txt

### FINAL-OG-COMBO Folder:
- VENDOR_MATCHING_PH_RESULTS.xlsx (214 rows of vendor comparisons)
- run_matching_PH_version.py
- README_PH_MATCHING.txt

### ORDER HIST. JUN-OCT 2025:
- ALL_INVOICES_20251105_0315.xlsx
- 85+ invoice images (JPEG format with background removed)
- Exposure backups of all images

### LARIAT BIBLE TEMPLATE:
- CSV templates (8 files):
  - catering_event_template.csv
  - equipment_maintenance_template.csv
  - inventory_template.csv
  - invoice_template.csv
  - order_guide_template.csv
  - recipe_cost_template.csv
  - spice_comparison_template.csv
  - vendor_product_comparison_template.csv
- CSV_TEMPLATES_README.md
- Professional Catering Kitchen Prep Requirements Guide (PDF)

### Image Archives:
- **heic/** - 241 HEIC format invoice images
- **ORDER HIST. JUN-OCT 2025/COMBO O.H./jpeg/** - 85 processed invoice JPEGs

### Python Scripts:
- PYTHON SCRIPTS/JPEG-2-CSV/EXTRACT_INVOICES.py
- heic/extract_all_invoices.py
- heic/extract_sysco.py

### Other Folders:
- lariat-bible/ - Desktop app project files
- drive-download-*/  - Additional backups
- Recipe_Book_Project/
- data/ - CSV data files

## Key Data Files Now Available

### For Recipe Manager:
✅ LARIAT MENU BIBLE PT.18 V3.xlsx
- 22 complete recipes
- All recipe sheets with ingredients and quantities

✅ LARIAT_SMART_COSTING_COMPLETE_1.xlsx
- 139 ingredients with pricing
- Ingredient Database sheet
- Multiple recipe costing sheets

### For Invoice Processing:
✅ 85+ invoice images (JPEG with background removed)
✅ Historical order data (Jun-Oct 2025)
✅ Invoice extraction Python scripts

### For Vendor Management:
✅ VENDOR_MATCHING_PH_RESULTS.xlsx
✅ Multi-vendor comparison data
✅ Price difference analysis

### For Catering/BEO:
✅ Kitchen Order guides
✅ Client invoices (McMahon, Bob Clauss)
✅ Event planning templates

## File Organization

```
desktop_app/data/uploaded/
├── BANQUET BEO/
│   ├── Kitchen Order guide 7_12_25.xlsx
│   ├── Lariat Invoice McMahon 7_18.xlsx
│   └── Lariat Invoice Bob Clauss 8_2 5-7pm.xlsx
└── LARIAT/
    ├── LARIAT MENU BIBLE PT.18 V3.xlsx
    ├── EMERGENCY ABSENCE.xlsx
    ├── QUARTER REPORT.xlsx
    ├── Lariat Recipe Book.docx
    ├── SMART COSTING 1/
    │   ├── LARIAT_SMART_COSTING_COMPLETE_1.xlsx
    │   ├── FINAL_SUMMARY.txt
    │   ├── HOW_TO_USE_GUIDE.txt
    │   ├── RECIPE_LIST.txt
    │   └── SCALING_FEATURE_GUIDE.txt
    ├── FINAL-OG-COMBO/
    │   ├── VENDOR_MATCHING_PH_RESULTS.xlsx
    │   ├── run_matching_PH_version.py
    │   └── README_PH_MATCHING.txt
    ├── ORDER HIST. JUN-OCT 2025/
    │   └── COMBO O.H./
    │       ├── ALL_INVOICES_20251105_0315.xlsx
    │       └── jpeg/ (85 invoice images)
    ├── LARIAT BIBLE TEMPLATE/
    │   ├── 8 CSV templates
    │   ├── CSV_TEMPLATES_README.md
    │   └── Professional Catering Kitchen Guide.pdf
    ├── heic/ (241 HEIC images)
    ├── PYTHON SCRIPTS/
    └── lariat-bible/ (project files)
```

## Integration Status

### Currently Integrated:
✅ Recipe Manager using Menu Bible and Smart Costing
✅ Ingredient Database (139 items)
✅ 22 Recipes with full ingredient lists

### Ready to Integrate:
- ⏳ Banquet/BEO order management
- ⏳ Invoice processing (85+ invoices)
- ⏳ Vendor matching and comparison
- ⏳ Order history analysis
- ⏳ Catering templates

## Next Steps

1. **Integrate BEO Data** - Add banquet event order processing
2. **Invoice Processing** - Import and analyze 85+ invoice images
3. **Vendor Comparison** - Integrate vendor matching results
4. **Order History** - Analyze June-October 2025 ordering patterns
5. **Template Management** - Make CSV templates accessible in app

## Data Usage

The uploaded files can now be accessed by the data importers:

```python
from data_importers import UnifiedImporter, ExcelImporter

# Access uploaded files
importer = UnifiedImporter(data_directory='data/uploaded/LARIAT')

# Import Menu Bible from uploaded location
excel_importer = ExcelImporter('data/uploaded/LARIAT')
recipes = excel_importer.import_menu_bible_recipes()

# Import Smart Costing
costing = excel_importer.import_smart_costing()
ingredients = excel_importer.import_ingredient_database()
```

## Upload Verification

✅ BANQUET BEO folder (40 KB, 3 files)
✅ LARIAT folder (1.4 GB, 785 files)
✅ Total: 788 files successfully copied
✅ All data accessible at `/Users/seanburdges/lariat-bible/desktop_app/data/uploaded/`

---

**Upload Date**: November 18, 2025
**Upload Location**: `/Users/seanburdges/lariat-bible/desktop_app/data/uploaded/`
**Total Size**: 1.4 GB
**Total Files**: 788
