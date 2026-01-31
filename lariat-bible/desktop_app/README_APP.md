# Lariat Bible - Complete Restaurant Management System 🍽️

## 🚀 Quick Start

**Double-click** `START_LARIAT_BIBLE.command` to launch the application!

Or run from terminal:
```bash
python3 lariat_bible_app.py
```

---

## 📋 What's Included

### Complete Desktop Application with 5 Tabs:

1. **📊 Dashboard** - System overview and quick actions
2. **📖 Recipes** - 53 recipes from Lariat Recipe Book
3. **🏪 Vendors** - 659 products (Sysco + Shamrock)
4. **📋 Orders** - Automated order generation
5. **📈 Analysis** - Purchase history insights

---

## ✨ Features

### Recipe Management
- ✅ 53 recipes from Word document
- ✅ Search and filter by category
- ✅ View ingredients and instructions
- ✅ Ingredient pricing from database

### Vendor Management
- ✅ 432 Sysco products
- ✅ 227 Shamrock products
- ✅ Price comparison tool
- ✅ Product matching across vendors (fuzzy search)
- ✅ Real pricing data

### Order Generation
- ✅ Select multiple recipes
- ✅ Set servings multiplier (2x, 3x, etc.)
- ✅ Automatic vendor consolidation
- ✅ Cost estimation
- ✅ Export to JSON

### Purchase Analysis
- ✅ 432 products with purchase history
- ✅ Top 20 most purchased items
- ✅ Trend identification
- ✅ Recommendations engine

---

## 🎯 How to Use

### 1. View Recipes
1. Click "📖 Recipes" tab
2. Browse recipe list or search
3. Select a recipe to view details
4. See ingredients with pricing

### 2. Compare Vendor Prices
1. Click "🏪 Vendors" tab
2. Click "Compare Prices" button
3. View price differences between Sysco & Shamrock
4. Identify savings opportunities

### 3. Generate Orders
1. Click "📋 Orders" tab
2. Select recipes (hold Cmd/Ctrl for multiple)
3. Set servings multiplier (e.g., 2.0 for double)
4. Click "Generate Order"
5. Review consolidated order
6. Export to JSON if needed

### 4. Analyze Purchases
1. Click "📈 Analysis" tab
2. Click "Run Analysis"
3. View top purchased products
4. Get recommendations
5. Export report

---

## 📊 What the App Shows

### Dashboard Statistics:
- Total Recipes: 53
- Vendor Products: 659
- Product Matches: 11
- Orders Generated: Track your orders

### Sample Data Loaded:
```
✓ 53 recipes from Lariat Recipe Book.docx
✓ 432 Sysco products
✓ 227 Shamrock products
✓ 139 ingredients with pricing
✓ 432 products with purchase history
```

---

## 🔧 Menu Options

### File Menu:
- **Load Order Guide** - Import new vendor data
- **Export Data** - Save current data
- **Exit** - Close application

### Tools Menu:
- **Generate Price Report** - Create price_comparison_report.txt
- **Run Purchase Analysis** - Analyze buying patterns
- **Refresh Data** - Reload all data

### Help Menu:
- **About** - Application info
- **Quick Start** - Usage guide

---

## 📁 File Structure

```
lariat-bible/desktop_app/
├── START_LARIAT_BIBLE.command  ← Double-click to launch!
├── lariat_bible_app.py          ← Main application
├── features/
│   ├── recipe_manager.py        ← Recipe system
│   └── vendor_manager.py        ← Vendor system
├── data_importers/
│   ├── order_guide_parser.py    ← Parser for Sysco/Shamrock
│   └── docx_importer.py         ← Recipe importer
└── data/
    └── vendor_order_guides/
        └── COMBO PH NO TOUCH (1).xlsx
```

---

## 💾 Data Sources

### Recipes:
- **Source**: `/Users/seanburdges/Desktop/LARIAT/Lariat Recipe Book.docx`
- **Format**: Word document
- **Count**: 53 recipes
- **Categories**: Soup/Base, Sauce, Salsa, Seasoning, Brine, etc.

### Vendor Products:
- **Source**: `data/vendor_order_guides/COMBO PH NO TOUCH (1).xlsx`
- **Vendors**: Sysco, Shamrock
- **Total**: 659 products
- **Data**: Codes, prices, brands, pack sizes, purchase history

### Ingredients:
- **Source**: Excel ingredient database
- **Count**: 139 ingredients
- **Data**: Pricing, conversions, costs

---

## 🎨 UI Overview

### Tab 1: Dashboard
```
┌─────────────────────────────────────────────┐
│  Lariat Bible Restaurant Management System  │
├─────────────────────────────────────────────┤
│  [Recipes: 53]  [Products: 659]            │
│  [Matches: 11]  [Orders: 0]                │
│                                             │
│  Quick Actions:                             │
│  [View Recipes] [Manage Vendors]           │
│  [Generate Order] [View Analysis]          │
│                                             │
│  Status Log:                                │
│  System initialized. Loading data...       │
│  ✓ Loaded 53 recipes                       │
│  ✓ Loaded 659 vendor products              │
└─────────────────────────────────────────────┘
```

### Tab 2: Recipes
```
┌─────────────────────────────────────────────┐
│  Recipe List          │  Recipe Details     │
│  ──────────────────   │  ─────────────────  │
│  □ Soup Base          │  Name: Soup Base    │
│  □ Mesa Melt          │  Category: Soup/Base│
│  □ Buttermilk Brine   │                     │
│  □ Lariat rub         │  Ingredients:       │
│  □ Queso/Mac Sauce    │  • 8 tbsp butter    │
│  ...                  │  • 3 onions         │
└─────────────────────────────────────────────┘
```

### Tab 3: Vendors
```
┌─────────────────────────────────────────────┐
│  [Vendor: All ▼] [Compare Prices] [Match]  │
├─────────────────────────────────────────────┤
│  Product List         │  Details            │
│  ──────────────────   │  ─────────────────  │
│  Sysco | Code | Desc  │  Price Comparison:  │
│  Sham  | Code | Price │  Results...         │
└─────────────────────────────────────────────┘
```

### Tab 4: Orders
```
┌─────────────────────────────────────────────┐
│  Select Recipes:                            │
│  ☑ Soup Base                                │
│  ☑ Mesa Melt                                │
│  ☐ Buttermilk Brine                         │
│                                             │
│  Servings: [2.0] [Generate Order]          │
├─────────────────────────────────────────────┤
│  Generated Order:                           │
│  Total Items: 20                            │
│  Sysco: 6 items                             │
│  Shamrock: 14 items                         │
│  Total: $1,243.92                           │
└─────────────────────────────────────────────┘
```

### Tab 5: Analysis
```
┌─────────────────────────────────────────────┐
│  [Run Analysis] [Export Report]            │
├─────────────────────────────────────────────┤
│  Top 20 Most Purchased Products:            │
│  1. SMITHFIELD - 81.9 units                │
│  2. WHITE MARBLE FARMS - 80.5 units        │
│  3. BUTCHERS BLOCK - 78.8 units            │
│  ...                                        │
│                                             │
│  Recommendations:                           │
│  • Consider consolidating orders           │
└─────────────────────────────────────────────┘
```

---

## 🔍 Advanced Features

### Price Comparison
- Fuzzy matching algorithm
- 11 high-confidence product matches
- Identifies cheaper vendor for each product
- Generates detailed reports

### Product Matching
- Multi-field scoring (description, brand, pack)
- 51-64% accuracy rates
- Suggests vendor alternatives

### Order Consolidation
- Combines ingredients from multiple recipes
- Groups by vendor (Sysco/Shamrock)
- Calculates quantities based on servings
- Tracks which recipes need each item

### Purchase Analysis
- Identifies top 20 purchased products
- Analyzes historical trends
- Generates actionable recommendations
- Supports data-driven decisions

---

## 💡 Tips & Tricks

### Recipe Search:
- Type in search box to filter recipes
- Select category from dropdown
- Double-click recipe for full details

### Vendor Comparison:
- Use "Match Products" to find same items across vendors
- Use "Compare Prices" to identify savings
- Export reports for meetings/analysis

### Order Generation:
- Select multiple recipes with Cmd/Ctrl+Click
- Adjust servings multiplier for events (e.g., 3.0 for triple)
- Review order before finalizing
- Export to JSON for vendor submission

### Analysis:
- Run analysis monthly to track trends
- Compare current vs historical purchases
- Use recommendations for optimization

---

## 🐛 Troubleshooting

### App Won't Launch:
```bash
# Check Python version
python3 --version  # Should be 3.7+

# Install dependencies
pip3 install openpyxl python-docx

# Run manually
cd /Users/seanburdges/lariat-bible/desktop_app
python3 lariat_bible_app.py
```

### Data Not Loading:
- Check that recipe file exists: `/Users/seanburdges/Desktop/LARIAT/Lariat Recipe Book.docx`
- Check that order guide exists: `data/vendor_order_guides/COMBO PH NO TOUCH (1).xlsx`
- Use File → Load Order Guide to reload

### Order Generation Issues:
- Ensure recipes are selected (check boxes)
- Verify servings multiplier is valid number
- Check that vendor data is loaded

---

## 📝 Export Formats

### Order Export (JSON):
```json
{
  "recipe_count": 3,
  "total_items": 20,
  "sysco_order": {...},
  "shamrock_order": {...},
  "estimated_total": 1243.92
}
```

### Analysis Export (TXT):
```
PURCHASE HISTORY ANALYSIS
==========================
Products with History: 432
Top Products: 20
...
```

---

## 🎓 For More Information

See comprehensive documentation:
- `VENDOR_MANAGEMENT_COMPLETE.md` - Full system docs
- `ORDER_GUIDE_TEMPLATES.md` - Template formats
- `QUICK_START_VENDOR_SYSTEM.md` - Quick reference
- `DOCX_INTEGRATION_COMPLETE.md` - Recipe system

---

## ✅ System Status

**Application**: ✅ Working
**Data Loaded**: ✅ Complete
**All Features**: ✅ Functional
**Production Ready**: ✅ Yes

---

## 🏆 Features Summary

| Feature | Status | Count |
|---------|--------|-------|
| Recipes | ✅ | 53 |
| Vendor Products | ✅ | 659 |
| Product Matches | ✅ | 11 |
| Price Comparisons | ✅ | Working |
| Order Generation | ✅ | Working |
| Purchase Analysis | ✅ | 432 histories |
| Export Functions | ✅ | JSON, TXT |

---

**Version**: 1.0
**Created**: November 2025
**Status**: Production Ready

🎉 **ENJOY YOUR LARIAT BIBLE APP!** 🎉
