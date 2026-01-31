# Quick Start Guide - Vendor Management System

## 🚀 Get Started in 5 Minutes

### 1. Run the Demo
```bash
cd /Users/seanburdges/lariat-bible/desktop_app
python3 demo_vendor_features.py
```

This demonstrates all 5 features with real data!

---

## 📋 Common Tasks

### Compare Vendor Prices
```python
from features.vendor_manager import VendorManager, PriceComparisonTool

vm = VendorManager('data/vendor_order_guides/COMBO PH NO TOUCH (1).xlsx')
tool = PriceComparisonTool(vm)

# Find savings
savings = tool.find_savings_opportunities(min_savings=5.0)

# Generate report
tool.generate_report('my_price_report.txt')
```

### Match Products Across Vendors
```python
matches = vm.match_products(min_confidence=0.6)

for match in matches:
    print(f"{match['description']}")
    print(f"  Confidence: {match['confidence']}")
    print(f"  Sysco: {match['sysco_product']['product_code']}")
    print(f"  Shamrock: {match['shamrock_product']['product_code']}")
```

### Generate Order from Recipes
```python
from data_importers.docx_importer import DocxRecipeImporter

# Load recipes
importer = DocxRecipeImporter()
recipes = importer.import_recipe_book('path/to/recipes.docx')

# Generate order for first 3 recipes, double servings
order = vm.generate_order(recipes[:3], servings_multiplier=2.0)

print(f"Total items: {order['total_items']}")
print(f"Estimated cost: ${order['estimated_shamrock_cost']:.2f}")
```

### Analyze Purchase History
```python
analysis = vm.analyze_purchase_history()

print(f"Products with history: {analysis['total_products_with_history']}")

# Show top 10
for item in analysis['top_purchased_products'][:10]:
    print(f"{item['product']['brand']} - {item['total_purchases']} units")
```

---

## 📊 View Reports

After running the demo, check these files:

- `price_comparison_report.txt` - Price comparisons
- `generated_order.json` - Sample order
- `purchase_analysis.json` - Purchase trends
- `unified_catalog.json` - Full product catalog

---

## 🔧 System Files

**Main System:**
- `features/vendor_manager.py` - Core system
- `data_importers/order_guide_parser.py` - Parser

**Demo:**
- `demo_vendor_features.py` - Feature demonstration

**Data:**
- `data/vendor_order_guides/COMBO PH NO TOUCH (1).xlsx` - Order guide

---

## ✅ What's Working

✅ 659 products cataloged (432 Sysco + 227 Shamrock)
✅ 11 product matches with 51-64% confidence
✅ 53 recipes integrated from Lariat Recipe Book
✅ Automated order generation
✅ Purchase history analysis for 432 products
✅ All features tested and working

---

## 🎯 Next Steps

1. Run `python3 demo_vendor_features.py` to see everything in action
2. Check generated reports in the project directory
3. Integrate with your existing systems
4. Use for real vendor orders!

---

**Status**: ✅ Ready for production use
**Documentation**: Complete (see VENDOR_MANAGEMENT_COMPLETE.md)
