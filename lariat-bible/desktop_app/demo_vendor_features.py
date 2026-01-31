"""
Vendor Management Features Demo
Demonstrates all 5 features with real data
"""

from features.vendor_manager import VendorManager, PriceComparisonTool
from data_importers.docx_importer import DocxRecipeImporter
import json

print("=" * 80)
print("LARIAT BIBLE - COMPREHENSIVE VENDOR MANAGEMENT DEMO")
print("=" * 80)

# Initialize
order_guide_path = "data/vendor_order_guides/COMBO PH NO TOUCH (1).xlsx"
vm = VendorManager(order_guide_path)

# ============================================================================
# FEATURE 1: PRICE COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE 1: PRICE COMPARISON")
print("=" * 80)

price_tool = PriceComparisonTool(vm)

print(f"\n✓ Sysco Products: {len(vm.sysco_products)}")
print(f"✓ Shamrock Products: {len(vm.shamrock_products)}")

# Generate price comparison report
price_tool.generate_report('price_comparison_report.txt')

print("\nSample Sysco Products (First 5):")
for i, product in enumerate(vm.sysco_products[:5], 1):
    print(f"{i}. {product.get('brand')} - {product.get('pack')} {product.get('size')}")
    print(f"   Code: {product.get('product_code')}")

print("\nSample Shamrock Products (First 5):")
for i, product in enumerate(vm.shamrock_products[:5], 1):
    print(f"{i}. {product.get('description')}")
    print(f"   Brand: {product.get('brand')} | Pack: {product.get('pack_size')}")
    print(f"   Price: ${product.get('price', 0):.2f} per {product.get('unit')}")

# ============================================================================
# FEATURE 2: PRODUCT MATCHING
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE 2: PRODUCT MATCHING (Fuzzy Search)")
print("=" * 80)

matches = vm.match_products(min_confidence=0.5)
print(f"\n✓ Found {len(matches)} potential product matches")

if matches:
    print("\nTop 10 Product Matches:")
    for i, match in enumerate(matches[:10], 1):
        print(f"\n{i}. {match['description']} (Confidence: {match['confidence']})")
        print(f"   Shamrock Code: {match['shamrock_product']['product_code']}")
        if match['sysco_product']:
            print(f"   Sysco Code: {match['sysco_product']['product_code']}")
            print(f"   Sysco Brand: {match['sysco_product'].get('brand')}")

# ============================================================================
# FEATURE 3: RECIPE INTEGRATION
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE 3: RECIPE INTEGRATION")
print("=" * 80)

# Load recipes from Word document
docx_path = "/Users/seanburdges/Desktop/LARIAT/Lariat Recipe Book.docx"
recipe_importer = DocxRecipeImporter()
recipes = recipe_importer.import_recipe_book(docx_path)

print(f"\n✓ Loaded {len(recipes)} recipes")

# Match first recipe to vendor products
if recipes:
    test_recipe = recipes[0]
    print(f"\nTesting Recipe: {test_recipe['name']}")
    print(f"Ingredients: {len(test_recipe['ingredients'])}")

    matched_recipe = vm.match_recipe_ingredients(test_recipe)

    print(f"\nIngredient Matching Results:")
    for ing_match in matched_recipe['ingredient_matches'][:5]:
        ingredient = ing_match['ingredient']
        print(f"\n• {ingredient['name']}")

        if ing_match['sysco_options']:
            print(f"  Sysco Options: {len(ing_match['sysco_options'])}")
            best = ing_match['sysco_options'][0]
            print(f"    Best: {best.get('brand')} - {best.get('pack')} {best.get('size')}")

        if ing_match['shamrock_options']:
            print(f"  Shamrock Options: {len(ing_match['shamrock_options'])}")
            best = ing_match['shamrock_options'][0]
            print(f"    Best: {best.get('description')} - ${best.get('price', 0):.2f}")

# ============================================================================
# FEATURE 4: ORDER GENERATION
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE 4: AUTOMATED ORDER GENERATION")
print("=" * 80)

# Generate order for first 3 recipes
order = vm.generate_order(recipes[:3], servings_multiplier=2.0)

print(f"\n✓ Order generated for {order['recipe_count']} recipes (2x servings)")
print(f"\nOrder Summary:")
print(f"  Total Items: {order['total_items']}")
print(f"  Sysco Items: {len(order['sysco_order'])}")
print(f"  Shamrock Items: {len(order['shamrock_order'])}")
print(f"  Estimated Sysco Cost: ${order['estimated_sysco_cost']:.2f}")
print(f"  Estimated Shamrock Cost: ${order['estimated_shamrock_cost']:.2f}")
print(f"  Total Estimated Cost: ${order['estimated_sysco_cost'] + order['estimated_shamrock_cost']:.2f}")

if order['sysco_order']:
    print(f"\nSample Sysco Order Items (First 3):")
    for i, (code, item) in enumerate(list(order['sysco_order'].items())[:3], 1):
        print(f"{i}. {item['product'].get('brand')}")
        print(f"   Quantity: {item['quantity']:.1f}")
        print(f"   For recipes: {', '.join(set(item['recipes']))}")

if order['shamrock_order']:
    print(f"\nSample Shamrock Order Items (First 3):")
    for i, (code, item) in enumerate(list(order['shamrock_order'].items())[:3], 1):
        print(f"{i}. {item['product'].get('description')}")
        print(f"   Quantity: {item['quantity']:.1f}")
        print(f"   Price: ${item['product'].get('price', 0):.2f} per {item['product'].get('unit')}")
        print(f"   For recipes: {', '.join(set(item['recipes']))}")

# Save order to JSON
with open('generated_order.json', 'w') as f:
    # Convert to serializable format
    order_export = {
        'recipe_count': order['recipe_count'],
        'total_items': order['total_items'],
        'estimated_sysco_cost': order['estimated_sysco_cost'],
        'estimated_shamrock_cost': order['estimated_shamrock_cost'],
        'sysco_items': len(order['sysco_order']),
        'shamrock_items': len(order['shamrock_order'])
    }
    json.dump(order_export, f, indent=2)

print("\n✓ Order saved to generated_order.json")

# ============================================================================
# FEATURE 5: PURCHASE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE 5: PURCHASE HISTORY ANALYSIS")
print("=" * 80)

analysis = vm.analyze_purchase_history()

print(f"\n✓ Products with Purchase History: {analysis['total_products_with_history']}")
print(f"✓ Top Products Analyzed: {len(analysis['top_purchased_products'])}")

if analysis['top_purchased_products']:
    print(f"\nTop 10 Most Purchased Products:")
    for i, item in enumerate(analysis['top_purchased_products'][:10], 1):
        product = item['product']
        print(f"{i}. {product.get('brand')} - {product.get('pack')} {product.get('size')}")
        print(f"   Total Purchases: {item['total_purchases']:.1f} units")

if analysis['recommendations']:
    print(f"\nRecommendations:")
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"{i}. {rec}")

# Save analysis
with open('purchase_analysis.json', 'w') as f:
    analysis_export = {
        'total_products_with_history': analysis['total_products_with_history'],
        'top_products_count': len(analysis['top_purchased_products']),
        'recommendations': analysis['recommendations']
    }
    json.dump(analysis_export, f, indent=2)

print("\n✓ Analysis saved to purchase_analysis.json")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("COMPREHENSIVE VENDOR MANAGEMENT - COMPLETE")
print("=" * 80)

print("\n✅ ALL 5 FEATURES DEMONSTRATED:")
print("  1. ✓ Price Comparison - Report generated")
print("  2. ✓ Product Matching - Fuzzy matching across vendors")
print("  3. ✓ Recipe Integration - Ingredients matched to products")
print("  4. ✓ Order Generation - Automated orders from recipes")
print("  5. ✓ Purchase Analysis - Historical data insights")

print("\nFiles Created:")
print("  • price_comparison_report.txt")
print("  • generated_order.json")
print("  • purchase_analysis.json")

print("\nSystem Ready for Production Use!")
print("=" * 80)
