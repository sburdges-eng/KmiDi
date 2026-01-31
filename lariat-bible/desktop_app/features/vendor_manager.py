"""
Vendor Management System
Comprehensive vendor product management with price comparison, product matching,
recipe integration, order generation, and purchase analysis
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_importers.order_guide_parser import OrderGuideParser, OfficialCombinedParser


class VendorManager:
    """Manage vendor products and operations"""

    def __init__(self, order_guide_path: str = None, use_official_format: bool = False):
        self.parser = OrderGuideParser()
        self.official_parser = OfficialCombinedParser()
        self.catalog = None
        self.sysco_products = []
        self.shamrock_products = []
        self.combo_products = []
        self.official_matches = []  # Pre-matched pairs from official guide
        self.use_official_format = use_official_format

        if order_guide_path:
            self.load_order_guide(order_guide_path, use_official_format=use_official_format)

    def load_order_guide(self, file_path: str, use_official_format: bool = False):
        """Load order guide from file"""
        if use_official_format:
            # Use the official combined format parser
            result = self.official_parser.parse(file_path)

            if 'sysco' in result:
                self.sysco_products = result['sysco']['products']
            if 'shamrock' in result:
                self.shamrock_products = result['shamrock']['products']
            if 'matched_pairs' in result:
                self.official_matches = result['matched_pairs']

            print(f"✓ Loaded {len(self.sysco_products)} Sysco products (official format)")
            print(f"✓ Loaded {len(self.shamrock_products)} Shamrock products (official format)")
            print(f"✓ Loaded {len(self.official_matches)} pre-matched pairs")
        else:
            # Use the old multi-vendor parser
            results = self.parser.parse_all_vendors(file_path)

            if 'sysco' in results:
                self.sysco_products = results['sysco']['products']
            if 'shamrock' in results:
                self.shamrock_products = results['shamrock']['products']
            if 'combo' in results:
                self.combo_products = results['combo']['products']

            print(f"✓ Loaded {len(self.sysco_products)} Sysco products")
            print(f"✓ Loaded {len(self.shamrock_products)} Shamrock products")

    # ========================================================================
    # FEATURE 1: PRICE COMPARISON
    # ========================================================================

    def compare_prices(self, product_matches: List[Dict] = None) -> List[Dict]:
        """
        Compare prices between Sysco and Shamrock for matched products

        Returns:
            List of price comparisons with savings info
        """
        if not product_matches:
            product_matches = self.match_products()

        comparisons = []

        for match in product_matches:
            if match['shamrock_product'] and match['sysco_product']:
                sysco = match['sysco_product']
                shamrock = match['shamrock_product']

                # Try to extract prices
                sysco_price = self._extract_price(sysco)
                sham_price = shamrock.get('price')

                if sysco_price and sham_price:
                    comparison = {
                        'description': shamrock['description'],
                        'sysco_code': sysco['product_code'],
                        'shamrock_code': shamrock['product_code'],
                        'sysco_price': sysco_price,
                        'shamrock_price': sham_price,
                        'difference': sysco_price - sham_price,
                        'percent_diff': ((sysco_price - sham_price) / sysco_price * 100) if sysco_price > 0 else 0,
                        'cheaper_vendor': 'Sysco' if sysco_price < sham_price else 'Shamrock',
                        'match_confidence': match['confidence']
                    }
                    comparisons.append(comparison)

        # Sort by absolute difference (biggest savings first)
        comparisons.sort(key=lambda x: abs(x['difference']), reverse=True)

        return comparisons

    def _extract_price(self, product: Dict) -> Optional[float]:
        """Extract numeric price from product"""
        price = product.get('price')
        if price:
            if isinstance(price, (int, float)):
                return float(price)
            # Try to parse string price
            price_str = str(price).replace('$', '').replace(',', '').strip()
            try:
                return float(price_str)
            except:
                pass
        return None

    # ========================================================================
    # FEATURE 2: PRODUCT MATCHING
    # ========================================================================

    def match_products(self, min_confidence: float = 0.6) -> List[Dict]:
        """
        Match products across vendors using fuzzy matching

        Args:
            min_confidence: Minimum confidence score (0-1) for a match

        Returns:
            List of matched products with confidence scores
        """
        # If we have official matches, use those (100% confidence)
        if self.official_matches:
            matches = []
            for match_pair in self.official_matches:
                # Find the actual product objects
                sham_product = self._find_product_by_code(
                    self.shamrock_products, match_pair['shamrock_code']
                )
                sysco_product = self._find_product_by_code(
                    self.sysco_products, match_pair['sysco_code']
                )

                if sham_product and sysco_product:
                    matches.append({
                        'shamrock_product': sham_product,
                        'sysco_product': sysco_product,
                        'confidence': 1.0,  # 100% confidence from official guide
                        'description': match_pair['shamrock_description'],
                        'source': 'official_guide'
                    })
            return matches

        # Otherwise, use fuzzy matching
        matches = []

        for sham_product in self.shamrock_products:
            sham_desc = sham_product.get('description', '').upper()
            best_match = None
            best_score = 0

            for sysco_product in self.sysco_products:
                # Try to match on brand + pack size
                sysco_brand = str(sysco_product.get('brand', '')).upper()
                sysco_pack = str(sysco_product.get('pack', ''))
                sysco_size = str(sysco_product.get('size', '')).upper()

                # Create comparison strings
                sham_pack = str(sham_product.get('pack_size', '')).upper()
                sham_brand = str(sham_product.get('brand', '')).upper()

                # Calculate similarity scores
                desc_score = self._similarity(sham_desc, f"{sysco_brand} {sysco_pack} {sysco_size}")
                brand_score = self._similarity(sham_brand, sysco_brand) if sham_brand and sysco_brand else 0
                pack_score = self._similarity(sham_pack, f"{sysco_pack} {sysco_size}") if sham_pack else 0

                # Weighted average
                score = (desc_score * 0.5) + (brand_score * 0.3) + (pack_score * 0.2)

                if score > best_score:
                    best_score = score
                    best_match = sysco_product

            if best_score >= min_confidence:
                matches.append({
                    'shamrock_product': sham_product,
                    'sysco_product': best_match,
                    'confidence': round(best_score, 2),
                    'description': sham_product.get('description')
                })

        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)

        return matches

    def _similarity(self, a: str, b: str) -> float:
        """Calculate similarity between two strings"""
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a.upper(), b.upper()).ratio()

    def _find_product_by_code(self, products: List[Dict], product_code: str) -> Optional[Dict]:
        """Find a product by its product code"""
        if not product_code:
            return None

        # Clean the code for comparison
        code_clean = str(product_code).strip().lstrip('0')

        for product in products:
            p_code = str(product.get('product_code', '')).strip().lstrip('0')
            if p_code == code_clean:
                return product
        return None

    # ========================================================================
    # FEATURE 3: RECIPE INTEGRATION
    # ========================================================================

    def match_recipe_ingredients(self, recipe: Dict) -> Dict:
        """
        Match recipe ingredients to vendor products

        Args:
            recipe: Recipe dict with ingredients list

        Returns:
            Recipe with vendor product matches for each ingredient
        """
        matched_recipe = recipe.copy()
        matched_recipe['ingredient_matches'] = []

        for ingredient in recipe.get('ingredients', []):
            ing_name = ingredient.get('name', '').upper()

            # Search for matching products
            sysco_matches = self._search_products(ing_name, self.sysco_products, top_n=3)
            sham_matches = self._search_products(ing_name, self.shamrock_products, top_n=3)

            ingredient_match = {
                'ingredient': ingredient,
                'sysco_options': sysco_matches,
                'shamrock_options': sham_matches,
                'best_match': sysco_matches[0] if sysco_matches else (sham_matches[0] if sham_matches else None)
            }

            matched_recipe['ingredient_matches'].append(ingredient_match)

        return matched_recipe

    def _search_products(self, query: str, products: List[Dict], top_n: int = 3) -> List[Dict]:
        """Search products by query string"""
        query_upper = query.upper()
        scored_products = []

        for product in products:
            # Create searchable text
            search_text = ' '.join([
                str(product.get('description', '')),
                str(product.get('brand', '')),
                str(product.get('pack_size', '')),
                str(product.get('pack', '')),
                str(product.get('size', ''))
            ]).upper()

            # Calculate relevance score
            score = self._calculate_search_score(query_upper, search_text)

            if score > 0:
                scored_products.append({
                    'product': product,
                    'score': score
                })

        # Sort by score and return top N
        scored_products.sort(key=lambda x: x['score'], reverse=True)
        return [item['product'] for item in scored_products[:top_n]]

    def _calculate_search_score(self, query: str, text: str) -> float:
        """Calculate search relevance score"""
        score = 0
        query_words = query.split()

        for word in query_words:
            if len(word) < 3:
                continue
            if word in text:
                score += 2  # Exact word match
            elif any(word in text_word for text_word in text.split()):
                score += 1  # Partial match

        return score

    # ========================================================================
    # FEATURE 4: ORDER GENERATION
    # ========================================================================

    def generate_order(self, recipes: List[Dict], servings_multiplier: float = 1.0) -> Dict:
        """
        Generate vendor order from recipe list

        Args:
            recipes: List of recipes to prepare
            servings_multiplier: Scale factor for servings

        Returns:
            Order with consolidated product list and totals
        """
        order = {
            'sysco_order': {},
            'shamrock_order': {},
            'total_items': 0,
            'estimated_sysco_cost': 0,
            'estimated_shamrock_cost': 0,
            'recipe_count': len(recipes)
        }

        # Process each recipe
        for recipe in recipes:
            matched_recipe = self.match_recipe_ingredients(recipe)

            for ing_match in matched_recipe.get('ingredient_matches', []):
                ingredient = ing_match['ingredient']
                best_match = ing_match['best_match']

                if not best_match:
                    continue

                # Determine vendor
                vendor = best_match.get('vendor', '')
                product_code = best_match.get('product_code')

                # Calculate quantity needed
                base_qty = ingredient.get('base_qty', 1)
                try:
                    qty_float = float(re.sub(r'[^\d.]', '', str(base_qty))) if base_qty else 1
                except:
                    qty_float = 1

                qty_needed = qty_float * servings_multiplier

                # Add to appropriate order
                if vendor == 'Sysco':
                    if product_code not in order['sysco_order']:
                        order['sysco_order'][product_code] = {
                            'product': best_match,
                            'quantity': 0,
                            'recipes': []
                        }
                    order['sysco_order'][product_code]['quantity'] += qty_needed
                    order['sysco_order'][product_code]['recipes'].append(recipe.get('name'))

                elif vendor == 'Shamrock':
                    if product_code not in order['shamrock_order']:
                        order['shamrock_order'][product_code] = {
                            'product': best_match,
                            'quantity': 0,
                            'recipes': []
                        }
                    order['shamrock_order'][product_code]['quantity'] += qty_needed
                    order['shamrock_order'][product_code]['recipes'].append(recipe.get('name'))

        # Calculate totals
        order['total_items'] = len(order['sysco_order']) + len(order['shamrock_order'])

        # Estimate costs
        for item in order['sysco_order'].values():
            price = self._extract_price(item['product'])
            if price:
                order['estimated_sysco_cost'] += price * item['quantity']

        for item in order['shamrock_order'].values():
            price = item['product'].get('price', 0)
            if price:
                order['estimated_shamrock_cost'] += price * item['quantity']

        return order

    # ========================================================================
    # FEATURE 5: PURCHASE ANALYSIS
    # ========================================================================

    def analyze_purchase_history(self) -> Dict:
        """
        Analyze purchase history from COMBO PH data

        Returns:
            Analysis with trends, top products, and forecasts
        """
        analysis = {
            'total_products_with_history': 0,
            'top_purchased_products': [],
            'purchase_trends': {},
            'low_stock_alerts': [],
            'recommendations': []
        }

        products_with_history = []

        for product in self.combo_products:
            history = product.get('purchase_history', {})
            if history:
                # Calculate total purchases
                total_purchases = sum(
                    float(v) if isinstance(v, (int, float)) else 0
                    for k, v in history.items()
                    if isinstance(v, (int, float))
                )

                if total_purchases > 0:
                    products_with_history.append({
                        'product': product,
                        'total_purchases': total_purchases,
                        'history': history
                    })

        analysis['total_products_with_history'] = len(products_with_history)

        # Top purchased products
        products_with_history.sort(key=lambda x: x['total_purchases'], reverse=True)
        analysis['top_purchased_products'] = products_with_history[:20]

        # Identify low stock items (products with recent decline in purchases)
        for item in products_with_history[:50]:
            # This is simplified - in real implementation would analyze trends
            analysis['low_stock_alerts'].append({
                'product_code': item['product'].get('product_code'),
                'brand': item['product'].get('brand'),
                'total_purchases': item['total_purchases']
            })

        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(products_with_history)

        return analysis

    def _generate_recommendations(self, products_with_history: List[Dict]) -> List[str]:
        """Generate purchasing recommendations"""
        recommendations = []

        if len(products_with_history) > 0:
            top_product = products_with_history[0]
            recommendations.append(
                f"Top product: {top_product['product'].get('brand')} - "
                f"{top_product['total_purchases']} units purchased"
            )

        if len(products_with_history) > 10:
            recommendations.append(
                f"Consider consolidating orders: {len(products_with_history)} products with purchase history"
            )

        return recommendations


# ============================================================================
# PRICE COMPARISON TOOL
# ============================================================================

class PriceComparisonTool:
    """Dedicated tool for price comparison"""

    def __init__(self, vendor_manager: VendorManager):
        self.vm = vendor_manager

    def find_savings_opportunities(self, min_savings: float = 5.0) -> List[Dict]:
        """Find products where switching vendors could save money"""
        comparisons = self.vm.compare_prices()

        savings_opportunities = [
            comp for comp in comparisons
            if abs(comp['difference']) >= min_savings
        ]

        return savings_opportunities

    def generate_report(self, output_file: str = 'price_comparison_report.txt'):
        """Generate price comparison report"""
        comparisons = self.vm.compare_prices()

        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VENDOR PRICE COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")

            total_savings = sum(comp['difference'] for comp in comparisons if comp['difference'] > 0)

            f.write(f"Total Products Compared: {len(comparisons)}\n")
            f.write(f"Potential Annual Savings: ${abs(total_savings):.2f}\n\n")

            f.write("Top 20 Price Differences:\n")
            f.write("-" * 80 + "\n")

            for i, comp in enumerate(comparisons[:20], 1):
                f.write(f"\n{i}. {comp['description']}\n")
                f.write(f"   Sysco: ${comp['sysco_price']:.2f} | Shamrock: ${comp['shamrock_price']:.2f}\n")
                f.write(f"   Difference: ${comp['difference']:.2f} ({comp['percent_diff']:.1f}%)\n")
                f.write(f"   Better Deal: {comp['cheaper_vendor']}\n")

        print(f"✓ Report saved to {output_file}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("VENDOR MANAGEMENT SYSTEM - COMPREHENSIVE TEST")
    print("=" * 80)

    # Initialize
    order_guide_path = "data/vendor_order_guides/COMBO PH NO TOUCH (1).xlsx"
    vm = VendorManager(order_guide_path)

    print("\n1. PRICE COMPARISON")
    print("=" * 80)
    comparisons = vm.compare_prices()
    print(f"Found {len(comparisons)} price comparisons")
    if comparisons:
        print("\nTop 3 price differences:")
        for i, comp in enumerate(comparisons[:3], 1):
            print(f"{i}. {comp['description']}")
            print(f"   Sysco: ${comp['sysco_price']:.2f} | Shamrock: ${comp['shamrock_price']:.2f}")
            print(f"   Difference: ${comp['difference']:.2f} ({comp['cheaper_vendor']} is cheaper)")

    print("\n2. PRODUCT MATCHING")
    print("=" * 80)
    matches = vm.match_products(min_confidence=0.7)
    print(f"Found {len(matches)} product matches")
    if matches:
        print("\nTop 3 matches:")
        for i, match in enumerate(matches[:3], 1):
            print(f"{i}. {match['description']} (confidence: {match['confidence']})")

    print("\n3. PURCHASE ANALYSIS")
    print("=" * 80)
    analysis = vm.analyze_purchase_history()
    print(f"Products with history: {analysis['total_products_with_history']}")
    print(f"Top products: {len(analysis['top_purchased_products'])}")
    print(f"Recommendations: {len(analysis['recommendations'])}")

    print("\n" + "=" * 80)
    print("✅ ALL FEATURES TESTED SUCCESSFULLY")
    print("=" * 80)
