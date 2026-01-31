"""
Lariat Bible - Main Data Integration Module
Coordinates all modules for comprehensive restaurant management
"""

import json
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

# Import all modules
from modules.menu.menu_item import MenuItem
from modules.recipes.recipe import Recipe, Ingredient, RecipeIngredient
from modules.order_guides.order_guide_manager import OrderGuideManager
from modules.equipment.equipment_manager import EquipmentManager, Equipment
from modules.vendor_analysis.comparator import VendorComparator


class LariatBible:
    """Main integration class for The Lariat restaurant management"""
    
    def __init__(self, data_path: str = "./data"):
        self.data_path = Path(data_path)
        
        # Initialize all managers
        self.menu_items: Dict[str, MenuItem] = {}
        self.recipes: Dict[str, Recipe] = {}
        self.ingredients: Dict[str, Ingredient] = {}
        self.order_guide_manager = OrderGuideManager()
        self.equipment_manager = EquipmentManager()
        self.vendor_comparator = VendorComparator()
        
        # Restaurant metrics
        self.monthly_catering_revenue = 28000
        self.monthly_restaurant_revenue = 20000
        self.target_catering_margin = 0.45
        self.target_restaurant_margin = 0.04
        
        # Load existing data if available
        self.load_data()
    
    def load_data(self):
        """Load all data from files if they exist"""
        # This would load from JSON files or database
        pass
    
    def save_data(self):
        """Save all data to files"""
        # This would save to JSON files or database
        pass
    
    # ========== INGREDIENT MANAGEMENT ==========
    
    def add_ingredient(self, ingredient: Ingredient) -> str:
        """Add or update an ingredient with vendor pricing"""
        # Calculate best price
        ingredient.calculate_best_price()
        
        # Store ingredient
        self.ingredients[ingredient.ingredient_id] = ingredient
        
        return f"Added ingredient: {ingredient.name} - Preferred vendor: {ingredient.preferred_vendor}"
    
    def update_ingredient_pricing(self, ingredient_id: str, vendor: str, new_price: float, 
                                 case_size: str = None) -> Dict:
        """Update pricing for an ingredient from a vendor"""
        if ingredient_id not in self.ingredients:
            return {'error': f'Ingredient {ingredient_id} not found'}
        
        ingredient = self.ingredients[ingredient_id]
        
        if vendor.upper() == "SYSCO":
            ingredient.sysco_price = new_price
            ingredient.sysco_last_updated = datetime.now()
            # Calculate unit price based on case size if provided
            if case_size:
                # Parse case size (e.g., "6/10#" = 60 lbs)
                # This is simplified - you'd want more robust parsing
                ingredient.sysco_unit_price = new_price / 60  # Example
        
        elif vendor.upper() == "SHAMROCK" or vendor.upper() == "SHAMROCK FOODS":
            ingredient.shamrock_price = new_price
            ingredient.shamrock_last_updated = datetime.now()
            if case_size:
                ingredient.shamrock_unit_price = new_price / 60  # Example
        
        # Recalculate best price
        result = ingredient.calculate_best_price()
        
        return {
            'ingredient': ingredient.name,
            'vendor_updated': vendor,
            'new_price': new_price,
            **result
        }
    
    # ========== RECIPE MANAGEMENT ==========
    
    def create_recipe_with_costing(self, recipe: Recipe) -> Dict:
        """Create a recipe and calculate its cost"""
        # Calculate costs
        total_cost = recipe.total_cost
        cost_per_portion = recipe.cost_per_portion
        
        # Store recipe
        self.recipes[recipe.recipe_id] = recipe
        
        # Get vendor impact analysis
        vendor_analysis = recipe.analyze_vendor_impact()
        
        return {
            'recipe': recipe.name,
            'total_cost': total_cost,
            'cost_per_portion': cost_per_portion,
            'yield': f"{recipe.yield_amount} {recipe.yield_unit}",
            'vendor_analysis': vendor_analysis,
            'suggested_menu_price': cost_per_portion / (1 - self.target_catering_margin)
        }
    
    def link_recipe_to_menu(self, recipe_id: str, menu_item_id: str) -> Dict:
        """Link a recipe to a menu item and update costing"""
        if recipe_id not in self.recipes:
            return {'error': f'Recipe {recipe_id} not found'}
        if menu_item_id not in self.menu_items:
            return {'error': f'Menu item {menu_item_id} not found'}
        
        recipe = self.recipes[recipe_id]
        menu_item = self.menu_items[menu_item_id]
        
        # Link recipe
        menu_item.recipe_id = recipe_id
        menu_item.food_cost = recipe.cost_per_portion
        
        # Update pricing recommendation
        pricing_analysis = menu_item.update_food_cost(recipe.cost_per_portion)
        
        return {
            'menu_item': menu_item.name,
            'recipe': recipe.name,
            'food_cost': recipe.cost_per_portion,
            **pricing_analysis
        }
    
    # ========== ORDER GUIDE COMPARISON ==========
    
    def import_order_guides(self, sysco_file: str = None, shamrock_file: str = None) -> Dict:
        """Import order guides from files (CSV, Excel, or JSON)"""
        results = {'sysco': 0, 'shamrock': 0}
        
        # This would read from actual files
        # For now, we'll use sample data
        
        # Sample SYSCO data
        if sysco_file:
            # Load from file
            pass
        else:
            # Use sample
            sample_sysco = [
                {
                    'item_code': 'SYS001',
                    'description': 'BEEF GROUND 80/20',
                    'pack_size': '10 LB',
                    'case_price': 45.99,
                    'unit_price': 4.599,
                    'unit': 'LB',
                    'category': 'MEAT'
                },
                {
                    'item_code': 'SYS002',
                    'description': 'CHICKEN BREAST BONELESS',
                    'pack_size': '40 LB',
                    'case_price': 89.99,
                    'unit_price': 2.249,
                    'unit': 'LB',
                    'category': 'POULTRY'
                }
            ]
            results['sysco'] = self.order_guide_manager.load_sysco_guide(sample_sysco)
        
        # Sample Shamrock data
        if shamrock_file:
            # Load from file
            pass
        else:
            sample_shamrock = [
                {
                    'item_code': 'SHA001',
                    'description': 'GROUND BEEF 80/20',
                    'pack_size': '10 LB',
                    'case_price': 32.50,  # 29.5% cheaper!
                    'unit_price': 3.25,
                    'unit': 'LB',
                    'category': 'MEAT'
                },
                {
                    'item_code': 'SHA002',
                    'description': 'CHICKEN BREAST BNLS',
                    'pack_size': '40 LB', 
                    'case_price': 63.50,  # Also cheaper!
                    'unit_price': 1.587,
                    'unit': 'LB',
                    'category': 'POULTRY'
                }
            ]
            results['shamrock'] = self.order_guide_manager.load_shamrock_guide(sample_shamrock)
        
        return results
    
    def run_comprehensive_comparison(self) -> Dict:
        """Run full vendor comparison and generate recommendations"""
        # Compare prices
        comparison_df = self.order_guide_manager.compare_prices()
        
        # Get category analysis
        category_analysis = self.order_guide_manager.get_category_analysis()
        
        # Generate recommendations
        recommendations = self.order_guide_manager.generate_purchase_recommendation()
        
        # Calculate impact on margins
        margin_impact = self.vendor_comparator.calculate_margin_impact(
            recommendations.get('estimated_monthly_savings', 4333)
        )
        
        return {
            'items_compared': len(comparison_df) if not comparison_df.empty else 0,
            'category_analysis': category_analysis,
            'recommendations': recommendations,
            'margin_impact': margin_impact,
            'export_status': self.order_guide_manager.export_comparison('data/vendor_comparison.xlsx')
        }
    
    # ========== MENU PRICING OPTIMIZATION ==========
    
    def optimize_menu_pricing(self) -> List[Dict]:
        """Analyze all menu items and suggest pricing changes"""
        optimization_results = []
        
        for menu_item in self.menu_items.values():
            if menu_item.food_cost > 0:
                current_margin = menu_item.margin
                target = self.target_catering_margin if menu_item.category == "Catering" else self.target_restaurant_margin
                
                if abs(current_margin - target) > 0.05:  # More than 5% off target
                    optimization_results.append({
                        'item': menu_item.name,
                        'category': menu_item.category,
                        'current_price': menu_item.menu_price,
                        'current_margin': current_margin,
                        'target_margin': target,
                        'suggested_price': menu_item.suggested_price,
                        'price_change': menu_item.suggested_price - menu_item.menu_price,
                        'action': 'INCREASE' if menu_item.suggested_price > menu_item.menu_price else 'DECREASE'
                    })
        
        # Sort by biggest opportunity
        optimization_results.sort(key=lambda x: abs(x['price_change']), reverse=True)
        
        return optimization_results
    
    # ========== REPORTING ==========
    
    def generate_executive_summary(self) -> str:
        """Generate comprehensive executive summary"""
        summary = []
        summary.append("=" * 60)
        summary.append("THE LARIAT BIBLE - EXECUTIVE SUMMARY")
        summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("=" * 60)
        
        # Financial Overview
        summary.append("\n📊 FINANCIAL OVERVIEW")
        summary.append("-" * 40)
        summary.append(f"Monthly Catering Revenue: ${self.monthly_catering_revenue:,}")
        summary.append(f"Monthly Restaurant Revenue: ${self.monthly_restaurant_revenue:,}")
        summary.append(f"Total Monthly Revenue: ${self.monthly_catering_revenue + self.monthly_restaurant_revenue:,}")
        
        # Vendor Analysis
        comparison = self.vendor_comparator.compare_vendors('Shamrock Foods', 'SYSCO')
        summary.append("\n💰 VENDOR OPTIMIZATION")
        summary.append("-" * 40)
        summary.append(f"Primary Vendor: Shamrock Foods")
        summary.append(f"Monthly Savings Potential: ${comparison:,.2f}")
        summary.append(f"Annual Savings Potential: ${comparison * 12:,.2f}")
        
        # Menu Analysis
        summary.append("\n🍽️ MENU ANALYSIS")
        summary.append("-" * 40)
        summary.append(f"Total Menu Items: {len(self.menu_items)}")
        summary.append(f"Total Recipes: {len(self.recipes)}")
        summary.append(f"Items Needing Price Adjustment: {len(self.optimize_menu_pricing())}")
        
        # Equipment Status
        equipment_summary = self.equipment_manager.get_equipment_summary()
        summary.append("\n🔧 EQUIPMENT STATUS")
        summary.append("-" * 40)
        summary.append(f"Total Equipment: {equipment_summary.get('total_equipment', 0)}")
        summary.append(f"Equipment Value: ${equipment_summary.get('depreciated_value', 0):,.2f}")
        summary.append(f"Maintenance Due: {equipment_summary.get('maintenance_due', 0)}")
        
        # Recommendations
        summary.append("\n📈 KEY RECOMMENDATIONS")
        summary.append("-" * 40)
        summary.append("1. Continue prioritizing Shamrock Foods for standard orders")
        summary.append("2. Review and adjust menu prices for items below target margins")
        summary.append("3. Schedule overdue equipment maintenance")
        summary.append("4. Focus on high-margin catering operations")
        
        return "\n".join(summary)
    
    def export_all_data(self, export_dir: str = "data/exports") -> Dict:
        """Export all data to files for backup/analysis"""
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        
        exports = {}
        
        # Export menu items
        menu_data = [item.to_dict() for item in self.menu_items.values()]
        menu_file = export_path / f"menu_items_{datetime.now().strftime('%Y%m%d')}.json"
        with open(menu_file, 'w') as f:
            json.dump(menu_data, f, indent=2, default=str)
        exports['menu'] = str(menu_file)
        
        # Export recipes
        recipe_data = []
        for recipe in self.recipes.values():
            recipe_dict = {
                'recipe_id': recipe.recipe_id,
                'name': recipe.name,
                'category': recipe.category,
                'total_cost': recipe.total_cost,
                'cost_per_portion': recipe.cost_per_portion,
                'ingredients': len(recipe.ingredients)
            }
            recipe_data.append(recipe_dict)
        
        recipe_file = export_path / f"recipes_{datetime.now().strftime('%Y%m%d')}.json"
        with open(recipe_file, 'w') as f:
            json.dump(recipe_data, f, indent=2, default=str)
        exports['recipes'] = str(recipe_file)
        
        # Export vendor comparison
        comparison_file = export_path / f"vendor_comparison_{datetime.now().strftime('%Y%m%d')}.xlsx"
        exports['vendor_comparison'] = self.order_guide_manager.export_comparison(str(comparison_file))
        
        # Export executive summary
        summary_file = export_path / f"executive_summary_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(summary_file, 'w') as f:
            f.write(self.generate_executive_summary())
        exports['summary'] = str(summary_file)
        
        return exports


# Create singleton instance
lariat_bible = LariatBible()
