"""
Recipe Management Module
Handles recipes, ingredients, and costing
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class Ingredient:
    """Individual ingredient with vendor pricing"""
    
    ingredient_id: str
    name: str
    category: str  # Protein, Produce, Dairy, Dry Goods, etc.
    
    # Purchasing Information
    unit_of_measure: str  # "lb", "oz", "each", "bunch", etc.
    case_size: str  # "6/10#", "25 lb bag", "12 each"
    
    # Vendor Pricing (this is the KEY for comparison)
    sysco_item_code: Optional[str] = None
    sysco_price: Optional[float] = None  # Price per case/unit
    sysco_unit_price: Optional[float] = None  # Price per unit of measure
    sysco_last_updated: Optional[datetime] = None
    
    shamrock_item_code: Optional[str] = None
    shamrock_price: Optional[float] = None  # Price per case/unit  
    shamrock_unit_price: Optional[float] = None  # Price per unit of measure
    shamrock_last_updated: Optional[datetime] = None
    
    # Preferred vendor based on price
    preferred_vendor: Optional[str] = None
    price_difference: Optional[float] = None
    price_difference_percent: Optional[float] = None
    
    # Storage
    storage_location: str = ""  # Walk-in, Freezer, Dry Storage, etc.
    shelf_life_days: int = 0
    
    def calculate_best_price(self) -> Dict:
        """Determine which vendor has better pricing"""
        if self.sysco_unit_price and self.shamrock_unit_price:
            if self.sysco_unit_price < self.shamrock_unit_price:
                self.preferred_vendor = "SYSCO"
                self.price_difference = self.shamrock_unit_price - self.sysco_unit_price
            else:
                self.preferred_vendor = "Shamrock Foods"
                self.price_difference = self.sysco_unit_price - self.shamrock_unit_price
            
            # Calculate percentage difference
            avg_price = (self.sysco_unit_price + self.shamrock_unit_price) / 2
            self.price_difference_percent = (self.price_difference / avg_price) * 100
            
            return {
                'ingredient': self.name,
                'preferred_vendor': self.preferred_vendor,
                'sysco_price': self.sysco_unit_price,
                'shamrock_price': self.shamrock_unit_price,
                'savings_per_unit': abs(self.price_difference),
                'savings_percent': abs(self.price_difference_percent)
            }
        return {
            'ingredient': self.name,
            'preferred_vendor': 'Insufficient data',
            'message': 'Need pricing from both vendors'
        }


@dataclass  
class RecipeIngredient:
    """Ingredient used in a recipe with quantity"""
    ingredient: Ingredient
    quantity: float
    unit: str  # Should match ingredient.unit_of_measure
    prep_instruction: str = ""  # "diced", "julienned", etc.
    
    @property
    def cost(self) -> float:
        """Calculate cost for this ingredient in the recipe"""
        if self.ingredient.preferred_vendor == "SYSCO" and self.ingredient.sysco_unit_price:
            return self.quantity * self.ingredient.sysco_unit_price
        elif self.ingredient.preferred_vendor == "Shamrock Foods" and self.ingredient.shamrock_unit_price:
            return self.quantity * self.ingredient.shamrock_unit_price
        # Fallback to whichever price is available
        elif self.ingredient.sysco_unit_price:
            return self.quantity * self.ingredient.sysco_unit_price
        elif self.ingredient.shamrock_unit_price:
            return self.quantity * self.ingredient.shamrock_unit_price
        return 0.0


@dataclass
class Recipe:
    """Complete recipe with ingredients and instructions"""
    
    recipe_id: str
    name: str
    category: str  # Appetizer, Entree, Side, Sauce, etc.
    
    # Recipe Details
    yield_amount: float
    yield_unit: str  # "portions", "oz", "cups", etc.
    portion_size: str
    prep_time_minutes: int = 0
    cook_time_minutes: int = 0
    
    # Ingredients
    ingredients: List[RecipeIngredient] = None
    
    # Instructions
    prep_instructions: List[str] = None
    cooking_instructions: List[str] = None
    
    # Storage & Shelf Life
    storage_instructions: str = ""
    shelf_life_days: int = 0
    
    # Metadata
    created_by: str = ""
    created_date: datetime = None
    last_modified: datetime = None
    notes: str = ""
    
    def __post_init__(self):
        """Initialize defaults"""
        if self.ingredients is None:
            self.ingredients = []
        if self.prep_instructions is None:
            self.prep_instructions = []
        if self.cooking_instructions is None:
            self.cooking_instructions = []
        if self.created_date is None:
            self.created_date = datetime.now()
        if self.last_modified is None:
            self.last_modified = datetime.now()
    
    @property
    def total_cost(self) -> float:
        """Calculate total recipe cost"""
        return sum(ingredient.cost for ingredient in self.ingredients)
    
    @property
    def cost_per_portion(self) -> float:
        """Calculate cost per portion"""
        if self.yield_amount == 0:
            return 0
        return self.total_cost / self.yield_amount
    
    def get_shopping_list(self, multiplier: float = 1.0) -> Dict[str, Dict]:
        """Generate shopping list with vendor recommendations"""
        shopping_list = {}
        
        for recipe_ingredient in self.ingredients:
            ing = recipe_ingredient.ingredient
            quantity_needed = recipe_ingredient.quantity * multiplier
            
            shopping_list[ing.name] = {
                'quantity_needed': quantity_needed,
                'unit': recipe_ingredient.unit,
                'preferred_vendor': ing.preferred_vendor,
                'vendor_item_code': (ing.sysco_item_code if ing.preferred_vendor == "SYSCO" 
                                    else ing.shamrock_item_code),
                'estimated_cost': recipe_ingredient.cost * multiplier,
                'savings_vs_other': ing.price_difference * quantity_needed if ing.price_difference else 0
            }
        
        return shopping_list
    
    def analyze_vendor_impact(self) -> Dict:
        """Analyze cost if using different vendors"""
        sysco_total = 0
        shamrock_total = 0
        mixed_total = 0  # Using preferred vendor for each item
        
        for recipe_ingredient in self.ingredients:
            ing = recipe_ingredient.ingredient
            qty = recipe_ingredient.quantity
            
            if ing.sysco_unit_price:
                sysco_total += qty * ing.sysco_unit_price
            if ing.shamrock_unit_price:
                shamrock_total += qty * ing.shamrock_unit_price
            
            # For mixed, use the preferred vendor
            if ing.preferred_vendor == "SYSCO" and ing.sysco_unit_price:
                mixed_total += qty * ing.sysco_unit_price
            elif ing.preferred_vendor == "Shamrock Foods" and ing.shamrock_unit_price:
                mixed_total += qty * ing.shamrock_unit_price
        
        return {
            'recipe': self.name,
            'sysco_only_cost': sysco_total,
            'shamrock_only_cost': shamrock_total,
            'optimized_cost': mixed_total,
            'savings_vs_sysco': sysco_total - mixed_total if sysco_total > 0 else 0,
            'savings_vs_shamrock': shamrock_total - mixed_total if shamrock_total > 0 else 0,
            'recommendation': 'Use mixed vendors for best pricing' if mixed_total < min(sysco_total, shamrock_total) else 'Single vendor might be simpler'
        }
