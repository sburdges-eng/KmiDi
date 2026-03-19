"""
Menu Item Model
Represents individual menu items with recipe links and pricing
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class MenuItem:
    """Individual menu item with full details"""
    
    # Basic Information
    item_id: str
    name: str
    category: str  # Appetizer, Entree, Dessert, Beverage, etc.
    subcategory: Optional[str] = None  # Beef, Chicken, Seafood, Vegetarian, etc.
    
    # Description & Display
    description: str = ""
    display_name: str = ""  # How it appears on printed menu
    
    # Pricing
    menu_price: float = 0.0  # Customer price
    food_cost: float = 0.0  # Actual cost to make
    target_margin: float = 0.30  # Target profit margin
    
    # Recipe Link
    recipe_id: Optional[str] = None
    portion_size: str = ""  # "8 oz", "1 plate", etc.
    
    # Availability
    available: bool = True
    seasonal: bool = False
    days_available: List[str] = None  # ["Monday", "Tuesday", etc.]
    meal_periods: List[str] = None  # ["Lunch", "Dinner", "Brunch"]
    
    # Dietary Information
    dietary_flags: List[str] = None  # ["Gluten-Free", "Vegetarian", "Spicy", etc.]
    allergens: List[str] = None  # ["Nuts", "Dairy", "Shellfish", etc.]
    
    # Popularity & Analytics
    popularity_score: int = 0  # 1-10 scale
    monthly_sales: int = 0
    
    # Timestamps
    created_date: datetime = None
    last_modified: datetime = None
    
    def __post_init__(self):
        """Initialize defaults after dataclass creation"""
        if self.days_available is None:
            self.days_available = ["Monday", "Tuesday", "Wednesday", "Thursday", 
                                  "Friday", "Saturday", "Sunday"]
        if self.meal_periods is None:
            self.meal_periods = ["Lunch", "Dinner"]
        if self.dietary_flags is None:
            self.dietary_flags = []
        if self.allergens is None:
            self.allergens = []
        if self.created_date is None:
            self.created_date = datetime.now()
        if self.last_modified is None:
            self.last_modified = datetime.now()
        if not self.display_name:
            self.display_name = self.name
    
    @property
    def margin(self) -> float:
        """Calculate actual profit margin"""
        if self.menu_price == 0:
            return 0
        return (self.menu_price - self.food_cost) / self.menu_price
    
    @property
    def margin_variance(self) -> float:
        """Calculate variance from target margin"""
        return self.margin - self.target_margin
    
    @property
    def suggested_price(self) -> float:
        """Calculate suggested price based on target margin"""
        if self.target_margin >= 1:
            return 0  # Invalid margin
        return self.food_cost / (1 - self.target_margin)
    
    def update_food_cost(self, new_cost: float) -> Dict:
        """Update food cost and return pricing analysis"""
        old_cost = self.food_cost
        self.food_cost = new_cost
        self.last_modified = datetime.now()
        
        return {
            'item': self.name,
            'old_cost': old_cost,
            'new_cost': new_cost,
            'cost_change': new_cost - old_cost,
            'current_price': self.menu_price,
            'current_margin': self.margin,
            'suggested_price': self.suggested_price,
            'price_adjustment_needed': abs(self.margin_variance) > 0.05
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/API"""
        return {
            'item_id': self.item_id,
            'name': self.name,
            'category': self.category,
            'subcategory': self.subcategory,
            'description': self.description,
            'display_name': self.display_name,
            'menu_price': self.menu_price,
            'food_cost': self.food_cost,
            'margin': self.margin,
            'target_margin': self.target_margin,
            'recipe_id': self.recipe_id,
            'portion_size': self.portion_size,
            'available': self.available,
            'seasonal': self.seasonal,
            'days_available': self.days_available,
            'meal_periods': self.meal_periods,
            'dietary_flags': self.dietary_flags,
            'allergens': self.allergens,
            'popularity_score': self.popularity_score,
            'monthly_sales': self.monthly_sales,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'last_modified': self.last_modified.isoformat() if self.last_modified else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MenuItem':
        """Create MenuItem from dictionary"""
        # Convert ISO strings back to datetime
        if 'created_date' in data and data['created_date']:
            data['created_date'] = datetime.fromisoformat(data['created_date'])
        if 'last_modified' in data and data['last_modified']:
            data['last_modified'] = datetime.fromisoformat(data['last_modified'])
        
        return cls(**data)
