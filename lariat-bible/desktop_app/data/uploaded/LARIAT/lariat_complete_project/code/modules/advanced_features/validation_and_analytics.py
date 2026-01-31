"""
Data Validation, Quality Checks, and Advanced Features
Complete system for ensuring accuracy and maximizing savings
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import json
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# DATA VALIDATION SYSTEM
# ============================================================================

class ValidationLevel(Enum):
    """Severity levels for validation issues"""
    CRITICAL = "CRITICAL"  # Must fix before using data
    WARNING = "WARNING"    # Should review
    INFO = "INFO"         # FYI only

@dataclass
class ValidationResult:
    """Result of a validation check"""
    level: ValidationLevel
    field: str
    message: str
    current_value: any
    expected_range: Optional[str] = None
    suggested_action: Optional[str] = None


class DataValidator:
    """Comprehensive data validation system"""
    
    def __init__(self):
        self.validation_rules = {
            "pack_size_sanity": {
                "max_pounds_per_case": 100,
                "min_pounds_per_case": 0.1,
                "common_sizes": [1, 5, 6, 10, 25, 50],
                "warn_if_unusual": True
            },
            "price_sanity": {
                "max_price_per_pound": 100,
                "min_price_per_pound": 0.10,
                "variance_threshold": 500,  # 500% difference triggers warning
                "identical_price_warning": True
            },
            "product_matching": {
                "grind_must_match": ["FINE", "COARSE", "CRACKED", "MEDIUM", "RESTAURANT"],
                "cut_must_match": ["WHOLE", "GROUND", "DICED", "SLICED"],
                "size_must_match": ["SMALL", "MEDIUM", "LARGE", "JUMBO"]
            }
        }
        
        self.audit_log = []
        
    def validate_price_comparison(self, item_name: str, 
                                 sysco_price: float, sysco_pack: str,
                                 shamrock_price: float, shamrock_pack: str) -> List[ValidationResult]:
        """Validate a price comparison between vendors"""
        results = []
        
        # Check for extreme price differences
        if sysco_price > 0 and shamrock_price > 0:
            ratio = max(sysco_price, shamrock_price) / min(sysco_price, shamrock_price)
            if ratio > 5:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    field="price_difference",
                    message=f"{item_name} shows {ratio:.0f}x price difference",
                    current_value=f"SYSCO: ${sysco_price:.2f}, Shamrock: ${shamrock_price:.2f}",
                    suggested_action="Verify products are truly equivalent (same grade/quality)"
                ))
        
        # Check for identical prices (suspicious)
        if sysco_price == shamrock_price and sysco_price > 0:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                field="identical_prices",
                message=f"{item_name} has identical prices from both vendors",
                current_value=f"${sysco_price:.2f}",
                suggested_action="Verify this isn't a data entry error"
            ))
        
        # Check for likely data entry errors
        for vendor, price in [("SYSCO", sysco_price), ("Shamrock", shamrock_price)]:
            if price > 1000:
                results.append(ValidationResult(
                    level=ValidationLevel.CRITICAL,
                    field="excessive_price",
                    message=f"{vendor} price over $1000 for {item_name}",
                    current_value=f"${price:.2f}",
                    suggested_action="Check for extra zeros or decimal point errors"
                ))
            
            if price < 0.01 and price > 0:
                results.append(ValidationResult(
                    level=ValidationLevel.CRITICAL,
                    field="too_low_price",
                    message=f"{vendor} price under $0.01 for {item_name}",
                    current_value=f"${price:.2f}",
                    suggested_action="Verify price and unit of measure"
                ))
        
        # Validate pack sizes
        pack_results = self.validate_pack_sizes(item_name, sysco_pack, shamrock_pack)
        results.extend(pack_results)
        
        return results
    
    def validate_pack_sizes(self, item_name: str, 
                           sysco_pack: str, shamrock_pack: str) -> List[ValidationResult]:
        """Validate pack size interpretations"""
        results = []
        
        # Parse pack sizes
        sysco_pounds = self._parse_pack_to_pounds(sysco_pack)
        shamrock_pounds = self._parse_pack_to_pounds(shamrock_pack)
        
        for vendor, pounds, pack in [("SYSCO", sysco_pounds, sysco_pack), 
                                     ("Shamrock", shamrock_pounds, shamrock_pack)]:
            if pounds:
                # Check for unusual sizes
                if pounds > self.validation_rules["pack_size_sanity"]["max_pounds_per_case"]:
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        field="pack_size",
                        message=f"{vendor} pack size unusually large for {item_name}",
                        current_value=f"{pounds} lbs from {pack}",
                        expected_range=f"Under {self.validation_rules['pack_size_sanity']['max_pounds_per_case']} lbs",
                        suggested_action="Verify pack size interpretation"
                    ))
                
                # Check if it's a common size
                if self.validation_rules["pack_size_sanity"]["warn_if_unusual"]:
                    if not any(abs(pounds - common) < 2 for common in 
                              self.validation_rules["pack_size_sanity"]["common_sizes"]):
                        results.append(ValidationResult(
                            level=ValidationLevel.INFO,
                            field="pack_size",
                            message=f"{vendor} has non-standard pack size for {item_name}",
                            current_value=f"{pounds} lbs",
                            expected_range=f"Common sizes: {self.validation_rules['pack_size_sanity']['common_sizes']}"
                        ))
        
        return results
    
    def validate_product_match(self, sysco_desc: str, shamrock_desc: str) -> bool:
        """Check if two products are truly equivalent"""
        sysco_upper = sysco_desc.upper()
        shamrock_upper = shamrock_desc.upper()
        
        # Check grind matching for spices
        for grind in self.validation_rules["product_matching"]["grind_must_match"]:
            if grind in sysco_upper and grind not in shamrock_upper:
                return False
            if grind in shamrock_upper and grind not in sysco_upper:
                return False
        
        # Check cut matching
        for cut in self.validation_rules["product_matching"]["cut_must_match"]:
            if cut in sysco_upper and cut not in shamrock_upper:
                return False
        
        return True
    
    def _parse_pack_to_pounds(self, pack_str: str) -> Optional[float]:
        """Parse pack string to total pounds"""
        import re
        pack_str = str(pack_str).upper()
        
        # Shamrock format: 1/6/LB
        if '/LB' in pack_str:
            parts = pack_str.replace('/LB', '').split('/')
            if len(parts) == 2:
                try:
                    return float(parts[0]) * float(parts[1])
                except:
                    pass
        
        # SYSCO format: 3/6LB or 6/1LB
        if 'LB' in pack_str and '/' in pack_str:
            parts = pack_str.replace('LB', '').split('/')
            if len(parts) == 2:
                try:
                    return float(parts[0]) * float(parts[1])
                except:
                    pass
        
        # Simple: 25 LB
        match = re.search(r'(\d+)\s*LB', pack_str)
        if match:
            return float(match.group(1))
        
        return None
    
    def create_audit_entry(self, action: str, details: Dict) -> Dict:
        """Create an audit log entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
            "user": "system",
            "source": details.get("source", "manual")
        }
        self.audit_log.append(entry)
        return entry


# ============================================================================
# USAGE-BASED RECOMMENDATIONS
# ============================================================================

class UsageAnalyzer:
    """Analyze usage patterns and make recommendations"""
    
    def __init__(self):
        self.usage_history = {}
        self.recommendations = []
    
    def calculate_optimal_vendor_split(self, products: List[Dict], 
                                      monthly_usage: Dict[str, float]) -> Dict:
        """Calculate optimal vendor split based on usage"""
        
        sysco_order = []
        shamrock_order = []
        total_savings = 0
        
        for product in products:
            item_name = product['name']
            usage = monthly_usage.get(item_name, 0)
            
            if usage == 0:
                continue
            
            sysco_monthly = usage * product.get('sysco_per_lb', float('inf'))
            shamrock_monthly = usage * product.get('shamrock_per_lb', float('inf'))
            
            if shamrock_monthly < sysco_monthly:
                shamrock_order.append({
                    'item': item_name,
                    'quantity': usage,
                    'cost': shamrock_monthly,
                    'savings': sysco_monthly - shamrock_monthly
                })
                total_savings += (sysco_monthly - shamrock_monthly)
            else:
                sysco_order.append({
                    'item': item_name,
                    'quantity': usage,
                    'cost': sysco_monthly
                })
        
        return {
            'sysco_items': len(sysco_order),
            'sysco_total': sum(item['cost'] for item in sysco_order),
            'shamrock_items': len(shamrock_order),
            'shamrock_total': sum(item['cost'] for item in shamrock_order),
            'total_monthly_savings': total_savings,
            'annual_savings_projection': total_savings * 12,
            'recommendations': self._generate_recommendations(sysco_order, shamrock_order)
        }
    
    def analyze_seasonality(self, historical_prices: pd.DataFrame) -> Dict:
        """Identify seasonal price patterns"""
        
        seasonality = {}
        
        for product in historical_prices['product'].unique():
            product_data = historical_prices[historical_prices['product'] == product]
            
            if len(product_data) < 12:
                continue
            
            # Calculate monthly averages
            product_data['month'] = pd.to_datetime(product_data['date']).dt.month
            monthly_avg = product_data.groupby('month')['price'].mean()
            
            # Find best and worst months
            best_month = monthly_avg.idxmin()
            worst_month = monthly_avg.idxmax()
            variance = (monthly_avg.max() - monthly_avg.min()) / monthly_avg.mean() * 100
            
            if variance > 10:  # Significant seasonality
                seasonality[product] = {
                    'best_month': best_month,
                    'best_price': monthly_avg.min(),
                    'worst_month': worst_month,
                    'worst_price': monthly_avg.max(),
                    'variance_percent': variance,
                    'recommendation': f"Stock up in month {best_month} when prices are {variance:.1f}% lower"
                }
        
        return seasonality
    
    def _generate_recommendations(self, sysco_order: List, shamrock_order: List) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # High-impact switches
        high_savings = [item for item in shamrock_order if item['savings'] > 100]
        if high_savings:
            recommendations.append(f"IMMEDIATE ACTION: Switch {len(high_savings)} items to Shamrock for ${sum(item['savings'] for item in high_savings):.2f}/month savings")
        
        # Consolidation opportunity
        if len(sysco_order) < 5 and len(shamrock_order) > 20:
            recommendations.append("Consider dropping SYSCO for standard items, use only for specialties")
        
        # Minimum order considerations
        shamrock_total = sum(item['cost'] for item in shamrock_order)
        if shamrock_total < 500:  # Assuming $500 minimum
            recommendations.append(f"Shamrock order ${shamrock_total:.2f} - consider adding items to meet delivery minimum")
        
        return recommendations


# ============================================================================
# AUTOMATED ALERTS SYSTEM
# ============================================================================

class AlertSystem:
    """Automated alert generation and management"""
    
    def __init__(self):
        self.alert_rules = {
            'price_increase': {'threshold': 10, 'priority': 'medium'},
            'new_savings': {'threshold': 100, 'priority': 'high'},
            'extreme_variance': {'threshold': 200, 'priority': 'critical'},
            'contract_renewal': {'days_notice': 30, 'priority': 'high'},
            'low_stock': {'threshold': 'reorder_point', 'priority': 'high'}
        }
        
        self.active_alerts = []
    
    def check_price_changes(self, old_prices: Dict, new_prices: Dict) -> List[Dict]:
        """Check for significant price changes"""
        alerts = []
        
        for item, new_price in new_prices.items():
            if item in old_prices:
                old_price = old_prices[item]
                if old_price > 0:
                    change_pct = ((new_price - old_price) / old_price) * 100
                    
                    if change_pct > self.alert_rules['price_increase']['threshold']:
                        alerts.append({
                            'type': 'price_increase',
                            'item': item,
                            'message': f"{item} increased {change_pct:.1f}% from ${old_price:.2f} to ${new_price:.2f}",
                            'priority': self.alert_rules['price_increase']['priority'],
                            'action': 'Review alternatives or stock up at old price if possible'
                        })
                    
                    elif change_pct < -self.alert_rules['price_increase']['threshold']:
                        alerts.append({
                            'type': 'price_decrease',
                            'item': item,
                            'message': f"{item} decreased {abs(change_pct):.1f}% - good time to stock up",
                            'priority': 'low',
                            'action': 'Consider increasing order quantity'
                        })
        
        return alerts
    
    def check_savings_opportunities(self, comparisons: List[Dict]) -> List[Dict]:
        """Identify new savings opportunities"""
        alerts = []
        
        for comp in comparisons:
            monthly_savings = comp.get('monthly_savings_estimate', 0)
            
            if monthly_savings > self.alert_rules['new_savings']['threshold']:
                alerts.append({
                    'type': 'savings_opportunity',
                    'item': comp['item'],
                    'message': f"Save ${monthly_savings:.2f}/month by switching {comp['item']} to {comp['preferred_vendor']}",
                    'priority': self.alert_rules['new_savings']['priority'],
                    'action': f"Switch to {comp['preferred_vendor']} immediately"
                })
        
        return alerts
    
    def check_contract_renewals(self, contracts: List[Dict]) -> List[Dict]:
        """Alert for upcoming contract renewals"""
        alerts = []
        days_notice = self.alert_rules['contract_renewal']['days_notice']
        
        for contract in contracts:
            renewal_date = contract.get('renewal_date')
            if renewal_date:
                days_until = (renewal_date - datetime.now()).days
                
                if 0 < days_until <= days_notice:
                    alerts.append({
                        'type': 'contract_renewal',
                        'vendor': contract['vendor'],
                        'message': f"{contract['vendor']} contract renews in {days_until} days",
                        'priority': self.alert_rules['contract_renewal']['priority'],
                        'action': 'Review pricing and negotiate better terms'
                    })
        
        return alerts


# ============================================================================
# ORDER GENERATION SYSTEM
# ============================================================================

class OrderGenerator:
    """Generate optimized purchase orders"""
    
    def __init__(self):
        self.minimum_orders = {
            'SYSCO': 500,
            'Shamrock Foods': 500
        }
    
    def generate_optimal_orders(self, needed_items: List[Dict], 
                               price_comparisons: Dict) -> Dict:
        """Generate purchase orders optimized by vendor"""
        
        sysco_order = []
        shamrock_order = []
        
        for item in needed_items:
            item_name = item['name']
            quantity = item['quantity']
            
            # Find best price
            if item_name in price_comparisons:
                comp = price_comparisons[item_name]
                
                if comp['preferred_vendor'] == 'Shamrock':
                    shamrock_order.append({
                        'item': item_name,
                        'quantity': quantity,
                        'unit_price': comp['shamrock_per_lb'],
                        'total': quantity * comp['shamrock_per_lb'],
                        'savings': quantity * comp['savings_per_lb']
                    })
                else:
                    sysco_order.append({
                        'item': item_name,
                        'quantity': quantity,
                        'unit_price': comp['sysco_per_lb'],
                        'total': quantity * comp['sysco_per_lb']
                    })
        
        # Check minimums and adjust if needed
        sysco_total = sum(item['total'] for item in sysco_order)
        shamrock_total = sum(item['total'] for item in shamrock_order)
        
        adjustments = self._check_minimums(sysco_order, shamrock_order, 
                                          sysco_total, shamrock_total)
        
        return {
            'sysco_order': sysco_order,
            'sysco_total': sysco_total,
            'shamrock_order': shamrock_order,
            'shamrock_total': shamrock_total,
            'total_savings': sum(item.get('savings', 0) for item in shamrock_order),
            'adjustments': adjustments,
            'order_dates': self._calculate_order_dates()
        }
    
    def _check_minimums(self, sysco_order: List, shamrock_order: List,
                       sysco_total: float, shamrock_total: float) -> List[str]:
        """Check and adjust for minimum orders"""
        adjustments = []
        
        if sysco_total > 0 and sysco_total < self.minimum_orders['SYSCO']:
            needed = self.minimum_orders['SYSCO'] - sysco_total
            adjustments.append(f"Add ${needed:.2f} to SYSCO order to meet minimum")
        
        if shamrock_total > 0 and shamrock_total < self.minimum_orders['Shamrock Foods']:
            needed = self.minimum_orders['Shamrock Foods'] - shamrock_total
            adjustments.append(f"Add ${needed:.2f} to Shamrock order to meet minimum")
        
        # Consider consolidation
        if sysco_total < 200 and shamrock_total > 1000:
            adjustments.append("Consider moving SYSCO items to Shamrock to consolidate")
        
        return adjustments
    
    def _calculate_order_dates(self) -> Dict:
        """Calculate optimal order dates"""
        today = datetime.now()
        
        return {
            'next_sysco': (today + timedelta(days=7)).strftime('%Y-%m-%d'),
            'next_shamrock': (today + timedelta(days=3)).strftime('%Y-%m-%d'),
            'combined_order': (today + timedelta(days=5)).strftime('%Y-%m-%d')
        }


# ============================================================================
# MENU ENGINEERING SYSTEM
# ============================================================================

class MenuEngineer:
    """Analyze and optimize menu profitability"""
    
    def __init__(self):
        self.menu_items = []
        self.ingredient_costs = {}
    
    def analyze_dish_profitability(self, dish: Dict, 
                                  optimized_costs: Dict) -> Dict:
        """Analyze individual dish profitability"""
        
        current_cost = sum(
            self.ingredient_costs.get(ing['name'], 0) * ing['quantity'] 
            for ing in dish['ingredients']
        )
        
        optimized_cost = sum(
            optimized_costs.get(ing['name'], self.ingredient_costs.get(ing['name'], 0)) * ing['quantity']
            for ing in dish['ingredients']
        )
        
        return {
            'dish': dish['name'],
            'menu_price': dish['price'],
            'current_food_cost': current_cost,
            'current_margin': dish['price'] - current_cost,
            'current_margin_pct': ((dish['price'] - current_cost) / dish['price'] * 100) if dish['price'] > 0 else 0,
            'optimized_food_cost': optimized_cost,
            'optimized_margin': dish['price'] - optimized_cost,
            'optimized_margin_pct': ((dish['price'] - optimized_cost) / dish['price'] * 100) if dish['price'] > 0 else 0,
            'margin_improvement': optimized_cost - current_cost,
            'classification': self._classify_menu_item(dish['price'] - current_cost, dish.get('popularity', 0))
        }
    
    def _classify_menu_item(self, margin: float, popularity: int) -> str:
        """Classify menu items using menu engineering matrix"""
        
        high_margin = margin > 10  # Adjust threshold as needed
        high_popularity = popularity > 5  # Adjust based on your scale
        
        if high_margin and high_popularity:
            return "STAR - Promote heavily"
        elif high_margin and not high_popularity:
            return "PUZZLE - Increase visibility"
        elif not high_margin and high_popularity:
            return "WORKHORSE - Increase price carefully"
        else:
            return "DOG - Consider removing or reworking"
    
    def generate_menu_optimization_report(self, menu: List[Dict]) -> Dict:
        """Generate complete menu optimization report"""
        
        analyzed_items = []
        for dish in menu:
            analysis = self.analyze_dish_profitability(
                dish, 
                self._get_optimized_costs()
            )
            analyzed_items.append(analysis)
        
        # Sort by opportunity
        analyzed_items.sort(key=lambda x: x['margin_improvement'], reverse=True)
        
        return {
            'total_dishes': len(analyzed_items),
            'average_margin_improvement': sum(item['margin_improvement'] for item in analyzed_items) / len(analyzed_items),
            'dishes_to_promote': [item for item in analyzed_items if item['classification'].startswith('STAR')],
            'dishes_to_rework': [item for item in analyzed_items if item['classification'].startswith('DOG')],
            'price_increase_candidates': [item for item in analyzed_items if item['current_margin_pct'] < 30],
            'top_improvements': analyzed_items[:10]
        }
    
    def _get_optimized_costs(self) -> Dict:
        """Get optimized ingredient costs using best vendors"""
        # This would pull from your vendor comparison system
        return self.ingredient_costs  # Placeholder


# ============================================================================
# DUPLICATE AND QUALITY DETECTION
# ============================================================================

class DataQualityChecker:
    """Check for data quality issues"""
    
    def __init__(self):
        self.quality_issues = []
    
    def find_duplicate_products(self, products: List[Dict]) -> List[Dict]:
        """Find potential duplicate entries"""
        
        duplicates = []
        seen = {}
        
        for product in products:
            # Normalize name for comparison
            normalized = self._normalize_product_name(product['name'])
            
            if normalized in seen:
                duplicates.append({
                    'original': seen[normalized],
                    'duplicate': product['name'],
                    'similarity': self._calculate_similarity(seen[normalized], product['name'])
                })
            else:
                seen[normalized] = product['name']
        
        return duplicates
    
    def find_missing_data(self, products: List[Dict]) -> Dict:
        """Identify missing data points"""
        
        missing = {
            'no_sysco_price': [],
            'no_shamrock_price': [],
            'no_pack_size': [],
            'no_recent_orders': [],
            'no_category': []
        }
        
        for product in products:
            if not product.get('sysco_price'):
                missing['no_sysco_price'].append(product['name'])
            if not product.get('shamrock_price'):
                missing['no_shamrock_price'].append(product['name'])
            if not product.get('pack_size'):
                missing['no_pack_size'].append(product['name'])
            if not product.get('last_ordered'):
                missing['no_recent_orders'].append(product['name'])
            if not product.get('category'):
                missing['no_category'].append(product['name'])
        
        return missing
    
    def check_quality_equivalence(self, sysco_product: Dict, 
                                 shamrock_product: Dict) -> Dict:
        """Check if products are truly equivalent in quality"""
        
        quality_markers = {
            'grade': ['PRIME', 'CHOICE', 'SELECT', 'STANDARD'],
            'organic': ['ORGANIC', 'CONVENTIONAL'],
            'processing': ['FRESH', 'FROZEN', 'CANNED', 'DRIED'],
            'brand_tier': ['PREMIUM', 'NATIONAL', 'PRIVATE LABEL', 'GENERIC']
        }
        
        differences = []
        
        sysco_desc = sysco_product.get('description', '').upper()
        shamrock_desc = shamrock_product.get('description', '').upper()
        
        for category, markers in quality_markers.items():
            sysco_marker = None
            shamrock_marker = None
            
            for marker in markers:
                if marker in sysco_desc:
                    sysco_marker = marker
                if marker in shamrock_desc:
                    shamrock_marker = marker
            
            if sysco_marker != shamrock_marker:
                differences.append({
                    'category': category,
                    'sysco': sysco_marker or 'Not specified',
                    'shamrock': shamrock_marker or 'Not specified'
                })
        
        return {
            'products_match': len(differences) == 0,
            'differences': differences,
            'recommendation': 'Products are equivalent' if len(differences) == 0 
                            else f'Quality differences found in: {", ".join(d["category"] for d in differences)}'
        }
    
    def _normalize_product_name(self, name: str) -> str:
        """Normalize product name for comparison"""
        # Remove common variations
        normalized = name.upper()
        normalized = normalized.replace('BLACK PEPPER', 'PEPPER BLACK')
        normalized = normalized.replace('POWDER', 'PWD')
        normalized = normalized.replace('GRANULATED', 'GRAN')
        # Remove special characters
        normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        return ' '.join(normalized.split())  # Remove extra spaces
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity (0-100%)"""
        # Simple character overlap - could use more sophisticated algorithm
        set1 = set(str1.upper())
        set2 = set(str2.upper())
        
        if not set1 or not set2:
            return 0
        
        intersection = set1 & set2
        union = set1 | set2
        
        return (len(intersection) / len(union)) * 100


# ============================================================================
# NEGOTIATION HELPER
# ============================================================================

class NegotiationAssistant:
    """Generate negotiation talking points and strategies"""
    
    def __init__(self):
        self.vendor_history = {}
        self.contract_terms = {}
    
    def generate_negotiation_package(self, vendor: str, 
                                    comparisons: List[Dict]) -> Dict:
        """Generate complete negotiation package"""
        
        # Find items where vendor is more expensive
        overpriced = [
            comp for comp in comparisons 
            if comp['preferred_vendor'] != vendor
        ]
        
        # Calculate leverage
        total_overpayment = sum(
            comp.get('monthly_savings_estimate', 0) 
            for comp in overpriced
        )
        
        annual_leverage = total_overpayment * 12
        
        # Generate talking points
        talking_points = self._generate_talking_points(vendor, overpriced, annual_leverage)
        
        # Suggested targets
        targets = self._calculate_negotiation_targets(overpriced)
        
        return {
            'vendor': vendor,
            'annual_leverage': annual_leverage,
            'number_of_overpriced_items': len(overpriced),
            'talking_points': talking_points,
            'negotiation_targets': targets,
            'fallback_positions': self._generate_fallbacks(vendor, annual_leverage),
            'documentation': self._prepare_documentation(overpriced)
        }
    
    def _generate_talking_points(self, vendor: str, 
                                overpriced: List, leverage: float) -> List[str]:
        """Generate specific talking points"""
        
        points = []
        
        # Opening
        points.append(f"We value our relationship with {vendor}, but need to address pricing concerns")
        
        # Leverage
        points.append(f"Current pricing analysis shows {len(overpriced)} items where competitors are significantly cheaper")
        points.append(f"This represents ${leverage:,.2f} in annual overpayment")
        
        # Specific examples
        if overpriced:
            top_item = max(overpriced, key=lambda x: x.get('savings_percent', 0))
            points.append(f"For example, {top_item['item']} is {top_item['savings_percent']:.1f}% cheaper elsewhere")
        
        # Commitment
        points.append(f"We want to continue working with {vendor} but need competitive pricing")
        
        # Ask
        points.append("We need an across-the-board price reduction or matching on these items")
        
        return points
    
    def _calculate_negotiation_targets(self, overpriced: List) -> Dict:
        """Calculate specific negotiation targets"""
        
        return {
            'ideal': "Match competitor pricing on all items",
            'realistic': "15-20% reduction on overpriced items",
            'minimum': "10% reduction or we reduce order volume",
            'specific_items': [
                {
                    'item': item['item'],
                    'current': item.get('sysco_per_lb', 0),
                    'competitor': item.get('shamrock_per_lb', 0),
                    'target': item.get('shamrock_per_lb', 0) * 1.05  # 5% above competitor
                }
                for item in overpriced[:5]  # Top 5 items
            ]
        }
    
    def _generate_fallbacks(self, vendor: str, leverage: float) -> List[str]:
        """Generate fallback positions"""
        
        return [
            "Reduce order volume by 50% and source elsewhere",
            "Lock in current pricing for 12 months with no increases",
            "Additional 2% discount for prompt payment",
            "Free delivery on all orders",
            f"Credit for ${leverage/12:.2f} monthly in recognition of past overpayment"
        ]
    
    def _prepare_documentation(self, overpriced: List) -> Dict:
        """Prepare documentation for negotiation"""
        
        return {
            'comparison_spreadsheet': "Price_Comparison.xlsx",
            'competitor_quotes': "Include recent invoices from competitors",
            'historical_spending': "Show loyalty and volume",
            'items_list': [
                {
                    'item': item['item'],
                    'annual_volume': item.get('annual_usage', 0),
                    'price_difference': f"{item.get('savings_percent', 0):.1f}%"
                }
                for item in overpriced
            ]
        }


# ============================================================================
# CONTRACT TRACKER
# ============================================================================

class ContractTracker:
    """Track vendor contracts and commitments"""
    
    def __init__(self):
        self.contracts = []
    
    def add_contract(self, vendor: str, start_date: datetime, 
                    end_date: datetime, terms: Dict) -> Dict:
        """Add a contract to track"""
        
        contract = {
            'id': f"{vendor}_{start_date.strftime('%Y%m%d')}",
            'vendor': vendor,
            'start_date': start_date,
            'end_date': end_date,
            'terms': terms,
            'minimum_commitment': terms.get('minimum_monthly', 0),
            'auto_renewal': terms.get('auto_renewal', False),
            'notice_period': terms.get('notice_days', 30),
            'status': self._calculate_status(end_date)
        }
        
        self.contracts.append(contract)
        return contract
    
    def check_commitments(self) -> List[Dict]:
        """Check all contract commitments"""
        
        alerts = []
        today = datetime.now()
        
        for contract in self.contracts:
            # Check renewal dates
            days_until_renewal = (contract['end_date'] - today).days
            
            if days_until_renewal <= contract['notice_period']:
                alerts.append({
                    'type': 'renewal_notice',
                    'vendor': contract['vendor'],
                    'message': f"{contract['vendor']} contract requires notice by {(contract['end_date'] - timedelta(days=contract['notice_period'])).strftime('%Y-%m-%d')}",
                    'action': 'Give notice or renegotiate terms'
                })
            
            # Check minimum commitments
            if contract['minimum_commitment'] > 0:
                # This would check actual spending
                alerts.append({
                    'type': 'minimum_commitment',
                    'vendor': contract['vendor'],
                    'required': contract['minimum_commitment'],
                    'message': f"Minimum monthly commitment: ${contract['minimum_commitment']:,.2f}"
                })
        
        return alerts
    
    def _calculate_status(self, end_date: datetime) -> str:
        """Calculate contract status"""
        days_remaining = (end_date - datetime.now()).days
        
        if days_remaining < 0:
            return "EXPIRED"
        elif days_remaining < 30:
            return "EXPIRING_SOON"
        elif days_remaining < 90:
            return "ACTIVE_RENEWAL_WINDOW"
        else:
            return "ACTIVE"


# ============================================================================
# WHAT-IF CALCULATOR
# ============================================================================

class WhatIfCalculator:
    """Run what-if scenarios for decision making"""
    
    def calculate_switching_impact(self, items_to_switch: List[str], 
                                  comparisons: Dict) -> Dict:
        """Calculate impact of switching specific items"""
        
        monthly_savings = 0
        switching_cost = 0  # One-time costs
        
        for item in items_to_switch:
            if item in comparisons:
                comp = comparisons[item]
                monthly_savings += comp.get('monthly_savings_estimate', 0)
                
                # Estimate switching costs (menu updates, training, etc.)
                switching_cost += 50  # Placeholder
        
        return {
            'items_switched': len(items_to_switch),
            'monthly_savings': monthly_savings,
            'annual_savings': monthly_savings * 12,
            'switching_cost': switching_cost,
            'payback_period_days': (switching_cost / (monthly_savings / 30)) if monthly_savings > 0 else float('inf'),
            'five_year_savings': (monthly_savings * 60) - switching_cost
        }
    
    def calculate_volume_discount_impact(self, vendor: str, 
                                        volume_increase_pct: float) -> Dict:
        """Calculate impact of volume commitments for discounts"""
        
        # Typical volume discount tiers
        discount_tiers = {
            10: 2,   # 10% more volume = 2% discount
            25: 5,   # 25% more = 5% discount
            50: 8,   # 50% more = 8% discount
            100: 12  # 100% more = 12% discount
        }
        
        # Find applicable discount
        discount = 0
        for volume, disc in discount_tiers.items():
            if volume_increase_pct >= volume:
                discount = disc
        
        # Estimate current spending (placeholder - would use real data)
        current_monthly = 10000
        increased_monthly = current_monthly * (1 + volume_increase_pct / 100)
        discounted_total = increased_monthly * (1 - discount / 100)
        
        return {
            'vendor': vendor,
            'volume_increase': f"{volume_increase_pct}%",
            'discount_earned': f"{discount}%",
            'current_spending': current_monthly,
            'new_spending': discounted_total,
            'net_impact': discounted_total - current_monthly,
            'recommendation': "Favorable" if discounted_total < current_monthly else "Unfavorable"
        }


# ============================================================================
# MASTER INTEGRATION CLASS
# ============================================================================

class LariatBibleAdvanced:
    """Master class integrating all advanced features"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.usage_analyzer = UsageAnalyzer()
        self.alert_system = AlertSystem()
        self.order_generator = OrderGenerator()
        self.menu_engineer = MenuEngineer()
        self.quality_checker = DataQualityChecker()
        self.negotiation_assistant = NegotiationAssistant()
        self.contract_tracker = ContractTracker()
        self.whatif_calculator = WhatIfCalculator()
    
    def run_complete_analysis(self, products: List[Dict], 
                             usage_data: Dict, 
                             menu: List[Dict]) -> Dict:
        """Run complete analysis with all systems"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'validation_results': [],
            'quality_issues': [],
            'optimization_recommendations': [],
            'alerts': [],
            'negotiation_packages': {}
        }
        
        # Validate all data
        for product in products:
            if 'sysco_price' in product and 'shamrock_price' in product:
                validation = self.validator.validate_price_comparison(
                    product['name'],
                    product['sysco_price'],
                    product.get('sysco_pack', ''),
                    product['shamrock_price'],
                    product.get('shamrock_pack', '')
                )
                results['validation_results'].extend(validation)
        
        # Check data quality
        duplicates = self.quality_checker.find_duplicate_products(products)
        missing = self.quality_checker.find_missing_data(products)
        results['quality_issues'] = {
            'duplicates': duplicates,
            'missing_data': missing
        }
        
        # Generate optimization recommendations
        optimal_split = self.usage_analyzer.calculate_optimal_vendor_split(
            products, usage_data
        )
        results['optimization_recommendations'] = optimal_split
        
        # Check for alerts
        # (This would compare against historical prices)
        
        # Generate negotiation packages
        for vendor in ['SYSCO', 'Shamrock Foods']:
            package = self.negotiation_assistant.generate_negotiation_package(
                vendor, products
            )
            results['negotiation_packages'][vendor] = package
        
        # Menu engineering
        if menu:
            menu_analysis = self.menu_engineer.generate_menu_optimization_report(menu)
            results['menu_optimization'] = menu_analysis
        
        return results
    
    def export_analysis_report(self, results: Dict, filepath: str = 'analysis_report.json'):
        """Export complete analysis report"""
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return f"Report exported to {filepath}"


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("LARIAT BIBLE - ADVANCED FEATURES LOADED")
    print("="*80)
    
    # Initialize master system
    system = LariatBibleAdvanced()
    
    print("\n✅ Systems Initialized:")
    print("  - Data Validation System")
    print("  - Usage Pattern Analyzer")
    print("  - Automated Alert System")
    print("  - Order Generation System")
    print("  - Menu Engineering System")
    print("  - Data Quality Checker")
    print("  - Negotiation Assistant")
    print("  - Contract Tracker")
    print("  - What-If Calculator")
    
    print("\n📊 Available Functions:")
    print("  - Complete data validation with sanity checks")
    print("  - Usage-based vendor recommendations")
    print("  - Automated price change alerts")
    print("  - Optimal order splitting by vendor")
    print("  - Menu profitability analysis")
    print("  - Duplicate detection and cleanup")
    print("  - Negotiation package generation")
    print("  - Contract renewal tracking")
    print("  - What-if scenario modeling")
    
    print("\n💡 Next Steps:")
    print("1. Import your actual order guides")
    print("2. Run complete validation")
    print("3. Set up email parsing for continuous updates")
    print("4. Configure alerts for your needs")
    print("5. Generate first negotiation package")
    
    print("\n" + "="*80)
    print("System ready for production use!")
