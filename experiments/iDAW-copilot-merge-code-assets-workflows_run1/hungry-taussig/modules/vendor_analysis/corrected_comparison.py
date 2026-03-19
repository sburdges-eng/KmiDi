"""
Corrected Vendor Price Comparison
Properly handles pack sizes and unit conversions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class CorrectedVendorComparison:
    """Accurate vendor price comparison with proper pack size handling"""
    
    def __init__(self):
        # Standard conversions
        self.conversions = {
            'LB_TO_OZ': 16,
            'GAL_TO_OZ': 128,
            'QT_TO_OZ': 32,
            'CAN_#10_OZ': 109,  # Standard #10 can
            'CAN_#5_OZ': 56,
            'CAN_#2.5_OZ': 28,
        }
        
        self.comparison_results = []
    
    def interpret_pack_size(self, pack_str: str) -> Dict:
        """
        Correctly interpret pack sizes based on ACTUAL vendor formats
        
        Rules:
        - Shamrock: 1/6/LB = 1 container of 6 lbs
        - SYSCO: 3/6LB = 3 containers of 6 lbs each = 18 lbs total
        - #10 = standard can (109 oz) ONLY with specific dimensions
        - Simple: 25 LB = 25 pounds total
        """
        import re
        pack_str = str(pack_str).upper().strip()
        
        # Dictionary to store parsed results
        result = {
            'original': pack_str,
            'total_pounds': None,
            'total_ounces': None,
            'containers': 1,
            'pounds_per_container': None,
            'unit_type': None
        }
        
        # SHAMROCK FORMAT: 1/6/LB or similar (containers/pounds/LB)
        shamrock_match = re.match(r'(\d+)/(\d+)/LB', pack_str)
        if shamrock_match:
            containers = int(shamrock_match.group(1))
            pounds_per = int(shamrock_match.group(2))
            result['containers'] = containers
            result['pounds_per_container'] = pounds_per
            result['unit_type'] = 'LB'
            result['total_pounds'] = containers * pounds_per
            result['total_ounces'] = containers * pounds_per * 16
            return result
        
        # SYSCO FORMAT: 3/6LB or 3/6# (containers/pounds each)
        sysco_match = re.match(r'(\d+)/(\d+)(LB|#)', pack_str)
        if sysco_match:
            containers = int(sysco_match.group(1))
            pounds_per = int(sysco_match.group(2))
            result['containers'] = containers
            result['pounds_per_container'] = pounds_per
            result['unit_type'] = 'LB'
            result['total_pounds'] = containers * pounds_per
            result['total_ounces'] = containers * pounds_per * 16
            return result
        
        # Check for #10 can specifically (standard can size)
        if '/#10' in pack_str:
            match = re.match(r'(\d+)/#10', pack_str)
            if match:
                count = int(match.group(1))
                result['containers'] = count
                result['unit_type'] = 'CAN#10'
                result['total_ounces'] = count * 109
                result['total_pounds'] = (count * 109) / 16
                return result
        
        # Simple pounds (25 LB, 50 LB, etc.)
        simple_lb = re.match(r'(\d+)\s*LB', pack_str)
        if simple_lb:
            pounds = int(simple_lb.group(1))
            result['containers'] = 1
            result['pounds_per_container'] = pounds
            result['unit_type'] = 'LB'
            result['total_pounds'] = pounds
            result['total_ounces'] = pounds * 16
            return result
        
        # Gallons
        gal_match = re.match(r'(\d+)/(\d+)\s*GAL', pack_str)
        if gal_match:
            containers = int(gal_match.group(1))
            gallons = int(gal_match.group(2))
            result['containers'] = containers
            result['unit_type'] = 'GAL'
            result['total_ounces'] = containers * gallons * 128
            return result
        
        # Cases/Each
        if any(unit in pack_str for unit in ['CS', 'CASE', 'EA', 'EACH']):
            match = re.search(r'(\d+)', pack_str)
            if match:
                count = int(match.group(1))
                result['containers'] = count
                result['unit_type'] = 'EACH'
                return result
        
        return result
    
    def calculate_price_per_unit(self, pack_size: str, case_price: float, 
                                 target_unit: str = 'LB') -> Optional[float]:
        """
        Calculate price per unit (default: per pound)
        
        Args:
            pack_size: Pack size string
            case_price: Total case price
            target_unit: Target unit for price calculation
        
        Returns:
            Price per target unit, or None if cannot convert
        """
        parsed = self.interpret_pack_size(pack_size)
        
        if target_unit == 'LB' and parsed['total_pounds']:
            return case_price / parsed['total_pounds']
        elif target_unit == 'OZ' and parsed['total_ounces']:
            return case_price / parsed['total_ounces']
        elif target_unit == 'EACH' and parsed['count']:
            return case_price / parsed['count']
        
        return None
    
    def compare_items(self, item_name: str, 
                     sysco_pack: str, sysco_price: float,
                     shamrock_pack: str, shamrock_price: float,
                     sysco_split_price: float = None) -> Dict:
        """
        Compare prices between vendors with correct pack size interpretation
        Including split pricing when available
        """
        # Calculate price per pound for each
        sysco_per_lb = self.calculate_price_per_unit(sysco_pack, sysco_price, 'LB')
        shamrock_per_lb = self.calculate_price_per_unit(shamrock_pack, shamrock_price, 'LB')
        
        # Calculate split price per pound if provided
        sysco_split_per_lb = None
        if sysco_split_price:
            parsed = self.interpret_pack_size(sysco_pack)
            if parsed['pounds_per_container']:
                sysco_split_per_lb = sysco_split_price / parsed['pounds_per_container']
        
        if sysco_per_lb and shamrock_per_lb:
            savings_per_lb = sysco_per_lb - shamrock_per_lb
            savings_pct = (savings_per_lb / sysco_per_lb) * 100 if sysco_per_lb > 0 else 0
            
            result = {
                'item': item_name,
                'sysco_pack': sysco_pack,
                'sysco_case_price': sysco_price,
                'sysco_per_lb': sysco_per_lb,
                'sysco_split_price': sysco_split_price,
                'sysco_split_per_lb': sysco_split_per_lb,
                'shamrock_pack': shamrock_pack,
                'shamrock_case_price': shamrock_price,
                'shamrock_per_lb': shamrock_per_lb,
                'savings_per_lb': savings_per_lb,
                'savings_percent': savings_pct,
                'preferred_vendor': 'Shamrock' if shamrock_per_lb < sysco_per_lb else 'SYSCO',
                'monthly_savings_estimate': savings_per_lb * 10  # Assume 10 lbs/month usage
            }
            
            # Compare with split pricing too
            if sysco_split_per_lb:
                split_savings = sysco_split_per_lb - shamrock_per_lb
                result['split_vs_shamrock_savings'] = split_savings
                result['split_savings_pct'] = (split_savings / sysco_split_per_lb) * 100 if sysco_split_per_lb > 0 else 0
            
            return result
        
        return None
    
    def recalculate_spice_savings(self) -> pd.DataFrame:
        """
        Recalculate the spice savings with CORRECT pack size interpretation
        """
        # Real examples from your data (ACTUALLY CORRECTED)
        spice_comparisons = [
            # Item, SYSCO pack, SYSCO price, Shamrock pack, Shamrock price
            ("Garlic Powder", "3/6LB", 213.19, "1/6/LB", 54.26),
            ("Black Pepper Ground", "6/1LB", 295.89, "25 LB", 95.88),
            ("Black Pepper Coarse", "6/1LB", 298.95, "25 LB", 79.71),
            ("Black Pepper Cracked", "6/1LB", 269.99, "25 LB", 76.90),
            ("Onion Powder", "6/1LB", 148.95, "25 LB", 39.80),
            ("Garlic Granulated", "5/1LB", 165.99, "25 LB", 67.47),
        ]
        
        results = []
        for item_data in spice_comparisons:
            comparison = self.compare_items(*item_data)
            if comparison:
                results.append(comparison)
        
        df = pd.DataFrame(results)
        
        # Add summary
        print("\n" + "="*80)
        print("CORRECTED SPICE PRICE COMPARISON")
        print("="*80)
        print("\nPack Size Format Guide:")
        print("- Shamrock '1/6/LB' = 1 container × 6 pounds = 6 lbs total")
        print("- SYSCO '3/6LB' = 3 containers × 6 pounds each = 18 lbs total")
        print("- SYSCO '6/1LB' = 6 containers × 1 pound each = 6 lbs total")
        print("- Simple '25 LB' = 25 pounds total")
        print("-"*80)
        
        for _, row in df.iterrows():
            print(f"\n{row['item']}")
            print(f"  SYSCO: {row['sysco_pack']} @ ${row['sysco_case_price']:.2f}")
            print(f"    = ${row['sysco_per_lb']:.2f} per pound")
            print(f"  Shamrock: {row['shamrock_pack']} @ ${row['shamrock_case_price']:.2f}")
            print(f"    = ${row['shamrock_per_lb']:.2f} per pound")
            print(f"  ACTUAL Savings: ${row['savings_per_lb']:.2f}/lb ({row['savings_percent']:.1f}%)")
            print(f"  Preferred: {row['preferred_vendor']}")
        
        print("\n" + "="*80)
        print("MONTHLY SAVINGS SUMMARY (Corrected)")
        print("="*80)
        total_monthly = df['monthly_savings_estimate'].sum()
        print(f"Estimated Monthly Savings on Spices: ${total_monthly:.2f}")
        print(f"Annual Projection: ${total_monthly * 12:.2f}")
        print("\nNOTE: Based on 10 lbs/month usage per spice type")
        
        return df


# Test the corrections
if __name__ == "__main__":
    comparator = CorrectedVendorComparison()
    
    # Test pack size interpretation
    print("PACK SIZE INTERPRETATION TESTS")
    print("="*50)
    
    test_packs = [
        "1/6/LB",    # Shamrock: 1 × 6 pounds = 6 lbs
        "3/6LB",     # SYSCO: 3 × 6 pounds = 18 lbs  
        "6/1LB",     # SYSCO: 6 × 1 pound = 6 lbs
        "25 LB",     # Simple: 25 pounds
        "50 LB",     # Simple: 50 pounds
        "6/#10",     # 6 × #10 cans = 654 oz
    ]
    
    for pack in test_packs:
        result = comparator.interpret_pack_size(pack)
        print(f"{pack:10} -> Total: {result['total_pounds']} lbs" 
              f" (or {result['total_ounces']} oz)")
    
    print("\n" + "="*50)
    
    # Run corrected spice comparison
    df = comparator.recalculate_spice_savings()
    
    # Save results
    df.to_csv('/root/lariat-bible/data/corrected_spice_comparison.csv', index=False)
    print("\n✅ Saved corrected comparison to corrected_spice_comparison.csv")
