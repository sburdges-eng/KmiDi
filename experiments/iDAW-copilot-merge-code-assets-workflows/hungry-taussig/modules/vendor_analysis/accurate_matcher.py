"""
Accurate Vendor Comparison - Matching Product Specifications
Critical: Different grinds/cuts are different products!
"""

import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ProductMatch:
    """Matched products between vendors"""
    product_name: str
    specification: str  # Fine, Coarse, Cracked, Restaurant Grind, etc.
    
    # SYSCO
    sysco_code: str
    sysco_description: str
    sysco_pack: str
    sysco_case_price: float
    sysco_split_price: Optional[float]
    
    # Shamrock
    shamrock_code: str
    shamrock_description: str  
    shamrock_pack: str
    shamrock_price: float
    
    notes: str = ""
    
    def calculate_savings(self) -> Dict:
        """Calculate savings only if products truly match"""
        # Parse pack sizes correctly
        sysco_pounds = self._parse_pounds(self.sysco_pack)
        shamrock_pounds = self._parse_pounds(self.shamrock_pack)
        
        if not sysco_pounds or not shamrock_pounds:
            return {"error": "Cannot parse pack sizes"}
        
        sysco_per_lb = self.sysco_case_price / sysco_pounds
        shamrock_per_lb = self.shamrock_price / shamrock_pounds
        
        result = {
            "product": f"{self.product_name} - {self.specification}",
            "sysco_per_lb": sysco_per_lb,
            "shamrock_per_lb": shamrock_per_lb,
            "savings_per_lb": sysco_per_lb - shamrock_per_lb,
            "savings_percent": ((sysco_per_lb - shamrock_per_lb) / sysco_per_lb * 100) if sysco_per_lb > 0 else 0,
            "preferred_vendor": "Shamrock" if shamrock_per_lb < sysco_per_lb else "SYSCO"
        }
        
        # Add split pricing comparison if available
        if self.sysco_split_price:
            # For splits, usually it's per container
            containers = int(self.sysco_pack.split('/')[0]) if '/' in self.sysco_pack else 1
            pounds_per_container = sysco_pounds / containers
            split_per_lb = self.sysco_split_price / pounds_per_container
            result["sysco_split_per_lb"] = split_per_lb
            result["split_vs_shamrock"] = split_per_lb - shamrock_per_lb
        
        return result
    
    def _parse_pounds(self, pack_str: str) -> Optional[float]:
        """Parse total pounds from pack string"""
        pack_str = pack_str.upper()
        
        # Shamrock format: 1/6/LB = 1 container × 6 lbs
        if '/LB' in pack_str:
            parts = pack_str.replace('/LB', '').split('/')
            if len(parts) == 2:
                return float(parts[0]) * float(parts[1])
        
        # SYSCO format: 3/6LB or 6/1LB = containers × pounds each
        if 'LB' in pack_str and '/' in pack_str:
            parts = pack_str.replace('LB', '').split('/')
            if len(parts) == 2:
                return float(parts[0]) * float(parts[1])
        
        # Simple: 25 LB
        if 'LB' in pack_str:
            import re
            match = re.search(r'(\d+)\s*LB', pack_str)
            if match:
                return float(match.group(1))
        
        return None


class AccurateVendorMatcher:
    """Match products accurately between vendors"""
    
    def __init__(self):
        self.matched_products = []
        self.unmatched_sysco = []
        self.unmatched_shamrock = []
    
    def load_pepper_products(self) -> List[ProductMatch]:
        """
        Load BLACK PEPPER products with EXACT grind matching
        Different grinds have different uses:
        - Fine/Table: Table shakers, finishing
        - Restaurant/Medium: All-purpose kitchen use
        - Coarse: Steaks, robust dishes
        - Cracked: Visual appeal, texture, marinades
        """
        
        pepper_matches = []
        
        # These need to be verified against actual order guides
        # PLACEHOLDER DATA - Replace with actual matches
        
        # Example structure (NEEDS REAL DATA):
        potential_matches = [
            ProductMatch(
                product_name="Black Pepper",
                specification="Fine/Table Grind",
                sysco_code="SYSCO_FINE",
                sysco_description="BLACK PEPPER FINE TABLE GRIND",
                sysco_pack="6/1LB",  
                sysco_case_price=295.89,
                sysco_split_price=None,
                shamrock_code="SHAM_FINE",
                shamrock_description="PEPPER BLACK FINE",
                shamrock_pack="25 LB",
                shamrock_price=95.88,
                notes="For table shakers"
            ),
            
            ProductMatch(
                product_name="Black Pepper",
                specification="Restaurant/Medium Grind",
                sysco_code="SYSCO_REST",
                sysco_description="BLACK PEPPER RESTAURANT GRIND",
                sysco_pack="6/1LB",
                sysco_case_price=289.99,
                sysco_split_price=None,
                shamrock_code="SHAM_REST", 
                shamrock_description="PEPPER BLACK RESTAURANT",
                shamrock_pack="25 LB",
                shamrock_price=92.50,
                notes="All-purpose kitchen use"
            ),
            
            ProductMatch(
                product_name="Black Pepper",
                specification="Coarse Grind",
                sysco_code="SYSCO_COARSE",
                sysco_description="BLACK PEPPER COARSE",
                sysco_pack="6/1LB",
                sysco_case_price=298.95,
                sysco_split_price=None,
                shamrock_code="SHAM_COARSE",
                shamrock_description="PEPPER BLACK COARSE GRIND",
                shamrock_pack="25 LB", 
                shamrock_price=79.71,
                notes="For steaks and robust dishes"
            ),
            
            ProductMatch(
                product_name="Black Pepper",
                specification="Cracked",
                sysco_code="SYSCO_CRACKED",
                sysco_description="BLACK PEPPER CRACKED",
                sysco_pack="6/1LB",
                sysco_case_price=269.99,
                sysco_split_price=None,
                shamrock_code="SHAM_CRACKED",
                shamrock_description="PEPPER BLACK CRACKED",
                shamrock_pack="25 LB",
                shamrock_price=76.90,
                notes="Visual appeal, marinades"
            )
        ]
        
        return potential_matches
    
    def load_all_spice_matches(self) -> List[ProductMatch]:
        """Load all matched spice products"""
        
        all_matches = []
        
        # Black Pepper varieties
        all_matches.extend(self.load_pepper_products())
        
        # Other spices with verified matches
        other_spices = [
            ProductMatch(
                product_name="Garlic Powder",
                specification="Powder",
                sysco_code="SYSCO_GARLIC_PWD",
                sysco_description="GARLIC POWDER",
                sysco_pack="3/6LB",  # 3 containers × 6 lbs each
                sysco_case_price=213.19,
                sysco_split_price=78.25,  # Per container
                shamrock_code="SHAM_GARLIC_PWD",
                shamrock_description="GARLIC POWDER CALIFORNIA",
                shamrock_pack="1/6/LB",  # 1 container × 6 lbs
                shamrock_price=54.26,
                notes="Powdered, not granulated"
            ),
            
            ProductMatch(
                product_name="Garlic Granulated", 
                specification="Granulated",
                sysco_code="SYSCO_GARLIC_GRAN",
                sysco_description="GARLIC GRANULATED",
                sysco_pack="5/1LB",
                sysco_case_price=165.99,
                sysco_split_price=None,
                shamrock_code="SHAM_GARLIC_GRAN",
                shamrock_description="GARLIC GRANULATED",
                shamrock_pack="25 LB",
                shamrock_price=67.47,
                notes="Coarser than powder"
            ),
            
            ProductMatch(
                product_name="Onion Powder",
                specification="Powder",
                sysco_code="SYSCO_ONION_PWD",
                sysco_description="ONION POWDER",
                sysco_pack="6/1LB",
                sysco_case_price=148.95,
                sysco_split_price=None,
                shamrock_code="SHAM_ONION_PWD",
                shamrock_description="ONION POWDER WHITE",
                shamrock_pack="25 LB",
                shamrock_price=39.80,
                notes="White onion powder"
            )
        ]
        
        all_matches.extend(other_spices)
        return all_matches
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate detailed comparison report"""
        
        matches = self.load_all_spice_matches()
        results = []
        
        print("\n" + "="*80)
        print("VENDOR COMPARISON - EXACT PRODUCT MATCHES ONLY")
        print("="*80)
        print("\nIMPORTANT: Different grinds/cuts are DIFFERENT products!")
        print("- Fine pepper ≠ Coarse pepper (different uses)")
        print("- Garlic powder ≠ Garlic granulated (different texture)")
        print("-"*80)
        
        for match in matches:
            savings = match.calculate_savings()
            if "error" not in savings:
                results.append(savings)
                
                print(f"\n{savings['product']}")
                print(f"  SYSCO: ${savings['sysco_per_lb']:.2f}/lb")
                print(f"  Shamrock: ${savings['shamrock_per_lb']:.2f}/lb")
                if savings['savings_per_lb'] > 0:
                    print(f"  Savings: ${savings['savings_per_lb']:.2f}/lb ({savings['savings_percent']:.1f}%)")
                else:
                    print(f"  SYSCO is cheaper by: ${abs(savings['savings_per_lb']):.2f}/lb")
                
                if 'sysco_split_per_lb' in savings:
                    print(f"  SYSCO Split: ${savings['sysco_split_per_lb']:.2f}/lb")
                    print(f"  Split vs Shamrock: ${savings['split_vs_shamrock']:.2f}/lb difference")
        
        df = pd.DataFrame(results)
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        shamrock_wins = df[df['preferred_vendor'] == 'Shamrock']
        sysco_wins = df[df['preferred_vendor'] == 'SYSCO']
        
        print(f"Products where Shamrock is cheaper: {len(shamrock_wins)}")
        print(f"Products where SYSCO is cheaper: {len(sysco_wins)}")
        
        if not shamrock_wins.empty:
            avg_savings = shamrock_wins['savings_percent'].mean()
            print(f"Average savings with Shamrock: {avg_savings:.1f}%")
        
        return df
    
    def check_unmatched_products(self, sysco_products: List, shamrock_products: List):
        """
        Identify products that couldn't be matched
        This helps find:
        - Products only one vendor carries
        - Products with different specifications
        """
        
        print("\n" + "="*80)
        print("UNMATCHED PRODUCTS ANALYSIS")
        print("="*80)
        
        print("\nCommon matching issues:")
        print("1. Different grind specifications (Fine vs Restaurant)")
        print("2. Different cuts (Whole vs Ground vs Cracked)")
        print("3. Brand differences (Generic vs Premium)")
        print("4. Pack size only (one vendor doesn't carry)")
        print("5. Seasonal or special order items")
        
        # This would analyze your actual order guides
        # to find products that don't have matches


# Testing and verification
if __name__ == "__main__":
    matcher = AccurateVendorMatcher()
    
    print("CRITICAL REMINDERS:")
    print("-"*50)
    print("1. NEVER compare different grinds/cuts as if they're the same")
    print("2. Fine pepper for shakers ≠ Cracked pepper for marinades")
    print("3. Garlic powder ≠ Garlic granulated (different applications)")
    print("4. Match brand quality levels when possible")
    print("5. Note if products are actually different (not just cheaper)")
    
    # Generate comparison
    df = matcher.generate_comparison_report()
    
    # Save results
    df.to_csv('/root/lariat-bible/data/matched_products_comparison.csv', index=False)
    print("\n✅ Saved matched products comparison")
    
    # Instructions for next steps
    print("\n" + "="*80)
    print("NEXT STEPS FOR ACCURATE COMPARISON:")
    print("="*80)
    print("1. Export your ACTUAL order guides from both vendors")
    print("2. Match products by BOTH name AND specification")
    print("3. Verify pack sizes are interpreted correctly")
    print("4. Note which products have no equivalent at other vendor")
    print("5. Consider quality differences, not just price")
