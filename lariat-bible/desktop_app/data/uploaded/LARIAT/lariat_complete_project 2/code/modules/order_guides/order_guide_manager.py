"""
Order Guide Module
Manages vendor catalogs and price comparisons
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

class OrderGuideManager:
    """Manage and compare order guides from different vendors"""
    
    def __init__(self):
        self.sysco_catalog = {}
        self.shamrock_catalog = {}
        self.comparison_results = []
        self.last_updated = {
            'sysco': None,
            'shamrock': None
        }
    
    def load_sysco_guide(self, data: List[Dict]) -> int:
        """
        Load SYSCO order guide data
        
        Expected format for each item:
        {
            'item_code': 'SYS123456',
            'description': 'BEEF GROUND 80/20 FRESH',
            'pack_size': '10 LB',
            'case_price': 45.99,
            'unit_price': 4.599,  # Price per pound
            'unit': 'LB',
            'category': 'MEAT'
        }
        """
        count = 0
        for item in data:
            self.sysco_catalog[item['item_code']] = {
                'vendor': 'SYSCO',
                'description': item['description'],
                'pack_size': item['pack_size'],
                'case_price': item['case_price'],
                'unit_price': item.get('unit_price', 0),
                'unit': item.get('unit', 'EACH'),
                'category': item.get('category', 'UNCATEGORIZED'),
                'last_updated': datetime.now()
            }
            count += 1
        
        self.last_updated['sysco'] = datetime.now()
        return count
    
    def load_shamrock_guide(self, data: List[Dict]) -> int:
        """
        Load Shamrock Foods order guide data
        
        Expected format matches SYSCO for consistency
        """
        count = 0
        for item in data:
            self.shamrock_catalog[item['item_code']] = {
                'vendor': 'Shamrock Foods',
                'description': item['description'],
                'pack_size': item['pack_size'],
                'case_price': item['case_price'],
                'unit_price': item.get('unit_price', 0),
                'unit': item.get('unit', 'EACH'),
                'category': item.get('category', 'UNCATEGORIZED'),
                'last_updated': datetime.now()
            }
            count += 1
        
        self.last_updated['shamrock'] = datetime.now()
        return count
    
    def find_matching_products(self, threshold: float = 0.8) -> List[Dict]:
        """
        Find products that appear in both catalogs
        Uses fuzzy matching on descriptions
        """
        matches = []
        
        # Simple matching based on description similarity
        # In production, you'd want fuzzy matching with libraries like fuzzywuzzy
        for sys_code, sys_item in self.sysco_catalog.items():
            sys_desc = sys_item['description'].lower()
            
            for sham_code, sham_item in self.shamrock_catalog.items():
                sham_desc = sham_item['description'].lower()
                
                # Basic matching - check if key words match
                sys_words = set(sys_desc.split())
                sham_words = set(sham_desc.split())
                
                # Calculate similarity
                intersection = sys_words & sham_words
                union = sys_words | sham_words
                
                if union:
                    similarity = len(intersection) / len(union)
                    
                    if similarity >= threshold:
                        matches.append({
                            'sysco_code': sys_code,
                            'sysco_description': sys_item['description'],
                            'shamrock_code': sham_code,
                            'shamrock_description': sham_item['description'],
                            'similarity_score': similarity
                        })
        
        return matches
    
    def compare_prices(self, matched_products: List[Dict] = None) -> pd.DataFrame:
        """
        Compare prices between matched products
        Returns a DataFrame with comparison results
        """
        if matched_products is None:
            matched_products = self.find_matching_products()
        
        comparisons = []
        
        for match in matched_products:
            sys_item = self.sysco_catalog[match['sysco_code']]
            sham_item = self.shamrock_catalog[match['shamrock_code']]
            
            # Calculate price differences
            case_diff = sys_item['case_price'] - sham_item['case_price']
            case_diff_pct = (case_diff / sys_item['case_price']) * 100 if sys_item['case_price'] > 0 else 0
            
            unit_diff = sys_item['unit_price'] - sham_item['unit_price']
            unit_diff_pct = (unit_diff / sys_item['unit_price']) * 100 if sys_item['unit_price'] > 0 else 0
            
            comparisons.append({
                'description': sys_item['description'],
                'category': sys_item['category'],
                'sysco_code': match['sysco_code'],
                'sysco_case_price': sys_item['case_price'],
                'sysco_unit_price': sys_item['unit_price'],
                'shamrock_code': match['shamrock_code'],
                'shamrock_case_price': sham_item['case_price'],
                'shamrock_unit_price': sham_item['unit_price'],
                'case_savings': abs(case_diff),
                'case_savings_pct': abs(case_diff_pct),
                'unit_savings': abs(unit_diff),
                'unit_savings_pct': abs(unit_diff_pct),
                'preferred_vendor': 'Shamrock Foods' if case_diff > 0 else 'SYSCO',
                'pack_size': sys_item['pack_size']
            })
        
        df = pd.DataFrame(comparisons)
        
        # Sort by savings potential
        if not df.empty:
            df = df.sort_values('case_savings', ascending=False)
        
        return df
    
    def get_category_analysis(self) -> Dict[str, Dict]:
        """
        Analyze pricing by category
        """
        df = self.compare_prices()
        
        if df.empty:
            return {}
        
        category_analysis = {}
        
        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            
            category_analysis[category] = {
                'total_items': len(cat_df),
                'shamrock_wins': len(cat_df[cat_df['preferred_vendor'] == 'Shamrock Foods']),
                'sysco_wins': len(cat_df[cat_df['preferred_vendor'] == 'SYSCO']),
                'avg_savings_pct': cat_df['case_savings_pct'].mean(),
                'total_potential_savings': cat_df['case_savings'].sum(),
                'top_savings_item': {
                    'description': cat_df.iloc[0]['description'] if not cat_df.empty else None,
                    'savings': cat_df.iloc[0]['case_savings'] if not cat_df.empty else 0
                }
            }
        
        return category_analysis
    
    def generate_purchase_recommendation(self, weekly_usage: Dict[str, float] = None) -> Dict:
        """
        Generate purchasing recommendations based on price comparisons
        
        Args:
            weekly_usage: Dict mapping item descriptions to weekly usage amounts
        """
        df = self.compare_prices()
        
        if df.empty:
            return {'error': 'No comparison data available'}
        
        recommendations = {
            'summary': {
                'total_items_compared': len(df),
                'shamrock_preferred': len(df[df['preferred_vendor'] == 'Shamrock Foods']),
                'sysco_preferred': len(df[df['preferred_vendor'] == 'SYSCO']),
            },
            'top_10_savings': [],
            'category_recommendations': self.get_category_analysis(),
            'estimated_monthly_savings': 0
        }
        
        # Get top 10 items with biggest savings
        top_10 = df.nlargest(10, 'case_savings')
        for _, row in top_10.iterrows():
            recommendations['top_10_savings'].append({
                'item': row['description'],
                'preferred_vendor': row['preferred_vendor'],
                'savings_per_case': row['case_savings'],
                'savings_percentage': row['case_savings_pct']
            })
        
        # Calculate estimated monthly savings if usage data provided
        if weekly_usage:
            monthly_savings = 0
            for desc, weekly_qty in weekly_usage.items():
                matching = df[df['description'].str.contains(desc, case=False)]
                if not matching.empty:
                    row = matching.iloc[0]
                    monthly_qty = weekly_qty * 4.33  # Average weeks per month
                    monthly_savings += row['case_savings'] * monthly_qty
            
            recommendations['estimated_monthly_savings'] = monthly_savings
        
        return recommendations
    
    def export_comparison(self, filepath: str = 'price_comparison.xlsx') -> str:
        """
        Export price comparison to Excel file
        """
        df = self.compare_prices()
        
        if df.empty:
            return "No data to export"
        
        # Create Excel writer with multiple sheets
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main comparison sheet
            df.to_excel(writer, sheet_name='Price Comparison', index=False)
            
            # Category analysis sheet
            cat_analysis = self.get_category_analysis()
            cat_df = pd.DataFrame.from_dict(cat_analysis, orient='index')
            cat_df.to_excel(writer, sheet_name='Category Analysis')
            
            # Top savings sheet
            top_savings = df.nlargest(20, 'case_savings')
            top_savings.to_excel(writer, sheet_name='Top 20 Savings', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Total Items Compared',
                    'Shamrock Better Price',
                    'SYSCO Better Price',
                    'Average Savings %',
                    'Total Potential Weekly Savings'
                ],
                'Value': [
                    len(df),
                    len(df[df['preferred_vendor'] == 'Shamrock Foods']),
                    len(df[df['preferred_vendor'] == 'SYSCO']),
                    f"{df['case_savings_pct'].mean():.2f}%",
                    f"${df['case_savings'].sum():.2f}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        return f"Comparison exported to {filepath}"
