"""
Email Order Confirmation Parser
Parses SYSCO and Shamrock order confirmation emails
"""

import re
import email
import imaplib
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from dataclasses import dataclass

@dataclass
class OrderItem:
    """Single line item from order confirmation"""
    vendor: str
    order_number: str
    date: datetime
    item_code: str
    description: str
    pack_size: str
    quantity_ordered: float
    unit_price: float
    extension: float
    
    @property
    def normalized_unit_price(self) -> float:
        """Convert to price per unit based on pack size"""
        return normalize_pack_size(self.pack_size, self.unit_price)


class PackSizeNormalizer:
    """Handles pack size interpretation and normalization"""
    
    # Standard can sizes in ounces
    CAN_SIZES = {
        '#10': 109,      # 109 oz (6.375" x 7")
        '#5': 56,        # 56 oz
        '#2.5': 28,      # 28 oz
        '#2': 20,        # 20 oz
        '#300': 15,      # 15 oz
        '#303': 16,      # 16 oz
    }
    
    @staticmethod
    def parse_pack_size(pack_str: str) -> Dict:
        """
        Parse pack size string into components
        Examples:
            '6/10#' -> 6 containers × 10 pounds
            '6/#10' -> 6 × #10 cans (109 oz each)
            '4/1 GAL' -> 4 × 1 gallon
            '25 LB' -> 25 pounds
        """
        pack_str = pack_str.upper().strip()
        
        # Check for can sizes first
        for can_size, ounces in PackSizeNormalizer.CAN_SIZES.items():
            if can_size in pack_str:
                # Found a can size
                match = re.match(r'(\d+)\s*/\s*' + re.escape(can_size), pack_str)
                if match:
                    count = int(match.group(1))
                    return {
                        'count': count,
                        'size': ounces,
                        'unit': 'OZ',
                        'total_ounces': count * ounces,
                        'total_pounds': (count * ounces) / 16
                    }
        
        # Check for X/Y# pattern (pounds)
        pound_match = re.match(r'(\d+)\s*/\s*(\d+)\s*#', pack_str)
        if pound_match:
            count = int(pound_match.group(1))
            pounds = int(pound_match.group(2))
            return {
                'count': count,
                'size': pounds,
                'unit': 'LB',
                'total_ounces': count * pounds * 16,
                'total_pounds': count * pounds
            }
        
        # Check for simple pounds
        lb_match = re.match(r'(\d+)\s*LB', pack_str)
        if lb_match:
            pounds = int(lb_match.group(1))
            return {
                'count': 1,
                'size': pounds,
                'unit': 'LB',
                'total_ounces': pounds * 16,
                'total_pounds': pounds
            }
        
        # Check for gallons
        gal_match = re.match(r'(\d+)\s*/\s*(\d+)\s*GAL', pack_str)
        if gal_match:
            count = int(gal_match.group(1))
            gallons = int(gal_match.group(2))
            return {
                'count': count,
                'size': gallons,
                'unit': 'GAL',
                'total_ounces': count * gallons * 128,
                'total_pounds': None  # Liquid measure
            }
        
        # Check for case/each
        case_match = re.match(r'(\d+)\s*/\s*(CS|CASE|EA|EACH)', pack_str)
        if case_match:
            count = int(case_match.group(1))
            return {
                'count': count,
                'size': 1,
                'unit': case_match.group(2),
                'total_ounces': None,
                'total_pounds': None
            }
        
        # Default - couldn't parse
        return {
            'count': 1,
            'size': 1,
            'unit': 'UNKNOWN',
            'total_ounces': None,
            'total_pounds': None,
            'original': pack_str
        }
    
    @staticmethod
    def normalize_to_price_per_pound(pack_str: str, case_price: float) -> Optional[float]:
        """Convert any pack size to price per pound"""
        parsed = PackSizeNormalizer.parse_pack_size(pack_str)
        
        if parsed['total_pounds']:
            return case_price / parsed['total_pounds']
        elif parsed['total_ounces']:
            return case_price / (parsed['total_ounces'] / 16)
        else:
            return None  # Can't convert to per pound


class EmailOrderParser:
    """Parse order confirmations from email"""
    
    def __init__(self, email_address: str, password: str, imap_server: str = "imap.gmail.com"):
        self.email_address = email_address
        self.password = password
        self.imap_server = imap_server
        self.normalizer = PackSizeNormalizer()
    
    def connect(self):
        """Connect to email server"""
        self.mail = imaplib.IMAP4_SSL(self.imap_server)
        self.mail.login(self.email_address, self.password)
        self.mail.select('inbox')
    
    def parse_sysco_email(self, email_body: str) -> List[OrderItem]:
        """Parse SYSCO order confirmation email"""
        items = []
        
        # SYSCO format patterns
        # Look for lines like: "123456  PEPPER BLACK GROUND  6/1#  2  $45.99  $91.98"
        pattern = r'(\d{6,7})\s+(.+?)\s+(\S+)\s+(\d+)\s+\$?([\d.]+)\s+\$?([\d.]+)'
        
        for match in re.finditer(pattern, email_body):
            item = OrderItem(
                vendor='SYSCO',
                order_number=self._extract_order_number(email_body, 'SYSCO'),
                date=datetime.now(),
                item_code=match.group(1),
                description=match.group(2).strip(),
                pack_size=match.group(3),
                quantity_ordered=float(match.group(4)),
                unit_price=float(match.group(5)),
                extension=float(match.group(6))
            )
            items.append(item)
        
        return items
    
    def parse_shamrock_email(self, email_body: str) -> List[OrderItem]:
        """Parse Shamrock Foods order confirmation email"""
        items = []
        
        # Shamrock format patterns
        # Adjust based on actual Shamrock email format
        pattern = r'(\d+)\s+(.+?)\s+(\S+)\s+(\d+)\s+\$?([\d.]+)\s+\$?([\d.]+)'
        
        for match in re.finditer(pattern, email_body):
            item = OrderItem(
                vendor='Shamrock Foods',
                order_number=self._extract_order_number(email_body, 'Shamrock'),
                date=datetime.now(),
                item_code=match.group(1),
                description=match.group(2).strip(),
                pack_size=match.group(3),
                quantity_ordered=float(match.group(4)),
                unit_price=float(match.group(5)),
                extension=float(match.group(6))
            )
            items.append(item)
        
        return items
    
    def _extract_order_number(self, email_body: str, vendor: str) -> str:
        """Extract order number from email"""
        if vendor == 'SYSCO':
            match = re.search(r'Order\s*#?\s*:?\s*(\d+)', email_body, re.IGNORECASE)
        else:  # Shamrock
            match = re.search(r'Confirmation\s*#?\s*:?\s*(\d+)', email_body, re.IGNORECASE)
        
        return match.group(1) if match else 'UNKNOWN'
    
    def fetch_recent_orders(self, days_back: int = 7) -> pd.DataFrame:
        """Fetch and parse recent order confirmation emails"""
        from datetime import timedelta
        
        # Search for emails from vendors
        date_criteria = (datetime.now() - timedelta(days=days_back)).strftime("%d-%b-%Y")
        
        all_items = []
        
        # Search SYSCO emails
        status, sysco_ids = self.mail.search(None, 
            f'(FROM "sysco.com" SINCE {date_criteria} SUBJECT "order")')
        
        if status == 'OK':
            for email_id in sysco_ids[0].split():
                status, data = self.mail.fetch(email_id, '(RFC822)')
                if status == 'OK':
                    email_body = data[0][1].decode('utf-8')
                    items = self.parse_sysco_email(email_body)
                    all_items.extend(items)
        
        # Search Shamrock emails
        status, shamrock_ids = self.mail.search(None,
            f'(FROM "shamrockfoods.com" SINCE {date_criteria} SUBJECT "confirmation")')
        
        if status == 'OK':
            for email_id in shamrock_ids[0].split():
                status, data = self.mail.fetch(email_id, '(RFC822)')
                if status == 'OK':
                    email_body = data[0][1].decode('utf-8')
                    items = self.parse_shamrock_email(email_body)
                    all_items.extend(items)
        
        # Convert to DataFrame
        if all_items:
            df = pd.DataFrame([vars(item) for item in all_items])
            
            # Add normalized price per pound
            df['price_per_pound'] = df.apply(
                lambda row: self.normalizer.normalize_to_price_per_pound(
                    row['pack_size'], row['unit_price']
                ), axis=1
            )
            
            return df
        else:
            return pd.DataFrame()
    
    def compare_vendor_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compare prices between vendors for matching items"""
        # Group by description (fuzzy matching would be better)
        comparison = []
        
        for desc in df['description'].unique():
            vendor_prices = df[df['description'] == desc].groupby('vendor').agg({
                'price_per_pound': 'mean',
                'unit_price': 'mean',
                'pack_size': 'first'
            })
            
            if len(vendor_prices) > 1 and 'price_per_pound' in vendor_prices.columns:
                sysco_price = vendor_prices.loc['SYSCO', 'price_per_pound'] if 'SYSCO' in vendor_prices.index else None
                shamrock_price = vendor_prices.loc['Shamrock Foods', 'price_per_pound'] if 'Shamrock Foods' in vendor_prices.index else None
                
                if sysco_price and shamrock_price:
                    comparison.append({
                        'description': desc,
                        'sysco_price_per_lb': sysco_price,
                        'shamrock_price_per_lb': shamrock_price,
                        'savings_per_lb': abs(sysco_price - shamrock_price),
                        'savings_percent': abs(sysco_price - shamrock_price) / max(sysco_price, shamrock_price) * 100,
                        'preferred_vendor': 'Shamrock Foods' if shamrock_price < sysco_price else 'SYSCO'
                    })
        
        return pd.DataFrame(comparison)


# Example usage
if __name__ == "__main__":
    # Test pack size parsing
    normalizer = PackSizeNormalizer()
    
    test_cases = [
        "6/10#",      # 6 × 10 pounds
        "6/#10",      # 6 × #10 cans
        "25 LB",      # 25 pounds
        "4/1 GAL",    # 4 × 1 gallon
        "12/CASE",    # 12 per case
    ]
    
    print("Pack Size Parsing Tests:")
    print("-" * 50)
    for pack in test_cases:
        result = normalizer.parse_pack_size(pack)
        print(f"{pack:15} -> {result}")
    
    # Test price normalization
    print("\nPrice Per Pound Calculations:")
    print("-" * 50)
    
    # Example: Black Pepper comparison (corrected)
    sysco_pepper = normalizer.normalize_to_price_per_pound("6/1#", 298.95)
    shamrock_pepper = normalizer.normalize_to_price_per_pound("25 LB", 79.71)
    
    print(f"SYSCO Black Pepper (6/1#) @ $298.95:")
    print(f"  = ${sysco_pepper:.2f} per pound")
    print(f"Shamrock Black Pepper (25 LB) @ $79.71:")
    print(f"  = ${shamrock_pepper:.2f} per pound")
    print(f"Actual savings: ${sysco_pepper - shamrock_pepper:.2f} per pound")
    print(f"Percentage difference: {((sysco_pepper - shamrock_pepper) / sysco_pepper * 100):.1f}%")
