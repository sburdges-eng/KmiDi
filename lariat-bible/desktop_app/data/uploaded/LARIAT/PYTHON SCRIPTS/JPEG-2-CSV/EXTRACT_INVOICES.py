#!/usr/bin/env python3
"""
LARIAT INVOICE TEXT EXTRACTION SCRIPT
Run this directly on your Mac to extract text from all invoice photos

INSTRUCTIONS:
1. Open Terminal on your Mac
2. Run: python3 /Users/seanburdges/Desktop/LARIAT/EXTRACT_INVOICES.py

This script will:
- Find all 122+ invoice photos
- Convert HEIC to JPG
- Extract text using OCR
- Save results to Excel + JSON
"""

import subprocess
import os
import sys
from pathlib import Path
import re
from datetime import datetime
import json

print("""
================================================================================
LARIAT INVOICE TEXT EXTRACTION
================================================================================
This will process all your invoice photos and extract:
- Vendor names (Shamrock, Sysco, US Foods, etc.)
- Dates
- Invoice/Order numbers  
- Totals
- Line items
- Full text content

Press ENTER to start extraction...
""")
input()

class InvoiceExtractor:
    def __init__(self):
        self.results = []
        self.temp_dir = "/tmp/lariat_invoices_temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def get_all_photos(self):
        """Find all invoice photos"""
        photos = []
        
        # Desktop folder
        desktop_base = "/Users/seanburdges/Desktop/LARIAT/ORDER HIST. JUN-OCT 2025"
        if os.path.exists(desktop_base):
            for file in os.listdir(desktop_base):
                if file.startswith("IMG_") and file.lower().endswith(".heic"):
                    photos.append(os.path.join(desktop_base, file))
        
        # Downloads folder  
        downloads_base = "/Users/seanburdges/Downloads"
        if os.path.exists(downloads_base):
            for file in os.listdir(downloads_base):
                if file.startswith("IMG_"):
                    if file.lower().endswith((".heic", ".jpg", ".jpeg", ".png")):
                        try:
                            num = int(re.search(r'IMG_(\d+)', file).group(1))
                            if 3475 <= num <= 3700:  # Invoice range
                                photos.append(os.path.join(downloads_base, file))
                        except:
                            pass
        
        # Remove duplicates
        unique = {}
        for p in photos:
            name = os.path.basename(p)
            if name not in unique:
                unique[name] = p
        
        return sorted(unique.values())
    
    def convert_heic(self, heic_path):
        """Convert HEIC to JPG"""
        jpg_path = os.path.join(self.temp_dir, Path(heic_path).stem + ".jpg")
        
        try:
            result = subprocess.run(
                ["sips", "-s", "format", "jpeg", heic_path, "--out", jpg_path],
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0 and os.path.exists(jpg_path):
                return jpg_path
        except Exception as e:
            print(f"    ⚠️  Conversion error: {e}")
        
        return None
    
    def extract_text(self, image_path):
        """Extract text using Mac's built-in Vision OCR"""
        try:
            # First try using tesseract if available
            result = subprocess.run(
                ["which", "tesseract"],
                capture_output=True
            )
            
            if result.returncode == 0:
                result = subprocess.run(
                    ["tesseract", image_path, "stdout"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    return result.stdout
            
            # Fallback: Use Python's pytesseract
            try:
                import pytesseract
                from PIL import Image
                return pytesseract.image_to_string(Image.open(image_path))
            except:
                pass
                
            print("    ⚠️  OCR not available. Install: brew install tesseract")
            return ""
            
        except Exception as e:
            print(f"    ⚠️  OCR error: {e}")
            return ""
    
    def parse_invoice(self, text, filename):
        """Extract structured data from text"""
        data = {
            'filename': filename,
            'raw_text': text[:5000],  # Limit text length
            'vendor': None,
            'date': None,
            'invoice_num': None,
            'total': None,
            'item_count': 0
        }
        
        # Vendor detection
        text_upper = text.upper()
        if 'SHAMROCK' in text_upper:
            data['vendor'] = 'Shamrock'
        elif 'SYSCO' in text_upper:
            data['vendor'] = 'Sysco'
        elif 'US FOODS' in text_upper or 'USFOODS' in text_upper:
            data['vendor'] = 'US Foods'
        
        # Date patterns
        date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text)
        if date_match:
            data['date'] = date_match.group(1)
        
        # Invoice number
        inv_match = re.search(r'(?:Invoice|Order|#)\s*[:#]?\s*(\d+)', text, re.IGNORECASE)
        if inv_match:
            data['invoice_num'] = inv_match.group(1)
        
        # Total
        total_match = re.search(r'(?:Total|Amount)[:\s]*\$?\s*([\d,]+\.\d{2})', text, re.IGNORECASE)
        if total_match:
            data['total'] = total_match.group(1).replace(',', '')
        
        # Count line items (rough estimate)
        lines = [l for l in text.split('\n') if re.search(r'\$\s*\d+\.\d{2}', l)]
        data['item_count'] = len(lines)
        
        return data
    
    def process_photo(self, photo_path, idx, total):
        """Process one photo"""
        filename = os.path.basename(photo_path)
        print(f"\n[{idx}/{total}] {filename}")
        
        # Convert if HEIC
        if photo_path.lower().endswith('.heic'):
            print("    Converting HEIC→JPG...")
            jpg_path = self.convert_heic(photo_path)
            if not jpg_path:
                return None
            process_path = jpg_path
        else:
            process_path = photo_path
        
        # Extract text
        print("    Running OCR...")
        text = self.extract_text(process_path)
        
        if len(text.strip()) < 20:
            print("    ⚠️  No text found")
            return None
        
        print(f"    ✓ Extracted {len(text)} chars")
        
        # Parse
        data = self.parse_invoice(text, filename)
        
        if data['vendor']:
            print(f"    Vendor: {data['vendor']}")
        if data['date']:
            print(f"    Date: {data['date']}")
        if data['total']:
            print(f"    Total: ${data['total']}")
        
        return data
    
    def save_results(self):
        """Save to files"""
        if not self.results:
            print("\nNo results to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "/Users/seanburdges/Desktop/LARIAT"
        
        # Save JSON
        json_path = f"{output_dir}/INVOICE_DATA_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✅ Saved: {json_path}")
        
        # Save CSV
        csv_path = f"{output_dir}/INVOICE_DATA_{timestamp}.csv"
        with open(csv_path, 'w') as f:
            f.write("Filename,Vendor,Date,Invoice_Number,Total,Item_Count\n")
            for r in self.results:
                f.write(f"{r['filename']},{r.get('vendor','')},{r.get('date','')},{r.get('invoice_num','')},{r.get('total','')},{r.get('item_count','')}\n")
        print(f"✅ Saved: {csv_path}")
        
        # Stats
        print(f"\n{'='*80}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*80}")
        print(f"Total invoices: {len(self.results)}")
        
        vendors = {}
        for r in self.results:
            v = r.get('vendor')
            if v:
                vendors[v] = vendors.get(v, 0) + 1
        
        if vendors:
            print(f"\nVendors:")
            for vendor, count in sorted(vendors.items()):
                print(f"  {vendor}: {count}")
        
        totals = [float(r['total']) for r in self.results if r.get('total')]
        if totals:
            print(f"\nTotal value extracted: ${sum(totals):,.2f}")
    
    def run(self):
        """Main process"""
        photos = self.get_all_photos()
        
        if not photos:
            print("\n❌ No photos found!")
            print("\nChecked:")
            print("  - /Users/seanburdges/Desktop/LARIAT/ORDER HIST. JUN-OCT 2025")
            print("  - /Users/seanburdges/Downloads")
            return
        
        print(f"\nFound {len(photos)} invoice photos\n")
        print("="*80)
        
        for idx, photo in enumerate(photos, 1):
            try:
                result = self.process_photo(photo, idx, len(photos))
                if result:
                    self.results.append(result)
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
        
        print("\n" + "="*80)
        self.save_results()

if __name__ == '__main__':
    try:
        extractor = InvoiceExtractor()
        extractor.run()
        print("\n✅ DONE! Check your LARIAT folder for the results.\n")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
