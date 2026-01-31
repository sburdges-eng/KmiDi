#!/usr/bin/env python3
"""
SYSCO INVOICE EXTRACTOR
Save this file to: Desktop/LARIAT/ORDER HIST. JUN-OCT 2025 COMBO O.H.
Then run: python3 extract_sysco.py
"""

import subprocess
import re
from pathlib import Path
import pandas as pd
from datetime import datetime

print("=" * 80)
print("SYSCO INVOICE EXTRACTOR")
print("=" * 80)

# Get current directory (where this script is saved)
current_dir = Path.cwd()
print(f"\nLooking for invoices in: {current_dir}")

# Find all image files (HEIC, JPEG, PNG)
all_images = []
all_images.extend(list(current_dir.glob("IMG_*.HEIC")))
all_images.extend(list(current_dir.glob("IMG_*.heic")))
all_images.extend(list(current_dir.glob("IMG_*.jpg")))
all_images.extend(list(current_dir.glob("IMG_*.JPG")))
all_images.extend(list(current_dir.glob("IMG_*.jpeg")))
all_images.extend(list(current_dir.glob("IMG_*.JPEG")))
all_images.extend(list(current_dir.glob("IMG_*.png")))
all_images.extend(list(current_dir.glob("IMG_*.PNG")))

if not all_images:
    print("\nERROR: No invoice photos found in this folder!")
    print("Make sure you saved this script in the same folder as your photos.")
    exit(1)

print(f"Found {len(all_images)} total invoice photos")

# Filter for Sysco invoices (IMG_3576 and higher)
sysco_images = []
for img in all_images:
    try:
        img_number = int(img.stem.split('_')[1])
        if img_number >= 3576:
            sysco_images.append(img)
    except:
        continue

sysco_images.sort()

print(f"Found {len(sysco_images)} Sysco invoice photos (IMG_3576+)")

if len(sysco_images) == 0:
    print("\nNo Sysco invoices found (need IMG_3576 or higher)")
    exit(1)

print(f"\nProcessing {len(sysco_images)} invoices...\n")

results = []

for idx, img in enumerate(sysco_images, 1):
    print(f"[{idx}/{len(sysco_images)}] {img.name}...", end=" ")
    
    try:
        # If it's HEIC, convert to JPG first
        if img.suffix.lower() in ['.heic']:
            jpg_path = img.with_suffix('.jpg')
            subprocess.run(
                ['sips', '-s', 'format', 'jpeg', str(img), '--out', str(jpg_path)],
                capture_output=True
            )
            process_image = jpg_path
        else:
            process_image = img
        
        # Run OCR
        result = subprocess.run(
            ['tesseract', str(process_image), 'stdout'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print("OCR FAILED")
            continue
        
        text = result.stdout
        
        if len(text.strip()) < 100:
            print("NO TEXT")
            continue
        
        text_upper = text.upper()
        
        # Extract data
        data = {
            'Vendor': 'Sysco',
            'Filename': img.name,
            'Invoice_Number': None,
            'Date': None,
            'Total': None,
            'Subtotal': None,
            'Tax': None
        }
        
        # Get invoice number
        inv_match = re.search(r'(?:INVOICE|ORDER)\s*#?\s*:?\s*(\d{7,})', text_upper)
        if inv_match:
            data['Invoice_Number'] = inv_match.group(1)
        
        # Get date
        date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', text)
        if date_match:
            data['Date'] = date_match.group(1)
        
        # Get total
        total_match = re.search(r'(?:TOTAL|INVOICE\s+TOTAL)\s*:?\s*\$?\s*([\d,]+\.\d{2})', text_upper)
        if total_match:
            data['Total'] = float(total_match.group(1).replace(',', ''))
        
        # Get subtotal
        sub_match = re.search(r'SUBTOTAL\s*:?\s*\$?\s*([\d,]+\.\d{2})', text_upper)
        if sub_match:
            data['Subtotal'] = float(sub_match.group(1).replace(',', ''))
        
        # Get tax
        tax_match = re.search(r'TAX\s*:?\s*\$?\s*([\d,]+\.\d{2})', text_upper)
        if tax_match:
            data['Tax'] = float(tax_match.group(1).replace(',', ''))
        
        results.append(data)
        
        # Print status
        status = "✓"
        if data['Invoice_Number']:
            status += f" #{data['Invoice_Number']}"
        if data['Total']:
            status += f" ${data['Total']:,.2f}"
        print(status)
        
    except Exception as e:
        print(f"ERROR: {e}")

if not results:
    print("\n❌ No data extracted!")
    print("Your photos may be too low quality for OCR.")
    print("Try using the manual template instead: SYSCO_DATA_SIMPLE.xlsx")
    exit(1)

# Save to Excel in the same folder
df = pd.DataFrame(results)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = current_dir / f'SYSCO_INVOICES_{timestamp}.xlsx'

df.to_excel(output_file, index=False, sheet_name='Sysco_Invoices')

print(f"\n{'='*80}")
print("✅ SUCCESS!")
print(f"{'='*80}")
print(f"\nExtracted {len(results)} invoices")
print(f"Saved to: {output_file.name}")
print(f"Full path: {output_file}")

# Print summary
invoices_with_numbers = df['Invoice_Number'].notna().sum()
invoices_with_dates = df['Date'].notna().sum()
invoices_with_totals = df['Total'].notna().sum()

print(f"\nData Quality:")
print(f"  Invoices with numbers: {invoices_with_numbers}/{len(df)}")
print(f"  Invoices with dates: {invoices_with_dates}/{len(df)}")
print(f"  Invoices with totals: {invoices_with_totals}/{len(df)}")

if df['Total'].notna().sum() > 0:
    print(f"\nFinancial Summary:")
    print(f"  Total Value: ${df['Total'].sum():,.2f}")
    print(f"  Average Invoice: ${df['Total'].mean():,.2f}")
    print(f"  Highest: ${df['Total'].max():,.2f}")
    print(f"  Lowest: ${df['Total'].min():,.2f}")

print(f"\n{'='*80}")
print("Next step: Open the Excel file to review the data!")
print(f"{'='*80}")
