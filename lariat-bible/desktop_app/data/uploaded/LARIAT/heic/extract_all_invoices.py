#!/usr/bin/env python3
"""
ALL VENDORS INVOICE EXTRACTOR
Extracts data from BOTH Shamrock AND Sysco invoices

Save to: Desktop/LARIAT/COMBO O.H.
Run: python3 extract_all_invoices.py

Shamrock: IMG_3527-3575
Sysco: IMG_3576-3663
"""

import subprocess
import re
from pathlib import Path
import pandas as pd
from datetime import datetime

print("=" * 80)
print("LARIAT INVOICE EXTRACTOR - ALL VENDORS")
print("=" * 80)

# Get current directory
current_dir = Path.cwd()
print(f"\nScript location: {current_dir}")

# Look for images in subfolders: heic, jpeg, png
all_images = []

# Search all subfolders
for subfolder in ['heic', 'jpeg', 'png']:
    folder_path = current_dir / subfolder
    if folder_path.exists():
        print(f"Searching: {subfolder}/ folder")
        for ext in ['*.HEIC', '*.heic', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
            all_images.extend(list(folder_path.glob(f"IMG_{ext}")))

# Also check main folder
for ext in ['*.HEIC', '*.heic', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
    all_images.extend(list(current_dir.glob(f"IMG_{ext}")))

if not all_images:
    print("\n❌ ERROR: No invoice photos found!")
    print("\nMake sure this script is in the COMBO O.H. folder")
    exit(1)

print(f"\n✓ Found {len(all_images)} total invoice photos")

# Separate Shamrock and Sysco
shamrock_images = []
sysco_images = []

for img in all_images:
    try:
        img_number = int(img.stem.split('_')[1])
        if 3527 <= img_number <= 3575:
            shamrock_images.append(img)
        elif img_number >= 3576:
            sysco_images.append(img)
    except:
        continue

shamrock_images.sort()
sysco_images.sort()

print(f"  Shamrock invoices: {len(shamrock_images)} (IMG_3527-3575)")
print(f"  Sysco invoices: {len(sysco_images)} (IMG_3576+)")

total_to_process = len(shamrock_images) + len(sysco_images)

if total_to_process == 0:
    print("\n❌ No invoices found in expected range!")
    exit(1)

print(f"\n🚀 Processing {total_to_process} invoices total...")
print("This will take several minutes...\n")

# Function to extract invoice data
def extract_invoice(img, vendor):
    """Extract data from a single invoice"""
    try:
        # Convert HEIC to JPG if needed
        if img.suffix.lower() in ['.heic']:
            jpg_path = img.with_suffix('.jpg')
            subprocess.run(
                ['sips', '-s', 'format', 'jpeg', str(img), '--out', str(jpg_path)],
                capture_output=True,
                timeout=30
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
        
        if result.returncode != 0 or len(result.stdout.strip()) < 100:
            return None
        
        text = result.stdout
        text_upper = text.upper()
        
        # Extract data
        data = {
            'Vendor': vendor,
            'Filename': img.name,
            'Folder': img.parent.name,
            'Invoice_Number': None,
            'Date': None,
            'Total': None,
            'Subtotal': None,
            'Tax': None
        }
        
        # Invoice number
        inv_match = re.search(r'(?:INVOICE|ORDER)\s*#?\s*:?\s*(\d{6,})', text_upper)
        if inv_match:
            data['Invoice_Number'] = inv_match.group(1)
        
        # Date
        date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', text)
        if date_match:
            data['Date'] = date_match.group(1)
        
        # Total
        total_patterns = [
            r'(?:TOTAL|INVOICE\s+TOTAL|AMOUNT\s+DUE)\s*:?\s*\$?\s*([\d,]+\.\d{2})',
            r'(?:GRAND\s+TOTAL)\s*:?\s*\$?\s*([\d,]+\.\d{2})'
        ]
        for pattern in total_patterns:
            match = re.search(pattern, text_upper)
            if match:
                data['Total'] = float(match.group(1).replace(',', ''))
                break
        
        # Subtotal
        sub_match = re.search(r'(?:SUB\s*TOTAL|SUBTOTAL)\s*:?\s*\$?\s*([\d,]+\.\d{2})', text_upper)
        if sub_match:
            data['Subtotal'] = float(sub_match.group(1).replace(',', ''))
        
        # Tax
        tax_match = re.search(r'(?:TAX|SALES\s+TAX)\s*:?\s*\$?\s*([\d,]+\.\d{2})', text_upper)
        if tax_match:
            data['Tax'] = float(tax_match.group(1).replace(',', ''))
        
        return data
        
    except Exception as e:
        return None

# Process Shamrock invoices
print("=" * 80)
print("PROCESSING SHAMROCK INVOICES")
print("=" * 80 + "\n")

shamrock_results = []
for idx, img in enumerate(shamrock_images, 1):
    print(f"[{idx}/{len(shamrock_images)}] {img.name}...", end=" ", flush=True)
    
    data = extract_invoice(img, 'Shamrock')
    
    if data:
        shamrock_results.append(data)
        status = "✓"
        if data['Invoice_Number']:
            status += f" #{data['Invoice_Number']}"
        if data['Total']:
            status += f" ${data['Total']:,.2f}"
        print(status)
    else:
        print("FAILED")

# Process Sysco invoices
print("\n" + "=" * 80)
print("PROCESSING SYSCO INVOICES")
print("=" * 80 + "\n")

sysco_results = []
for idx, img in enumerate(sysco_images, 1):
    print(f"[{idx}/{len(sysco_images)}] {img.name}...", end=" ", flush=True)
    
    data = extract_invoice(img, 'Sysco')
    
    if data:
        sysco_results.append(data)
        status = "✓"
        if data['Invoice_Number']:
            status += f" #{data['Invoice_Number']}"
        if data['Total']:
            status += f" ${data['Total']:,.2f}"
        print(status)
    else:
        print("FAILED")

# Save to Excel
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = current_dir / f'ALL_INVOICES_{timestamp}.xlsx'

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Shamrock sheet
    if shamrock_results:
        df_shamrock = pd.DataFrame(shamrock_results)
        df_shamrock.to_excel(writer, sheet_name='Shamrock', index=False)
        print(f"\n✓ Shamrock: {len(df_shamrock)} invoices saved")
    
    # Sysco sheet
    if sysco_results:
        df_sysco = pd.DataFrame(sysco_results)
        df_sysco.to_excel(writer, sheet_name='Sysco', index=False)
        print(f"✓ Sysco: {len(df_sysco)} invoices saved")
    
    # Combined sheet
    all_results = shamrock_results + sysco_results
    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all.to_excel(writer, sheet_name='All_Invoices', index=False)
        print(f"✓ Combined: {len(df_all)} invoices saved")

print(f"\n{'='*80}")
print("✅ SUCCESS!")
print(f"{'='*80}")
print(f"\nFile saved: {output_file.name}")
print(f"Full path: {output_file}")

# Print summary statistics
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

if shamrock_results:
    df_sham = pd.DataFrame(shamrock_results)
    print(f"\n📦 SHAMROCK:")
    print(f"  Total invoices: {len(df_sham)}")
    print(f"  With invoice numbers: {df_sham['Invoice_Number'].notna().sum()}")
    print(f"  With dates: {df_sham['Date'].notna().sum()}")
    print(f"  With totals: {df_sham['Total'].notna().sum()}")
    if df_sham['Total'].notna().sum() > 0:
        print(f"  Total value: ${df_sham['Total'].sum():,.2f}")
        print(f"  Average invoice: ${df_sham['Total'].mean():,.2f}")

if sysco_results:
    df_sys = pd.DataFrame(sysco_results)
    print(f"\n🚚 SYSCO:")
    print(f"  Total invoices: {len(df_sys)}")
    print(f"  With invoice numbers: {df_sys['Invoice_Number'].notna().sum()}")
    print(f"  With dates: {df_sys['Date'].notna().sum()}")
    print(f"  With totals: {df_sys['Total'].notna().sum()}")
    if df_sys['Total'].notna().sum() > 0:
        print(f"  Total value: ${df_sys['Total'].sum():,.2f}")
        print(f"  Average invoice: ${df_sys['Total'].mean():,.2f}")

if shamrock_results and sysco_results:
    total_value = df_sham['Total'].sum() + df_sys['Total'].sum()
    print(f"\n💰 GRAND TOTAL: ${total_value:,.2f}")

print(f"\n{'='*80}")
print("Open the Excel file to see all your invoice data!")
print(f"{'='*80}")
