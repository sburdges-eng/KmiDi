#!/usr/bin/env python3
"""Review script for RECOVERY_OPS deduplication implementation."""
import json
from pathlib import Path

output_dir = Path("docs/RECOVERY_OPS_DEDUPE")

# Load all data
with open(output_dir / "dedupe_statistics.json") as f:
    stats = json.load(f)

with open(output_dir / "preservation_map.json") as f:
    preservation = json.load(f)

with open(output_dir / "RECOVERY_OPS_DELETE_MANIFEST.json") as f:
    manifest = json.load(f)

with open(output_dir / "verification_results.json") as f:
    verification = json.load(f)

print("=" * 80)
print("RECOVERY_OPS Deduplication Implementation Review")
print("=" * 80)

print("\n1. ANALYSIS COMPLETENESS")
print("-" * 80)
print(f"✅ Duplicate groups identified: {stats['duplicate_groups']}")
print(f"✅ Total duplicate files: {stats['total_duplicate_files']}")
print(f"✅ Files to preserve: {stats['files_to_preserve']}")
print(f"✅ Files to delete: {stats['files_to_delete']}")
print(f"✅ Space savings: {stats['space_savings']['gb']:.2f} GB ({stats['space_savings']['mb']:.2f} MB)")

print("\n2. PRESERVATION STRATEGY")
print("-" * 80)
# Check preservation locations
preserved_locations = {}
for hash_val, info in preservation['preservation_map'].items():
    path = info['preserved_path']
    if '/CANONICAL/' in path:
        preserved_locations['CANONICAL'] = preserved_locations.get('CANONICAL', 0) + 1
    elif '/ARCHIVE/' in path:
        preserved_locations['ARCHIVE'] = preserved_locations.get('ARCHIVE', 0) + 1
    else:
        preserved_locations['Other'] = preserved_locations.get('Other', 0) + 1

print("Preserved files by location:")
for loc, count in sorted(preserved_locations.items(), key=lambda x: x[1], reverse=True):
    print(f"  {loc}: {count}")

print("\n3. VERIFICATION STATUS")
print("-" * 80)
print(f"✅ Files verified: {verification['verified_ok']}")
print(f"⚠️  Warnings: {len(verification['warnings'])}")
print(f"❌ Errors: {len(verification['errors'])}")
if verification['errors']:
    print("  ⚠️  CRITICAL: Errors found - deletion should not proceed")
else:
    print("  ✅ No errors - safe to proceed")

print("\n4. SAFETY CHECKS")
print("-" * 80)
safety_issues = []

# Check if any preserved files are in deletion list
preserved_paths = {info['preserved_path'] for info in preservation['preservation_map'].values()}
deletion_paths = {f['path'] for f in manifest['files']}
overlap = preserved_paths & deletion_paths

if overlap:
    safety_issues.append(f"CRITICAL: {len(overlap)} preserved files also in deletion list!")
else:
    print("✅ No preserved files in deletion list")

# Check if any files outside RECOVERY_OPS
outside_recovery = [f for f in manifest['files'] if not f['path'].startswith('RECOVERY_OPS/')]
if outside_recovery:
    safety_issues.append(f"CRITICAL: {len(outside_recovery)} files outside RECOVERY_OPS!")
else:
    print("✅ All files are within RECOVERY_OPS")

# Check vendor files
vendor_count = sum(1 for f in manifest['files'] 
                   if any(v in f['path'].lower() for v in ['venv', 'site-packages', 'node_modules']))
if vendor_count > 0:
    print(f"⚠️  Warning: {vendor_count} vendor files in deletion list (should be excluded)")
else:
    print("✅ No vendor files in deletion list")

if safety_issues:
    print("\n❌ SAFETY ISSUES FOUND:")
    for issue in safety_issues:
        print(f"  - {issue}")
else:
    print("\n✅ All safety checks passed")

print("\n5. FILE BREAKDOWN")
print("-" * 80)
print("Files to delete by location:")
for loc, count in sorted(stats['by_location_deletion'].items(), 
                        key=lambda x: x[1], reverse=True):
    print(f"  {loc}: {count:,} files")

print("\n6. RECOMMENDATIONS")
print("-" * 80)
if len(verification['errors']) == 0 and len(safety_issues) == 0:
    print("✅ Implementation is safe and ready for use")
    print("✅ All verification checks passed")
    print("✅ No safety issues detected")
    print("\nNext steps:")
    print("  1. Review RECOVERY_OPS_DEDUPE_REPORT.md")
    print("  2. Review DRY_RUN_REPORT.md")
    print("  3. When ready: python3 scripts/recovery_ops/safe_dedupe.py --live")
else:
    print("⚠️  Issues found - review before proceeding")
