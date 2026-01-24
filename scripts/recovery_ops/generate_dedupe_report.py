#!/usr/bin/env python3
"""
Generate comprehensive deduplication report for RECOVERY_OPS.
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def generate_dedupe_report():
    """Generate the deduplication report."""
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "docs" / "RECOVERY_OPS_DEDUPE"
    
    print("=" * 80)
    print("Generating RECOVERY_OPS Deduplication Report")
    print("=" * 80)
    
    # Load data
    with open(output_dir / "preservation_map.json") as f:
        preservation_data = json.load(f)
    
    with open(output_dir / "dedupe_statistics.json") as f:
        stats = json.load(f)
    
    with open(output_dir / "RECOVERY_OPS_DELETE_MANIFEST.json") as f:
        manifest = json.load(f)
    
    # Generate report
    report_path = output_dir / "RECOVERY_OPS_DEDUPE_REPORT.md"
    
    with open(report_path, 'w') as f:
        f.write("# RECOVERY_OPS Safe Deduplication Report\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"This report identifies **{stats['duplicate_groups']} duplicate groups** ")
        f.write(f"containing **{stats['total_duplicate_files']} KmiDi-related files** ")
        f.write(f"in the RECOVERY_OPS directory.\n\n")
        
        f.write(f"- **Files to preserve:** {stats['files_to_preserve']}\n")
        f.write(f"- **Files to delete:** {stats['files_to_delete']}\n")
        f.write(f"- **Estimated space savings:** {stats['space_savings']['gb']:.2f} GB ")
        f.write(f"({stats['space_savings']['mb']:.2f} MB)\n\n")
        
        # Statistics
        f.write("## Statistics\n\n")
        f.write("### Files by Location\n\n")
        f.write("| Location | Total Files | Files to Delete |\n")
        f.write("|----------|-------------|-----------------|\n")
        for location in sorted(stats['by_location'].keys()):
            total = stats['by_location'][location]
            to_delete = stats['by_location_deletion'].get(location, 0)
            f.write(f"| {location} | {total:,} | {to_delete:,} |\n")
        
        # Preservation Strategy
        f.write("\n## Preservation Strategy\n\n")
        f.write("Files are preserved based on the following priority:\n\n")
        f.write("1. **CANONICAL directories** - Highest priority (canonical copies)\n")
        f.write("2. **Most recently modified** - More current versions\n")
        f.write("3. **Shortest path** - Less nested locations\n")
        f.write("4. **Non-ARCHIVE directories** - Active files over archived\n\n")
        
        # Sample duplicate groups
        f.write("## Sample Duplicate Groups\n\n")
        f.write("### Top 20 Duplicate Groups by File Count\n\n")
        
        # Group deletion candidates by hash
        groups_by_hash = defaultdict(list)
        for file_info in manifest['files']:
            groups_by_hash[file_info['hash']].append(file_info)
        
        # Sort by group size
        sorted_groups = sorted(groups_by_hash.items(), 
                              key=lambda x: len(x[1]), reverse=True)[:20]
        
        for i, (hash_val, files) in enumerate(sorted_groups, 1):
            preserved = preservation_data['preservation_map'].get(hash_val, {})
            preserved_path = preserved.get('preserved_path', 'Unknown')
            
            f.write(f"#### Group {i}: {len(files)} duplicates\n\n")
            f.write(f"**Preserved:** `{preserved_path}`\n")
            f.write(f"**Reason:** {preserved.get('preserved_reason', 'N/A')}\n\n")
            f.write("**Duplicates to delete:**\n\n")
            f.write("| Path | Size (KB) | Location |\n")
            f.write("|------|-----------|----------|\n")
            for file_info in files[:10]:  # Top 10 per group
                size_kb = file_info['size'] / 1024
                f.write(f"| `{file_info['path'][:70]}` | {size_kb:.1f} | {file_info['location']} |\n")
            if len(files) > 10:
                f.write(f"| ... and {len(files) - 10} more | | |\n")
            f.write("\n")
        
        # Risk Assessment
        f.write("## Risk Assessment\n\n")
        f.write("### Low Risk (Safe to Delete)\n\n")
        f.write("- Files in ARCHIVE directories (backup copies)\n")
        f.write("- Files with preserved copies in CANONICAL directories\n")
        f.write("- Vendor/system files (excluded from analysis)\n\n")
        
        f.write("### Medium Risk (Review Before Deletion)\n\n")
        f.write("- Files in Desktop directories (may be active)\n")
        f.write("- Files in KmiDi subdirectories (may be referenced)\n")
        f.write("- Recently modified files (check if active)\n\n")
        
        f.write("### High Risk (Preserve)\n\n")
        f.write("- Files in CANONICAL directories (always preserved)\n")
        f.write("- Only copy of a file (never deleted)\n")
        f.write("- Files outside RECOVERY_OPS (never touched)\n\n")
        
        # Deletion Summary
        f.write("## Deletion Summary\n\n")
        f.write(f"Total files marked for deletion: **{stats['files_to_delete']:,}**\n\n")
        f.write("### By Location\n\n")
        f.write("| Location | Files to Delete | Space (MB) |\n")
        f.write("|----------|-----------------|------------|\n")
        
        location_sizes = defaultdict(int)
        for file_info in manifest['files']:
            location_sizes[file_info['location']] += file_info['size']
        
        for location in sorted(location_sizes.keys(), 
                              key=lambda x: location_sizes[x], reverse=True):
            count = stats['by_location_deletion'].get(location, 0)
            size_mb = location_sizes[location] / (1024 ** 2)
            f.write(f"| {location} | {count:,} | {size_mb:.2f} |\n")
        
        # Next Steps
        f.write("\n## Next Steps\n\n")
        f.write("1. **Review this report** - Verify preservation strategy\n")
        f.write("2. **Run verification** - `python3 scripts/recovery_ops/verify_dedupe_manifest.py`\n")
        f.write("3. **Run dry-run** - `python3 scripts/recovery_ops/dry_run_dedupe.py`\n")
        f.write("4. **Review dry-run results** - Check what would be deleted\n")
        f.write("5. **Execute deletion** - `python3 scripts/recovery_ops/safe_dedupe.py`\n\n")
        
        f.write("## Safety Features\n\n")
        f.write("- All deletions are logged\n")
        f.write("- Preservation targets verified before deletion\n")
        f.write("- Hash verification confirms duplicates\n")
        f.write("- Rollback mechanism available\n")
        f.write("- Dry-run available for testing\n\n")
    
    print(f"\n✅ Report generated: {report_path}")
    return report_path

if __name__ == "__main__":
    generate_dedupe_report()
