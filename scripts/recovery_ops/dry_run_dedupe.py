#!/usr/bin/env python3
"""
Dry-run deletion simulation for RECOVERY_OPS deduplication.
Shows what would be deleted without actually deleting anything.
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

HOME = Path("/Users/seanburdges")

def dry_run_dedupe():
    """Simulate deletion without actually deleting."""
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "docs" / "RECOVERY_OPS_DEDUPE"
    
    print("=" * 80)
    print("RECOVERY_OPS Deduplication Dry-Run")
    print("=" * 80)
    print("\n⚠️  DRY-RUN MODE - No files will be deleted\n")
    
    # Load manifest
    manifest_path = output_dir / "RECOVERY_OPS_DELETE_MANIFEST.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    files_to_delete = manifest['files']
    
    print(f"Simulating deletion of {len(files_to_delete)} files...\n")
    
    # Statistics
    stats = {
        'total_files': len(files_to_delete),
        'total_size_bytes': 0,
        'total_size_gb': 0,
        'by_location': defaultdict(lambda: {'count': 0, 'size': 0}),
        'by_extension': defaultdict(lambda: {'count': 0, 'size': 0}),
        'files_that_would_be_deleted': []
    }
    
    # Simulate deletion
    for i, file_info in enumerate(files_to_delete):
        file_path = HOME / file_info['path']
        
        # Check if file exists
        if not file_path.exists():
            continue
        
        # Get actual file size
        try:
            file_size = file_path.stat().st_size
        except Exception:
            file_size = file_info.get('size', 0)
        
        stats['total_size_bytes'] += file_size
        stats['by_location'][file_info['location']]['count'] += 1
        stats['by_location'][file_info['location']]['size'] += file_size
        
        # Extension
        ext = Path(file_info['path']).suffix.lower() or 'no_extension'
        stats['by_extension'][ext]['count'] += 1
        stats['by_extension'][ext]['size'] += file_size
        
        # Track file
        stats['files_that_would_be_deleted'].append({
            'path': file_info['path'],
            'size': file_size,
            'location': file_info['location'],
            'preserved_path': file_info['preserved_path']
        })
    
    stats['total_size_gb'] = stats['total_size_bytes'] / (1024 ** 3)
    stats['by_location'] = {
        k: {'count': v['count'], 'size_mb': v['size'] / (1024 ** 2)}
        for k, v in stats['by_location'].items()
    }
    stats['by_extension'] = {
        k: {'count': v['count'], 'size_mb': v['size'] / (1024 ** 2)}
        for k, v in stats['by_extension'].items()
    }
    
    # Print summary
    print("=" * 80)
    print("Dry-Run Summary")
    print("=" * 80)
    print(f"\nFiles that would be deleted: {stats['total_files']:,}")
    print(f"Space that would be freed: {stats['total_size_gb']:.2f} GB")
    print(f"                      ({stats['total_size_bytes'] / (1024 ** 2):.2f} MB)\n")
    
    print("By Location:")
    print("-" * 80)
    for location in sorted(stats['by_location'].keys(), 
                          key=lambda x: stats['by_location'][x]['size_mb'], reverse=True):
        loc_stats = stats['by_location'][location]
        print(f"  {location:15} {loc_stats['count']:5,} files  {loc_stats['size_mb']:8.2f} MB")
    
    print("\nBy File Extension (Top 10):")
    print("-" * 80)
    sorted_exts = sorted(stats['by_extension'].items(), 
                        key=lambda x: x[1]['size_mb'], reverse=True)[:10]
    for ext, ext_stats in sorted_exts:
        ext_display = ext if ext != 'no_extension' else '(no ext)'
        print(f"  {ext_display:15} {ext_stats['count']:5,} files  {ext_stats['size_mb']:8.2f} MB")
    
    # Sample files
    print("\n" + "=" * 80)
    print("Sample Files That Would Be Deleted (First 20)")
    print("=" * 80)
    print("\n| Path | Size (KB) | Location | Preserved In |\n")
    print("|------|-----------|----------|--------------|")
    for file_info in stats['files_that_would_be_deleted'][:20]:
        size_kb = file_info['size'] / 1024
        path_display = file_info['path'][:60] + '...' if len(file_info['path']) > 60 else file_info['path']
        preserved_display = file_info['preserved_path'][:40] + '...' if len(file_info['preserved_path']) > 40 else file_info['preserved_path']
        print(f"| `{path_display}` | {size_kb:.1f} | {file_info['location']} | `{preserved_display}` |")
    
    # Save dry-run results
    dry_run_path = output_dir / "dry_run_results.json"
    stats['generated_at'] = datetime.now().isoformat()
    stats['dry_run'] = True
    
    with open(dry_run_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Generate markdown report
    report_path = output_dir / "DRY_RUN_REPORT.md"
    with open(report_path, 'w') as f:
        f.write("# RECOVERY_OPS Deduplication Dry-Run Report\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write("⚠️ **DRY-RUN MODE** - No files were actually deleted\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Files that would be deleted:** {stats['total_files']:,}\n")
        f.write(f"- **Space that would be freed:** {stats['total_size_gb']:.2f} GB\n")
        f.write(f"- **Space that would be freed:** {stats['total_size_bytes'] / (1024 ** 2):.2f} MB\n\n")
        
        f.write("## By Location\n\n")
        f.write("| Location | Files | Size (MB) |\n")
        f.write("|----------|-------|-----------|\n")
        for location in sorted(stats['by_location'].keys(), 
                              key=lambda x: stats['by_location'][x]['size_mb'], reverse=True):
            loc_stats = stats['by_location'][location]
            f.write(f"| {location} | {loc_stats['count']:,} | {loc_stats['size_mb']:.2f} |\n")
        
        f.write("\n## By File Extension (Top 20)\n\n")
        f.write("| Extension | Files | Size (MB) |\n")
        f.write("|-----------|-------|-----------|\n")
        for ext, ext_stats in sorted_exts[:20]:
            ext_display = ext if ext != 'no_extension' else '(no ext)'
            f.write(f"| {ext_display} | {ext_stats['count']:,} | {ext_stats['size_mb']:.2f} |\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("1. Review this dry-run report\n")
        f.write("2. Verify the files listed are safe to delete\n")
        f.write("3. If satisfied, run: `python3 scripts/recovery_ops/safe_dedupe.py`\n\n")
    
    print(f"\n✅ Dry-run complete!")
    print(f"   - Results saved to: {dry_run_path}")
    print(f"   - Report saved to: {report_path}")
    print(f"\n⚠️  No files were actually deleted (dry-run mode)")

if __name__ == "__main__":
    dry_run_dedupe()
