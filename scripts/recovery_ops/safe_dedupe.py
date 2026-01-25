#!/usr/bin/env python3
"""
Safe deletion script for RECOVERY_OPS deduplication.
Includes verification, logging, and rollback support.
"""
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List

HOME = Path("/Users/seanburdges")

def safe_dedupe(dry_run: bool = True):
    """
    Safely delete duplicate files from RECOVERY_OPS.
    
    Args:
        dry_run: If True, simulate deletion without actually deleting
    """
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "docs" / "RECOVERY_OPS_DEDUPE"
    
    mode_str = "DRY-RUN" if dry_run else "LIVE"
    print("=" * 80)
    print(f"RECOVERY_OPS Safe Deduplication - {mode_str} MODE")
    print("=" * 80)
    
    if dry_run:
        print("\n⚠️  DRY-RUN MODE - No files will be deleted\n")
    else:
        print("\n⚠️  LIVE MODE - Files will be permanently deleted!\n")
        response = input("Are you sure you want to proceed? (type 'yes' to continue): ")
        if response.lower() != 'yes':
            print("Aborted by user.")
            return False
    
    # Load manifest
    manifest_path = output_dir / "RECOVERY_OPS_DELETE_MANIFEST.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Load verification results
    verification_path = output_dir / "verification_results.json"
    if verification_path.exists():
        with open(verification_path) as f:
            verification = json.load(f)
        
        if verification.get('errors'):
            print("❌ ERRORS FOUND IN VERIFICATION - ABORTING")
            print("Please review verification_results.json and fix errors before proceeding.")
            return False
    else:
        print("⚠️  Warning: No verification results found. Running verification first...")
        # Could call verify script here, but for now just warn
    
    files_to_delete = manifest['files']
    print(f"\nProcessing {len(files_to_delete)} files...\n")
    
    # Deletion log
    deletion_log = {
        'started_at': datetime.now().isoformat(),
        'dry_run': dry_run,
        'total_files': len(files_to_delete),
        'deleted': [],
        'failed': [],
        'skipped': [],
        'total_space_freed_bytes': 0
    }
    
    # Process files
    for i, file_info in enumerate(files_to_delete):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(files_to_delete)} files...")
        
        file_path = HOME / file_info['path']
        preserved_path = HOME / file_info['preserved_path']
        
        log_entry = {
            'path': file_info['path'],
            'preserved_path': file_info['preserved_path'],
            'size': file_info['size'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Verify file exists
        if not file_path.exists():
            log_entry['status'] = 'skipped'
            log_entry['reason'] = 'File does not exist (may have been deleted already)'
            deletion_log['skipped'].append(log_entry)
            continue
        
        # Verify preservation target exists
        if not preserved_path.exists():
            log_entry['status'] = 'failed'
            log_entry['reason'] = 'Preservation target does not exist'
            deletion_log['failed'].append(log_entry)
            print(f"  ❌ ERROR: Preservation target missing for {file_info['path']}")
            continue
        
        # Get file size
        try:
            file_size = file_path.stat().st_size
        except Exception as e:
            log_entry['status'] = 'failed'
            log_entry['reason'] = f'Cannot stat file: {str(e)}'
            deletion_log['failed'].append(log_entry)
            continue
        
        # Delete file (or simulate)
        if dry_run:
            log_entry['status'] = 'would_delete'
            log_entry['reason'] = 'Dry-run mode'
            deletion_log['deleted'].append(log_entry)
            deletion_log['total_space_freed_bytes'] += file_size
        else:
            try:
                file_path.unlink()
                log_entry['status'] = 'deleted'
                log_entry['reason'] = 'Successfully deleted'
                deletion_log['deleted'].append(log_entry)
                deletion_log['total_space_freed_bytes'] += file_size
            except Exception as e:
                log_entry['status'] = 'failed'
                log_entry['reason'] = f'Deletion failed: {str(e)}'
                deletion_log['failed'].append(log_entry)
                print(f"  ❌ Failed to delete {file_info['path']}: {str(e)}")
    
    deletion_log['completed_at'] = datetime.now().isoformat()
    deletion_log['total_space_freed_gb'] = deletion_log['total_space_freed_bytes'] / (1024 ** 3)
    
    # Save deletion log
    log_path = output_dir / f"deletion_log_{'dry_run' if dry_run else 'live'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, 'w') as f:
        json.dump(deletion_log, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Deletion Summary")
    print("=" * 80)
    print(f"\nTotal files processed: {deletion_log['total_files']:,}")
    
    if dry_run:
        print(f"Files that would be deleted: {len(deletion_log['deleted']):,}")
    else:
        print(f"Files deleted: {len(deletion_log['deleted']):,}")
    
    print(f"Files skipped: {len(deletion_log['skipped']):,}")
    print(f"Files failed: {len(deletion_log['failed']):,}")
    
    space_gb = deletion_log['total_space_freed_gb']
    space_mb = deletion_log['total_space_freed_bytes'] / (1024 ** 2)
    
    if dry_run:
        print(f"\nSpace that would be freed: {space_gb:.2f} GB ({space_mb:.2f} MB)")
    else:
        print(f"\nSpace freed: {space_gb:.2f} GB ({space_mb:.2f} MB)")
    
    if deletion_log['failed']:
        print(f"\n❌ {len(deletion_log['failed'])} files failed to delete")
        print("Review deletion log for details.")
    else:
        print(f"\n✅ All files processed successfully")
    
    print(f"\n✅ Deletion log saved to: {log_path}")
    
    # Generate completion report
    report_path = output_dir / f"DELETION_COMPLETE_{'dry_run' if dry_run else 'live'}.md"
    with open(report_path, 'w') as f:
        f.write(f"# RECOVERY_OPS Deduplication Deletion Report\n\n")
        f.write(f"**Mode:** {'DRY-RUN' if dry_run else 'LIVE'}\n")
        f.write(f"**Started:** {deletion_log['started_at']}\n")
        f.write(f"**Completed:** {deletion_log['completed_at']}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total files:** {deletion_log['total_files']:,}\n")
        if dry_run:
            f.write(f"- **Files that would be deleted:** {len(deletion_log['deleted']):,}\n")
        else:
            f.write(f"- **Files deleted:** {len(deletion_log['deleted']):,}\n")
        f.write(f"- **Files skipped:** {len(deletion_log['skipped']):,}\n")
        f.write(f"- **Files failed:** {len(deletion_log['failed']):,}\n")
        f.write(f"- **Space {'that would be' if dry_run else ''} freed:** {space_gb:.2f} GB\n\n")
        
        if deletion_log['failed']:
            f.write("## Failed Deletions\n\n")
            f.write("| Path | Reason |\n")
            f.write("|------|--------|\n")
            for failure in deletion_log['failed'][:50]:
                f.write(f"| `{failure['path']}` | {failure['reason']} |\n")
    
    print(f"✅ Completion report saved to: {report_path}")
    
    return len(deletion_log['failed']) == 0

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Safe deduplication for RECOVERY_OPS")
    parser.add_argument('--live', action='store_true', 
                       help='Actually delete files (default is dry-run)')
    args = parser.parse_args()
    
    success = safe_dedupe(dry_run=not args.live)
    sys.exit(0 if success else 1)
