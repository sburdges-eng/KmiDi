#!/usr/bin/env python3
"""
Verify deduplication manifest before deletion.
Checks file existence, hash verification, and preservation targets.
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

HOME = Path("/Users/seanburdges")

def sha256_file(file_path: Path, max_bytes: int = 5 * 1024 * 1024) -> str:
    """Calculate SHA-256 hash of file."""
    try:
        hash_obj = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                hash_obj.update(chunk)
                if hash_obj.digest_size * 2 > max_bytes:
                    break
        return hash_obj.hexdigest()
    except Exception as e:
        return f"ERROR: {str(e)}"

def verify_dedupe_manifest():
    """Verify the deletion manifest."""
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "docs" / "RECOVERY_OPS_DEDUPE"
    
    print("=" * 80)
    print("Verifying RECOVERY_OPS Deduplication Manifest")
    print("=" * 80)
    
    # Load manifest
    manifest_path = output_dir / "RECOVERY_OPS_DELETE_MANIFEST.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    preservation_path = output_dir / "preservation_map.json"
    with open(preservation_path) as f:
        preservation_data = json.load(f)
    
    files_to_delete = manifest['files']
    print(f"\nVerifying {len(files_to_delete)} files marked for deletion...\n")
    
    # Verification results
    results = {
        'generated_at': datetime.now().isoformat(),
        'total_files': len(files_to_delete),
        'verified_ok': 0,
        'errors': [],
        'warnings': []
    }
    
    verified_files = []
    
    for i, file_info in enumerate(files_to_delete):
        if (i + 1) % 100 == 0:
            print(f"  Verified {i + 1}/{len(files_to_delete)} files...")
        
        file_path = HOME / file_info['path']
        preserved_path = HOME / file_info['preserved_path']
        hash_val = file_info['hash']
        
        verification = {
            'path': file_info['path'],
            'preserved_path': file_info['preserved_path'],
            'status': 'pending',
            'errors': [],
            'warnings': []
        }
        
        # Check 1: File exists
        if not file_path.exists():
            verification['status'] = 'error'
            verification['errors'].append(f"File does not exist: {file_path}")
            results['errors'].append(verification)
            continue
        
        # Check 2: Preservation target exists
        if not preserved_path.exists():
            verification['status'] = 'error'
            verification['errors'].append(f"Preservation target does not exist: {preserved_path}")
            results['errors'].append(verification)
            continue
        
        # Check 3: Preservation target is not marked for deletion
        preserved_in_delete_list = any(
            f['path'] == file_info['preserved_path'] 
            for f in files_to_delete
        )
        if preserved_in_delete_list:
            verification['status'] = 'error'
            verification['errors'].append(f"Preservation target is also marked for deletion!")
            results['errors'].append(verification)
            continue
        
        # Check 4: File is actually a duplicate (hash verification)
        try:
            file_hash = sha256_file(file_path)
            preserved_hash = sha256_file(preserved_path)
            
            if file_hash != preserved_hash:
                verification['status'] = 'warning'
                verification['warnings'].append(
                    f"Hash mismatch: file hash={file_hash[:16]}..., "
                    f"preserved hash={preserved_hash[:16]}..."
                )
                results['warnings'].append(verification)
            else:
                verification['status'] = 'ok'
                verification['file_hash'] = file_hash[:16] + '...'
                results['verified_ok'] += 1
        except Exception as e:
            verification['status'] = 'warning'
            verification['warnings'].append(f"Hash verification failed: {str(e)}")
            results['warnings'].append(verification)
        
        # Check 5: File size matches
        try:
            file_size = file_path.stat().st_size
            preserved_size = preserved_path.stat().st_size
            if file_size != preserved_size:
                verification['warnings'].append(
                    f"Size mismatch: {file_size} vs {preserved_size} bytes"
                )
        except Exception as e:
            verification['warnings'].append(f"Size check failed: {str(e)}")
        
        verified_files.append(verification)
    
    results['verified_files'] = verified_files
    
    # Save verification results
    verification_path = output_dir / "verification_results.json"
    with open(verification_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("Verification Summary")
    print("=" * 80)
    print(f"Total files: {results['total_files']}")
    print(f"✅ Verified OK: {results['verified_ok']}")
    print(f"⚠️  Warnings: {len(results['warnings'])}")
    print(f"❌ Errors: {len(results['errors'])}")
    
    if results['errors']:
        print(f"\n❌ ERRORS FOUND - DO NOT PROCEED WITH DELETION")
        print("\nFirst 10 errors:")
        for error in results['errors'][:10]:
            print(f"  - {error['path']}")
            for err_msg in error['errors']:
                print(f"    {err_msg}")
    else:
        print(f"\n✅ No critical errors found")
    
    if results['warnings']:
        print(f"\n⚠️  WARNINGS:")
        print(f"First 10 warnings:")
        for warning in results['warnings'][:10]:
            print(f"  - {warning['path']}")
            for warn_msg in warning['warnings']:
                print(f"    {warn_msg}")
    
    print(f"\n✅ Verification results saved to: {verification_path}")
    
    return len(results['errors']) == 0

if __name__ == "__main__":
    import sys
    success = verify_dedupe_manifest()
    sys.exit(0 if success else 1)
