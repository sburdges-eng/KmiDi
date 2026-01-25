#!/usr/bin/env python3
"""
Analyze KmiDi-related duplicates in RECOVERY_OPS directory.
Identifies duplicates, categorizes by location, and determines preservation priority.
"""
import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

KMIDI_KEYWORDS = ['kmidi', 'music', 'kelly', 'penta', 'brain']
VENDOR_EXCLUSIONS = ['venv', 'site-packages', 'node_modules', '.cache', 'build', 'dist', 
                     '.gradle', '.cargo', '.rustup', '.npm', '__pycache__', '.pytest_cache']

def is_vendor_path(path: str) -> bool:
    """Check if path is in a vendor/system directory."""
    path_lower = path.lower()
    return any(exclusion in path_lower for exclusion in VENDOR_EXCLUSIONS)

def is_kmidi_related(path: str) -> bool:
    """Check if path is KmiDi-related."""
    path_lower = path.lower()
    return any(keyword in path_lower for keyword in KMIDI_KEYWORDS)

def get_location_category(path: str) -> str:
    """Categorize path by location in RECOVERY_OPS."""
    if '/CANONICAL/' in path:
        return 'CANONICAL'
    elif '/ARCHIVE/' in path:
        return 'ARCHIVE'
    elif '/Desktop/' in path:
        return 'Desktop'
    elif '/KmiDi/' in path:
        return 'KmiDi'
    elif '/sbdrive/' in path:
        return 'sbdrive'
    elif '/Downloads/' in path:
        return 'Downloads'
    else:
        return 'Other'

def preservation_priority(path: str, modified: Optional[str] = None) -> Tuple[int, str]:
    """
    Calculate preservation priority.
    Returns (priority_score, reason) where lower score = higher priority.
    """
    path_lower = path.lower()
    
    # Priority 1: CANONICAL directories
    if '/canonical/' in path_lower:
        return (1, "CANONICAL directory")
    
    # Priority 2: Most recent (will be sorted separately)
    # Priority 3: Shortest path (less nested)
    depth = path.count('/')
    
    # Priority 4: NOT in ARCHIVE
    if '/archive/' in path_lower:
        return (400 + depth, "ARCHIVE directory")
    
    # Priority 3: Shortest path for non-archive
    return (100 + depth, f"Non-archive, depth={depth}")

def analyze_recovery_dupes():
    """Main analysis function."""
    repo_root = Path(__file__).resolve().parents[2]
    disco_dir = repo_root / "docs" / "DISCOVERY"
    output_dir = repo_root / "docs" / "RECOVERY_OPS_DEDUPE"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("RECOVERY_OPS KmiDi-Related Duplicate Analysis")
    print("=" * 80)
    
    # Load duplicate data
    print("\n1. Loading duplicate data...")
    with open(disco_dir / "content_analysis.json") as f:
        content_data = json.load(f)
    
    duplicates = content_data.get('duplicates', {})
    print(f"   Loaded {len(duplicates)} duplicate groups")
    
    # Load file metadata from discovery database
    print("\n2. Loading file metadata...")
    with open(disco_dir / "discovery_database.json") as f:
        db_data = json.load(f)
    
    # Build file metadata lookup
    file_metadata = {}
    for category in ['code', 'docs', 'models']:
        inventory = db_data.get('inventories', {}).get(category, {})
        for file_list in inventory.values():
            if isinstance(file_list, list):
                for file_info in file_list:
                    rel_path = file_info.get('rel_path', '')
                    if rel_path:
                        file_metadata[rel_path] = file_info
    
    print(f"   Loaded metadata for {len(file_metadata)} files")
    
    # Filter to RECOVERY_OPS KmiDi-related duplicates
    print("\n3. Filtering to RECOVERY_OPS KmiDi-related duplicates...")
    recovery_kmidi_dupes = {}
    
    for hash_val, paths in duplicates.items():
        # Filter to RECOVERY_OPS paths
        recovery_paths = [p for p in paths if p.startswith('RECOVERY_OPS/')]
        if not recovery_paths:
            continue
        
        # Filter to KmiDi-related and exclude vendor
        kmidi_paths = []
        for p in recovery_paths:
            if is_kmidi_related(p) and not is_vendor_path(p):
                kmidi_paths.append(p)
        
        if kmidi_paths:
            recovery_kmidi_dupes[hash_val] = kmidi_paths
    
    print(f"   Found {len(recovery_kmidi_dupes)} KmiDi-related duplicate groups")
    total_files = sum(len(paths) for paths in recovery_kmidi_dupes.values())
    print(f"   Total duplicate files: {total_files}")
    
    # Categorize by location
    print("\n4. Categorizing by location...")
    location_counts = defaultdict(int)
    location_groups = defaultdict(list)
    
    for hash_val, paths in recovery_kmidi_dupes.items():
        for path in paths:
            location = get_location_category(path)
            location_counts[location] += 1
        location_groups[hash_val] = paths
    
    print("   By location:")
    for loc in sorted(location_counts.keys(), key=lambda x: location_counts[x], reverse=True):
        print(f"     {loc}: {location_counts[loc]} files")
    
    # Determine preservation priority
    print("\n5. Determining preservation priority...")
    preservation_map = {}
    deletion_candidates = []
    
    for hash_val, paths in recovery_kmidi_dupes.items():
        # Get metadata for all paths
        paths_with_meta = []
        for path in paths:
            meta = file_metadata.get(path, {})
            priority_score, reason = preservation_priority(path, meta.get('modified'))
            paths_with_meta.append({
                'path': path,
                'priority_score': priority_score,
                'reason': reason,
                'modified': meta.get('modified', ''),
                'size': meta.get('size', 0),
                'location': get_location_category(path)
            })
        
        # Sort by priority (lower score = higher priority)
        paths_with_meta.sort(key=lambda x: (x['priority_score'], x['modified']), reverse=True)
        
        # First item is preserved, rest are deletion candidates
        preserved = paths_with_meta[0]
        preservation_map[hash_val] = {
            'preserved_path': preserved['path'],
            'preserved_reason': preserved['reason'],
            'preserved_modified': preserved['modified'],
            'preserved_size': preserved['size'],
            'duplicate_count': len(paths_with_meta)
        }
        
        # Mark others for deletion
        for dup in paths_with_meta[1:]:
            deletion_candidates.append({
                'hash': hash_val,
                'path': dup['path'],
                'preserved_path': preserved['path'],
                'preserved_reason': preserved['reason'],
                'size': dup['size'],
                'modified': dup['modified'],
                'location': dup['location'],
                'deletion_reason': f"Duplicate of preserved file (priority: {preserved['reason']})"
            })
    
    print(f"   Files to preserve: {len(preservation_map)}")
    print(f"   Files to delete: {len(deletion_candidates)}")
    
    # Calculate space savings
    total_size_to_delete = sum(c['size'] for c in deletion_candidates)
    size_gb = total_size_to_delete / (1024 ** 3)
    print(f"\n6. Space savings estimate: {size_gb:.2f} GB")
    
    # Save results
    print("\n7. Saving results...")
    
    # Preservation map
    preservation_output = {
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'duplicate_groups': len(preservation_map),
            'total_duplicate_files': total_files,
            'files_to_preserve': len(preservation_map),
            'files_to_delete': len(deletion_candidates),
            'estimated_space_savings_gb': round(size_gb, 2),
            'location_breakdown': dict(location_counts)
        },
        'preservation_map': preservation_map
    }
    
    with open(output_dir / "preservation_map.json", 'w') as f:
        json.dump(preservation_output, f, indent=2)
    
    # Deletion manifest
    deletion_manifest = {
        'generated_at': datetime.now().isoformat(),
        'total_files': len(deletion_candidates),
        'estimated_space_savings_bytes': total_size_to_delete,
        'estimated_space_savings_gb': round(size_gb, 2),
        'files': deletion_candidates
    }
    
    with open(output_dir / "RECOVERY_OPS_DELETE_MANIFEST.json", 'w') as f:
        json.dump(deletion_manifest, f, indent=2)
    
    # Statistics
    stats = {
        'generated_at': datetime.now().isoformat(),
        'duplicate_groups': len(recovery_kmidi_dupes),
        'total_duplicate_files': total_files,
        'files_to_preserve': len(preservation_map),
        'files_to_delete': len(deletion_candidates),
        'space_savings': {
            'bytes': total_size_to_delete,
            'gb': round(size_gb, 2),
            'mb': round(total_size_to_delete / (1024 ** 2), 2)
        },
        'by_location': dict(location_counts),
        'by_location_deletion': defaultdict(int)
    }
    
    for candidate in deletion_candidates:
        stats['by_location_deletion'][candidate['location']] += 1
    
    stats['by_location_deletion'] = dict(stats['by_location_deletion'])
    
    with open(output_dir / "dedupe_statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Analysis complete!")
    print(f"   - Preservation map: {output_dir / 'preservation_map.json'}")
    print(f"   - Deletion manifest: {output_dir / 'RECOVERY_OPS_DELETE_MANIFEST.json'}")
    print(f"   - Statistics: {output_dir / 'dedupe_statistics.json'}")
    
    return preservation_map, deletion_candidates, stats

if __name__ == "__main__":
    analyze_recovery_dupes()
