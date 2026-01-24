#!/usr/bin/env python3
"""
Detailed analysis of large files (1,002 files).
Categorize, assess value, and provide cleanup recommendations.
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

def categorize_large_file(file_info):
    """Categorize a large file by type and value."""
    path = file_info.get('path', '').lower()
    name = file_info.get('name', '').lower()
    ext = file_info.get('extension', '').lower()
    
    # Categories
    category = 'unknown'
    subcategory = None
    
    # Dataset files
    if any(x in path for x in ['dataset', 'data', 'training', 'train']):
        if ext in ['.tar', '.gz', '.zip', '.7z', '.rar']:
            category = 'dataset_archive'
        elif ext in ['.h5', '.hdf5', '.npz', '.npy']:
            category = 'dataset_binary'
        else:
            category = 'dataset'
    
    # Audio/media files
    elif ext in ['.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg']:
        category = 'audio'
        if 'sample' in path or 'example' in path:
            subcategory = 'sample'
        elif 'training' in path or 'train' in path:
            subcategory = 'training_data'
        else:
            subcategory = 'media'
    
    # Video files
    elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        category = 'video'
    
    # Archives
    elif ext in ['.tar', '.gz', '.zip', '.7z', '.rar', '.bz2']:
        category = 'archive'
        if 'backup' in path or 'backup' in name:
            subcategory = 'backup'
        elif 'download' in path:
            subcategory = 'download'
        else:
            subcategory = 'general'
    
    # Database files
    elif ext in ['.db', '.sqlite', '.sqlite3']:
        category = 'database'
    
    # Log files
    elif ext in ['.log'] or 'log' in name:
        category = 'log'
    
    # Build artifacts
    elif any(x in path for x in ['build/', 'dist/', 'target/', '.gradle', '.cargo']):
        category = 'build_artifact'
    
    # Vendor/system files
    elif any(x in path for x in ['site-packages', 'node_modules', '.venv', 'venv', '.cache']):
        category = 'vendor'
    
    # Model files (large ones)
    elif ext in ['.pt', '.pth', '.onnx', '.pb', '.h5', '.ckpt']:
        category = 'model'
    
    # KmiDi-related
    is_kmidi = any(k in path for k in ['kmidi', 'music', 'kelly', 'penta', 'brain'])
    
    # Value assessment
    value = 'low'
    if is_kmidi:
        value = 'high'
    elif category in ['dataset', 'dataset_archive', 'dataset_binary', 'model']:
        value = 'medium'
    elif category in ['vendor', 'build_artifact', 'log']:
        value = 'low'
    
    return {
        'category': category,
        'subcategory': subcategory,
        'is_kmidi_related': is_kmidi,
        'value': value
    }

def main():
    disco_dir = Path("docs/DISCOVERY")
    db_path = disco_dir / "discovery_database.json"
    
    print("=" * 80)
    print("Large Files Detailed Analysis")
    print("=" * 80)
    
    # Load database
    with open(db_path) as f:
        db = json.load(f)
    
    large_files = db.get('inventories', {}).get('large_files', [])
    print(f"\nAnalyzing {len(large_files):,} large files...\n")
    
    # Categorize and analyze
    categorized = []
    by_category = defaultdict(list)
    by_value = defaultdict(list)
    kmidi_related = []
    
    for file_info in large_files:
        analysis = categorize_large_file(file_info)
        file_analysis = {
            **file_info,
            **analysis
        }
        categorized.append(file_analysis)
        by_category[analysis['category']].append(file_analysis)
        by_value[analysis['value']].append(file_analysis)
        if analysis['is_kmidi_related']:
            kmidi_related.append(file_analysis)
    
    # Summary statistics
    print("Summary by Category:")
    for category in sorted(by_category.keys()):
        files = by_category[category]
        total_size = sum(f.get('size', 0) for f in files)
        total_size_gb = total_size / (1024**3)
        print(f"  {category}: {len(files):,} files, {total_size_gb:.2f} GB")
    
    print(f"\nSummary by Value:")
    for value in ['high', 'medium', 'low']:
        files = by_value[value]
        total_size = sum(f.get('size', 0) for f in files)
        total_size_gb = total_size / (1024**3)
        print(f"  {value}: {len(files):,} files, {total_size_gb:.2f} GB")
    
    print(f"\nKmiDi-related: {len(kmidi_related):,} files")
    
    # Top files by category
    top_by_category = {}
    for category, files in by_category.items():
        top_by_category[category] = sorted(files, key=lambda x: x.get('size', 0), reverse=True)[:20]
    
    # Save analysis
    output = {
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'total_files': len(large_files),
            'total_size_gb': sum(f.get('size', 0) for f in large_files) / (1024**3),
            'by_category': {k: len(v) for k, v in by_category.items()},
            'by_value': {k: len(v) for k, v in by_value.items()},
            'kmidi_related': len(kmidi_related)
        },
        'categorized_files': categorized[:500],  # Top 500
        'top_by_category': {k: [
            {
                'path': f.get('path'),
                'rel_path': f.get('rel_path'),
                'size_mb': round(f.get('size', 0) / (1024**2), 2),
                'value': f.get('value'),
                'is_kmidi_related': f.get('is_kmidi_related')
            }
            for f in files[:20]
        ] for k, files in top_by_category.items()}
    }
    
    output_path = disco_dir / "large_files_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Generate markdown report
    md_path = disco_dir / "large_files_analysis.md"
    with open(md_path, 'w') as f:
        f.write("# Large Files Detailed Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        
        total_size_gb = sum(file_info.get('size', 0) for file_info in large_files) / (1024**3)
        f.write(f"## Summary\n\n")
        f.write(f"- **Total files:** {len(large_files):,}\n")
        f.write(f"- **Total size:** {total_size_gb:.2f} GB\n")
        f.write(f"- **KmiDi-related:** {len(kmidi_related):,} files\n\n")
        
        f.write("## Files by Category\n\n")
        f.write("| Category | Count | Total Size (GB) |\n")
        f.write("|----------|-------|-----------------|\n")
        for category in sorted(by_category.keys()):
            files = by_category[category]
            total_size = sum(f.get('size', 0) for f in files)
            total_size_gb = total_size / (1024**3)
            f.write(f"| {category} | {len(files):,} | {total_size_gb:.2f} |\n")
        
        f.write("\n## Files by Value\n\n")
        f.write("| Value | Count | Total Size (GB) |\n")
        f.write("|-------|-------|-----------------|\n")
        for value in ['high', 'medium', 'low']:
            files = by_value[value]
            total_size = sum(f.get('size', 0) for f in files)
            total_size_gb = total_size / (1024**3)
            f.write(f"| {value} | {len(files):,} | {total_size_gb:.2f} |\n")
        
        f.write("\n## High-Value Files (KmiDi-Related)\n\n")
        f.write("| File | Size (GB) | Category | Path |\n")
        f.write("|------|-----------|----------|------|\n")
        for file_info in sorted(kmidi_related, key=lambda x: x.get('size', 0), reverse=True)[:50]:
            rel_path = file_info.get('rel_path', '')
            size_gb = file_info.get('size', 0) / (1024**3)
            category = file_info.get('category', 'unknown')
            f.write(f"| `{file_info.get('name', '')}` | {size_gb:.2f} | {category} | `{rel_path[:80]}` |\n")
        
        f.write("\n## Top Files by Category\n\n")
        for category in sorted(top_by_category.keys()):
            files = top_by_category[category]
            if not files:
                continue
            f.write(f"### {category.title()} (Top 20)\n\n")
            f.write("| File | Size (GB) | Value | Path |\n")
            f.write("|------|-----------|-------|------|\n")
            for file_info in files:
                rel_path = file_info.get('rel_path', '')
                size_gb = file_info.get('size', 0) / (1024**3)
                value = file_info.get('value', 'unknown')
                f.write(f"| `{file_info.get('name', '')}` | {size_gb:.2f} | {value} | `{rel_path[:70]}` |\n")
            f.write("\n")
        
        f.write("## Cleanup Recommendations\n\n")
        f.write("### Low Value (Consider Deletion)\n\n")
        low_value = sorted(by_value['low'], key=lambda x: x.get('size', 0), reverse=True)[:50]
        for file_info in low_value:
            rel_path = file_info.get('rel_path', '')
            size_gb = file_info.get('size', 0) / (1024**3)
            category = file_info.get('category', 'unknown')
            f.write(f"- `{rel_path}` ({size_gb:.2f} GB, {category})\n")
    
    print(f"\n✅ Analysis saved to:")
    print(f"   - {output_path}")
    print(f"   - {md_path}")

if __name__ == "__main__":
    main()
